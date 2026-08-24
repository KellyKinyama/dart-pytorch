/// AirLLM-style per-expert streaming for a single MoE FFN layer,
/// factored out of the demo bins so the same code powers both the
/// synthetic-weights runner ([bin/moe_streaming_demo.dart]) and the
/// real-weights runners
/// ([bin/deepseek_v2_moe_layer_stream.dart], [bin/qwen15_moe_layer_stream.dart]).
///
/// Given a [ShardedSafeTensorsReader] and the HF key prefix for the
/// layer's `mlp` submodule (e.g. `"model.layers.5.mlp"`), it can:
///
///   * Load the router weight (`gate.weight`) with the right
///     transposition for `x @ router`.
///   * Run the router on an input batch and return **only** the
///     union of top-K expert indices — no side effects on any
///     module's expert-load counters or routing bias.
///   * Load a single routed expert's SwiGLU triplet
///     (`experts.{j}.{gate,up,down}_proj.weight`) in fp16.
///   * Optionally load a fused shared expert (DeepSeek-V2 style:
///     `shared_experts.*`) or a scalar-gated shared expert
///     (Qwen2-MoE style: `shared_expert.*` + `shared_expert_gate.weight`).
///   * Report `bytes_saved` vs. loading the full routed set.
///
/// Handles both `GateFunction.softmax` (DeepSeek-V2, Qwen2-MoE) and
/// `GateFunction.sigmoid` (DeepSeek-V3), with optional top-K
/// renormalization.
library;

import 'dart:math' as math;

import '../tensor/tensor.dart';
import 'moe.dart' show GateFunction;
import 'safetensors_reader.dart';

/// Bytes-plan of a per-expert streaming forward — how many experts
/// would be streamed and how many bytes vs. loading the full set.
class MoeStreamingReport {
  final int numExperts;
  final int topK;
  final int streamedExperts;
  final int expertBytes;
  const MoeStreamingReport({
    required this.numExperts,
    required this.topK,
    required this.streamedExperts,
    required this.expertBytes,
  });
  int get streamedBytes => streamedExperts * expertBytes;
  int get totalBytes => numExperts * expertBytes;
  int get savedBytes => totalBytes - streamedBytes;
  double get savedPercent => 100.0 * savedBytes / totalBytes;
}

/// Weights of one routed expert loaded from disk. Shapes follow the
/// HF layout: `wGate`, `wUp` are `[hidden, D]`; `wDown` is
/// `[D, hidden]`. All fp16-preserved if the source is fp16.
class RoutedExpertWeights {
  final Tensor wGate;
  final Tensor wUp;
  final Tensor wDown;
  const RoutedExpertWeights(this.wGate, this.wUp, this.wDown);
}

class MoeStreamingLayer {
  /// HF key prefix ending in `.mlp` — e.g. `"model.layers.5.mlp"`.
  final String prefix;

  /// Total routed experts (`numRoutedExperts`).
  final int numExperts;

  /// Top-K experts picked per token by the router.
  final int topK;

  /// Router gate function (Softmax for V2 / Qwen; sigmoid for V3).
  final GateFunction gate;

  /// If true, top-K weights are row-normalized so they sum to 1.
  /// V3 default; V2 / Qwen leave this false.
  final bool renormalize;

  /// Streaming source.
  final ShardedSafeTensorsReader reader;

  MoeStreamingLayer({
    required this.prefix,
    required this.numExperts,
    required this.topK,
    required this.reader,
    this.gate = GateFunction.softmax,
    this.renormalize = false,
  });

  String get routerKey => '$prefix.gate.weight';
  String routedExpertGateKey(int j) => '$prefix.experts.$j.gate_proj.weight';
  String routedExpertUpKey(int j) => '$prefix.experts.$j.up_proj.weight';
  String routedExpertDownKey(int j) => '$prefix.experts.$j.down_proj.weight';

  /// Read the router weight and return it transposed to `[D, E]` so
  /// `x @ routerW` computes gate logits directly.
  ///
  /// HF stores `gate.weight` as `[E, D]`. If [keepFp16] is true the
  /// transpose is done on the fp32-promoted values (safetensors
  /// preserves fp16 only for direct reads, not through
  /// [Tensor.transpose]). For the router that's cheap — E is small.
  Tensor loadRouter({bool keepFp16 = false}) {
    final w = reader.readTensor(routerKey, keepFp16: keepFp16);
    return w.transpose();
  }

  /// Load one routed expert's SwiGLU triplet.
  RoutedExpertWeights loadRoutedExpert(int j, {bool keepFp16 = true}) =>
      RoutedExpertWeights(
        reader.readTensor(routedExpertGateKey(j), keepFp16: keepFp16),
        reader.readTensor(routedExpertUpKey(j), keepFp16: keepFp16),
        reader.readTensor(routedExpertDownKey(j), keepFp16: keepFp16),
      );

  /// Per-expert byte count on disk (sum of the 3 SwiGLU tensors).
  /// Cheap — reads header entries only.
  int expertBytes({int probeExpert = 0}) {
    int size(String name) {
      final entry = reader.entry(name);
      if (entry == null) {
        throw StateError('no such tensor: $name');
      }
      return entry.dataEnd - entry.dataStart;
    }

    return size(routedExpertGateKey(probeExpert)) +
        size(routedExpertUpKey(probeExpert)) +
        size(routedExpertDownKey(probeExpert));
  }

  /// Full routing pass. Returns:
  ///
  ///   * `flat`: raw gate scores `[T * E]` in row-major (post-softmax
  ///     / -sigmoid).
  ///   * `union`: set of expert indices any token in the batch picked
  ///     in its top-K. This is the set to stream.
  ///   * `topK`: per-token top-K expert indices `[T][K]`.
  ///   * `combineWeights`: per-expert per-token weight for the final
  ///     weighted sum. `[T * E]` in row-major, with zeros for
  ///     non-top-K entries. Optionally renormalized so the K
  ///     non-zero entries per row sum to 1.
  ///
  /// Given `routerW` `[D, E]` (from [loadRouter]) and input
  /// `x` `[T, D]`, one forward pass through this method is enough
  /// to plan the streaming for a whole forward through the MoE FFN.
  MoeRoutingDecision route(Tensor routerW, Tensor x) {
    if (x.shape.length != 2) {
      throw ArgumentError('MoeStreamingLayer.route: x must be [T, D]');
    }
    final t = x.shape[0];
    final e = numExperts;
    final k = topK < e ? topK : e;
    final logits = x.matmul(routerW); // [T, E]
    final scores =
        gate == GateFunction.softmax ? logits.softmax() : logits.sigmoid();
    final flat = scores.toList();

    final union = <int>{};
    final perTokenTopK = List<List<int>>.generate(t, (_) => <int>[]);
    final combine = List<double>.filled(t * e, 0.0);
    for (int i = 0; i < t; i++) {
      final indexed = List<MapEntry<int, double>>.generate(
        e,
        (j) => MapEntry(j, flat[i * e + j]),
      );
      indexed.sort((a, b) => b.value.compareTo(a.value));
      double rowSum = 0.0;
      for (int r = 0; r < k; r++) {
        final j = indexed[r].key;
        final w = flat[i * e + j];
        combine[i * e + j] = w;
        rowSum += w;
        perTokenTopK[i].add(j);
        union.add(j);
      }
      if (renormalize && rowSum > 0) {
        for (int r = 0; r < k; r++) {
          final j = perTokenTopK[i][r];
          combine[i * e + j] /= rowSum;
        }
      }
    }
    return MoeRoutingDecision(
      scores: flat,
      union: union,
      topKPerToken: perTokenTopK,
      combineWeights: combine,
      numExperts: e,
      topK: k,
      tokens: t,
    );
  }

  /// Bytes plan given a [MoeRoutingDecision].
  MoeStreamingReport report(MoeRoutingDecision decision, {int? bytesPerExpert}) =>
      MoeStreamingReport(
        numExperts: numExperts,
        topK: topK,
        streamedExperts: decision.union.length,
        expertBytes: bytesPerExpert ?? expertBytes(),
      );
}

class MoeRoutingDecision {
  /// Raw router scores in `[T * E]` row-major order (post
  /// softmax/sigmoid).
  final List<double> scores;

  /// Union of expert indices any token in the batch chose in its
  /// top-K. Streaming loads exactly these experts.
  final Set<int> union;

  /// Per-token top-K expert indices, `topKPerToken[t]` has length K.
  final List<List<int>> topKPerToken;

  /// Per-token per-expert combining weight, `[T * E]` row-major.
  /// Non-top-K entries are zero. Optionally renormalized so each
  /// row's top-K weights sum to 1.
  final List<double> combineWeights;

  final int numExperts;
  final int topK;
  final int tokens;

  const MoeRoutingDecision({
    required this.scores,
    required this.union,
    required this.topKPerToken,
    required this.combineWeights,
    required this.numExperts,
    required this.topK,
    required this.tokens,
  });

  /// Deterministic sorted view of [union] for reporting.
  List<int> get sortedUnion => union.toList()..sort();

  /// Per-token combining weight for expert [j] as a length-T list.
  List<double> weightForExpert(int j) {
    final out = List<double>.filled(tokens, 0.0);
    for (int i = 0; i < tokens; i++) {
      out[i] = combineWeights[i * numExperts + j];
    }
    return out;
  }
}

/// Utility: SwiGLU forward for a single expert given fp16-or-fp32
/// weights in the HF layout. `x` is `[T, D]`; weights are `wGate`,
/// `wUp` `[hidden, D]` and `wDown` `[D, hidden]`. All matmuls are
/// done as `x @ w.transpose()`.
Tensor swiGluForward(Tensor x, RoutedExpertWeights w) {
  final gate = x.matmul(w.wGate.transpose());
  final up = x.matmul(w.wUp.transpose());
  final act = gate * gate.sigmoid();
  return (act * up).matmul(w.wDown.transpose());
}

/// Same, but taking three raw tensors (useful when the shared expert
/// weights come from a different HF prefix).
Tensor swiGluForwardRaw(Tensor x, Tensor gate, Tensor up, Tensor down) {
  final g = x.matmul(gate.transpose());
  final u = x.matmul(up.transpose());
  final act = g * g.sigmoid();
  return (act * u).matmul(down.transpose());
}

// `math` is used indirectly via `Tensor.softmax()` — keep the import
// so future extensions (temperature scaling, gumbel top-K, etc.) can
// use `math.log` etc. without re-importing.
// ignore: unused_element
math.Random _rngPlaceholder() => math.Random(0);
