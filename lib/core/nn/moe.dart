/// Mixture-of-Experts feed-forward block.
///
/// DeepSeek-V3-style routing: for each token the router picks the top-K
/// of `numRoutedExperts` "sparse" experts and combines their outputs
/// weighted by the gating scores. `numShared` always-on shared experts
/// are added on top (unweighted). The combined output is returned in
/// place of a dense FFN block.
///
/// The gating function is configurable via [gateFunction]:
///   * [GateFunction.softmax] (default) — classical softmax router.
///   * [GateFunction.sigmoid] — per-expert independent sigmoid gate,
///     as used in the DeepSeek-V3 paper and the aux-loss-free paper
///     (Wang et al. 2024, arXiv:2408.15664). The paper reports sigmoid
///     outperforms softmax under equal load balance.
///
/// If [renormalizeTopK] is true the K selected weights are divided by
/// their per-token sum so each token's expert contributions sum to 1.
/// Mixtral does this unconditionally; DeepSeek-V3 does it for sigmoid.
/// Default is `true` for `sigmoid` and `false` for `softmax` (matches
/// both references).
///
/// Load balancing follows the aux-loss-free recipe: per-expert
/// additive biases nudge the top-K comparison so under-utilized
/// experts win more often. The biases are non-differentiable and are
/// added ONLY to the top-K sort key, never to the weights used to
/// combine expert outputs. Call [MoEFeedForward.updateRoutingBias]
/// once per training batch (Algorithm 1 of the paper) — the counter
/// is per-batch, not per-epoch.
///
/// [biasUpdateRule] selects between:
///   * [BiasUpdateRule.sign] (default, paper's main variant):
///     `b_i += u * sign(mean_load - load_i)`. Better perplexity.
///   * [BiasUpdateRule.proportional]:
///     `b_i += u * (mean_load - load_i) / mean_load`. Slightly better
///     load balance but slightly worse perplexity per §4.3 of the
///     paper.
///
/// Design notes:
///   * The router (`gateW`) and both routed / shared experts are
///     ordinary trainable parameters — gradient flows through the
///     gating function into `gateW`, and through the expert forward
///     passes into their weights.
///   * The discrete top-K choice is made on CPU (Dart) and applied as
///     a `[T, E]` 0/1 mask multiplied into `gateScores`. The mask is a
///     non-differentiable straight-through stop; only the K selected
///     scores per row contribute to the output. When the module lives
///     on GPU this incurs one small `[T, E]` device→host copy per
///     forward — negligible for typical sequence lengths.
///   * Column-broadcasting of `masked_scores[:, e]` to `[T, embedDim]`
///     uses a precomputed one-hot selector matmul (`[T, E] @ [E, D]`)
///     — this keeps everything in existing tensor ops with correct
///     autograd, and runs on the same device as the input.
///   * All experts are evaluated on all tokens (dense compute). Real
///     sparse-MoE implementations (DeepSeek-V3, Mixtral) only run the
///     top-K experts per token via gather/scatter; that is a much
///     bigger rework and is not done here.
///   * 2D input `[T, embedDim]` only. Higher-rank inputs should be
///     reshaped by the caller. Runs on CPU or GPU (matches the device
///     passed to the constructor).
library;

import 'dart:math' as math;

import '../tensor/tensor.dart';
import 'linear.dart';
import 'module.dart';

/// Gating function used by [MoEFeedForward] to turn router logits into
/// per-expert scores.
enum GateFunction { softmax, sigmoid }

/// Bias update rule for [MoEFeedForward.updateRoutingBias]. See
/// aux-loss-free paper (arXiv:2408.15664) §4.3.
enum BiasUpdateRule { sign, proportional }

/// Activation used inside an [Expert]. `relu` and `silu`
/// (`x * sigmoid(x)`) both have fwd+bwd on CPU and GPU.
enum ExpertActivation { relu, silu }

/// Feed-forward body inside an [Expert].
///
/// * [ExpertVariant.mlp] — the classical two-layer MLP
///   `x -> w2(act(w1(x)))`.
/// * [ExpertVariant.swiGlu] — the SwiGLU gated body used by both
///   DeepSeek-V3 (`inference/model.py`) and Mixtral
///   (`transformers/models/mixtral/modeling_mixtral.py`):
///   `x -> w2(silu(w1(x)) * w3(x))`. Adds one extra `w3` projection
///   of shape `[hiddenDim, dim]`. Uses SiLU regardless of
///   [ExpertActivation] (the gate itself is the nonlinearity).
enum ExpertVariant { mlp, swiGlu }

/// A single expert. Either a two-layer MLP (`variant = mlp`, default)
/// or a SwiGLU gated FFN (`variant = swiGlu`), matching DeepSeek-V3
/// and Mixtral.
class Expert extends Module {
  final int dim;
  final int hiddenDim;
  final ExpertActivation activation;
  final ExpertVariant variant;
  final Linear w1;
  final Linear w2;

  /// Gate projection `[dim, hiddenDim]`, only allocated when
  /// [variant] is [ExpertVariant.swiGlu].
  final Linear? w3;

  Expert(
    this.dim,
    this.hiddenDim, {
    Device device = Device.CPU,
    int seed = 0,
    this.activation = ExpertActivation.relu,
    this.variant = ExpertVariant.mlp,
  }) : w1 = Linear(dim, hiddenDim, device: device, seed: seed),
       w2 = Linear(hiddenDim, dim, device: device, seed: seed + 1),
       w3 = variant == ExpertVariant.swiGlu
           ? Linear(dim, hiddenDim, device: device, seed: seed + 2)
           : null;

  Tensor call(Tensor x) {
    if (variant == ExpertVariant.swiGlu) {
      // SwiGLU: w2(silu(w1(x)) * w3(x)). Always uses silu — the gate
      // itself is the nonlinearity, per DeepSeek-V3 / Mixtral.
      final gate = w1(x);
      final up = w3!(x);
      final silu = gate * gate.sigmoid();
      return w2(silu * up);
    }
    final h = w1(x);
    final a = switch (activation) {
      ExpertActivation.relu => h.relu(),
      ExpertActivation.silu => h * h.sigmoid(),
    };
    return w2(a);
  }

  @override
  List<Tensor> parameters() => [
    ...w1.parameters(),
    ...w2.parameters(),
    if (w3 != null) ...w3!.parameters(),
  ];

  @override
  List<Module> submodules() => [w1, w2, if (w3 != null) w3!];
}

class MoEFeedForward extends Module {
  final int embedDim;
  final int numRoutedExperts;
  final int numSharedExperts;
  final int topK;
  final int expertHiddenDim;
  final double biasUpdateRate;
  final ExpertActivation activation;
  final ExpertVariant expertVariant;
  final GateFunction gateFunction;
  final BiasUpdateRule biasUpdateRule;

  /// When true, the K selected weights are divided by their per-token
  /// sum so each row of the routing weights sums to 1.
  final bool renormalizeTopK;

  /// Router weights `[embedDim, numRoutedExperts]`.
  final Tensor gateW;

  final List<Expert> routedExperts;
  final List<Expert> sharedExperts;

  /// Per-expert additive bias used only during top-K selection (not in
  /// the differentiable weighting output). Non-trainable — updated by
  /// [updateRoutingBias].
  final List<double> routingBias;

  /// Running expert-load counters (# tokens routed to each expert
  /// since the last [updateRoutingBias] call).
  final List<int> _expertLoad;

  /// Precomputed one-hot selectors `[E, embedDim]`. Selector `j` has
  /// its `j`-th row all ones, all other rows zero. Multiplying
  /// `[T, E] @ selector_j` gives `[T, embedDim]` whose every column
  /// equals the `j`-th column of the input — cheap column broadcast
  /// through matmul with correct autograd.
  final List<Tensor> _selectors;

  /// `[E, 1]` all-ones, used to compute per-row sums via
  /// `[T, E] @ ones_e1 -> [T, 1]` when [renormalizeTopK] is true.
  final Tensor? _onesE1;

  /// `[1, E]` all-ones, used to broadcast a `[T, 1]` per-row scalar
  /// back to `[T, E]` via `[T, 1] @ ones_1e -> [T, E]`.
  final Tensor? _ones1E;

  MoEFeedForward({
    required this.embedDim,
    required this.numRoutedExperts,
    required this.numSharedExperts,
    required this.topK,
    required this.expertHiddenDim,
    this.biasUpdateRate = 0.001,
    Device device = Device.CPU,
    int seed = 0,
    this.activation = ExpertActivation.relu,
    this.expertVariant = ExpertVariant.mlp,
    this.gateFunction = GateFunction.softmax,
    this.biasUpdateRule = BiasUpdateRule.sign,
    bool? renormalizeTopK,
  }) : renormalizeTopK =
           renormalizeTopK ?? (gateFunction == GateFunction.sigmoid),
       gateW = _initGate(embedDim, numRoutedExperts, seed, device),
       routedExperts = List<Expert>.generate(
         numRoutedExperts,
         (i) => Expert(
           embedDim,
           expertHiddenDim,
           device: device,
           seed: seed + 1000 + i * 100,
           activation: activation,
           variant: expertVariant,
         ),
       ),
       sharedExperts = List<Expert>.generate(
         numSharedExperts,
         (i) => Expert(
           embedDim,
           expertHiddenDim,
           device: device,
           seed: seed + 500000 + i * 100,
           activation: activation,
           variant: expertVariant,
         ),
       ),
       routingBias = List<double>.filled(numRoutedExperts, 0.0),
       _expertLoad = List<int>.filled(numRoutedExperts, 0),
       _selectors = _buildSelectors(numRoutedExperts, embedDim, device),
       _onesE1 = (renormalizeTopK ?? (gateFunction == GateFunction.sigmoid))
           ? Tensor.fill([numRoutedExperts, 1], 1.0, device: device)
           : null,
       _ones1E = (renormalizeTopK ?? (gateFunction == GateFunction.sigmoid))
           ? Tensor.fill([1, numRoutedExperts], 1.0, device: device)
           : null {
    if (topK <= 0 || topK > numRoutedExperts) {
      throw ArgumentError(
        'MoEFeedForward: topK ($topK) must be in [1, numRoutedExperts '
        '($numRoutedExperts)]',
      );
    }
  }

  static Tensor _initGate(int inDim, int e, int seed, Device device) {
    final rng = math.Random(seed);
    final bound = 1.0 / math.sqrt(inDim);
    final vals = List<double>.generate(
      inDim * e,
      (_) => (rng.nextDouble() * 2 - 1) * bound,
    );
    return Tensor.fromList(
      [inDim, e],
      vals,
      requiresGrad: true,
      device: device,
    );
  }

  static List<Tensor> _buildSelectors(int e, int d, Device device) {
    final list = <Tensor>[];
    for (int j = 0; j < e; j++) {
      final vals = List<double>.filled(e * d, 0.0);
      for (int c = 0; c < d; c++) {
        vals[j * d + c] = 1.0;
      }
      list.add(
        Tensor.fromList([e, d], vals, requiresGrad: false, device: device),
      );
    }
    return list;
  }

  Tensor call(Tensor x) {
    if (x.shape.length != 2 || x.shape[1] != embedDim) {
      throw ArgumentError(
        'MoEFeedForward: expected 2D input [T, $embedDim]; got ${x.shape}',
      );
    }
    final t = x.shape[0];
    final e = numRoutedExperts;

    final gateLogits = x.matmul(gateW); // [T, E]
    final gateScores = switch (gateFunction) {
      GateFunction.softmax => gateLogits.softmax(),
      GateFunction.sigmoid => gateLogits.sigmoid(),
    };

    // Discrete top-K with routing bias (CPU). Bias is added ONLY to
    // the sort key, never to the weights that combine expert outputs
    // — matches DeepSeek-V3 and the aux-loss-free paper.
    final scoresFlat = gateScores.toList();
    final maskVals = List<double>.filled(t * e, 0.0);
    final k = topK < e ? topK : e;
    for (int i = 0; i < t; i++) {
      final indexed = List<MapEntry<int, double>>.generate(
        e,
        (j) => MapEntry(j, scoresFlat[i * e + j] + routingBias[j]),
      );
      indexed.sort((a, b) => b.value.compareTo(a.value));
      for (int r = 0; r < k; r++) {
        final j = indexed[r].key;
        maskVals[i * e + j] = 1.0;
        _expertLoad[j]++;
      }
    }
    final mask = Tensor.fromList([t, e], maskVals, device: x.device);
    var masked = gateScores * mask; // [T, E]

    if (renormalizeTopK) {
      // Row-normalize so each token's K selected weights sum to 1.
      // rowSum: [T, E] @ [E, 1] -> [T, 1]; then broadcast back to
      // [T, E] via matmul with [1, E] all-ones. `+ eps` for safety
      // (top-K sum is always positive but this is cheap insurance).
      final rowSum = masked.matmul(_onesE1!); // [T, 1]
      final rowSumBcast = rowSum.matmul(_ones1E!); // [T, E]
      masked = masked / (rowSumBcast + 1e-9);
    }

    Tensor? acc;
    for (int j = 0; j < numRoutedExperts; j++) {
      final gateCol = masked.matmul(_selectors[j]); // [T, embedDim]
      final expertOut = routedExperts[j](x); // [T, embedDim]
      final weighted = gateCol * expertOut;
      acc = acc == null ? weighted : acc + weighted;
    }
    for (final s in sharedExperts) {
      final o = s(x);
      acc = acc == null ? o : acc + o;
    }
    return acc!;
  }

  /// Aux-loss-free bias update (Wang et al. 2024, Algorithm 1). Call
  /// this once per training batch — the counter is per-batch, not
  /// per-epoch. The update follows [biasUpdateRule]:
  ///
  ///   * [BiasUpdateRule.sign] (default, paper's main variant):
  ///     `b_i += u * sign(mean_load - load_i)`. Best perplexity.
  ///   * [BiasUpdateRule.proportional]:
  ///     `b_i += u * (mean_load - load_i) / mean_load`. Slightly
  ///     better load balance, slightly worse perplexity per §4.3 of
  ///     the paper.
  ///
  /// Resets the running load counters.
  void updateRoutingBias() {
    final total = _expertLoad.fold<int>(0, (a, b) => a + b);
    if (total == 0) return;
    final mean = total / numRoutedExperts;
    if (mean == 0) return;
    for (int j = 0; j < numRoutedExperts; j++) {
      final e = mean - _expertLoad[j];
      final delta = switch (biasUpdateRule) {
        BiasUpdateRule.sign =>
          e > 0
              ? 1.0
              : e < 0
              ? -1.0
              : 0.0,
        BiasUpdateRule.proportional => e / mean,
      };
      routingBias[j] += biasUpdateRate * delta;
      _expertLoad[j] = 0;
    }
  }

  /// Snapshot of the current per-expert load counters (read-only copy).
  List<int> get expertLoad => List<int>.unmodifiable(_expertLoad);

  @override
  List<Tensor> parameters() => [
    gateW,
    for (final e in routedExperts) ...e.parameters(),
    for (final e in sharedExperts) ...e.parameters(),
  ];

  @override
  List<Module> submodules() => [...routedExperts, ...sharedExperts];
}
