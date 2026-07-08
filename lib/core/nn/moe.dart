/// Mixture-of-Experts feed-forward block.
///
/// DeepSeek-V3-style routing: for each token the router picks the top-K
/// of `numRoutedExperts` "sparse" experts and combines their outputs
/// weighted by the (softmax-of-gate-logits) probabilities. `numShared`
/// always-on shared experts are added on top. The combined output is
/// returned in place of a dense FFN block.
///
/// Load balancing follows the DeepSeek-V3 "aux-loss-free" recipe:
/// per-expert additive biases nudge under-utilized experts up and
/// over-utilized experts down in the top-K comparison. The biases are
/// non-differentiable; call [MoEFeedForward.updateRoutingBias] once
/// per epoch (or whatever cadence you prefer) to adjust them from the
/// running load counters.
///
/// Design notes:
///   * The router (`gateW`) and both routed / shared experts are
///     ordinary trainable parameters — gradient flows through the
///     softmax of the gate logits into `gateW`, and through the
///     expert forward passes into their weights.
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
///   * 2D input `[T, embedDim]` only. Higher-rank inputs should be
///     reshaped by the caller. Runs on CPU or GPU (matches the device
///     passed to the constructor).
library;

import 'dart:math' as math;

import '../tensor/tensor.dart';
import 'linear.dart';
import 'module.dart';

/// Activation used inside an [Expert]. `relu` runs its backward on CPU
/// (matches classical MoE); `silu` (`x * sigmoid(x)`) has fwd+bwd on
/// both CPU and GPU and is required if the module lives on GPU.
enum ExpertActivation { relu, silu }

/// A single expert — a two-layer MLP `x -> act(x @ W1.T) @ W2.T`.
class Expert extends Module {
  final int dim;
  final int hiddenDim;
  final ExpertActivation activation;
  final Linear w1;
  final Linear w2;

  Expert(
    this.dim,
    this.hiddenDim, {
    Device device = Device.CPU,
    int seed = 0,
    this.activation = ExpertActivation.relu,
  }) : w1 = Linear(dim, hiddenDim, device: device, seed: seed),
       w2 = Linear(hiddenDim, dim, device: device, seed: seed + 1);

  Tensor call(Tensor x) {
    final h = w1(x);
    final a = switch (activation) {
      ExpertActivation.relu => h.relu(),
      ExpertActivation.silu => h * h.sigmoid(),
    };
    return w2(a);
  }

  @override
  List<Tensor> parameters() => [...w1.parameters(), ...w2.parameters()];

  @override
  List<Module> submodules() => [w1, w2];
}

class MoEFeedForward extends Module {
  final int embedDim;
  final int numRoutedExperts;
  final int numSharedExperts;
  final int topK;
  final int expertHiddenDim;
  final double biasUpdateRate;
  final ExpertActivation activation;

  /// Router weights `[embedDim, numRoutedExperts]`.
  final Tensor gateW;

  final List<Expert> routedExperts;
  final List<Expert> sharedExperts;

  /// Per-expert additive bias used only during top-K selection (not in
  /// the differentiable softmax output). Non-trainable — updated by
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
  }) : gateW = _initGate(embedDim, numRoutedExperts, seed, device),
       routedExperts = List<Expert>.generate(
         numRoutedExperts,
         (i) => Expert(
           embedDim,
           expertHiddenDim,
           device: device,
           seed: seed + 1000 + i * 100,
           activation: activation,
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
         ),
       ),
       routingBias = List<double>.filled(numRoutedExperts, 0.0),
       _expertLoad = List<int>.filled(numRoutedExperts, 0),
       _selectors = _buildSelectors(numRoutedExperts, embedDim, device) {
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
    final gateScores = gateLogits.softmax(); // [T, E]

    // Discrete top-K with routing bias (CPU).
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
    final masked = gateScores * mask; // [T, E]

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

  /// DeepSeek-V3 aux-loss-free bias update. Call this at whatever
  /// cadence balances your workload (per-epoch is common). Resets the
  /// running load counters.
  void updateRoutingBias() {
    final total = _expertLoad.fold<int>(0, (a, b) => a + b);
    if (total == 0) return;
    final mean = total / numRoutedExperts;
    if (mean == 0) return;
    for (int j = 0; j < numRoutedExperts; j++) {
      final delta = (mean - _expertLoad[j]) / mean;
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
