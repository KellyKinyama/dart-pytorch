/// Attention-Free Transformer (AFT) attention module.
///
/// Wraps [TensorAft.aftFull] with `Linear` projections for Q, K, V
/// and a learnable position-bias matrix `posBias` of shape
/// `[maxSeqLen, maxSeqLen]`. Produces an output of the same shape as
/// the input `[T, embedDim]`.
///
/// Set `masked: true` for a causal (decoder-only) variant.
///
/// Notes:
///   * Runs on CPU or GPU (matches the `device:` passed to the
///     constructor). The underlying `TensorAft.aftFull` +
///     `sliceTopLeft` both have GPU kernels.
///   * 2D input `[T, embedDim]` only — no 3D batched fast-path yet
///     (call it in a batch loop from higher-level code if needed).
///   * The position bias is a single `[maxSeqLen, maxSeqLen]` trainable
///     tensor; for shorter sequences we slice to `[T, T]` at runtime.
library;

import 'dart:math' as math;

import '../../tensor/tensor.dart';
import '../linear.dart';
import '../module.dart';

class AFTAttention extends Module {
  final int embedDim;
  final int maxSeqLen;
  final bool masked;

  final Linear wq;
  final Linear wk;
  final Linear wv;

  /// Learnable position-bias matrix, shape `[maxSeqLen, maxSeqLen]`.
  final Tensor posBias;

  AFTAttention(
    this.embedDim, {
    required this.maxSeqLen,
    this.masked = false,
    bool bias = false,
    Device device = Device.CPU,
    int seed = 0,
  }) : wq = Linear(embedDim, embedDim, bias: bias, device: device, seed: seed),
       wk = Linear(
         embedDim,
         embedDim,
         bias: bias,
         device: device,
         seed: seed + 1000,
       ),
       wv = Linear(
         embedDim,
         embedDim,
         bias: bias,
         device: device,
         seed: seed + 2000,
       ),
       posBias = _initPosBias(maxSeqLen, seed + 3000, device);

  static Tensor _initPosBias(int n, int seed, Device device) {
    final rng = math.Random(seed);
    final data = List<double>.generate(
      n * n,
      (_) => (rng.nextDouble() * 2 - 1) * 0.02,
    );
    return Tensor.fromList(
      [n, n],
      data,
      requiresGrad: true,
      device: device,
    );
  }

  /// Forward pass. `x` is `[T, embedDim]` (2D single sequence).
  Tensor call(Tensor x) {
    if (x.shape.length != 2) {
      throw ArgumentError(
        'AFTAttention: expected 2D input [T, embedDim]; got ${x.shape}',
      );
    }
    if (x.shape[1] != embedDim) {
      throw ArgumentError(
        'AFTAttention: last dim ${x.shape[1]} != embedDim $embedDim',
      );
    }
    final t = x.shape[0];
    if (t > maxSeqLen) {
      throw ArgumentError(
        'AFTAttention: seqLen $t exceeds maxSeqLen $maxSeqLen',
      );
    }
    final q = wq(x);
    final k = wk(x);
    final v = wv(x);
    final w = t == maxSeqLen ? posBias : TensorAft.sliceTopLeft(posBias, t, t);
    return TensorAft.aftFull(q, k, v, w, masked: masked);
  }

  @override
  List<Tensor> parameters() => [
    ...wq.parameters(),
    ...wk.parameters(),
    ...wv.parameters(),
    posBias,
  ];

  @override
  List<Module> submodules() => [wq, wk, wv];
}
