/// Multi-head self / cross attention (2D single-sequence).
///
/// Uses `numHeads` parallel `Linear` projections for Q, K, V (each of
/// shape `[embedDim, headDim]`), runs single-head SDPA per head, then
/// concatenates the outputs and applies an output `Linear` projection.
///
/// Only supports `embedDim % numHeads == 0`. Sequence tensors are 2D
/// `[N, embedDim]` (single batch).
library;

import '../tensor/tensor.dart';
import 'dropout.dart';
import 'linear.dart';
import 'module.dart';

class MultiHeadAttention extends Module {
  final int embedDim;
  final int numHeads;
  final int headDim;
  final List<Linear> wq;
  final List<Linear> wk;
  final List<Linear> wv;
  final Linear wo;
  final Dropout? attnDropout;

  MultiHeadAttention(
    this.embedDim,
    this.numHeads, {
    bool bias = false,
    double dropoutP = 0.0,
    Device device = Device.CPU,
    int seed = 0,
  }) : headDim = embedDim ~/ numHeads,
       wq = List<Linear>.generate(
         numHeads,
         (h) => Linear(
           embedDim,
           embedDim ~/ numHeads,
           bias: bias,
           device: device,
           seed: seed + h,
         ),
       ),
       wk = List<Linear>.generate(
         numHeads,
         (h) => Linear(
           embedDim,
           embedDim ~/ numHeads,
           bias: bias,
           device: device,
           seed: seed + 1000 + h,
         ),
       ),
       wv = List<Linear>.generate(
         numHeads,
         (h) => Linear(
           embedDim,
           embedDim ~/ numHeads,
           bias: bias,
           device: device,
           seed: seed + 2000 + h,
         ),
       ),
       wo = Linear(
         embedDim,
         embedDim,
         bias: bias,
         device: device,
         seed: seed + 3000,
       ),
       attnDropout = dropoutP > 0.0 ? Dropout(dropoutP) : null {
    if (embedDim % numHeads != 0) {
      throw ArgumentError(
        'MultiHeadAttention: embedDim ($embedDim) must be divisible by '
        'numHeads ($numHeads)',
      );
    }
  }

  /// Forward pass. `x` shape `[N, embedDim]`. Optional additive `mask`
  /// of shape `[N, N]` (same-shape for all heads).
  Tensor call(Tensor x, {Tensor? mask}) {
    final heads = <Tensor>[];
    for (int h = 0; h < numHeads; h++) {
      final q = wq[h](x);
      final k = wk[h](x);
      final v = wv[h](x);
      heads.add(q.scaledDotProductAttention(k, v, mask: mask));
    }
    final concatted = TensorConcat.concat(heads, axis: 1);
    var out = wo(concatted);
    if (attnDropout != null) {
      out = attnDropout!(out);
    }
    return out;
  }

  @override
  List<Tensor> parameters() => [
    for (final l in wq) ...l.parameters(),
    for (final l in wk) ...l.parameters(),
    for (final l in wv) ...l.parameters(),
    ...wo.parameters(),
  ];

  @override
  List<Module> submodules() => [
    ...wq,
    ...wk,
    ...wv,
    wo,
    if (attnDropout != null) attnDropout!,
  ];
}
