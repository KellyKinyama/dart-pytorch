/// Multi-head cross-attention.
///
/// Queries come from the "target" side (`xq`, e.g. the decoder input),
/// keys and values come from the "memory" side (`xkv`, e.g. the
/// encoder output). Query and memory may have different sequence
/// lengths and even different embedding sizes:
///
///     xq   : [Sq, embedDim]      or  [B, Sq, embedDim]
///     xkv  : [Skv, kvEmbedDim]   or  [B, Skv, kvEmbedDim]
///     out  : [Sq, embedDim]      or  [B, Sq, embedDim]
///
/// Layout mirrors [MultiHeadAttention]: `numHeads` parallel `Linear`
/// projections for Q, K, V (each producing `headDim = embedDim /
/// numHeads`), single-head SDPA per head, concat, output projection.
///
/// Unlike self-attention, this module never causal-masks — the memory
/// (encoder output) is fully observable. No KV cache: the memory is
/// fixed for the whole decoding pass, so cache-append semantics do
/// not apply.
library;

import '../tensor/tensor.dart';
import 'dropout.dart';
import 'linear.dart';
import 'module.dart';

class MultiHeadCrossAttention extends Module {
  final int embedDim;
  final int kvEmbedDim;
  final int numHeads;
  final int headDim;
  final List<Linear> wq;
  final List<Linear> wk;
  final List<Linear> wv;
  final Linear wo;
  final Dropout? attnDropout;

  MultiHeadCrossAttention(
    this.embedDim,
    this.kvEmbedDim,
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
           kvEmbedDim,
           embedDim ~/ numHeads,
           bias: bias,
           device: device,
           seed: seed + 1000 + h,
         ),
       ),
       wv = List<Linear>.generate(
         numHeads,
         (h) => Linear(
           kvEmbedDim,
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
        'MultiHeadCrossAttention: embedDim ($embedDim) must be '
        'divisible by numHeads ($numHeads)',
      );
    }
  }

  /// Forward pass. `xq` is the query side, `xkv` the key/value
  /// (memory) side. Both must be 2D or both 3D; when 3D, batch sizes
  /// must match.
  Tensor call(Tensor xq, Tensor xkv) {
    if (xq.shape.length != xkv.shape.length) {
      throw ArgumentError(
        'MultiHeadCrossAttention: xq and xkv must share rank; '
        'got xq=${xq.shape} xkv=${xkv.shape}',
      );
    }
    if (xq.shape.length != 2 && xq.shape.length != 3) {
      throw ArgumentError(
        'MultiHeadCrossAttention: xq/xkv must be 2D or 3D; '
        'got rank ${xq.shape.length}',
      );
    }
    if (xq.shape.last != embedDim) {
      throw ArgumentError(
        'MultiHeadCrossAttention: xq last dim must be $embedDim; '
        'got ${xq.shape}',
      );
    }
    if (xkv.shape.last != kvEmbedDim) {
      throw ArgumentError(
        'MultiHeadCrossAttention: xkv last dim must be $kvEmbedDim; '
        'got ${xkv.shape}',
      );
    }
    if (xq.shape.length == 3) {
      if (xq.shape[0] != xkv.shape[0]) {
        throw ArgumentError(
          'MultiHeadCrossAttention: batched xq/xkv must share batch '
          'size; got xq=${xq.shape} xkv=${xkv.shape}',
        );
      }
      return _callBatched(xq, xkv);
    }

    final heads = <Tensor>[];
    for (int h = 0; h < numHeads; h++) {
      final q = wq[h](xq);
      final k = wk[h](xkv);
      final v = wv[h](xkv);
      heads.add(q.scaledDotProductAttention(k, v));
    }
    final concatted = TensorConcat.concat(heads, axis: 1);
    var out = wo(concatted);
    if (attnDropout != null) {
      out = attnDropout!(out);
    }
    return out;
  }

  Tensor _callBatched(Tensor xq, Tensor xkv) {
    final b = xq.shape[0];
    final sq = xq.shape[1];
    final skv = xkv.shape[1];
    final heads = <Tensor>[];
    for (int h = 0; h < numHeads; h++) {
      final q = wq[h](xq); // [B, Sq, headDim]
      final k = wk[h](xkv); // [B, Skv, headDim]
      final v = wv[h](xkv);
      final qFlat = q.reshape([b * sq, headDim]);
      final kFlat = k.reshape([b * skv, headDim]);
      final vFlat = v.reshape([b * skv, headDim]);
      final qSplits = TensorConcat.splitRows(qFlat, sq);
      final kSplits = TensorConcat.splitRows(kFlat, skv);
      final vSplits = TensorConcat.splitRows(vFlat, skv);
      final perBatch = <Tensor>[];
      for (int i = 0; i < b; i++) {
        perBatch.add(
          qSplits[i].scaledDotProductAttention(kSplits[i], vSplits[i]),
        );
      }
      heads.add(TensorConcat.concat(perBatch, axis: 0)); // [B*Sq, headDim]
    }
    final concatted = TensorConcat.concat(heads, axis: 1); // [B*Sq, embedDim]
    var out = wo(concatted.reshape([b, sq, embedDim]));
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
