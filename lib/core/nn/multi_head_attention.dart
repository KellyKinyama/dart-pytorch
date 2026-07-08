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
import 'kv_cache.dart';
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

  /// Forward pass. Accepts either 2D `[N, embedDim]` (single
  /// sequence) or 3D `[B, N, embedDim]` (batched — attention is
  /// computed independently per batch element and stacked).
  ///
  /// Optional additive `mask` of shape `[N, N]` (same-shape for all
  /// heads, shared across batches when batched).
  ///
  /// If `cache` is provided (autoregressive fast path), per-head K/V
  /// are appended to the cache and attention is computed against the
  /// full cached history. Two combinations are valid:
  ///   * empty cache + optional causal `mask` (prompt fill: `Q = K = V
  ///     = [N, D]` — the mask is required for causality when `N > 1`);
  ///   * non-empty cache + no `mask` (single-token append: `Q = [1, D]`
  ///     attends to `K = V = [T+1, D]`, all cached positions are past
  ///     context and therefore always attendable).
  ///
  /// Batched input (`shape.length == 3`) does not support `cache` —
  /// KV caching is inherently per-sequence.
  Tensor call(Tensor x, {Tensor? mask, MHACache? cache}) {
    final isBatched = x.shape.length == 3;
    if (isBatched && cache != null) {
      throw ArgumentError(
        'MultiHeadAttention: KV cache is not supported for batched '
        '(3D) input; run per-sequence for cached generation.',
      );
    }
    if (cache != null && mask != null && cache.seqLen > 0) {
      throw ArgumentError(
        'MultiHeadAttention: cannot pass mask when appending to a '
        'non-empty cache (mask shape would be incompatible with '
        'grown K/V)',
      );
    }

    if (isBatched) {
      return _callBatched(x, mask: mask);
    }

    final heads = <Tensor>[];
    for (int h = 0; h < numHeads; h++) {
      final q = wq[h](x);
      var k = wk[h](x);
      var v = wv[h](x);
      if (cache != null) {
        k = cache.appendK(h, k);
        v = cache.appendV(h, v);
      }
      heads.add(q.scaledDotProductAttention(k, v, mask: mask));
    }
    final concatted = TensorConcat.concat(heads, axis: 1);
    var out = wo(concatted);
    if (attnDropout != null) {
      out = attnDropout!(out);
    }
    return out;
  }

  /// Batched forward: for each head, project [B,S,D] -> [B,S,headDim]
  /// (Linear handles 3D internally), split into B sequences, run
  /// per-sequence SDPA sharing the same mask, and stitch back into
  /// [B,S,headDim]. Heads are concatenated on the packed [B*S, ...]
  /// view (axis=1) and reshaped back to [B,S,embedDim].
  Tensor _callBatched(Tensor x, {Tensor? mask}) {
    final b = x.shape[0];
    final s = x.shape[1];
    final heads = <Tensor>[];
    for (int h = 0; h < numHeads; h++) {
      final q = wq[h](x); // [B, S, headDim]
      final k = wk[h](x);
      final v = wv[h](x);
      final qFlat = q.reshape([b * s, headDim]);
      final kFlat = k.reshape([b * s, headDim]);
      final vFlat = v.reshape([b * s, headDim]);
      final qSplits = TensorConcat.splitRows(qFlat, s);
      final kSplits = TensorConcat.splitRows(kFlat, s);
      final vSplits = TensorConcat.splitRows(vFlat, s);
      final perBatch = <Tensor>[];
      for (int i = 0; i < b; i++) {
        perBatch.add(
          qSplits[i].scaledDotProductAttention(
            kSplits[i],
            vSplits[i],
            mask: mask,
          ),
        );
      }
      heads.add(TensorConcat.concat(perBatch, axis: 0)); // [B*S, headDim]
    }
    final concatted = TensorConcat.concat(heads, axis: 1); // [B*S, embedDim]
    var out = wo(concatted.reshape([b, s, embedDim])); // [B, S, embedDim]
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
