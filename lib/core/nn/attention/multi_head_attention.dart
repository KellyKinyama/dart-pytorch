/// Multi-head self / cross attention (2D single-sequence).
///
/// Uses `numHeads` parallel `Linear` projections for Q and `numKvHeads`
/// projections for K and V (each of shape `[embedDim, headDim]`), runs
/// single-head SDPA per Q head (with the corresponding K/V shared
/// across a group of Q heads when `numKvHeads < numHeads`), then
/// concatenates the outputs and applies an output `Linear` projection.
///
/// When `numKvHeads == numHeads` this is plain multi-head attention
/// (the default and what GPT-2 / GPT-J / Pythia use). When
/// `numKvHeads < numHeads` this is grouped-query attention (GQA) as
/// used by Llama / Mistral / Qwen — `numHeads` must be a multiple of
/// `numKvHeads`, and Q heads `[g*R, (g+1)*R)` share KV head `g` where
/// `R = numHeads / numKvHeads`.
///
/// Only supports `embedDim % numHeads == 0`. Sequence tensors are 2D
/// `[N, embedDim]` (single batch).
library;

import '../../tensor/tensor.dart';
import '../dropout.dart';
import '../kv_cache.dart';
import '../linear.dart';
import '../module.dart';
import '../rotary.dart';

class MultiHeadAttention extends Module {
  final int embedDim;
  final int numHeads;
  final int numKvHeads;
  final int numHeadGroups; // = numHeads ~/ numKvHeads
  final int headDim;
  final List<Linear> wq;
  final List<Linear> wk;
  final List<Linear> wv;
  final Linear wo;
  final Dropout? attnDropout;

  /// Optional rotary positional embedding cache. When non-null,
  /// [call] applies RoPE to Q and K per head after their linear
  /// projections and before the SDPA. The `startPos` argument to
  /// [call] tells RoPE the absolute position of the first Q/K row
  /// (matters for KV-cache generation, where each appended token
  /// sits at a new absolute position).
  RopeCache? rope;

  MultiHeadAttention(
    this.embedDim,
    this.numHeads, {
    int? numKvHeads,
    bool bias = false,
    double dropoutP = 0.0,
    Device device = Device.CPU,
    int seed = 0,
  }) : numKvHeads = numKvHeads ?? numHeads,
       numHeadGroups = numHeads ~/ (numKvHeads ?? numHeads),
       headDim = embedDim ~/ numHeads,
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
         numKvHeads ?? numHeads,
         (h) => Linear(
           embedDim,
           embedDim ~/ numHeads,
           bias: bias,
           device: device,
           seed: seed + 1000 + h,
         ),
       ),
       wv = List<Linear>.generate(
         numKvHeads ?? numHeads,
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
    final kv = numKvHeads ?? numHeads;
    if (numHeads % kv != 0) {
      throw ArgumentError(
        'MultiHeadAttention: numHeads ($numHeads) must be divisible by '
        'numKvHeads ($kv) for GQA',
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
  Tensor call(Tensor x, {Tensor? mask, MHACache? cache, int startPos = 0}) {
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
    if (isBatched && rope != null) {
      throw ArgumentError(
        'MultiHeadAttention: RoPE not supported for batched (3D) '
        'inputs; use the 2D single-sequence path.',
      );
    }
    if (isBatched && numKvHeads != numHeads) {
      throw ArgumentError(
        'MultiHeadAttention: GQA (numKvHeads=$numKvHeads < '
        'numHeads=$numHeads) is only implemented for the 2D '
        'single-sequence path.',
      );
    }

    if (isBatched) {
      return _callBatched(x, mask: mask);
    }

    // Project K/V once per KV head and (optionally) apply RoPE.
    // Then append into cache once per KV head — Q heads that share
    // this KV head will read the same appended slice.
    final ks = List<Tensor>.filled(numKvHeads, x);
    final vs = List<Tensor>.filled(numKvHeads, x);
    for (int kh = 0; kh < numKvHeads; kh++) {
      var k = wk[kh](x);
      var v = wv[kh](x);
      if (rope != null) {
        k = rope!.apply(k, startPos: startPos);
      }
      if (cache != null) {
        k = cache.appendK(kh, k);
        v = cache.appendV(kh, v);
      }
      ks[kh] = k;
      vs[kh] = v;
    }

    final heads = <Tensor>[];
    for (int h = 0; h < numHeads; h++) {
      var q = wq[h](x);
      if (rope != null) {
        q = rope!.apply(q, startPos: startPos);
      }
      final kh = h ~/ numHeadGroups; // Q-head -> KV-head grouping
      heads.add(q.scaledDotProductAttention(ks[kh], vs[kh], mask: mask));
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
