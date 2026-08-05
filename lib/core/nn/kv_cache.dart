/// Per-block KV cache for autoregressive attention.
///
/// A [MHACache] holds the running K and V tensors for every head of
/// one [MultiHeadAttention] layer. During autoregressive sampling the
/// forward pass appends the K/V produced from each new token to these
/// buffers via a row-wise concat, letting the next step reuse all past
/// projections instead of recomputing them.
///
/// An [EncoderCache] is a list of [MHACache]s (one per block) — the
/// state carried across steps by a full [TransformerEncoder] /
/// [GPT.generate] loop.
///
/// The caches are strictly inference-time state: their tensors are
/// created without `requiresGrad`. Building a fresh cache
/// (`MHACache.empty` / `EncoderCache.forGpt`) makes every entry
/// null; the first `append` sets it, subsequent `append`s concat the
/// new rows onto the existing buffer.
library;

import '../tensor/tensor.dart';

class MHACache {
  /// Number of KV-head slots stored. For standard MHA this equals the
  /// number of Q heads; for GQA (Llama / Mistral / Qwen) it is smaller
  /// than the Q-head count and each cached slot is shared by several
  /// Q heads via head-grouping.
  final int numKvHeads;

  /// Per-KV-head running K, shape `[T_seen, headDim]`. `null` until
  /// the first append.
  final List<Tensor?> k;

  /// Per-KV-head running V, shape `[T_seen, headDim]`. `null` until
  /// the first append.
  final List<Tensor?> v;

  MHACache.empty(this.numKvHeads)
    : k = List<Tensor?>.filled(numKvHeads, null, growable: false),
      v = List<Tensor?>.filled(numKvHeads, null, growable: false);

  /// Length of the cached sequence so far (0 if empty). All KV heads
  /// share the same T, so we read from KV head 0.
  int get seqLen => k[0]?.shape[0] ?? 0;

  /// Append `[N_new, headDim]` K and V for a single KV head. Returns
  /// the updated concatenated tensor (also stored in the cache).
  Tensor appendK(int kvHead, Tensor newK) {
    final prev = k[kvHead];
    final next = prev == null
        ? newK
        : TensorConcat.concat([prev, newK], axis: 0);
    k[kvHead] = next;
    return next;
  }

  Tensor appendV(int kvHead, Tensor newV) {
    final prev = v[kvHead];
    final next = prev == null
        ? newV
        : TensorConcat.concat([prev, newV], axis: 0);
    v[kvHead] = next;
    return next;
  }
}

class EncoderCache {
  final List<MHACache> layers;

  EncoderCache(this.layers);

  /// Fresh cache sized for a stack of `numLayers` blocks each with
  /// `numKvHeads` KV-head slots (equal to the Q-head count for
  /// standard MHA; smaller for GQA).
  factory EncoderCache.empty(int numLayers, int numKvHeads) => EncoderCache(
    List<MHACache>.generate(
      numLayers,
      (_) => MHACache.empty(numKvHeads),
      growable: false,
    ),
  );

  /// Length of the sequence cached so far. Reads from layer 0.
  int get seqLen => layers.isEmpty ? 0 : layers[0].seqLen;
}
