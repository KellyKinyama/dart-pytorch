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
  final int numHeads;

  /// Per-head running K, shape `[T_seen, headDim]`. `null` until the
  /// first append.
  final List<Tensor?> k;

  /// Per-head running V, shape `[T_seen, headDim]`. `null` until the
  /// first append.
  final List<Tensor?> v;

  MHACache.empty(this.numHeads)
    : k = List<Tensor?>.filled(numHeads, null, growable: false),
      v = List<Tensor?>.filled(numHeads, null, growable: false);

  /// Length of the cached sequence so far (0 if empty). All heads
  /// share the same T, so we read from head 0.
  int get seqLen => k[0]?.shape[0] ?? 0;

  /// Append `[N_new, headDim]` K and V for a single head. Returns the
  /// updated concatenated tensor (also stored in the cache).
  Tensor appendK(int head, Tensor newK) {
    final prev = k[head];
    final next = prev == null
        ? newK
        : TensorConcat.concat([prev, newK], axis: 0);
    k[head] = next;
    return next;
  }

  Tensor appendV(int head, Tensor newV) {
    final prev = v[head];
    final next = prev == null
        ? newV
        : TensorConcat.concat([prev, newV], axis: 0);
    v[head] = next;
    return next;
  }
}

class EncoderCache {
  final List<MHACache> layers;

  EncoderCache(this.layers);

  /// Fresh cache sized for a stack of `numLayers` blocks each with
  /// `numHeads` heads.
  factory EncoderCache.empty(int numLayers, int numHeads) => EncoderCache(
    List<MHACache>.generate(
      numLayers,
      (_) => MHACache.empty(numHeads),
      growable: false,
    ),
  );

  /// Length of the sequence cached so far. Reads from layer 0.
  int get seqLen => layers.isEmpty ? 0 : layers[0].seqLen;
}
