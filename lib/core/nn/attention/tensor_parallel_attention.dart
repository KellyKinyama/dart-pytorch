/// Tensor-parallel multi-head attention — one attention layer's heads
/// spread across several GPUs in a single process (Megatron-style).
///
/// Sharding: the `numKvHeads` KV heads (and the Q heads that map to
/// them) are partitioned into contiguous groups, one group per GPU.
/// Each GPU holds its heads' Q/K/V weights, runs scaled-dot-product
/// attention for those heads entirely **on-card**, and projects its
/// slice of the concatenated head outputs with its shard of the output
/// weight. The per-GPU partial projections are summed (all-reduced) into
/// the final `[N, embedDim]` result — so only the input activation and
/// the output partials cross GPU boundaries; attention itself never does.
///
/// This mirrors [MultiHeadAttention] numerically. It is **inference
/// oriented** (weights held detached, like the other tensor-parallel
/// layers), covers the 2D `[N, embedDim]` single-sequence path, supports
/// GQA and an optional additive mask. RoPE, dropout, KV-cache, and the
/// batched 3D path are intentionally out of scope for this first cut —
/// see doc/multi_gpu_roadmap.md.
library;

import '../../tensor/tensor.dart';
import '../kv_cache.dart';
import '../module.dart';
import '../rotary.dart';
import 'multi_head_attention.dart';

/// Even split of `total` into `parts` contiguous chunks; earlier chunks
/// take the remainder. Returns `parts + 1` boundary offsets.
List<int> _splitOffsets(int total, int parts) {
  final offs = <int>[0];
  final base = total ~/ parts;
  var rem = total % parts;
  var acc = 0;
  for (var i = 0; i < parts; i++) {
    acc += base + (rem > 0 ? 1 : 0);
    if (rem > 0) rem--;
    offs.add(acc);
  }
  return offs;
}

/// Column-slice `[c0, c1)` of a rank-2 CPU tensor into a fresh CPU
/// tensor (complements [Tensor.sliceRows]).
Tensor _sliceCols(Tensor w2d, int c0, int c1) {
  final rows = w2d.shape[0];
  final cols = w2d.shape[1];
  final src = w2d.toFloat32List();
  final w = c1 - c0;
  final out = List<double>.filled(rows * w, 0);
  for (var r = 0; r < rows; r++) {
    for (var j = 0; j < w; j++) {
      out[r * w + j] = src[r * cols + c0 + j];
    }
  }
  return Tensor.fromList([rows, w], out);
}

/// One GPU's slice of the attention layer: the Q heads it owns, the KV
/// heads those map to, and the matching output-projection columns —
/// all resident on [device].
class _AttnShard {
  _AttnShard({
    required this.device,
    required this.qHead0,
    required this.kvHead0,
    required this.numHeadGroups,
    required this.wqW,
    required this.wqB,
    required this.wkW,
    required this.wkB,
    required this.wvW,
    required this.wvB,
    required this.woShard,
    this.rope,
  });

  final int device;

  /// Global index of this shard's first Q head and first KV head, so a
  /// local Q head `i` maps to local KV head `(qHead0 + i) ~/ groups - kvHead0`.
  final int qHead0;
  final int kvHead0;
  final int numHeadGroups;

  final List<Tensor> wqW; // per local Q head: [headDim, embedDim] on device
  final List<Tensor?> wqB; // per local Q head: [1, headDim] or null
  final List<Tensor> wkW; // per local KV head
  final List<Tensor?> wkB;
  final List<Tensor> wvW;
  final List<Tensor?> wvB;

  /// Output-projection shard: `[embedDim, qHeads*headDim]` on device.
  final Tensor woShard;

  /// Device-local RoPE cache (null when the layer has no rotary embed).
  final RopeCache? rope;

  int get qCount => wqW.length;
}

/// KV cache for a [TensorParallelMultiHeadAttention] layer: one
/// [MHACache] per attention shard (holding that shard's KV heads on its
/// own GPU). Build one with [TensorParallelMultiHeadAttention.newCache].
class TPMHACache {
  final List<MHACache> shards;
  TPMHACache(this.shards);

  /// Cached sequence length so far (all shards share it).
  int get seqLen => shards.isEmpty ? 0 : shards[0].seqLen;
}

class TensorParallelMultiHeadAttention extends Module {
  final int embedDim;
  final int numHeads;
  final int numKvHeads;
  final int numHeadGroups;
  final int headDim;
  final List<_AttnShard> shards;

  /// Full output-projection bias `[1, embedDim]` on [outputDevice], or null.
  final Tensor? woBias;
  final int outputDevice;

  /// Whether the sharded weights/biases are autograd leaves (trainable).
  final bool trainable;

  TensorParallelMultiHeadAttention._(
    this.embedDim,
    this.numHeads,
    this.numKvHeads,
    this.numHeadGroups,
    this.headDim,
    this.shards,
    this.woBias,
    this.outputDevice,
    this.trainable,
  );

  /// Build a tensor-parallel copy of [mha], slicing its per-head weights
  /// onto [devices] (defaults to all visible GPUs). Pass `trainable: true`
  /// to make the shards autograd leaves (weight grads stay local to each
  /// card; only the activation grad crosses GPUs).
  factory TensorParallelMultiHeadAttention.fromAttention(
    MultiHeadAttention mha, {
    List<int>? devices,
    int? outputDevice,
    bool trainable = false,
  }) {
    final devs = devices ?? List<int>.generate(Tensor.gpuCount, (i) => i);
    if (devs.isEmpty) {
      throw ArgumentError('TensorParallelMultiHeadAttention: no devices');
    }
    final embedDim = mha.embedDim;
    final numHeads = mha.numHeads;
    final numKvHeads = mha.numKvHeads;
    final groups = mha.numHeadGroups;
    final headDim = mha.headDim;
    final outDev = outputDevice ?? devs.first;

    // Shard along KV heads so each GPU owns whole KV heads plus the Q
    // heads that attend to them. Use at most numKvHeads devices.
    final g = devs.length < numKvHeads ? devs.length : numKvHeads;
    final kvOffs = _splitOffsets(numKvHeads, g);

    // Pull every weight to the host once for bit-exact slicing.
    Tensor cpu(Tensor t) => t.to(Device.CPU);
    final wqWcpu = [for (final l in mha.wq) cpu(l.weight)];
    final wqBcpu = [for (final l in mha.wq) l.bias == null ? null : cpu(l.bias!)];
    final wkWcpu = [for (final l in mha.wk) cpu(l.weight)];
    final wkBcpu = [for (final l in mha.wk) l.bias == null ? null : cpu(l.bias!)];
    final wvWcpu = [for (final l in mha.wv) cpu(l.weight)];
    final wvBcpu = [for (final l in mha.wv) l.bias == null ? null : cpu(l.bias!)];
    final woWcpu = cpu(mha.wo.weight); // [embedDim, embedDim]

    final shards = <_AttnShard>[];
    for (var d = 0; d < g; d++) {
      final kv0 = kvOffs[d];
      final kv1 = kvOffs[d + 1];
      if (kv1 <= kv0) continue; // more devices than KV heads
      final dev = devs[d];
      final q0 = kv0 * groups;
      final q1 = kv1 * groups;

      final wqW = <Tensor>[];
      final wqB = <Tensor?>[];
      for (var h = q0; h < q1; h++) {
        final w = wqWcpu[h].toGpu(dev);
        if (trainable) w.requiresGrad = true;
        wqW.add(w);
        final b = wqBcpu[h]?.toGpu(dev);
        if (trainable && b != null) b.requiresGrad = true;
        wqB.add(b);
      }
      final wkW = <Tensor>[];
      final wkB = <Tensor?>[];
      final wvW = <Tensor>[];
      final wvB = <Tensor?>[];
      for (var kh = kv0; kh < kv1; kh++) {
        final kw = wkWcpu[kh].toGpu(dev);
        final vw = wvWcpu[kh].toGpu(dev);
        if (trainable) {
          kw.requiresGrad = true;
          vw.requiresGrad = true;
        }
        wkW.add(kw);
        wvW.add(vw);
        final kb = wkBcpu[kh]?.toGpu(dev);
        final vb = wvBcpu[kh]?.toGpu(dev);
        if (trainable) {
          if (kb != null) kb.requiresGrad = true;
          if (vb != null) vb.requiresGrad = true;
        }
        wkB.add(kb);
        wvB.add(vb);
      }
      // Output-projection columns for this shard's Q heads.
      final woShard = _sliceCols(woWcpu, q0 * headDim, q1 * headDim).toGpu(dev);
      if (trainable) woShard.requiresGrad = true;

      shards.add(
        _AttnShard(
          device: dev,
          qHead0: q0,
          kvHead0: kv0,
          numHeadGroups: groups,
          wqW: wqW,
          wqB: wqB,
          wkW: wkW,
          wkB: wkB,
          wvW: wvW,
          wvB: wvB,
          woShard: woShard,
          rope: mha.rope?.onGpu(dev),
        ),
      );
    }

    final woBias = mha.wo.bias == null ? null : cpu(mha.wo.bias!).toGpu(outDev);
    if (trainable && woBias != null) woBias.requiresGrad = true;
    return TensorParallelMultiHeadAttention._(
      embedDim,
      numHeads,
      numKvHeads,
      groups,
      headDim,
      shards,
      woBias,
      outDev,
      trainable,
    );
  }

  /// A fresh, empty KV cache matching this layer's shard/KV-head layout.
  TPMHACache newCache() =>
      TPMHACache([for (final s in shards) MHACache.empty(s.wkW.length)]);

  /// Forward over a 2D `[N, embedDim]` sequence with optional additive
  /// mask `[N, N]`. [startPos] is the absolute position of the first row
  /// (for RoPE); when a [cache] is given, positions continue from the
  /// cached length and each new token's K/V is appended per shard.
  /// Returns `[N, embedDim]` on [outputDevice].
  Tensor call(Tensor x, {Tensor? mask, TPMHACache? cache, int startPos = 0}) {
    if (x.shape.length != 2 || x.shape[1] != embedDim) {
      throw ArgumentError(
        'TensorParallelMultiHeadAttention: expected [N, $embedDim]; '
        'got ${x.shape}',
      );
    }
    if (cache != null && cache.shards.length != shards.length) {
      throw ArgumentError(
        'cache has ${cache.shards.length} shards; layer has ${shards.length}',
      );
    }
    if (cache != null && cache.seqLen > 0 && mask != null) {
      throw ArgumentError(
        'cannot pass a mask when appending to a non-empty cache',
      );
    }
    final pos = cache != null ? cache.seqLen : startPos;

    Tensor? acc;
    for (var si = 0; si < shards.length; si++) {
      final s = shards[si];
      final dev = s.device;
      final sc = cache?.shards[si];
      final xg = x.toGpu(dev);
      final maskg = mask?.toGpu(dev);
      final partial = Tensor.onGpu(dev, () {
        // Project K/V once per local KV head (append to cache if present).
        final ks = <Tensor>[];
        final vs = <Tensor>[];
        for (var i = 0; i < s.wkW.length; i++) {
          var k = xg.matmul(s.wkW[i].transpose());
          if (s.wkB[i] != null) k = k + s.wkB[i]!;
          if (s.rope != null) k = s.rope!.apply(k, startPos: pos);
          if (sc != null) k = sc.appendK(i, k);
          var v = xg.matmul(s.wvW[i].transpose());
          if (s.wvB[i] != null) v = v + s.wvB[i]!;
          if (sc != null) v = sc.appendV(i, v);
          ks.add(k);
          vs.add(v);
        }
        // Per Q head: project, SDPA against its KV head.
        final heads = <Tensor>[];
        for (var i = 0; i < s.qCount; i++) {
          var q = xg.matmul(s.wqW[i].transpose());
          if (s.wqB[i] != null) q = q + s.wqB[i]!;
          if (s.rope != null) q = s.rope!.apply(q, startPos: pos);
          final globalH = s.qHead0 + i;
          final localKv = (globalH ~/ s.numHeadGroups) - s.kvHead0;
          heads.add(
            q.scaledDotProductAttention(ks[localKv], vs[localKv], mask: maskg),
          );
        }
        final concat = TensorConcat.concat(heads, axis: 1);
        // Row-parallel output projection: this shard's contribution.
        return concat.matmul(s.woShard.transpose());
      });
      final onOut = partial.toGpu(outputDevice);
      acc = acc == null
          ? onOut
          : Tensor.onGpu(outputDevice, () => acc! + onOut);
    }
    var out = acc!;
    if (woBias != null) {
      final b = woBias!;
      out = Tensor.onGpu(outputDevice, () => out + b);
    }
    return out;
  }

  @override
  List<Tensor> parameters() {
    if (!trainable) return const [];
    final ps = <Tensor>[];
    for (final s in shards) {
      ps.addAll(s.wqW);
      ps.addAll(s.wkW);
      ps.addAll(s.wvW);
      ps.add(s.woShard);
      for (final b in s.wqB) {
        if (b != null) ps.add(b);
      }
      for (final b in s.wkB) {
        if (b != null) ps.add(b);
      }
      for (final b in s.wvB) {
        if (b != null) ps.add(b);
      }
    }
    if (woBias != null) ps.add(woBias!);
    return ps;
  }
}
