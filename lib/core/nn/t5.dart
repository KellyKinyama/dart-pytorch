/// T5 (Google, 2020) encoder-decoder architecture.
///
/// Implements the text-to-text T5 model family. Supports:
///
///   * **T5-v1.0** (`t5-small`, `t5-base`, `t5-large`) — plain ReLU FFN.
///   * **T5-v1.1** (`google/t5-v1_1-*`) — gated-GELU FFN.
///   * **FLAN-T5** (`google/flan-t5-*`) — same as T5-v1.1 arch, just
///     instruction-tuned weights.
///
/// Key T5-specific pieces:
///
///   * **Learned relative-position bias** per head, bucketed by
///     |j - i|, added to attention scores. Only the first block per
///     stack owns the bias parameters; other blocks reuse it.
///   * **T5 layer-norm** — RMSNorm with no mean-centering, no bias,
///     epsilon=1e-6. Same math as [RMSNorm].
///   * **No attention biases** anywhere (Q/K/V/O and FFN linears are
///     bias-free).
///   * **Unscaled attention scores** — T5 does NOT divide by
///     sqrt(d_kv). This module cancels the SDPA scale by pre-scaling
///     Q with sqrt(d_kv), so the effective attention weight is
///     `Q @ K^T + bias`.
///   * **Independent `d_kv`** — head dim is *not* forced to
///     `d_model / num_heads`. E.g. FLAN-T5-small has `d_model=512`,
///     `num_heads=6`, `d_kv=64` (heads project to 384, out projects
///     back to 512).
///   * **Rescaled LM head** when tied embeddings — pre-multiply the
///     last hidden by `1 / sqrt(d_model)`.
///
/// Special tokens (T5 conventions):
///   `pad_token_id = 0`, `eos_token_id = 1`,
///   `decoder_start_token_id = 0` (the pad id, on purpose).
library;

import 'dart:math' as math;

import '../tensor/tensor.dart';
import 'embedding.dart';
import 'kv_cache.dart';
import 'linear.dart';
import 'masks.dart';
import 'module.dart';
import 'rms_norm.dart';

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

enum T5FfnActivation {
  /// `w_o(relu(w_i(x)))` — T5-v1.0.
  relu,

  /// `w_o(gelu(w_i(x)) * w_g(x))` — T5-v1.1 / FLAN-T5.
  gatedGelu,
}

class T5Config {
  final int vocabSize;
  final int dModel;
  final int dFf;
  final int dKv;
  final int numLayers;
  final int numDecoderLayers;
  final int numHeads;
  final int relativeAttentionNumBuckets;
  final int relativeAttentionMaxDistance;
  final double layerNormEps;
  final bool tieWordEmbeddings;
  final T5FfnActivation feedForwardProj;
  final int maxCtx;
  final Device device;
  final int seed;

  const T5Config({
    required this.vocabSize,
    required this.dModel,
    required this.dFf,
    required this.dKv,
    required this.numLayers,
    required this.numDecoderLayers,
    required this.numHeads,
    this.relativeAttentionNumBuckets = 32,
    this.relativeAttentionMaxDistance = 128,
    this.layerNormEps = 1e-6,
    this.tieWordEmbeddings = true,
    this.feedForwardProj = T5FfnActivation.relu,
    this.maxCtx = 512,
    this.device = Device.CPU,
    this.seed = 0,
  });
}

// ---------------------------------------------------------------------------
// Relative-position bias
// ---------------------------------------------------------------------------

/// Learned per-head relative-position bias, bucketed by |j - i|.
///
/// The bias is an `[num_buckets, num_heads]` embedding table; for a
/// pair of positions `(i, j)` the bucket function maps the signed
/// relative offset into `[0, num_buckets)` and the resulting
/// `[num_heads]` slice is added to attention scores. `bidirectional`
/// controls the bucketing:
///
///   * `true` (encoder self-attn): sign of `j - i` selects half of the
///     bucket range; symmetric.
///   * `false` (decoder self-attn): only non-positive offsets are
///     bucketed (future positions are masked out anyway).
class T5RelativeBias extends Module {
  final int numBuckets;
  final int numHeads;
  final int maxDistance;
  final bool bidirectional;
  final Embedding table;

  /// CPU snapshot of `table.weight` — populated on first bias call,
  /// invalidated by [invalidateCache]. Amortises the GPU->CPU
  /// round-trip that would otherwise happen on every attention call.
  List<double>? _tableCache;

  T5RelativeBias({
    required this.numBuckets,
    required this.numHeads,
    required this.maxDistance,
    required this.bidirectional,
    Device device = Device.CPU,
    int seed = 0,
  }) : table = Embedding(numBuckets, numHeads, device: device, seed: seed);

  /// Drop the cached CPU snapshot. Call after the loader updates
  /// the underlying `table.weight`; otherwise the cache is
  /// snapshot-once-then-frozen (fine for inference).
  void invalidateCache() {
    _tableCache = null;
  }

  List<double> _tableData() => _tableCache ??= table.weight.toList();

  /// Returns a `[nq, nk]` additive mask if the caller is single-head,
  /// or `[num_heads, nq, nk]` when treated per-head. This impl folds
  /// per-head into a list of `[nq, nk]` mask tensors (one per head)
  /// because [T5Attention] loops heads.
  List<Tensor> maskPerHead(int nq, int nk) {
    final buckets = List<int>.filled(nq * nk, 0);
    for (int i = 0; i < nq; i++) {
      for (int j = 0; j < nk; j++) {
        final rel = j - i;
        buckets[i * nk + j] = _bucket(rel);
      }
    }
    final tableData = _tableData();
    final out = <Tensor>[];
    for (int h = 0; h < numHeads; h++) {
      final vals = List<double>.filled(nq * nk, 0);
      for (int i = 0; i < nq; i++) {
        for (int j = 0; j < nk; j++) {
          final b = buckets[i * nk + j];
          vals[i * nk + j] = tableData[b * numHeads + h];
        }
      }
      out.add(Tensor.fromList([nq, nk], vals, device: table.weight.device));
    }
    return out;
  }

  /// Cached-decode variant. Returns per-head `[1, kLen]` bias tensors
  /// for a single query token at position `qPos` attending to keys
  /// `[0, kLen)`. Used by [T5Decoder.callCached].
  List<Tensor> maskPerHeadSingleQ(int qPos, int kLen) {
    final tableData = _tableData();
    final out = <Tensor>[];
    for (int h = 0; h < numHeads; h++) {
      final vals = List<double>.filled(kLen, 0);
      for (int j = 0; j < kLen; j++) {
        vals[j] = tableData[_bucket(j - qPos) * numHeads + h];
      }
      out.add(Tensor.fromList([1, kLen], vals, device: table.weight.device));
    }
    return out;
  }

  int _bucket(int relPos) {
    var ret = 0;
    var n = relPos;
    int numB = numBuckets;
    if (bidirectional) {
      numB = numB ~/ 2;
      if (n > 0) ret += numB;
      n = n.abs();
    } else {
      n = -math.min(n, 0);
    }
    final maxExact = numB ~/ 2;
    final isSmall = n < maxExact;
    if (isSmall) return ret + n;
    final logDist = math.log(n / maxExact) / math.log(maxDistance / maxExact);
    final large = maxExact + (logDist * (numB - maxExact)).floor();
    return ret + math.min(large, numB - 1);
  }

  @override
  List<Tensor> parameters() => table.parameters();

  @override
  List<Module> submodules() => [table];
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

/// T5 attention — self or cross. Independent `d_kv`, no biases, no
/// attention-score scaling (cancelled by pre-scaling Q with
/// sqrt(d_kv)). Additive relative bias is expected pre-computed by
/// the caller and passed via `relativeBiasPerHead` (per-head list of
/// `[nq, nk]` tensors).
class T5Attention extends Module {
  final int dModel;
  final int kvDim;
  final int numHeads;
  final int dKv;
  final List<Linear> wq;
  final List<Linear> wk;
  final List<Linear> wv;
  final Linear wo;
  final double qScale;

  T5Attention({
    required this.dModel,
    required this.kvDim,
    required this.numHeads,
    required this.dKv,
    Device device = Device.CPU,
    int seed = 0,
  }) : qScale = math.sqrt(dKv.toDouble()),
       wq = List<Linear>.generate(
         numHeads,
         (h) =>
             Linear(dModel, dKv, bias: false, device: device, seed: seed + h),
       ),
       wk = List<Linear>.generate(
         numHeads,
         (h) => Linear(
           kvDim,
           dKv,
           bias: false,
           device: device,
           seed: seed + 1000 + h,
         ),
       ),
       wv = List<Linear>.generate(
         numHeads,
         (h) => Linear(
           kvDim,
           dKv,
           bias: false,
           device: device,
           seed: seed + 2000 + h,
         ),
       ),
       wo = Linear(
         numHeads * dKv,
         dModel,
         bias: false,
         device: device,
         seed: seed + 3000,
       );

  /// `xq`: `[Nq, dModel]`. `xkv`: `[Nk, kvDim]` (pass `xq` for self-
  /// attention). `mask`: optional additive mask shared across heads
  /// (e.g. causal). `relativeBiasPerHead`: optional per-head list of
  /// `[Nq, Nk]` tensors added to attention scores.
  Tensor call(
    Tensor xq,
    Tensor xkv, {
    Tensor? mask,
    List<Tensor>? relativeBiasPerHead,
  }) {
    if (relativeBiasPerHead != null && relativeBiasPerHead.length != numHeads) {
      throw ArgumentError(
        'T5Attention: relativeBiasPerHead must have $numHeads tensors; '
        'got ${relativeBiasPerHead.length}',
      );
    }
    final heads = <Tensor>[];
    for (int h = 0; h < numHeads; h++) {
      // Pre-scale Q to cancel SDPA's built-in 1/sqrt(d_kv) — T5 uses
      // unscaled attention scores.
      final q = wq[h](xq) * qScale;
      final k = wk[h](xkv);
      final v = wv[h](xkv);
      Tensor? effectiveMask = mask;
      if (relativeBiasPerHead != null) {
        effectiveMask = effectiveMask == null
            ? relativeBiasPerHead[h]
            : effectiveMask + relativeBiasPerHead[h];
      }
      heads.add(q.scaledDotProductAttention(k, v, mask: effectiveMask));
    }
    final concat = TensorConcat.concat(heads, axis: 1);
    return wo(concat);
  }

  /// Cached self-attention for autoregressive decoding.
  /// `xqSingle`: `[1, dModel]` — the current token's normed hidden.
  /// Appends this token's K/V to `cache`, then Q attends to the full
  /// cached K/V.
  Tensor callCachedSelf(
    Tensor xqSingle, {
    required MHACache cache,
    List<Tensor>? relativeBiasPerHead,
  }) {
    final heads = <Tensor>[];
    for (int h = 0; h < numHeads; h++) {
      final q = wq[h](xqSingle) * qScale;
      final kNew = wk[h](xqSingle);
      final vNew = wv[h](xqSingle);
      final kFull = cache.appendK(h, kNew);
      final vFull = cache.appendV(h, vNew);
      final bias = relativeBiasPerHead == null ? null : relativeBiasPerHead[h];
      heads.add(q.scaledDotProductAttention(kFull, vFull, mask: bias));
    }
    final concat = TensorConcat.concat(heads, axis: 1);
    return wo(concat);
  }

  /// Cached cross-attention for autoregressive decoding.
  /// `xqSingle`: `[1, dModel]`. K/V come from the precomputed
  /// [T5CrossAttnCache] (see [primeCross]).
  Tensor callCachedCross(Tensor xqSingle, T5CrossAttnCache cache) {
    final heads = <Tensor>[];
    for (int h = 0; h < numHeads; h++) {
      final q = wq[h](xqSingle) * qScale;
      heads.add(q.scaledDotProductAttention(cache.k[h], cache.v[h]));
    }
    final concat = TensorConcat.concat(heads, axis: 1);
    return wo(concat);
  }

  /// Precompute per-head K/V from a fixed encoder `memory` tensor
  /// `[Nk, kvDim]`. Called once per generation, cached across all
  /// decoder steps.
  T5CrossAttnCache primeCross(Tensor memory) {
    final ks = <Tensor>[];
    final vs = <Tensor>[];
    for (int h = 0; h < numHeads; h++) {
      ks.add(wk[h](memory));
      vs.add(wv[h](memory));
    }
    return T5CrossAttnCache(ks, vs);
  }

  @override
  List<Tensor> parameters() => [
    for (final l in wq) ...l.parameters(),
    for (final l in wk) ...l.parameters(),
    for (final l in wv) ...l.parameters(),
    ...wo.parameters(),
  ];

  @override
  List<Module> submodules() => [...wq, ...wk, ...wv, wo];
}

/// Precomputed cross-attention K/V for one decoder block. Populated
/// once from the encoder memory on the first decode step, then
/// reused unchanged for every subsequent step.
class T5CrossAttnCache {
  final List<Tensor> k;
  final List<Tensor> v;
  T5CrossAttnCache(this.k, this.v);
}

/// Per-block cache state for one T5 decoder layer: a running
/// self-attention KV cache plus a lazily-populated cross-attention
/// cache.
class T5DecoderBlockCache {
  final MHACache selfAttn;
  T5CrossAttnCache? crossAttn;
  T5DecoderBlockCache(int numHeads) : selfAttn = MHACache.empty(numHeads);
}

/// Whole-stack cache for autoregressive T5 decoding. `seqLen` is the
/// number of tokens already fed through the decoder (i.e., the
/// position of the next Q).
class T5DecoderCache {
  final List<T5DecoderBlockCache> blocks;
  int seqLen;
  T5DecoderCache(int numLayers, int numHeads)
    : blocks = List.generate(
        numLayers,
        (_) => T5DecoderBlockCache(numHeads),
        growable: false,
      ),
      seqLen = 0;
}

// ---------------------------------------------------------------------------
// FFN
// ---------------------------------------------------------------------------

class T5Ffn extends Module {
  final T5FfnActivation activation;
  final Linear wi0; // "wi" for relu; "wi_0" for gated (the pre-act path)
  final Linear? wi1; // "wi_1" for gated (the gate path), null for relu
  final Linear wo;

  T5Ffn({
    required int dModel,
    required int dFf,
    required this.activation,
    Device device = Device.CPU,
    int seed = 0,
  }) : wi0 = Linear(dModel, dFf, bias: false, device: device, seed: seed),
       wi1 = activation == T5FfnActivation.gatedGelu
           ? Linear(
               dModel,
               dFf,
               bias: false,
               device: device,
               seed: seed + 100000,
             )
           : null,
       wo = Linear(
         dFf,
         dModel,
         bias: false,
         device: device,
         seed: seed + 200000,
       );

  Tensor call(Tensor x) {
    switch (activation) {
      case T5FfnActivation.relu:
        return wo(wi0(x).relu());
      case T5FfnActivation.gatedGelu:
        final act = _geluTanh(wi0(x));
        final gate = wi1!(x);
        return wo(act * gate);
    }
  }

  static Tensor _geluTanh(Tensor x) {
    const c = 0.7978845608028654; // sqrt(2 / pi)
    final inner = (x + x.pow(3) * 0.044715) * c;
    return x * (inner.tanh() + 1.0) * 0.5;
  }

  @override
  List<Tensor> parameters() => [
    ...wi0.parameters(),
    if (wi1 != null) ...wi1!.parameters(),
    ...wo.parameters(),
  ];

  @override
  List<Module> submodules() => [wi0, if (wi1 != null) wi1!, wo];
}

// ---------------------------------------------------------------------------
// Encoder block
// ---------------------------------------------------------------------------

class T5EncoderBlock extends Module {
  final RMSNorm selfAttnNorm;
  final T5Attention selfAttn;
  final RMSNorm ffnNorm;
  final T5Ffn ffn;

  T5EncoderBlock({required T5Config cfg, required int seed})
    : selfAttnNorm = RMSNorm(
        cfg.dModel,
        eps: cfg.layerNormEps,
        device: cfg.device,
      ),
      selfAttn = T5Attention(
        dModel: cfg.dModel,
        kvDim: cfg.dModel,
        numHeads: cfg.numHeads,
        dKv: cfg.dKv,
        device: cfg.device,
        seed: seed,
      ),
      ffnNorm = RMSNorm(cfg.dModel, eps: cfg.layerNormEps, device: cfg.device),
      ffn = T5Ffn(
        dModel: cfg.dModel,
        dFf: cfg.dFf,
        activation: cfg.feedForwardProj,
        device: cfg.device,
        seed: seed + 10000,
      );

  /// `x: [N, dModel]`. `relativeBias`: per-head bias tensors from the
  /// shared encoder [T5RelativeBias] (may be null on layer 0 during
  /// param dump; production callers always pass a valid list).
  Tensor call(Tensor x, {required List<Tensor>? relativeBias}) {
    final a = selfAttn(
      selfAttnNorm(x),
      selfAttnNorm(x),
      relativeBiasPerHead: relativeBias,
    );
    final h = x + a;
    final f = ffn(ffnNorm(h));
    return h + f;
  }

  @override
  List<Tensor> parameters() => [
    ...selfAttnNorm.parameters(),
    ...selfAttn.parameters(),
    ...ffnNorm.parameters(),
    ...ffn.parameters(),
  ];

  @override
  List<Module> submodules() => [selfAttnNorm, selfAttn, ffnNorm, ffn];
}

// ---------------------------------------------------------------------------
// Decoder block
// ---------------------------------------------------------------------------

class T5DecoderBlock extends Module {
  final RMSNorm selfAttnNorm;
  final T5Attention selfAttn;
  final RMSNorm crossAttnNorm;
  final T5Attention crossAttn;
  final RMSNorm ffnNorm;
  final T5Ffn ffn;

  T5DecoderBlock({required T5Config cfg, required int seed})
    : selfAttnNorm = RMSNorm(
        cfg.dModel,
        eps: cfg.layerNormEps,
        device: cfg.device,
      ),
      selfAttn = T5Attention(
        dModel: cfg.dModel,
        kvDim: cfg.dModel,
        numHeads: cfg.numHeads,
        dKv: cfg.dKv,
        device: cfg.device,
        seed: seed,
      ),
      crossAttnNorm = RMSNorm(
        cfg.dModel,
        eps: cfg.layerNormEps,
        device: cfg.device,
      ),
      crossAttn = T5Attention(
        dModel: cfg.dModel,
        kvDim: cfg.dModel,
        numHeads: cfg.numHeads,
        dKv: cfg.dKv,
        device: cfg.device,
        seed: seed + 5000,
      ),
      ffnNorm = RMSNorm(cfg.dModel, eps: cfg.layerNormEps, device: cfg.device),
      ffn = T5Ffn(
        dModel: cfg.dModel,
        dFf: cfg.dFf,
        activation: cfg.feedForwardProj,
        device: cfg.device,
        seed: seed + 10000,
      );

  /// `x: [Nq, dModel]` decoder input. `memory: [Nk, dModel]` encoder
  /// output. `selfCausalMask`: pre-built causal `[Nq, Nq]` mask.
  /// `selfRelativeBias`: per-head bias for the decoder self-attn.
  Tensor call(
    Tensor x, {
    required Tensor memory,
    required Tensor selfCausalMask,
    required List<Tensor>? selfRelativeBias,
  }) {
    final a = selfAttn(
      selfAttnNorm(x),
      selfAttnNorm(x),
      mask: selfCausalMask,
      relativeBiasPerHead: selfRelativeBias,
    );
    var h = x + a;
    final c = crossAttn(crossAttnNorm(h), memory);
    h = h + c;
    final f = ffn(ffnNorm(h));
    return h + f;
  }

  /// Cached forward for autoregressive decoding. `xSingle`:
  /// `[1, dModel]` — the current token's residual input. Uses the
  /// self-attn cache in [blockCache], and lazily primes the cross-
  /// attn cache from `memory` on the first call.
  Tensor callCached(
    Tensor xSingle, {
    required Tensor memory,
    required T5DecoderBlockCache blockCache,
    required List<Tensor> selfRelativeBias,
  }) {
    final normedForSelf = selfAttnNorm(xSingle);
    final a = selfAttn.callCachedSelf(
      normedForSelf,
      cache: blockCache.selfAttn,
      relativeBiasPerHead: selfRelativeBias,
    );
    var h = xSingle + a;
    blockCache.crossAttn ??= crossAttn.primeCross(memory);
    final c = crossAttn.callCachedCross(
      crossAttnNorm(h),
      blockCache.crossAttn!,
    );
    h = h + c;
    final f = ffn(ffnNorm(h));
    return h + f;
  }

  @override
  List<Tensor> parameters() => [
    ...selfAttnNorm.parameters(),
    ...selfAttn.parameters(),
    ...crossAttnNorm.parameters(),
    ...crossAttn.parameters(),
    ...ffnNorm.parameters(),
    ...ffn.parameters(),
  ];

  @override
  List<Module> submodules() => [
    selfAttnNorm,
    selfAttn,
    crossAttnNorm,
    crossAttn,
    ffnNorm,
    ffn,
  ];
}

// ---------------------------------------------------------------------------
// Encoder + Decoder stacks
// ---------------------------------------------------------------------------

class T5Encoder extends Module {
  final T5Config config;
  final Embedding tokenEmbedding;
  final T5RelativeBias relativeBias;
  final List<T5EncoderBlock> blocks;
  final RMSNorm finalNorm;

  T5Encoder(this.config, Embedding sharedEmbedding)
    : tokenEmbedding = sharedEmbedding,
      relativeBias = T5RelativeBias(
        numBuckets: config.relativeAttentionNumBuckets,
        numHeads: config.numHeads,
        maxDistance: config.relativeAttentionMaxDistance,
        bidirectional: true,
        device: config.device,
        seed: config.seed + 700000,
      ),
      blocks = <T5EncoderBlock>[],
      finalNorm = RMSNorm(
        config.dModel,
        eps: config.layerNormEps,
        device: config.device,
      ) {
    for (int i = 0; i < config.numLayers; i++) {
      blocks.add(
        T5EncoderBlock(cfg: config, seed: config.seed + 100000 + i * 1000),
      );
    }
  }

  /// `tokens: [N]` int token ids. Returns encoder hidden `[N, dModel]`.
  Tensor call(Tensor tokens) {
    if (tokens.shape.length != 1) {
      throw ArgumentError('T5Encoder: expected [N]; got ${tokens.shape}');
    }
    final n = tokens.shape[0];
    var h = tokenEmbedding(tokens); // [N, dModel]
    final biasPerHead = relativeBias.maskPerHead(n, n);
    for (final b in blocks) {
      h = b(h, relativeBias: biasPerHead);
    }
    return finalNorm(h);
  }

  @override
  List<Tensor> parameters() => [
    ...relativeBias.parameters(),
    for (final b in blocks) ...b.parameters(),
    ...finalNorm.parameters(),
  ];

  @override
  List<Module> submodules() => [relativeBias, ...blocks, finalNorm];
}

class T5Decoder extends Module {
  final T5Config config;
  final Embedding tokenEmbedding;
  final T5RelativeBias relativeBias;
  final List<T5DecoderBlock> blocks;
  final RMSNorm finalNorm;

  T5Decoder(this.config, Embedding sharedEmbedding)
    : tokenEmbedding = sharedEmbedding,
      relativeBias = T5RelativeBias(
        numBuckets: config.relativeAttentionNumBuckets,
        numHeads: config.numHeads,
        maxDistance: config.relativeAttentionMaxDistance,
        bidirectional: false,
        device: config.device,
        seed: config.seed + 800000,
      ),
      blocks = <T5DecoderBlock>[],
      finalNorm = RMSNorm(
        config.dModel,
        eps: config.layerNormEps,
        device: config.device,
      ) {
    for (int i = 0; i < config.numDecoderLayers; i++) {
      blocks.add(
        T5DecoderBlock(cfg: config, seed: config.seed + 300000 + i * 1000),
      );
    }
  }

  /// `tokens: [Nq]` decoder input token ids. `memory: [Nk, dModel]`
  /// encoder output. Returns decoder hidden `[Nq, dModel]`.
  Tensor call(Tensor tokens, {required Tensor memory}) {
    if (tokens.shape.length != 1) {
      throw ArgumentError('T5Decoder: expected [Nq]; got ${tokens.shape}');
    }
    final nq = tokens.shape[0];
    var h = tokenEmbedding(tokens);
    final causal = causalMask(nq, device: h.device);
    final biasPerHead = relativeBias.maskPerHead(nq, nq);
    for (final b in blocks) {
      h = b(
        h,
        memory: memory,
        selfCausalMask: causal,
        selfRelativeBias: biasPerHead,
      );
    }
    return finalNorm(h);
  }

  /// Cached forward for autoregressive decoding. Feeds a single new
  /// token id, using and updating [cache]. Returns the decoder
  /// hidden `[1, dModel]` for that token. Advances `cache.seqLen`.
  Tensor callCached(
    int newTokenId, {
    required Tensor memory,
    required T5DecoderCache cache,
  }) {
    if (cache.blocks.length != blocks.length) {
      throw ArgumentError(
        'T5Decoder.callCached: cache has ${cache.blocks.length} blocks; '
        'model has ${blocks.length}',
      );
    }
    final qPos = cache.seqLen;
    final tokens = Tensor.fromList(
      [1],
      [newTokenId.toDouble()],
      device: config.device,
    );
    var h = tokenEmbedding(tokens);
    final biasPerHead = relativeBias.maskPerHeadSingleQ(qPos, qPos + 1);
    for (int i = 0; i < blocks.length; i++) {
      h = blocks[i].callCached(
        h,
        memory: memory,
        blockCache: cache.blocks[i],
        selfRelativeBias: biasPerHead,
      );
    }
    cache.seqLen = qPos + 1;
    return finalNorm(h);
  }

  @override
  List<Tensor> parameters() => [
    ...relativeBias.parameters(),
    for (final b in blocks) ...b.parameters(),
    ...finalNorm.parameters(),
  ];

  @override
  List<Module> submodules() => [relativeBias, ...blocks, finalNorm];
}

// ---------------------------------------------------------------------------
// Full T5 model
// ---------------------------------------------------------------------------

class T5Model extends Module {
  final T5Config config;
  final Embedding sharedEmbedding;
  final T5Encoder encoder;
  final T5Decoder decoder;

  /// Always allocated. The loader may point this at the checkpoint's
  /// `lm_head.weight` (untied) or copy `shared.weight` into it
  /// (tied). See [useUntiedLmHead].
  final Linear lmHead;

  /// The `1/sqrt(dModel)` rescale HF applies when embeddings are
  /// tied. Used at inference iff [useUntiedLmHead] is false.
  final double lmHeadScale;

  /// Flipped to `true` by the loader when the checkpoint ships a
  /// distinct `lm_head.weight`. When true, [logitsLastToken] skips
  /// the `1/sqrt(dModel)` rescale (mirrors HF's behaviour for
  /// checkpoints where `shared.weight != lm_head.weight`, e.g.
  /// `google/flan-t5-*`).
  bool useUntiedLmHead;

  T5Model._(
    this.config,
    this.sharedEmbedding,
    this.encoder,
    this.decoder,
    this.lmHead,
    this.lmHeadScale,
    this.useUntiedLmHead,
  );

  factory T5Model(T5Config config) {
    final shared = Embedding(
      config.vocabSize,
      config.dModel,
      device: config.device,
      seed: config.seed,
    );
    final encoder = T5Encoder(config, shared);
    final decoder = T5Decoder(config, shared);
    final lmHead = Linear(
      config.dModel,
      config.vocabSize,
      bias: false,
      device: config.device,
      seed: config.seed + 999000,
    );
    final scale = 1.0 / math.sqrt(config.dModel);
    // Default: assume tied per config. Loader flips to untied if it
    // finds lm_head.weight in the checkpoint.
    return T5Model._(
      config,
      shared,
      encoder,
      decoder,
      lmHead,
      scale,
      /*useUntiedLmHead=*/ !config.tieWordEmbeddings,
    );
  }

  /// Runs the encoder over `srcTokens` and returns the `[N, dModel]`
  /// memory tensor. Call once, feed the result to [decodeStep] or
  /// [generate].
  Tensor encode(List<int> srcTokens) {
    final t = Tensor.fromList(
      [srcTokens.length],
      srcTokens.map((i) => i.toDouble()).toList(),
      device: config.device,
    );
    return encoder(t);
  }

  /// One decoder forward pass. `tgtTokens` is the growing decoded
  /// prefix (starting with `[decoderStartTokenId]`, typically 0).
  /// Returns `[vocabSize]` logits for the last position.
  Tensor logitsLastToken(List<int> tgtTokens, Tensor memory) {
    final t = Tensor.fromList(
      [tgtTokens.length],
      tgtTokens.map((i) => i.toDouble()).toList(),
      device: config.device,
    );
    var h = decoder(t, memory: memory);
    if (!useUntiedLmHead) {
      h = h * lmHeadScale;
    }
    // Slice last row.
    final n = tgtTokens.length;
    final d = config.dModel;
    final flat = h.toList();
    final lastRow = List<double>.filled(d, 0);
    final base = (n - 1) * d;
    for (int j = 0; j < d; j++) {
      lastRow[j] = flat[base + j];
    }
    final lastT = Tensor.fromList([1, d], lastRow, device: h.device);
    return lmHead(lastT).reshape([config.vocabSize]);
  }

  /// Greedy generation. Encodes once, then decodes up to
  /// [maxNewTokens] tokens, stopping on `eosTokenId` (default 1).
  /// Uses the fast KV-cached decoder path by default; pass
  /// `useCache: false` to fall back to the recompute-everything
  /// path (useful for A/B numerical checks).
  List<int> generate(
    List<int> srcTokens, {
    int maxNewTokens = 32,
    int decoderStartTokenId = 0,
    int eosTokenId = 1,
    bool useCache = true,
  }) {
    final memory = encode(srcTokens);
    if (!useCache) {
      final out = <int>[decoderStartTokenId];
      for (int step = 0; step < maxNewTokens; step++) {
        final logits = logitsLastToken(out, memory);
        final data = logits.toList();
        var bestIdx = 0;
        var bestVal = data[0];
        for (int i = 1; i < data.length; i++) {
          if (data[i] > bestVal) {
            bestVal = data[i];
            bestIdx = i;
          }
        }
        out.add(bestIdx);
        if (bestIdx == eosTokenId) break;
      }
      return out;
    }
    return _generateCached(
      memory: memory,
      maxNewTokens: maxNewTokens,
      decoderStartTokenId: decoderStartTokenId,
      eosTokenId: eosTokenId,
    );
  }

  List<int> _generateCached({
    required Tensor memory,
    required int maxNewTokens,
    required int decoderStartTokenId,
    required int eosTokenId,
  }) {
    final cache = T5DecoderCache(config.numDecoderLayers, config.numHeads);
    final out = <int>[decoderStartTokenId];
    var feed = decoderStartTokenId;
    for (int step = 0; step < maxNewTokens; step++) {
      var h = decoder.callCached(feed, memory: memory, cache: cache);
      if (!useUntiedLmHead) {
        h = h * lmHeadScale;
      }
      final logits = lmHead(h).reshape([config.vocabSize]);
      final data = logits.toList();
      var bestIdx = 0;
      var bestVal = data[0];
      for (int i = 1; i < data.length; i++) {
        if (data[i] > bestVal) {
          bestVal = data[i];
          bestIdx = i;
        }
      }
      out.add(bestIdx);
      if (bestIdx == eosTokenId) break;
      feed = bestIdx;
    }
    return out;
  }

  @override
  List<Tensor> parameters() => [
    ...sharedEmbedding.parameters(),
    ...encoder.parameters(),
    ...decoder.parameters(),
    ...lmHead.parameters(),
  ];

  @override
  List<Module> submodules() => [sharedEmbedding, encoder, decoder, lmHead];
}
