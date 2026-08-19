/// Marian NMT (Helsinki-NLP Opus-MT) encoder-decoder architecture.
///
/// Ports the HuggingFace `MarianMTModel` for the Opus-MT translation
/// family (~74 M dense params for the base config). Architecturally
/// a classic Vaswani-post-LN Transformer with two twists:
///
///   * Sinusoidal position embeddings **stored in the checkpoint**
///     (not recomputed), added to token embeddings, both encoder and
///     decoder.
///   * A `final_logits_bias` `[1, vocab]` added to the LM head output
///     (Bart legacy, inherited via MarianForConditionalGeneration).
///
/// Other Marian-isms compared to T5:
///   * Post-LN (`x' = LN(x + Sub(x))`), not pre-LN.
///   * Standard `LayerNorm` with bias (not RMSNorm).
///   * All Q/K/V/O and FFN Linears have biases.
///   * Standard SDPA scale (`1/sqrt(d_kv)`) — no unscaled attention.
///   * Plain FFN `w2(silu(w1(x) + b1)) + b2` — not gated.
///   * Input embeddings scaled by `sqrt(d_model)` (opt-in via
///     `scaleEmbeddings`).
///   * Tied encoder/decoder/lm_head embeddings.
library;

import 'dart:math' as math;

import '../tensor/tensor.dart';
import 'embedding.dart';
import 'kv_cache.dart';
import 'layer_norm.dart';
import 'linear.dart';
import 'masks.dart';
import 'module.dart';

enum MarianActivation { silu, relu, gelu }

class MarianConfig {
  final int vocabSize;
  final int dModel;
  final int ffnDim;
  final int numLayers;
  final int numDecoderLayers;
  final int numHeads;
  final int maxPositionEmbeddings;
  final int padTokenId;
  final int eosTokenId;
  final int decoderStartTokenId;
  final double layerNormEps;
  final bool scaleEmbeddings;
  final bool addFinalLayerNorm;
  final MarianActivation activation;
  final Device device;
  final int seed;

  const MarianConfig({
    required this.vocabSize,
    this.dModel = 512,
    this.ffnDim = 2048,
    this.numLayers = 6,
    this.numDecoderLayers = 6,
    this.numHeads = 8,
    this.maxPositionEmbeddings = 512,
    required this.padTokenId,
    this.eosTokenId = 0,
    required this.decoderStartTokenId,
    this.layerNormEps = 1e-5,
    this.scaleEmbeddings = true,
    this.addFinalLayerNorm = false,
    this.activation = MarianActivation.silu,
    this.device = Device.CPU,
    this.seed = 0,
  });

  int get headDim => dModel ~/ numHeads;
}

// ---------------------------------------------------------------------------
// Attention (self + cross)
// ---------------------------------------------------------------------------

/// Per-head Q/K/V/O attention with biases, standard SDPA scaling.
/// Used for both self- and cross-attention.
class MarianAttention extends Module {
  final int dModel;
  final int numHeads;
  final int headDim;
  final List<Linear> wq;
  final List<Linear> wk;
  final List<Linear> wv;
  final Linear wo;

  MarianAttention({
    required this.dModel,
    required this.numHeads,
    Device device = Device.CPU,
    int seed = 0,
  }) : headDim = dModel ~/ numHeads,
       wq = List<Linear>.generate(
         numHeads,
         (h) => Linear(
           dModel,
           dModel ~/ numHeads,
           bias: true,
           device: device,
           seed: seed + h,
         ),
       ),
       wk = List<Linear>.generate(
         numHeads,
         (h) => Linear(
           dModel,
           dModel ~/ numHeads,
           bias: true,
           device: device,
           seed: seed + 1000 + h,
         ),
       ),
       wv = List<Linear>.generate(
         numHeads,
         (h) => Linear(
           dModel,
           dModel ~/ numHeads,
           bias: true,
           device: device,
           seed: seed + 2000 + h,
         ),
       ),
       wo = Linear(
         dModel,
         dModel,
         bias: true,
         device: device,
         seed: seed + 3000,
       );

  /// Full-sequence forward. `xq`: `[Nq, dModel]`, `xkv`: `[Nk, dModel]`.
  /// Optional additive `mask` (e.g. causal) broadcast to `[Nq, Nk]`.
  Tensor call(Tensor xq, Tensor xkv, {Tensor? mask}) {
    final heads = <Tensor>[];
    for (int h = 0; h < numHeads; h++) {
      final q = wq[h](xq);
      final k = wk[h](xkv);
      final v = wv[h](xkv);
      heads.add(q.scaledDotProductAttention(k, v, mask: mask));
    }
    final concat = TensorConcat.concat(heads, axis: 1);
    return wo(concat);
  }

  /// Cached self-attention. Appends new K/V to `cache`, then Q
  /// attends to full cached K/V.
  Tensor callCachedSelf(Tensor xqSingle, {required MHACache cache}) {
    final heads = <Tensor>[];
    for (int h = 0; h < numHeads; h++) {
      final q = wq[h](xqSingle);
      final kNew = wk[h](xqSingle);
      final vNew = wv[h](xqSingle);
      final kFull = cache.appendK(h, kNew);
      final vFull = cache.appendV(h, vNew);
      heads.add(q.scaledDotProductAttention(kFull, vFull));
    }
    final concat = TensorConcat.concat(heads, axis: 1);
    return wo(concat);
  }

  /// Cached cross-attention. K/V come from the precomputed
  /// [MarianCrossAttnCache].
  Tensor callCachedCross(Tensor xqSingle, MarianCrossAttnCache cache) {
    final heads = <Tensor>[];
    for (int h = 0; h < numHeads; h++) {
      final q = wq[h](xqSingle);
      heads.add(q.scaledDotProductAttention(cache.k[h], cache.v[h]));
    }
    final concat = TensorConcat.concat(heads, axis: 1);
    return wo(concat);
  }

  /// Precompute per-head K/V from a fixed encoder `memory` [Nk, dModel].
  MarianCrossAttnCache primeCross(Tensor memory) {
    final ks = <Tensor>[];
    final vs = <Tensor>[];
    for (int h = 0; h < numHeads; h++) {
      ks.add(wk[h](memory));
      vs.add(wv[h](memory));
    }
    return MarianCrossAttnCache(ks, vs);
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

/// Precomputed cross-attn K/V for one Marian decoder block.
class MarianCrossAttnCache {
  final List<Tensor> k;
  final List<Tensor> v;
  MarianCrossAttnCache(this.k, this.v);
}

class MarianDecoderBlockCache {
  final MHACache selfAttn;
  MarianCrossAttnCache? crossAttn;
  MarianDecoderBlockCache(int numHeads) : selfAttn = MHACache.empty(numHeads);
  MarianDecoderBlockCache._from(this.selfAttn, this.crossAttn);

  /// Shallow clone. Cross-attn K/V (encoder-derived, immutable
  /// across decode steps) are shared by reference.
  MarianDecoderBlockCache clone() =>
      MarianDecoderBlockCache._from(selfAttn.clone(), crossAttn);
}

class MarianDecoderCache {
  final List<MarianDecoderBlockCache> blocks;
  int seqLen;
  MarianDecoderCache(int numLayers, int numHeads)
    : blocks = List.generate(
        numLayers,
        (_) => MarianDecoderBlockCache(numHeads),
        growable: false,
      ),
      seqLen = 0;
  MarianDecoderCache._from(this.blocks, this.seqLen);

  MarianDecoderCache clone() =>
      MarianDecoderCache._from([for (final b in blocks) b.clone()], seqLen);
}

// ---------------------------------------------------------------------------
// FFN (plain, not gated)
// ---------------------------------------------------------------------------

class MarianFfn extends Module {
  final MarianActivation activation;
  final Linear fc1;
  final Linear fc2;

  MarianFfn({
    required int dModel,
    required int ffnDim,
    required this.activation,
    Device device = Device.CPU,
    int seed = 0,
  }) : fc1 = Linear(dModel, ffnDim, bias: true, device: device, seed: seed),
       fc2 = Linear(
         ffnDim,
         dModel,
         bias: true,
         device: device,
         seed: seed + 200000,
       );

  Tensor call(Tensor x) {
    var h = fc1(x);
    switch (activation) {
      case MarianActivation.silu:
        h = h * h.sigmoid();
        break;
      case MarianActivation.relu:
        h = h.relu();
        break;
      case MarianActivation.gelu:
        h = _geluTanh(h);
        break;
    }
    return fc2(h);
  }

  static Tensor _geluTanh(Tensor x) {
    const c = 0.7978845608028654;
    final inner = (x + x.pow(3) * 0.044715) * c;
    return x * (inner.tanh() + 1.0) * 0.5;
  }

  @override
  List<Tensor> parameters() => [...fc1.parameters(), ...fc2.parameters()];

  @override
  List<Module> submodules() => [fc1, fc2];
}

// ---------------------------------------------------------------------------
// Encoder + decoder blocks (post-LN)
// ---------------------------------------------------------------------------

class MarianEncoderBlock extends Module {
  final MarianAttention selfAttn;
  final LayerNorm selfAttnLn;
  final MarianFfn ffn;
  final LayerNorm finalLn;

  MarianEncoderBlock({required MarianConfig cfg, required int seed})
    : selfAttn = MarianAttention(
        dModel: cfg.dModel,
        numHeads: cfg.numHeads,
        device: cfg.device,
        seed: seed,
      ),
      selfAttnLn = LayerNorm(
        cfg.dModel,
        eps: cfg.layerNormEps,
        device: cfg.device,
      ),
      ffn = MarianFfn(
        dModel: cfg.dModel,
        ffnDim: cfg.ffnDim,
        activation: cfg.activation,
        device: cfg.device,
        seed: seed + 10000,
      ),
      finalLn = LayerNorm(
        cfg.dModel,
        eps: cfg.layerNormEps,
        device: cfg.device,
      );

  Tensor call(Tensor x) {
    var h = selfAttnLn(x + selfAttn(x, x));
    h = finalLn(h + ffn(h));
    return h;
  }

  @override
  List<Tensor> parameters() => [
    ...selfAttn.parameters(),
    ...selfAttnLn.parameters(),
    ...ffn.parameters(),
    ...finalLn.parameters(),
  ];

  @override
  List<Module> submodules() => [selfAttn, selfAttnLn, ffn, finalLn];
}

class MarianDecoderBlock extends Module {
  final MarianAttention selfAttn;
  final LayerNorm selfAttnLn;
  final MarianAttention crossAttn;
  final LayerNorm crossAttnLn;
  final MarianFfn ffn;
  final LayerNorm finalLn;

  MarianDecoderBlock({required MarianConfig cfg, required int seed})
    : selfAttn = MarianAttention(
        dModel: cfg.dModel,
        numHeads: cfg.numHeads,
        device: cfg.device,
        seed: seed,
      ),
      selfAttnLn = LayerNorm(
        cfg.dModel,
        eps: cfg.layerNormEps,
        device: cfg.device,
      ),
      crossAttn = MarianAttention(
        dModel: cfg.dModel,
        numHeads: cfg.numHeads,
        device: cfg.device,
        seed: seed + 5000,
      ),
      crossAttnLn = LayerNorm(
        cfg.dModel,
        eps: cfg.layerNormEps,
        device: cfg.device,
      ),
      ffn = MarianFfn(
        dModel: cfg.dModel,
        ffnDim: cfg.ffnDim,
        activation: cfg.activation,
        device: cfg.device,
        seed: seed + 10000,
      ),
      finalLn = LayerNorm(
        cfg.dModel,
        eps: cfg.layerNormEps,
        device: cfg.device,
      );

  Tensor call(
    Tensor x, {
    required Tensor memory,
    required Tensor selfCausalMask,
  }) {
    var h = selfAttnLn(x + selfAttn(x, x, mask: selfCausalMask));
    h = crossAttnLn(h + crossAttn(h, memory));
    h = finalLn(h + ffn(h));
    return h;
  }

  Tensor callCached(
    Tensor xSingle, {
    required Tensor memory,
    required MarianDecoderBlockCache blockCache,
  }) {
    var h = selfAttnLn(
      xSingle + selfAttn.callCachedSelf(xSingle, cache: blockCache.selfAttn),
    );
    blockCache.crossAttn ??= crossAttn.primeCross(memory);
    h = crossAttnLn(h + crossAttn.callCachedCross(h, blockCache.crossAttn!));
    h = finalLn(h + ffn(h));
    return h;
  }

  @override
  List<Tensor> parameters() => [
    ...selfAttn.parameters(),
    ...selfAttnLn.parameters(),
    ...crossAttn.parameters(),
    ...crossAttnLn.parameters(),
    ...ffn.parameters(),
    ...finalLn.parameters(),
  ];

  @override
  List<Module> submodules() => [
    selfAttn,
    selfAttnLn,
    crossAttn,
    crossAttnLn,
    ffn,
    finalLn,
  ];
}

// ---------------------------------------------------------------------------
// Encoder + decoder stacks
// ---------------------------------------------------------------------------

class MarianEncoder extends Module {
  final MarianConfig config;
  final Embedding sharedEmbedding;

  /// Sinusoidal position embedding table. Loaded from the safetensors
  /// (`model.encoder.embed_positions.weight`) instead of recomputed.
  final Tensor positionEmbeddings;
  final List<MarianEncoderBlock> blocks;
  final LayerNorm? finalLn;
  final double embedScale;

  MarianEncoder(this.config, this.sharedEmbedding)
    : positionEmbeddings = Tensor.fill(
        [config.maxPositionEmbeddings, config.dModel],
        0.0,
        device: config.device,
      ),
      blocks = <MarianEncoderBlock>[],
      finalLn = config.addFinalLayerNorm
          ? LayerNorm(
              config.dModel,
              eps: config.layerNormEps,
              device: config.device,
            )
          : null,
      embedScale = config.scaleEmbeddings
          ? math.sqrt(config.dModel.toDouble())
          : 1.0 {
    for (int i = 0; i < config.numLayers; i++) {
      blocks.add(
        MarianEncoderBlock(cfg: config, seed: config.seed + 100000 + i * 1000),
      );
    }
  }

  /// `tokens`: `[N]` int ids. Returns `[N, dModel]`.
  Tensor call(Tensor tokens) {
    if (tokens.shape.length != 1) {
      throw ArgumentError('MarianEncoder: expected [N]; got ${tokens.shape}');
    }
    final n = tokens.shape[0];
    var h = sharedEmbedding(tokens) * embedScale;
    h = h + _positionSlice(n);
    for (final b in blocks) {
      h = b(h);
    }
    if (finalLn != null) h = finalLn!(h);
    return h;
  }

  Tensor _positionSlice(int n) {
    if (n > config.maxPositionEmbeddings) {
      throw ArgumentError(
        'MarianEncoder: seq length $n > maxPositionEmbeddings '
        '${config.maxPositionEmbeddings}',
      );
    }
    // Row-slice of the positional table.
    final all = positionEmbeddings.toList();
    final d = config.dModel;
    final vals = List<double>.filled(n * d, 0);
    for (int i = 0; i < n * d; i++) {
      vals[i] = all[i];
    }
    return Tensor.fromList([n, d], vals, device: positionEmbeddings.device);
  }

  @override
  List<Tensor> parameters() => [
    for (final b in blocks) ...b.parameters(),
    if (finalLn != null) ...finalLn!.parameters(),
  ];

  @override
  List<Module> submodules() => [...blocks, if (finalLn != null) finalLn!];
}

class MarianDecoder extends Module {
  final MarianConfig config;
  final Embedding sharedEmbedding;
  final Tensor positionEmbeddings;
  final List<MarianDecoderBlock> blocks;
  final LayerNorm? finalLn;
  final double embedScale;

  MarianDecoder(this.config, this.sharedEmbedding)
    : positionEmbeddings = Tensor.fill(
        [config.maxPositionEmbeddings, config.dModel],
        0.0,
        device: config.device,
      ),
      blocks = <MarianDecoderBlock>[],
      finalLn = config.addFinalLayerNorm
          ? LayerNorm(
              config.dModel,
              eps: config.layerNormEps,
              device: config.device,
            )
          : null,
      embedScale = config.scaleEmbeddings
          ? math.sqrt(config.dModel.toDouble())
          : 1.0 {
    for (int i = 0; i < config.numDecoderLayers; i++) {
      blocks.add(
        MarianDecoderBlock(cfg: config, seed: config.seed + 300000 + i * 1000),
      );
    }
  }

  /// Non-cached forward. `tokens`: `[Nq]`. Returns `[Nq, dModel]`.
  Tensor call(Tensor tokens, {required Tensor memory}) {
    if (tokens.shape.length != 1) {
      throw ArgumentError('MarianDecoder: expected [Nq]; got ${tokens.shape}');
    }
    final nq = tokens.shape[0];
    var h = sharedEmbedding(tokens) * embedScale;
    h = h + _positionSlice(nq);
    final causal = causalMask(nq, device: h.device);
    for (final b in blocks) {
      h = b(h, memory: memory, selfCausalMask: causal);
    }
    if (finalLn != null) h = finalLn!(h);
    return h;
  }

  /// Cached decoder step. Feeds a single new token id and updates
  /// `cache.seqLen`.
  Tensor callCached(
    int newTokenId, {
    required Tensor memory,
    required MarianDecoderCache cache,
  }) {
    if (cache.blocks.length != blocks.length) {
      throw ArgumentError(
        'MarianDecoder.callCached: cache/model layer count mismatch',
      );
    }
    final qPos = cache.seqLen;
    if (qPos >= config.maxPositionEmbeddings) {
      throw ArgumentError(
        'MarianDecoder.callCached: qPos $qPos exceeds max_position_embeddings '
        '${config.maxPositionEmbeddings}',
      );
    }
    final tokens = Tensor.fromList(
      [1],
      [newTokenId.toDouble()],
      device: config.device,
    );
    var h = sharedEmbedding(tokens) * embedScale;
    h = h + _positionRow(qPos);
    for (int i = 0; i < blocks.length; i++) {
      h = blocks[i].callCached(h, memory: memory, blockCache: cache.blocks[i]);
    }
    if (finalLn != null) h = finalLn!(h);
    cache.seqLen = qPos + 1;
    return h;
  }

  Tensor _positionSlice(int n) {
    if (n > config.maxPositionEmbeddings) {
      throw ArgumentError(
        'MarianDecoder: seq length $n > maxPositionEmbeddings '
        '${config.maxPositionEmbeddings}',
      );
    }
    final all = positionEmbeddings.toList();
    final d = config.dModel;
    final vals = List<double>.filled(n * d, 0);
    for (int i = 0; i < n * d; i++) {
      vals[i] = all[i];
    }
    return Tensor.fromList([n, d], vals, device: positionEmbeddings.device);
  }

  Tensor _positionRow(int pos) {
    final all = positionEmbeddings.toList();
    final d = config.dModel;
    final vals = List<double>.filled(d, 0);
    final base = pos * d;
    for (int j = 0; j < d; j++) {
      vals[j] = all[base + j];
    }
    return Tensor.fromList([1, d], vals, device: positionEmbeddings.device);
  }

  @override
  List<Tensor> parameters() => [
    for (final b in blocks) ...b.parameters(),
    if (finalLn != null) ...finalLn!.parameters(),
  ];

  @override
  List<Module> submodules() => [...blocks, if (finalLn != null) finalLn!];
}

// ---------------------------------------------------------------------------
// Full model
// ---------------------------------------------------------------------------

class MarianModel extends Module {
  final MarianConfig config;
  final Embedding sharedEmbedding;
  final MarianEncoder encoder;
  final MarianDecoder decoder;

  /// `[1, vocab]` bias added to logits before argmax. Loaded from
  /// `final_logits_bias` in the safetensors (Bart legacy).
  final Tensor finalLogitsBias;

  MarianModel._(
    this.config,
    this.sharedEmbedding,
    this.encoder,
    this.decoder,
    this.finalLogitsBias,
  );

  factory MarianModel(MarianConfig config) {
    final shared = Embedding(
      config.vocabSize,
      config.dModel,
      device: config.device,
      seed: config.seed,
    );
    final enc = MarianEncoder(config, shared);
    final dec = MarianDecoder(config, shared);
    final bias = Tensor.fill([1, config.vocabSize], 0.0, device: config.device);
    return MarianModel._(config, shared, enc, dec, bias);
  }

  Tensor encode(List<int> srcTokens) {
    final t = Tensor.fromList(
      [srcTokens.length],
      srcTokens.map((i) => i.toDouble()).toList(),
      device: config.device,
    );
    return encoder(t);
  }

  /// Non-cached: rebuilds decoder over the whole prefix each call.
  Tensor logitsLastToken(List<int> tgtTokens, Tensor memory) {
    final t = Tensor.fromList(
      [tgtTokens.length],
      tgtTokens.map((i) => i.toDouble()).toList(),
      device: config.device,
    );
    final h = decoder(t, memory: memory);
    final n = tgtTokens.length;
    final d = config.dModel;
    final flat = h.toList();
    final lastRow = List<double>.filled(d, 0);
    final base = (n - 1) * d;
    for (int j = 0; j < d; j++) {
      lastRow[j] = flat[base + j];
    }
    final lastT = Tensor.fromList([1, d], lastRow, device: h.device);
    final logits = lastT.matmul(sharedEmbedding.weight.transpose());
    return (logits + finalLogitsBias).reshape([config.vocabSize]);
  }

  /// Greedy generation (KV cached by default).
  List<int> generate(
    List<int> srcTokens, {
    int maxNewTokens = 40,
    int? decoderStartTokenId,
    int? eosTokenId,
    bool useCache = true,
  }) {
    final memory = encode(srcTokens);
    final startId = decoderStartTokenId ?? config.decoderStartTokenId;
    final endId = eosTokenId ?? config.eosTokenId;
    if (!useCache) {
      final out = <int>[startId];
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
        if (bestIdx == endId) break;
      }
      return out;
    }
    return _generateCached(
      memory: memory,
      maxNewTokens: maxNewTokens,
      startId: startId,
      endId: endId,
    );
  }

  List<int> _generateCached({
    required Tensor memory,
    required int maxNewTokens,
    required int startId,
    required int endId,
  }) {
    final cache = MarianDecoderCache(config.numDecoderLayers, config.numHeads);
    final out = <int>[startId];
    var feed = startId;
    for (int step = 0; step < maxNewTokens; step++) {
      final h = decoder.callCached(feed, memory: memory, cache: cache);
      // h: [1, dModel] -> logits [1, vocab] via tied embedding.
      final logits =
          h.matmul(sharedEmbedding.weight.transpose()) + finalLogitsBias;
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
      if (bestIdx == endId) break;
      feed = bestIdx;
    }
    return out;
  }

  /// Beam-search generation with length normalization.
  ///
  /// Maintains [numBeams] parallel hypotheses each with its own KV
  /// cache. At every step, expands each active beam to `numBeams`
  /// candidate continuations (top-K logits per beam), keeps the
  /// global top-`numBeams` by cumulative log-probability, and forks
  /// caches (via [MarianDecoderCache.clone]) when two children share
  /// a parent.
  ///
  /// Finished beams (those that emitted `eosTokenId`) are stashed
  /// with a length-normalized score `sum_log_prob / len^lengthPenalty`
  /// (Wu et al. 2016). Generation stops when `numBeams` beams have
  /// finished OR the active pool is empty OR `maxNewTokens` is hit.
  /// Returns the best finished beam's token sequence.
  ///
  /// Falls back to greedy when `numBeams == 1`.
  List<int> generateBeam(
    List<int> srcTokens, {
    int numBeams = 4,
    int maxNewTokens = 60,
    double lengthPenalty = 0.6,
    int? decoderStartTokenId,
    int? eosTokenId,
  }) {
    if (numBeams < 1) {
      throw ArgumentError('generateBeam: numBeams must be >= 1');
    }
    if (numBeams == 1) {
      return generate(
        srcTokens,
        maxNewTokens: maxNewTokens,
        decoderStartTokenId: decoderStartTokenId,
        eosTokenId: eosTokenId,
      );
    }
    final memory = encode(srcTokens);
    final startId = decoderStartTokenId ?? config.decoderStartTokenId;
    final endId = eosTokenId ?? config.eosTokenId;

    // Active beams. Score is cumulative log-prob (not normalized).
    var active = <_Beam>[
      _Beam(
        seq: [startId],
        cache: MarianDecoderCache(config.numDecoderLayers, config.numHeads),
        score: 0.0,
      ),
    ];
    final finished = <_Beam>[];

    for (int step = 0; step < maxNewTokens; step++) {
      // Score every (beam, next_token) candidate.
      final candidates = <_Candidate>[];
      for (int b = 0; b < active.length; b++) {
        final beam = active[b];
        final feed = beam.seq.last;
        // Advance beam's cache by feeding its last token.
        final h = decoder.callCached(feed, memory: memory, cache: beam.cache);
        final logits =
            h.matmul(sharedEmbedding.weight.transpose()) + finalLogitsBias;
        final data = logits.toList();
        final logProbs = _logSoftmax(data);
        // Top-K per beam is sufficient (any lower rank can't survive
        // the global top-K prune).
        final topPerBeam = _topKIndices(logProbs, numBeams);
        for (final idx in topPerBeam) {
          candidates.add(
            _Candidate(
              beamIdx: b,
              tokenId: idx,
              score: beam.score + logProbs[idx],
            ),
          );
        }
      }

      // Global top-K.
      candidates.sort((a, b) => b.score.compareTo(a.score));
      final next = <_Beam>[];
      for (final c in candidates) {
        if (next.length >= numBeams) break;
        final parent = active[c.beamIdx];
        final newSeq = List<int>.of(parent.seq)..add(c.tokenId);
        if (c.tokenId == endId) {
          // Finished — length-normalized score.
          final len = newSeq.length - 1; // exclude the start token
          final norm = c.score / math.pow(len.toDouble(), lengthPenalty);
          finished.add(_Beam(seq: newSeq, cache: parent.cache, score: norm));
          continue;
        }
        // Fork the cache — parent may be used by another top candidate.
        next.add(
          _Beam(seq: newSeq, cache: parent.cache.clone(), score: c.score),
        );
      }

      if (finished.length >= numBeams) break;
      if (next.isEmpty) break;
      active = next;
    }

    // Prefer finished beams; fall back to best active (with length norm).
    if (finished.isEmpty) {
      for (final b in active) {
        final len = b.seq.length - 1;
        final norm = b.score / math.pow(len.toDouble(), lengthPenalty);
        finished.add(_Beam(seq: b.seq, cache: b.cache, score: norm));
      }
    }
    finished.sort((a, b) => b.score.compareTo(a.score));
    return finished.first.seq;
  }

  /// Row-wise log-softmax over a single row of length V. Numerically
  /// stable via max-subtraction.
  static List<double> _logSoftmax(List<double> logits) {
    var maxV = logits[0];
    for (int i = 1; i < logits.length; i++) {
      if (logits[i] > maxV) maxV = logits[i];
    }
    var sumExp = 0.0;
    for (int i = 0; i < logits.length; i++) {
      sumExp += math.exp(logits[i] - maxV);
    }
    final logSum = math.log(sumExp);
    final out = List<double>.filled(logits.length, 0);
    for (int i = 0; i < logits.length; i++) {
      out[i] = (logits[i] - maxV) - logSum;
    }
    return out;
  }

  /// Indices of the top-K values in `data`, in descending order.
  static List<int> _topKIndices(List<double> data, int k) {
    final n = data.length;
    final kk = k < n ? k : n;
    final idx = List<int>.generate(n, (i) => i);
    // Partial sort is nice but not built-in; full sort is fine at
    // vocab sizes we care about (< 60k) and k around 4-8.
    idx.sort((a, b) => data[b].compareTo(data[a]));
    return idx.sublist(0, kk);
  }

  @override
  List<Tensor> parameters() => [
    ...sharedEmbedding.parameters(),
    ...encoder.parameters(),
    ...decoder.parameters(),
  ];

  @override
  List<Module> submodules() => [sharedEmbedding, encoder, decoder];
}

class _Beam {
  final List<int> seq;
  final MarianDecoderCache cache;
  final double score;
  _Beam({required this.seq, required this.cache, required this.score});
}

class _Candidate {
  final int beamIdx;
  final int tokenId;
  final double score;
  _Candidate({
    required this.beamIdx,
    required this.tokenId,
    required this.score,
  });
}
