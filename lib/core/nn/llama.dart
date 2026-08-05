/// Llama-style causal language model.
///
/// Architectural pieces vs. [GPT] (GPT-2) / [PythiaModel] (GPT-NeoX):
///
///   * **Pre-RMSNorm** on both attention and FFN sub-blocks (no bias,
///     no mean-subtract) — see [RMSNorm].
///   * **RoPE full-rotation** applied per KV/Q head, base is
///     `500_000.0` for Llama 3 (`10_000.0` for Llama 1/2).
///   * **Grouped-Query Attention** — `numKvHeads` is separate from
///     `numHeads` (see [MultiHeadAttention]). For Llama 3.2 1B/3B and
///     Llama 3 8B this is `8` vs. `numHeads == 32`.
///   * **SwiGLU** FFN — three bias-free `Linear`s
///     (`gate_proj`, `up_proj`, `down_proj`), see [SwiGluFfn].
///   * **Sequential residual** — `x + attn(rmsNorm(x))` then
///     `x + swiglu(rmsNorm(x))` (Pythia is parallel).
///   * **Weight-tied lm_head** by default (Llama 3.2 1B/3B tie
///     `embed_tokens ↔ lm_head`; Llama 3 8B does not — pass
///     `tieWeights: false`).
///   * **No biases anywhere**.
///
/// Same 2D single-sequence convention as [GPT.generate] / [PythiaModel].
library;

import 'dart:math' as math;

import '../tensor/tensor.dart';
import 'attention/multi_head_attention.dart';
import 'embedding.dart';
import 'ffn/swiglu.dart';
import 'kv_cache.dart';
import 'linear.dart';
import 'masks.dart';
import 'module.dart';
import 'rms_norm.dart';
import 'rotary.dart';

class LlamaConfig {
  final int vocabSize;
  final int maxCtx;
  final int embedDim;
  final int numLayers;
  final int numHeads;
  final int numKvHeads;
  final int ffnDim;
  final double ropeBase;
  final double rmsNormEps;
  final bool tieWeights;
  final Device device;
  final int seed;

  LlamaConfig({
    required this.vocabSize,
    required this.maxCtx,
    required this.embedDim,
    required this.numLayers,
    required this.numHeads,
    int? numKvHeads,
    required this.ffnDim,
    this.ropeBase = 500000.0,
    this.rmsNormEps = 1e-5,
    this.tieWeights = true,
    this.device = Device.CPU,
    this.seed = 0,
  }) : numKvHeads = numKvHeads ?? numHeads {
    if (embedDim % numHeads != 0) {
      throw ArgumentError(
        'LlamaConfig: embedDim ($embedDim) must be divisible by '
        'numHeads ($numHeads)',
      );
    }
    if (numHeads % (numKvHeads ?? numHeads) != 0) {
      throw ArgumentError(
        'LlamaConfig: numHeads ($numHeads) must be divisible by '
        'numKvHeads (${numKvHeads ?? numHeads}) for GQA',
      );
    }
  }
}

class LlamaBlock extends Module {
  final RMSNorm attnNorm;
  final RMSNorm ffnNorm;
  final MultiHeadAttention attn;
  final SwiGluFfn ffn;

  LlamaBlock(
    int embedDim,
    int numHeads, {
    required int numKvHeads,
    required int ffnDim,
    required RopeCache rope,
    double rmsNormEps = 1e-5,
    Device device = Device.CPU,
    int seed = 0,
  }) : attnNorm = RMSNorm(embedDim, eps: rmsNormEps, device: device),
       ffnNorm = RMSNorm(embedDim, eps: rmsNormEps, device: device),
       attn = MultiHeadAttention(
         embedDim,
         numHeads,
         numKvHeads: numKvHeads,
         bias: false,
         device: device,
         seed: seed,
       ),
       ffn = SwiGluFfn(embedDim, ffnDim, device: device, seed: seed + 500000) {
    attn.rope = rope;
  }

  Tensor call(Tensor x, {Tensor? mask, MHACache? cache, int startPos = 0}) {
    final a = attn(attnNorm(x), mask: mask, cache: cache, startPos: startPos);
    final h = x + a;
    final m = ffn(ffnNorm(h));
    return h + m;
  }

  @override
  List<Tensor> parameters() => [
    ...attnNorm.parameters(),
    ...ffnNorm.parameters(),
    ...attn.parameters(),
    ...ffn.parameters(),
  ];

  @override
  List<Module> submodules() => [attnNorm, ffnNorm, attn, ffn];
}

class Llama extends Module {
  final LlamaConfig config;

  final Embedding embedIn;
  final List<LlamaBlock> blocks;
  final RMSNorm finalNorm;
  final RopeCache rope;

  /// Only populated when `config.tieWeights == false`. When tied, the
  /// output head is computed inline as `h @ embedIn.weight.T`.
  final Linear? untiedHead;

  Llama(this.config)
    : embedIn = Embedding(
        config.vocabSize,
        config.embedDim,
        device: config.device,
        seed: config.seed,
      ),
      finalNorm = RMSNorm(
        config.embedDim,
        eps: config.rmsNormEps,
        device: config.device,
      ),
      rope = RopeCache(
        maxCtx: config.maxCtx,
        headDim: config.embedDim ~/ config.numHeads,
        base: config.ropeBase,
        device: config.device,
      ),
      untiedHead = config.tieWeights
          ? null
          : Linear(
              config.embedDim,
              config.vocabSize,
              bias: false,
              device: config.device,
              seed: config.seed + 900000,
            ),
      blocks = <LlamaBlock>[] {
    for (int i = 0; i < config.numLayers; i++) {
      blocks.add(
        LlamaBlock(
          config.embedDim,
          config.numHeads,
          numKvHeads: config.numKvHeads,
          ffnDim: config.ffnDim,
          rope: rope,
          rmsNormEps: config.rmsNormEps,
          device: config.device,
          seed: config.seed + 100000 + i * 1000,
        ),
      );
    }
  }

  /// Forward pass. `tokens` is a 1D `[seqLen]` tensor of class
  /// indices as float32. Output is `[seqLen, vocab]`.
  Tensor call(Tensor tokens) {
    if (tokens.shape.length != 1) {
      throw ArgumentError(
        'Llama: tokens must be 1D [seqLen]; got ${tokens.shape}',
      );
    }
    final n = tokens.shape.last;
    if (n > config.maxCtx) {
      throw ArgumentError('Llama: seqLen $n exceeds maxCtx ${config.maxCtx}');
    }
    return _forward(tokens, startPos: 0, cache: null);
  }

  Tensor _forward(Tensor tokens, {required int startPos, EncoderCache? cache}) {
    final n = tokens.shape.last;
    if (startPos + n > config.maxCtx) {
      throw ArgumentError(
        'Llama: window [$startPos, ${startPos + n}) exceeds '
        'maxCtx=${config.maxCtx}',
      );
    }
    var h = embedIn(tokens); // [N, D]
    final mask = n > 1 ? causalMask(n, device: h.device) : null;
    for (int i = 0; i < blocks.length; i++) {
      final cacheI = cache?.layers[i];
      h = blocks[i](h, mask: mask, cache: cacheI, startPos: startPos);
    }
    h = finalNorm(h);
    if (config.tieWeights) {
      return h.matmul(embedIn.weight.transpose());
    }
    return untiedHead!(h);
  }

  /// Autoregressive sampling. Same signature and behaviour as
  /// `GPT.generate`.
  List<double> generate(
    List<double> prompt, {
    required int maxNewTokens,
    double temperature = 1.0,
    int? topK,
    math.Random? rng,
    bool useCache = true,
  }) {
    if (prompt.isEmpty) {
      throw ArgumentError('Llama.generate: prompt must be non-empty');
    }
    if (useCache && prompt.length > config.maxCtx) {
      throw ArgumentError(
        'Llama.generate(useCache: true): prompt length '
        '${prompt.length} exceeds maxCtx ${config.maxCtx}. Pass '
        'useCache: false to enable sliding-window truncation.',
      );
    }
    final r = rng ?? math.Random();
    final wasTraining = training;
    eval();
    try {
      return Tensor.noGrad(
        () => useCache
            ? _generateCached(prompt, maxNewTokens, temperature, topK, r)
            : _generateNoCache(prompt, maxNewTokens, temperature, topK, r),
      );
    } finally {
      if (wasTraining) train();
    }
  }

  List<double> _generateCached(
    List<double> prompt,
    int maxNewTokens,
    double temperature,
    int? topK,
    math.Random rng,
  ) {
    final v = config.vocabSize;
    final cache = EncoderCache.empty(config.numLayers, config.numKvHeads);
    final out = List<double>.of(prompt);

    final promptCtx = Tensor.fromList(
      [prompt.length],
      prompt,
      device: config.device,
    );
    var logits = _forward(promptCtx, startPos: 0, cache: cache).toList();
    var lastBase = (prompt.length - 1) * v;
    var row = List<double>.generate(v, (i) => logits[lastBase + i]);
    var next = _sampleFromLogits(row, temperature, topK, rng);
    out.add(next.toDouble());

    for (int step = 1; step < maxNewTokens; step++) {
      if (cache.seqLen >= config.maxCtx) break;
      final tokTensor = Tensor.fromList(
        [1],
        [next.toDouble()],
        device: config.device,
      );
      logits = _forward(
        tokTensor,
        startPos: cache.seqLen,
        cache: cache,
      ).toList();
      row = List<double>.generate(v, (i) => logits[i]);
      next = _sampleFromLogits(row, temperature, topK, rng);
      out.add(next.toDouble());
    }
    return out;
  }

  List<double> _generateNoCache(
    List<double> prompt,
    int maxNewTokens,
    double temperature,
    int? topK,
    math.Random rng,
  ) {
    final v = config.vocabSize;
    final out = List<double>.of(prompt);
    for (int step = 0; step < maxNewTokens; step++) {
      final start = out.length > config.maxCtx ? out.length - config.maxCtx : 0;
      final ctxList = out.sublist(start);
      final ctx = Tensor.fromList(
        [ctxList.length],
        ctxList,
        device: config.device,
      );
      final logits = call(ctx).toList();
      final lastBase = (ctxList.length - 1) * v;
      final row = List<double>.generate(v, (i) => logits[lastBase + i]);
      final next = _sampleFromLogits(row, temperature, topK, rng);
      out.add(next.toDouble());
    }
    return out;
  }

  int _sampleFromLogits(
    List<double> logits,
    double temperature,
    int? topK,
    math.Random rng,
  ) {
    final v = logits.length;
    final greedy = temperature <= 0.0 || (topK != null && topK <= 1);
    if (greedy) {
      var bestI = 0;
      var bestV = logits[0];
      for (int i = 1; i < v; i++) {
        if (logits[i] > bestV) {
          bestV = logits[i];
          bestI = i;
        }
      }
      return bestI;
    }
    final scaled = List<double>.generate(v, (i) => logits[i] / temperature);
    if (topK != null && topK < v) {
      final sorted = List<int>.generate(v, (i) => i)
        ..sort((a, b) => scaled[b].compareTo(scaled[a]));
      final keep = sorted.take(topK).toSet();
      for (int i = 0; i < v; i++) {
        if (!keep.contains(i)) scaled[i] = double.negativeInfinity;
      }
    }
    var maxV = scaled[0];
    for (int i = 1; i < v; i++) {
      if (scaled[i] > maxV) maxV = scaled[i];
    }
    final exps = List<double>.generate(
      v,
      (i) => scaled[i] == double.negativeInfinity
          ? 0.0
          : math.exp(scaled[i] - maxV),
    );
    var sum = 0.0;
    for (final e in exps) {
      sum += e;
    }
    final u = rng.nextDouble() * sum;
    var acc = 0.0;
    for (int i = 0; i < v; i++) {
      acc += exps[i];
      if (u <= acc) return i;
    }
    return v - 1;
  }

  @override
  List<Tensor> parameters() => [
    ...embedIn.parameters(),
    for (final b in blocks) ...b.parameters(),
    ...finalNorm.parameters(),
    if (untiedHead != null) ...untiedHead!.parameters(),
  ];

  @override
  List<Module> submodules() => [
    embedIn,
    ...blocks,
    finalNorm,
    if (untiedHead != null) untiedHead!,
  ];
}
