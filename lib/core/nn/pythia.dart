/// GPT-NeoX / Pythia causal language model.
///
/// The main structural differences vs. our [GPT] (GPT-2):
///
///   * **No learned positional embedding table.** Positions are
///     injected via rotary embeddings ([RopeCache]) applied to Q and K
///     inside each attention layer.
///   * **Parallel residual block** — instead of the sequential
///     `x -> ln1 -> attn -> +x -> ln2 -> mlp -> +x` GPT-2 uses, each
///     block computes `attn` and `mlp` in parallel from independent
///     LayerNorms of the same input:
///     `y = x + attn(input_layernorm(x)) + mlp(post_attention_layernorm(x))`
///     (this is HF's `use_parallel_residual: True` mode).
///   * **Untied output head.** Pythia keeps a separate `embed_out`
///     `[vocab, hidden]` rather than tying to `embed_in`.
///
/// Everything else (biased Q/K/V/output projections, biased 4x MLP,
/// causal mask, KV cache, sampling loop) matches our GPT-2 stack.
///
/// The activation used here is the tanh approximation of GELU
/// ([Activation.geluTanh]). Pythia originally uses exact GELU
/// (`erf`-based), but the two agree to within a few parts in 1e4 —
/// perfectly fine for inference next-token argmax.
library;

import 'dart:math' as math;

import '../tensor/tensor.dart';
import 'attention/multi_head_attention.dart';
import 'embedding.dart';
import 'kv_cache.dart';
import 'layer_norm.dart';
import 'linear.dart';
import 'masks.dart';
import 'module.dart';
import 'rotary.dart';

class PythiaConfig {
  final int vocabSize;
  final int maxCtx;
  final int embedDim;
  final int numLayers;
  final int numHeads;
  final int ffnDim; // stored as resolved value (4 * embedDim if unspecified)
  final double ropeBase; // 10000 for Pythia
  final double rotaryPct; // 0.25 for Pythia; 1.0 for LLaMA-style
  final Device device;
  final int seed;

  PythiaConfig({
    required this.vocabSize,
    required this.maxCtx,
    required this.embedDim,
    required this.numLayers,
    required this.numHeads,
    int? ffnDim,
    this.ropeBase = 10000.0,
    this.rotaryPct = 0.25,
    this.device = Device.CPU,
    this.seed = 0,
  }) : ffnDim = ffnDim ?? 4 * embedDim;
}

class PythiaBlock extends Module {
  final LayerNorm inputLn;
  final LayerNorm postAttnLn;
  final MultiHeadAttention attn;
  final Linear ffn1; // dense_h_to_4h
  final Linear ffn2; // dense_4h_to_h

  PythiaBlock(
    int embedDim,
    int numHeads, {
    int? ffnDim,
    required RopeCache rope,
    Device device = Device.CPU,
    int seed = 0,
  }) : inputLn = LayerNorm(embedDim, device: device),
       postAttnLn = LayerNorm(embedDim, device: device),
       attn = MultiHeadAttention(
         embedDim,
         numHeads,
         bias: true,
         device: device,
         seed: seed,
       ),
       ffn1 = Linear(
         embedDim,
         ffnDim ?? 4 * embedDim,
         bias: true,
         device: device,
         seed: seed + 10000,
       ),
       ffn2 = Linear(
         ffnDim ?? 4 * embedDim,
         embedDim,
         bias: true,
         device: device,
         seed: seed + 20000,
       ) {
    attn.rope = rope;
  }

  Tensor call(Tensor x, {Tensor? mask, MHACache? cache, int startPos = 0}) {
    final a = attn(inputLn(x), mask: mask, cache: cache, startPos: startPos);
    final m = ffn2(_geluTanh(ffn1(postAttnLn(x))));
    return x + a + m;
  }

  static Tensor _geluTanh(Tensor x) {
    // 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    const c = 0.7978845608028654;
    final inner = (x + x.pow(3.0) * 0.044715) * c;
    return x * 0.5 * (inner.tanh() + 1.0);
  }

  @override
  List<Tensor> parameters() => [
    ...inputLn.parameters(),
    ...postAttnLn.parameters(),
    ...attn.parameters(),
    ...ffn1.parameters(),
    ...ffn2.parameters(),
  ];

  @override
  List<Module> submodules() => [inputLn, postAttnLn, attn, ffn1, ffn2];
}

/// Resolve GPT-NeoX-style `rotary_pct` into an integer `rotaryDim`.
/// Rounded down and then made even (RoPE needs pairs); clamped to
/// `[2, headDim]`. For `pct == 1.0` returns `headDim`.
int _computeRotaryDim(int headDim, double pct) {
  if (pct <= 0.0) {
    throw ArgumentError('rotaryPct ($pct) must be > 0');
  }
  if (pct >= 1.0) return headDim;
  var r = (headDim * pct).floor();
  if (r.isOdd) r -= 1;
  if (r < 2) r = 2;
  return r;
}

class PythiaModel extends Module {
  final PythiaConfig config;

  final Embedding embedIn;
  final List<PythiaBlock> blocks;
  final LayerNorm finalLn;
  final Linear embedOut;
  final RopeCache rope;

  PythiaModel(this.config)
    : embedIn = Embedding(
        config.vocabSize,
        config.embedDim,
        device: config.device,
        seed: config.seed,
      ),
      finalLn = LayerNorm(config.embedDim, device: config.device),
      embedOut = Linear(
        config.embedDim,
        config.vocabSize,
        bias: false,
        device: config.device,
        seed: config.seed + 900000,
      ),
      rope = RopeCache(
        maxCtx: config.maxCtx,
        headDim: config.embedDim ~/ config.numHeads,
        rotaryDim: _computeRotaryDim(
          config.embedDim ~/ config.numHeads,
          config.rotaryPct,
        ),
        base: config.ropeBase,
        device: config.device,
      ),
      blocks = <PythiaBlock>[] {
    // Rope must be built before blocks so we can hand it to each MHA.
    for (int i = 0; i < config.numLayers; i++) {
      blocks.add(
        PythiaBlock(
          config.embedDim,
          config.numHeads,
          ffnDim: config.ffnDim,
          rope: rope,
          device: config.device,
          seed: config.seed + 100000 + i * 1000,
        ),
      );
    }
  }

  /// Forward pass. `tokens` is a 1D `[seqLen]` tensor of class
  /// indices as float32 (matches [GPT]). Output is `[seqLen, vocab]`.
  Tensor call(Tensor tokens) {
    if (tokens.shape.length != 1) {
      throw ArgumentError(
        'PythiaModel: tokens must be 1D [seqLen]; got ${tokens.shape}',
      );
    }
    return _forward(tokens, startPos: 0, cache: null);
  }

  Tensor _forward(Tensor tokens, {required int startPos, EncoderCache? cache}) {
    final n = tokens.shape.last;
    if (startPos + n > config.maxCtx) {
      throw ArgumentError(
        'PythiaModel: window [$startPos, ${startPos + n}) exceeds '
        'maxCtx=${config.maxCtx}',
      );
    }
    var h = embedIn(tokens); // [N, D]
    final mask = n > 1 ? causalMask(n, device: h.device) : null;
    for (int i = 0; i < blocks.length; i++) {
      final cacheI = cache?.layers[i];
      h = blocks[i](h, mask: mask, cache: cacheI, startPos: startPos);
    }
    h = finalLn(h);
    return embedOut(h);
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
      throw ArgumentError('PythiaModel.generate: prompt must be non-empty');
    }
    if (useCache && prompt.length > config.maxCtx) {
      throw ArgumentError(
        'PythiaModel.generate(useCache: true): prompt length '
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
    final cache = EncoderCache.empty(config.numLayers, config.numHeads);
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
    ...finalLn.parameters(),
    ...embedOut.parameters(),
  ];

  @override
  List<Module> submodules() => [embedIn, ...blocks, finalLn, embedOut];
}
