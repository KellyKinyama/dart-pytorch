/// GPT-J causal language model (EleutherAI, 2021).
///
/// GPT-J was the first widely-used open replica of a GPT-3-class
/// model. Architecturally it sits between GPT-2 and GPT-NeoX:
///
///   * **Rotary positional embedding**, applied only to the first
///     `rotaryDim` dimensions of each head (64 out of 256 for the 6B
///     model). Uses the **interleaved-pair** convention:
///     `q_rot[..., 2i]   = q[..., 2i]*cos - q[..., 2i+1]*sin`
///     `q_rot[..., 2i+1] = q[..., 2i]*sin + q[..., 2i+1]*cos`.
///     This is *different* from GPT-NeoX / LLaMA / Pythia which use
///     "half-split" rotary. The [GPTJHFLoader] handles this by
///     permuting the Q/K weight rows at load time so our shared
///     half-split [RopeCache] produces the mathematically identical
///     result inside attention (see loader for the derivation).
///
///   * **Parallel residual block with a single shared LayerNorm.**
///     Where Pythia has two LNs (`input_layernorm` and
///     `post_attention_layernorm`), GPT-J uses one:
///     ```
///     h = ln(x)
///     y = x + attn(h) + mlp(h)
///     ```
///     Both attn and mlp read from the *same* normalised input, and
///     both are added into the residual — an important departure
///     from GPT-2's sequential `x -> attn -> +x -> mlp -> +x`.
///
///   * **No bias on attention Q/K/V/out projections.** The MLP
///     retains biases, and (unlike Pythia) the untied `lm_head`
///     *also* has a bias.
///
///   * Untied output head (`lm_head.weight` [V, D] + `lm_head.bias`).
///
///   * Uses the standard GPT-2 byte-level BPE tokenizer, but with
///     the vocabulary padded to 50400 for GPU-friendly matmul shapes.
///
/// The activation is the tanh approximation of GELU (`gelu_new`),
/// same as Pythia and GPT-2.
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

class GPTJConfig {
  final int vocabSize;
  final int maxCtx;
  final int embedDim;
  final int numLayers;
  final int numHeads;
  final int ffnDim; // 4 * embedDim by default
  final int rotaryDim; // 64 for GPT-J 6B
  final double ropeBase; // 10000 for GPT-J
  final Device device;
  final int seed;

  GPTJConfig({
    required this.vocabSize,
    required this.maxCtx,
    required this.embedDim,
    required this.numLayers,
    required this.numHeads,
    required this.rotaryDim,
    int? ffnDim,
    this.ropeBase = 10000.0,
    this.device = Device.CPU,
    this.seed = 0,
  }) : ffnDim = ffnDim ?? 4 * embedDim {
    if (rotaryDim <= 0 || rotaryDim.isOdd) {
      throw ArgumentError('rotaryDim ($rotaryDim) must be a positive even int');
    }
    if (embedDim % numHeads != 0) {
      throw ArgumentError('embedDim must be divisible by numHeads');
    }
    final headDim = embedDim ~/ numHeads;
    if (rotaryDim > headDim) {
      throw ArgumentError(
        'rotaryDim ($rotaryDim) must be <= headDim ($headDim)',
      );
    }
  }
}

/// GPT-J transformer block — parallel residual with a single shared LN.
class GPTJBlock extends Module {
  final LayerNorm ln;
  final MultiHeadAttention attn;
  final Linear ffn1; // fc_in
  final Linear ffn2; // fc_out

  GPTJBlock(
    int embedDim,
    int numHeads, {
    int? ffnDim,
    required RopeCache rope,
    Device device = Device.CPU,
    int seed = 0,
  }) : ln = LayerNorm(embedDim, device: device),
       attn = MultiHeadAttention(
         embedDim,
         numHeads,
         bias: false, // GPT-J has no bias on Q/K/V/out
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
    final h = ln(x);
    final a = attn(h, mask: mask, cache: cache, startPos: startPos);
    final m = ffn2(_geluTanh(ffn1(h)));
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
    ...ln.parameters(),
    ...attn.parameters(),
    ...ffn1.parameters(),
    ...ffn2.parameters(),
  ];

  @override
  List<Module> submodules() => [ln, attn, ffn1, ffn2];
}

class GPTJModel extends Module {
  final GPTJConfig config;

  final Embedding wte;
  final List<GPTJBlock> blocks;
  final LayerNorm finalLn;
  final Linear lmHead; // has bias in GPT-J
  final RopeCache rope;

  GPTJModel(this.config)
    : wte = Embedding(
        config.vocabSize,
        config.embedDim,
        device: config.device,
        seed: config.seed,
      ),
      finalLn = LayerNorm(config.embedDim, device: config.device),
      lmHead = Linear(
        config.embedDim,
        config.vocabSize,
        bias: true, // unusual — GPT-J's lm_head has a bias
        device: config.device,
        seed: config.seed + 900000,
      ),
      rope = RopeCache(
        maxCtx: config.maxCtx,
        headDim: config.embedDim ~/ config.numHeads,
        rotaryDim: config.rotaryDim,
        base: config.ropeBase,
        device: config.device,
      ),
      blocks = <GPTJBlock>[] {
    for (int i = 0; i < config.numLayers; i++) {
      blocks.add(
        GPTJBlock(
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

  /// Forward pass. `tokens` is a 1D `[seqLen]` float tensor of token
  /// ids. Output is `[seqLen, vocab]` logits.
  Tensor call(Tensor tokens) {
    if (tokens.shape.length != 1) {
      throw ArgumentError(
        'GPTJModel: tokens must be 1D [seqLen]; got ${tokens.shape}',
      );
    }
    return _forward(tokens, startPos: 0, cache: null);
  }

  Tensor _forward(Tensor tokens, {required int startPos, EncoderCache? cache}) {
    final n = tokens.shape.last;
    if (startPos + n > config.maxCtx) {
      throw ArgumentError(
        'GPTJModel: window [$startPos, ${startPos + n}) exceeds '
        'maxCtx=${config.maxCtx}',
      );
    }
    var h = wte(tokens); // [N, D]
    final mask = n > 1 ? causalMask(n, device: h.device) : null;
    for (int i = 0; i < blocks.length; i++) {
      final cacheI = cache?.layers[i];
      h = blocks[i](h, mask: mask, cache: cacheI, startPos: startPos);
    }
    h = finalLn(h);
    return lmHead(h);
  }

  /// Autoregressive sampling — same interface as `PythiaModel.generate`.
  List<double> generate(
    List<double> prompt, {
    required int maxNewTokens,
    double temperature = 1.0,
    int? topK,
    math.Random? rng,
    bool useCache = true,
  }) {
    if (prompt.isEmpty) {
      throw ArgumentError('GPTJModel.generate: prompt must be non-empty');
    }
    if (useCache && prompt.length > config.maxCtx) {
      throw ArgumentError(
        'GPTJModel.generate(useCache: true): prompt length '
        '${prompt.length} exceeds maxCtx ${config.maxCtx}.',
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
    ...wte.parameters(),
    for (final b in blocks) ...b.parameters(),
    ...finalLn.parameters(),
    ...lmHead.parameters(),
  ];

  @override
  List<Module> submodules() => [wte, ...blocks, finalLn, lmHead];
}
