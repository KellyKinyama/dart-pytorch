/// A minimal GPT-style causal language model.
///
/// Compared to [TransformerLM] this module follows the GPT-2 recipe:
///
///   * **Learned** positional embeddings of shape `[maxCtx, embedDim]`
///     (not sinusoidal).
///   * An embedding **dropout** applied to `tokenEmb + posEmb` before
///     the encoder.
///   * The output head is **weight-tied** to the token embedding when
///     `tieWeights: true` (default). No separate `Linear` and no head
///     bias — the token-embedding matrix `[V, D]` is transposed and
///     used to project the final hidden state `[N, D]` to logits
///     `[N, V]`. Because the same tensor appears in both the embedding
///     lookup and the head matmul, its gradient accumulates
///     contributions from both paths automatically.
///   * A trailing `LayerNorm` inside the [TransformerEncoder]
///     (`finalNorm: true`, standard for pre-LN GPT).
///   * A [generate] method that samples autoregressively with greedy,
///     temperature, or top-k modes and truncates context to `maxCtx`.
///
/// Same 2D-only, single-sequence tensor convention as the rest of the
/// library: `tokens` is a 1D `[seqLen]` tensor of class indices stored
/// as float32.
library;

import 'dart:math' as math;

import '../tensor/tensor.dart';
import 'dropout.dart';
import 'embedding.dart';
import 'kv_cache.dart';
import 'linear.dart';
import 'masks.dart';
import 'module.dart';
import 'positional.dart';
import 'transformer_encoder.dart';

class GPTConfig {
  final int vocabSize;
  final int maxCtx;
  final int embedDim;
  final int numLayers;
  final int numHeads;
  final int? ffnDim;
  final double dropoutP;
  final bool tieWeights;
  final Device device;
  final int seed;

  const GPTConfig({
    required this.vocabSize,
    required this.maxCtx,
    required this.embedDim,
    required this.numLayers,
    required this.numHeads,
    this.ffnDim,
    this.dropoutP = 0.0,
    this.tieWeights = true,
    this.device = Device.CPU,
    this.seed = 0,
  });
}

class GPT extends Module {
  final GPTConfig config;

  final Embedding tokenEmb;
  final LearnedPositionalEmbedding posEmb;
  final Dropout embedDrop;
  final TransformerEncoder encoder;

  /// Only populated when `config.tieWeights == false`. When tied, the
  /// output head is computed inline as `h @ tokenEmb.weight.T`.
  final Linear? untiedHead;

  GPT(this.config)
    : tokenEmb = Embedding(
        config.vocabSize,
        config.embedDim,
        device: config.device,
        seed: config.seed,
      ),
      posEmb = LearnedPositionalEmbedding(
        config.maxCtx,
        config.embedDim,
        device: config.device,
        seed: config.seed + 50000,
      ),
      embedDrop = Dropout(config.dropoutP),
      encoder = TransformerEncoder(
        config.numLayers,
        config.embedDim,
        config.numHeads,
        ffnDim: config.ffnDim,
        dropoutP: config.dropoutP,
        device: config.device,
        seed: config.seed + 100000,
      ),
      untiedHead = config.tieWeights
          ? null
          : Linear(
              config.embedDim,
              config.vocabSize,
              bias: false,
              device: config.device,
              seed: config.seed + 900000,
            );

  /// Forward pass. `tokens` is `[seqLen]`; returns logits `[seqLen, vocabSize]`.
  /// A causal mask is applied inside the encoder.
  Tensor call(Tensor tokens) {
    if (tokens.shape.length != 1) {
      throw ArgumentError(
        'GPT: tokens must be 1D [seqLen]; got ${tokens.shape}',
      );
    }
    final n = tokens.shape[0];
    if (n > config.maxCtx) {
      throw ArgumentError('GPT: seqLen $n exceeds maxCtx ${config.maxCtx}');
    }
    return _forward(tokens, startPos: 0, cache: null);
  }

  /// Internal shared forward. Used by [call] (no cache, `startPos=0`,
  /// causal mask), by the initial prompt fill in [generate]
  /// (cache empty, causal mask), and by subsequent single-token steps
  /// in [generate] (cache non-empty, `startPos = cache.seqLen`, no mask).
  Tensor _forward(Tensor tokens, {required int startPos, EncoderCache? cache}) {
    final n = tokens.shape[0];
    var h = tokenEmb(tokens); // [N, D]
    h = posEmb(h, startPos: startPos); // [N, D]
    h = embedDrop(h);
    // Only need a causal mask when we're processing multiple new
    // tokens simultaneously. Single-token steps (autoregressive
    // append) attend only to past cached positions.
    final mask = n > 1 ? causalMask(n, device: h.device) : null;
    h = encoder(h, mask: mask, cache: cache); // [N, D]
    if (config.tieWeights) {
      // logits = h @ W_e^T   -> shared weight, same graph node.
      return h.matmul(tokenEmb.weight.transpose());
    }
    return untiedHead!(h);
  }

  /// Autoregressive sampling. Returns the full sequence `prompt +
  /// generated tokens` as a Dart list of doubles (matches the float32
  /// index convention used everywhere else). The model is put into
  /// [eval] mode for the duration of the call and restored on exit.
  ///
  /// * `maxNewTokens` — how many tokens to append.
  /// * `temperature` — logits are divided by this. `0` (or negative)
  ///   forces greedy argmax regardless of `topK`. Default `1.0`.
  /// * `topK` — if non-null and `> 0`, restricts sampling to the top-K
  ///   logits at each step (others get probability 0).
  /// * `rng` — optional seeded RNG for reproducible sampling.
  /// * `useCache` — enable the KV-cache fast path (default `true`).
  ///   Each new token becomes an O(N) rather than O(N^2) forward.
  ///   When `false`, every step re-runs the full context — useful for
  ///   parity testing and for prompts longer than `maxCtx` where the
  ///   context needs to slide.
  List<double> generate(
    List<double> prompt, {
    required int maxNewTokens,
    double temperature = 1.0,
    int? topK,
    math.Random? rng,
    bool useCache = true,
  }) {
    if (prompt.isEmpty) {
      throw ArgumentError('GPT.generate: prompt must be non-empty');
    }
    if (useCache && prompt.length > config.maxCtx) {
      throw ArgumentError(
        'GPT.generate(useCache: true): prompt length ${prompt.length} '
        'exceeds maxCtx ${config.maxCtx}. Pass useCache: false to '
        'enable sliding-window truncation.',
      );
    }
    final r = rng ?? math.Random();
    final wasTraining = training;
    eval();
    try {
      return useCache
          ? _generateCached(prompt, maxNewTokens, temperature, topK, r)
          : _generateNoCache(prompt, maxNewTokens, temperature, topK, r);
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

    // Initial prompt fill — one big forward that populates the cache.
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

    // Autoregressive single-token steps.
    for (int step = 1; step < maxNewTokens; step++) {
      if (cache.seqLen >= config.maxCtx) break; // cache is full
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
      // [1, V] — the whole row is the "last" row.
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
      // Take at most maxCtx trailing tokens as the context window.
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
    // Greedy: temperature <= 0 or topK == 1 both collapse to argmax.
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
    // Scale by temperature.
    final scaled = List<double>.generate(v, (i) => logits[i] / temperature);
    // Top-k filter — set everything not in the top-k to -inf.
    if (topK != null && topK < v) {
      final sorted = List<int>.generate(v, (i) => i)
        ..sort((a, b) => scaled[b].compareTo(scaled[a]));
      final keep = sorted.take(topK).toSet();
      for (int i = 0; i < v; i++) {
        if (!keep.contains(i)) scaled[i] = double.negativeInfinity;
      }
    }
    // Numerically-stable softmax.
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
    // Multinomial draw via CDF.
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
    ...tokenEmb.parameters(),
    ...posEmb.parameters(),
    ...encoder.parameters(),
    if (untiedHead != null) ...untiedHead!.parameters(),
  ];

  @override
  List<Module> submodules() => [
    tokenEmb,
    posEmb,
    embedDrop,
    encoder,
    if (untiedHead != null) untiedHead!,
  ];
}
