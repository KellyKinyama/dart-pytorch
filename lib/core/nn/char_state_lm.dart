/// Context-free, state-based character language model — a decoder-only
/// transformer whose self-attention is replaced by **linear attention**
/// in its recurrent form ("Transformers are RNNs", Katharopoulos et al.
/// 2020). Instead of attending over the whole context window, each block
/// carries a fixed-size state `(S ∈ ℝ^{D×D}, z ∈ ℝ^D)` that summarises all
/// history, so:
///
///   * there is **no context window** and **no positional embedding** —
///     token order is encoded entirely by the order of state updates;
///   * generation is **O(1) time and memory per character**, with
///     unbounded effective context.
///
/// Linear attention recurrence (per block, single head, feature map
/// `φ(x) = relu(x) + 1`):
///
///   S ← S + φ(kₜ)ᵀ vₜ            (outer product, [D×D])
///   z ← z + φ(kₜ)                 ([1×D])
///   outₜ = φ(qₜ) · S / (φ(qₜ) · z)
///
/// Training uses the exact same step function unrolled over a sequence,
/// so the trained model and the streaming generator are identical.
library;

import 'dart:math' as math;

import '../tensor/tensor.dart';
import 'embedding.dart';
import 'layer_norm.dart';
import 'linear.dart';
import 'module.dart';

const double _eps = 1e-6;

/// Fixed-size recurrent state for one [LinearAttention] block — one
/// `(S, z)` pair per head.
class _LinAttnState {
  _LinAttnState(this.s, this.z);
  final List<Tensor> s; // H × [headDim, headDim]
  final List<Tensor> z; // H × [1, headDim]

  factory _LinAttnState.zeros(int numHeads, int headDim, Device device) =>
      _LinAttnState(
        [
          for (var h = 0; h < numHeads; h++)
            Tensor.fill([headDim, headDim], 0.0, device: device),
        ],
        [
          for (var h = 0; h < numHeads; h++)
            Tensor.fill([1, headDim], 0.0, device: device),
        ],
      );
}

/// Multi-head causal linear attention with an explicit recurrent state
/// and an optional RWKV/RetNet-style per-head decay gate.
///
/// With `useDecay`, each head applies a learnable scalar `λ_h ∈ (0, 1)`
/// to its state every step (`S ← λ_h·S + φ(k)ᵀv`), letting the head
/// forget at its own timescale — heads are initialised across a spread of
/// decays for multi-scale memory. With `useDecay = false` this is plain
/// linear attention (`λ = 1`, perfect memory).
class LinearAttention extends Module {
  final int dim;
  final int numHeads;
  final int headDim;
  final bool useDecay;

  // Per-head projections (mirrors the repo's MultiHeadAttention layout so
  // no strided column slicing is needed): each maps D → headDim.
  final List<Linear> wq;
  final List<Linear> wk;
  final List<Linear> wv;
  final Linear wo;

  /// Per-head decay logits `[1, 1]`; `λ_h = sigmoid(decayRaw[h])`.
  final List<Tensor> decayRaw;

  LinearAttention(
    this.dim, {
    this.numHeads = 4,
    this.useDecay = true,
    Device device = Device.CPU,
    int seed = 0,
  })  : headDim = dim ~/ numHeads,
        wq = [
          for (var h = 0; h < numHeads; h++)
            Linear(dim, dim ~/ numHeads,
                bias: false, device: device, seed: seed + h),
        ],
        wk = [
          for (var h = 0; h < numHeads; h++)
            Linear(dim, dim ~/ numHeads,
                bias: false, device: device, seed: seed + 100 + h),
        ],
        wv = [
          for (var h = 0; h < numHeads; h++)
            Linear(dim, dim ~/ numHeads,
                bias: false, device: device, seed: seed + 200 + h),
        ],
        wo = Linear(dim, dim, bias: false, device: device, seed: seed + 300),
        decayRaw = [
          for (var h = 0; h < numHeads; h++)
            Tensor.fromList(
              [1, 1],
              // Spread initial decays across [~0.90, ~0.999] for
              // multi-timescale heads (logit of the target decay).
              [_decayLogit(0.90 + 0.099 * h / math.max(numHeads - 1, 1))],
              requiresGrad: true,
              device: device,
            ),
        ] {
    if (dim % numHeads != 0) {
      throw ArgumentError(
        'LinearAttention: dim ($dim) must be divisible by numHeads '
        '($numHeads)',
      );
    }
  }

  static double _decayLogit(double d) => math.log(d / (1.0 - d));

  // Positive feature map φ(x) = relu(x) + 1 (elu+1 substitute).
  static Tensor _phi(Tensor x) => x.relu() + 1.0;

  /// Parallel training form: run the recurrence over a `[N, D]` sequence
  /// and return the `[N, D]` attention output.
  Tensor forwardSeq(Tensor x) {
    final n = x.shape[0];
    final headOuts = <Tensor>[];
    for (var h = 0; h < numHeads; h++) {
      final q = TensorConcat.splitRows(_phi(wq[h](x)), 1);
      final k = TensorConcat.splitRows(_phi(wk[h](x)), 1);
      final v = TensorConcat.splitRows(wv[h](x), 1);
      final lam = useDecay ? decayRaw[h].sigmoid() : null; // [1, 1]

      var s = Tensor.fill([headDim, headDim], 0.0, device: x.device);
      var z = Tensor.fill([1, headDim], 0.0, device: x.device);
      final outs = <Tensor>[];
      for (var t = 0; t < n; t++) {
        s = (lam != null ? s * lam : s) + k[t].transpose().matmul(v[t]);
        z = (lam != null ? z * lam : z) + k[t];
        outs.add(q[t].matmul(s) / ((q[t] * z).sum() + _eps));
      }
      headOuts.add(TensorConcat.concat(outs, axis: 0)); // [N, headDim]
    }
    return wo(TensorConcat.concat(headOuts, axis: 1)); // [N, D]
  }

  /// Streaming form: advance the per-head recurrent [state] by one token
  /// `[1, D]` and return this step's output `[1, D]`. Call inside
  /// [Tensor.noGrad].
  Tensor step(Tensor x1, _LinAttnState state) {
    final heads = <Tensor>[];
    for (var h = 0; h < numHeads; h++) {
      final qf = _phi(wq[h](x1));
      final kf = _phi(wk[h](x1));
      final vv = wv[h](x1);
      final lam = useDecay ? decayRaw[h].sigmoid() : null;
      var s = state.s[h];
      var z = state.z[h];
      s = (lam != null ? s * lam : s) + kf.transpose().matmul(vv);
      z = (lam != null ? z * lam : z) + kf;
      state.s[h] = s;
      state.z[h] = z;
      heads.add(qf.matmul(s) / ((qf * z).sum() + _eps));
    }
    return wo(TensorConcat.concat(heads, axis: 1)); // [1, D]
  }

  @override
  List<Tensor> parameters() => [
        for (final l in wq) ...l.parameters(),
        for (final l in wk) ...l.parameters(),
        for (final l in wv) ...l.parameters(),
        ...wo.parameters(),
        if (useDecay) ...decayRaw,
      ];

  @override
  List<Module> submodules() => [...wq, ...wk, ...wv, wo];
}

/// Pre-LN decoder block: `h = x + attn(ln1(x)); y = h + ffn(ln2(h))`,
/// with linear (state-based) attention.
class RecurrentDecoderBlock extends Module {
  final LayerNorm ln1;
  final LayerNorm ln2;
  final LinearAttention attn;
  final Linear ffn1;
  final Linear ffn2;

  RecurrentDecoderBlock(int dim,
      {int numHeads = 4,
      bool useDecay = true,
      int? ffnDim,
      Device device = Device.CPU,
      int seed = 0})
      : ln1 = LayerNorm(dim, device: device),
        ln2 = LayerNorm(dim, device: device),
        attn = LinearAttention(dim,
            numHeads: numHeads,
            useDecay: useDecay,
            device: device,
            seed: seed),
        ffn1 = Linear(dim, ffnDim ?? dim * 4,
            device: device, seed: seed + 10),
        ffn2 = Linear(ffnDim ?? dim * 4, dim,
            device: device, seed: seed + 11);

  Tensor callSeq(Tensor x) {
    final h = x + attn.forwardSeq(ln1(x));
    return h + ffn2(ffn1(ln2(h)).relu());
  }

  Tensor step(Tensor x1, _LinAttnState state) {
    final h = x1 + attn.step(ln1(x1), state);
    return h + ffn2(ffn1(ln2(h)).relu());
  }

  @override
  List<Tensor> parameters() => [
        ...ln1.parameters(),
        ...ln2.parameters(),
        ...attn.parameters(),
        ...ffn1.parameters(),
        ...ffn2.parameters(),
      ];

  @override
  List<Module> submodules() => [ln1, ln2, attn, ffn1, ffn2];
}

/// Context-free character language model: embedding → stacked
/// state-based decoder blocks → final norm → tied output head.
class CharStateLM extends Module {
  final int vocabSize;
  final int embedDim;
  final int numHeads;
  final bool useDecay;
  final Device device;

  final Embedding tokenEmbed;
  final List<RecurrentDecoderBlock> blocks;
  final LayerNorm finalNorm;

  CharStateLM({
    required this.vocabSize,
    this.embedDim = 128,
    int numLayers = 2,
    this.numHeads = 4,
    this.useDecay = true,
    int? ffnDim,
    this.device = Device.CPU,
    int seed = 0,
  })  : tokenEmbed =
            Embedding(vocabSize, embedDim, device: device, seed: seed),
        blocks = List<RecurrentDecoderBlock>.generate(
          numLayers,
          (i) => RecurrentDecoderBlock(embedDim,
              numHeads: numHeads,
              useDecay: useDecay,
              ffnDim: ffnDim,
              device: device,
              seed: seed + 1000 * (i + 1)),
        ),
        finalNorm = LayerNorm(embedDim, device: device);

  /// Teacher-forced forward over a `[N]` id sequence → logits `[N, V]`.
  Tensor callSeq(Tensor ids) {
    var h = tokenEmbed(ids); // [N, D]
    for (final b in blocks) {
      h = b.callSeq(h);
    }
    h = finalNorm(h);
    return h.matmul(tokenEmbed.weight.transpose()); // tied head → [N, V]
  }

  Tensor _stepLogits(int id, List<_LinAttnState> states) {
    var h = tokenEmbed(Tensor.fromList([1], [id.toDouble()], device: device));
    for (var i = 0; i < blocks.length; i++) {
      h = blocks[i].step(h, states[i]);
    }
    h = finalNorm(h);
    return h.matmul(tokenEmbed.weight.transpose()); // [1, V]
  }

  /// Autoregressively extend [promptIds] by [maxNewTokens] characters,
  /// carrying the recurrent state (O(1) per step, no context window).
  /// `temperature <= 0` is greedy argmax.
  List<int> generate(
    List<int> promptIds,
    int maxNewTokens, {
    double temperature = 1.0,
    math.Random? rng,
  }) {
    if (promptIds.isEmpty) {
      throw ArgumentError('generate: promptIds must be non-empty');
    }
    final r = rng ?? math.Random();
    return Tensor.noGrad(() {
      final states = [
        for (var i = 0; i < blocks.length; i++)
          _LinAttnState.zeros(numHeads, embedDim ~/ numHeads, device),
      ];
      final out = List<int>.of(promptIds);
      Tensor logits = _stepLogits(promptIds.first, states);
      for (var i = 1; i < promptIds.length; i++) {
        logits = _stepLogits(promptIds[i], states);
      }
      for (var i = 0; i < maxNewTokens; i++) {
        final next = _sample(logits.toList(), temperature, r);
        out.add(next);
        logits = _stepLogits(next, states);
      }
      return out;
    });
  }

  static int _sample(List<double> logits, double temperature, math.Random r) {
    if (temperature <= 0) {
      var best = 0;
      for (var i = 1; i < logits.length; i++) {
        if (logits[i] > logits[best]) best = i;
      }
      return best;
    }
    var maxLogit = logits[0];
    for (final v in logits) {
      if (v > maxLogit) maxLogit = v;
    }
    var sum = 0.0;
    final probs = List<double>.filled(logits.length, 0);
    for (var i = 0; i < logits.length; i++) {
      probs[i] = math.exp((logits[i] - maxLogit) / temperature);
      sum += probs[i];
    }
    var u = r.nextDouble() * sum;
    for (var i = 0; i < probs.length; i++) {
      u -= probs[i];
      if (u <= 0) return i;
    }
    return probs.length - 1;
  }

  @override
  List<Tensor> parameters() => [
        ...tokenEmbed.parameters(),
        for (final b in blocks) ...b.parameters(),
        ...finalNorm.parameters(),
      ];

  @override
  List<Module> submodules() => [tokenEmbed, ...blocks, finalNorm];
}
