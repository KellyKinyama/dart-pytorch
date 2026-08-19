/// ESM-2 (Lin et al. 2023) — Facebook / Meta's protein language model.
///
/// A Roberta-style pre-LN Transformer encoder with:
///
///   * **Rotary position embeddings** on Q/K (base = 10 000) — no
///     learned absolute position embeddings, matching HF
///     `position_embedding_type: "rotary"`.
///   * **Pre-LN attention** (LayerNorm before the self-attention
///     projections) and **pre-LN FFN** (LayerNorm before the
///     intermediate → activation → output stack).
///   * **Exact GELU** activation (erf-based, not the tanh approx).
///   * **Post-encoder LayerNorm** on the final hidden states
///     (`emb_layer_norm_after`).
///   * **Fixed 33-token vocab** — 20 canonical amino acids, 5 ambiguity
///     codes, 4 special tokens (`<cls>`, `<pad>`, `<eos>`, `<unk>`),
///     `<mask>`, `<null_1>`, `.`, `-`. See [esm2Vocab] below.
///
/// Ships two configs:
///
///   * `esm2_t6_8mConfig`   — `facebook/esm2_t6_8M_UR50D`   (~8M params)
///   * `esm2_t12_35mConfig` — `facebook/esm2_t12_35M_UR50D` (~35M params)
///
/// Both are fine on a 6 GB laptop GPU. Bigger checkpoints (150M / 650M
/// / 3B / 15B) share the same architecture — only the config numbers
/// change.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../tensor/tensor.dart';
import 'attention/multi_head_attention.dart';
import 'embedding.dart';
import 'layer_norm.dart';
import 'linear.dart';
import 'module.dart';
import 'rotary.dart';

class ESM2Config {
  final int vocabSize;
  final int maxCtx;
  final int embedDim;
  final int numLayers;
  final int numHeads;
  final int ffnDim;
  final double layerNormEps;
  final double ropeBase;
  final Device device;
  final int seed;

  const ESM2Config({
    required this.vocabSize,
    required this.maxCtx,
    required this.embedDim,
    required this.numLayers,
    required this.numHeads,
    required this.ffnDim,
    this.layerNormEps = 1e-5,
    this.ropeBase = 10000.0,
    this.device = Device.CPU,
    this.seed = 0,
  });
}

class ESM2Layer extends Module {
  final LayerNorm attnLn; // pre-LN before attention
  final MultiHeadAttention attn; // biased Q/K/V + O
  final LayerNorm ffnLn; // pre-LN before FFN
  final Linear ffnIntermediate; // [ffn, D]
  final Linear ffnOutput; // [D, ffn]

  ESM2Layer({
    required int embedDim,
    required int numHeads,
    required int ffnDim,
    required RopeCache rope,
    double layerNormEps = 1e-5,
    Device device = Device.CPU,
    int seed = 0,
  }) : attnLn = LayerNorm(embedDim, eps: layerNormEps, device: device),
       attn = MultiHeadAttention(
         embedDim,
         numHeads,
         bias: true,
         device: device,
         seed: seed,
       ),
       ffnLn = LayerNorm(embedDim, eps: layerNormEps, device: device),
       ffnIntermediate = Linear(
         embedDim,
         ffnDim,
         bias: true,
         device: device,
         seed: seed + 100_000,
       ),
       ffnOutput = Linear(
         ffnDim,
         embedDim,
         bias: true,
         device: device,
         seed: seed + 200_000,
       ) {
    attn.rope = rope;
  }

  Tensor call(Tensor x, {Tensor? mask}) {
    final a = attn(attnLn(x), mask: mask);
    final h = x + a;
    final m = ffnOutput(_geluExact(ffnIntermediate(ffnLn(h))));
    return h + m;
  }

  /// erf-based GELU: `0.5 · x · (1 + erf(x / √2))`. Matches HF's
  /// default GELU exactly (Roberta / ESM-2 use `gelu`, not `gelu_new`).
  static Tensor _geluExact(Tensor x) {
    const invSqrt2 = 0.7071067811865475;
    final data = x.toFloat32List();
    final out = Float32List(data.length);
    for (int i = 0; i < data.length; i++) {
      out[i] = 0.5 * data[i] * (1.0 + _erf(data[i] * invSqrt2));
    }
    return Tensor.fromFloat32List(x.shape, out, device: x.device);
  }

  // Abramowitz & Stegun approximation of erf, max abs error ~1.5e-7.
  static double _erf(double x) {
    final sign = x < 0 ? -1.0 : 1.0;
    x = x.abs();
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;
    final t = 1.0 / (1.0 + p * x);
    final y =
        1.0 -
        (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x);
    return sign * y;
  }

  @override
  List<Tensor> parameters() => [
    ...attnLn.parameters(),
    ...attn.parameters(),
    ...ffnLn.parameters(),
    ...ffnIntermediate.parameters(),
    ...ffnOutput.parameters(),
  ];

  @override
  List<Module> submodules() => [
    attnLn,
    attn,
    ffnLn,
    ffnIntermediate,
    ffnOutput,
  ];
}

class ESM2Model extends Module {
  final ESM2Config config;

  final Embedding embedIn; // word_embeddings [V, D]
  final List<ESM2Layer> layers;
  final LayerNorm finalLn; // emb_layer_norm_after
  final RopeCache rope;

  ESM2Model(this.config)
    : embedIn = Embedding(
        config.vocabSize,
        config.embedDim,
        device: config.device,
        seed: config.seed,
      ),
      finalLn = LayerNorm(
        config.embedDim,
        eps: config.layerNormEps,
        device: config.device,
      ),
      rope = RopeCache(
        maxCtx: config.maxCtx,
        headDim: config.embedDim ~/ config.numHeads,
        base: config.ropeBase,
        device: config.device,
      ),
      layers = <ESM2Layer>[] {
    for (int i = 0; i < config.numLayers; i++) {
      layers.add(
        ESM2Layer(
          embedDim: config.embedDim,
          numHeads: config.numHeads,
          ffnDim: config.ffnDim,
          rope: rope,
          layerNormEps: config.layerNormEps,
          device: config.device,
          seed: config.seed + 1_000_000 + i * 1_000,
        ),
      );
    }
  }

  /// Forward pass. `tokens` is a 1D `[seqLen]` tensor of protein-token
  /// indices as float32 (typically `[<cls>, aa1, aa2, ..., aaN, <eos>]`).
  /// Output is `[seqLen, embedDim]` per-residue hidden states.
  Tensor call(Tensor tokens) {
    if (tokens.shape.length != 1) {
      throw ArgumentError(
        'ESM2Model: tokens must be 1D [seqLen]; got ${tokens.shape}',
      );
    }
    final n = tokens.shape.last;
    if (n == 0) {
      throw ArgumentError('ESM2Model: empty sequence');
    }
    if (n > config.maxCtx) {
      throw ArgumentError(
        'ESM2Model: seqLen $n exceeds maxCtx ${config.maxCtx}',
      );
    }
    var h = embedIn(tokens);
    for (final layer in layers) {
      h = layer(h);
    }
    return finalLn(h);
  }

  /// Mean-pool the per-residue hidden states to a single `[embedDim]`
  /// protein-level embedding. Common ESM-2 usage for downstream
  /// property classifiers / regressors.
  Tensor meanPool(Tensor tokens) {
    final h = call(tokens); // [N, D]
    final n = h.shape[0];
    final d = h.shape[1];
    final data = Tensor.noGrad(() => h.toList());
    final out = List<double>.filled(d, 0);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < d; j++) {
        out[j] += data[i * d + j];
      }
    }
    for (int j = 0; j < d; j++) {
      out[j] /= n;
    }
    return Tensor.fromList([d], out, device: h.device);
  }

  @override
  List<Tensor> parameters() => [
    ...embedIn.parameters(),
    for (final l in layers) ...l.parameters(),
    ...finalLn.parameters(),
  ];

  @override
  List<Module> submodules() => [embedIn, ...layers, finalLn];
}

/// ESM-2's fixed 33-token vocab (same as HF `facebook/esm2_*`).
/// Index 0 is `<cls>`, index 1 is `<pad>`, index 2 is `<eos>`, index
/// 3 is `<unk>`, indices 4–27 are the 24 amino-acid / ambiguity codes,
/// 28 is `.`, 29 is `-`, 30 is `<null_1>`, 32 is `<mask>`.
///
/// The typical inference tokenisation is
/// `[<cls>] + amino_acid_indices + [<eos>]`.
const List<String> esm2Vocab = [
  '<cls>', '<pad>', '<eos>', '<unk>', //
  'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', //
  'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', //
  'X', 'B', 'U', 'Z', //
  'O', '.', '-', '<null_1>', '<mask>',
];

/// Encode a raw amino-acid string (e.g. "MSVAK...") into a 1-D
/// `List<int>` of ESM-2 vocab indices with `<cls>` prepended and
/// `<eos>` appended. Unknown characters map to `<unk>` (index 3).
List<int> encodeProteinSequence(String seq) {
  final map = <String, int>{
    for (int i = 0; i < esm2Vocab.length; i++) esm2Vocab[i]: i,
  };
  final out = <int>[map['<cls>']!];
  for (final ch in seq.toUpperCase().split('')) {
    out.add(map[ch] ?? map['<unk>']!);
  }
  out.add(map['<eos>']!);
  return out;
}
