/// **Aspirational** port of the SkyReels-V2 / Wan diffusion transformer.
///
/// This is a Dart translation of the transformer backbone in
/// [`SkyworkAI/SkyReels-V2/skyreels_v2_infer/modules/transformer.py`](https://github.com/SkyworkAI/SkyReels-V2/blob/main/skyreels_v2_infer/modules/transformer.py)
/// aka `WanModel`. It exists to prove that the AirLLM-style
/// layer-streaming pattern in this repo generalises to video-DiT
/// backbones — **not** to actually render videos on the current
/// hardware. The reference model requires 14.7 GB VRAM for the
/// smallest 1.3B variant at 540P; a 6 GB GPU cannot host it, and the
/// end-to-end pipeline needs several supporting modules that are out
/// of scope here (see the TODO list below).
///
/// ## What is implemented
///
///   * [SkyReelsV2Config] — dims/heads/layers matching the reference
///     `WanModel(**kwargs)` — 1.3B and 14B presets.
///   * [SkyReelsV2Block] — one transformer block with adaptive
///     modulation (AdaLN-single), self-attention, cross-attention,
///     and gated FFN. Uses this repo's [MultiHeadAttention] and
///     [MultiHeadCrossAttention] with tanh-approx GELU.
///   * [SkyReelsV2Model] — patch-embed (as a placeholder [Linear],
///     not a real Conv3d), text-embed (2-layer MLP), time-embed
///     (sinusoidal → MLP → 6-way modulation projection), N blocks,
///     and the modulated output head.
///
/// ## What is NOT implemented (intentionally, for scope)
///
///   * **Conv3D patch embedding.** The reference uses
///     `Conv3d(in_dim=16, dim, kernel=patch_size, stride=patch_size)`
///     with `patch_size=(1,2,2)`. Here [SkyReelsV2Model] takes a
///     `[num_tokens, in_dim * prod(patch_size)]` tensor and applies a
///     linear projection instead. Correct for pre-patched inputs,
///     wrong for raw video.
///   * **3D RoPE (`rope_apply` with time/height/width frequency
///     partitions).** Skipped entirely — self-attention runs
///     RoPE-free. Bit-exact video output would require the WanModel
///     `freqs = [rope(d - 4*(d/6)), rope(2*(d/6)), rope(2*(d/6))]`
///     ternary split and 3D grid application.
///   * **Q/K RMSNorm inside attention (`qk_norm=True`).** SkyReels
///     applies RMSNorm to Q and K *after* projection and before
///     scaled-dot-product. Our [MultiHeadAttention] doesn't expose
///     that hook. Adding it means a custom attention implementation.
///   * **I2V cross-attention (`WanI2VCrossAttention` with separate
///     `k_img`/`v_img` on the first 257 image tokens).** T2V-only
///     here.
///   * **Diffusion Forcing per-token noise schedule** and the
///     block-wise causal attention mask used for autoregressive
///     video extension.
///   * **Teacache** residual-caching acceleration and the FPS
///     embedding branch.
///   * **VAE (AutoencoderKLWan), T5 text encoder, UniPC scheduler,
///     flow-matching sampler**, video I/O.
///
/// See [doc/skyreels_v2_port.md](../../../doc/skyreels_v2_port.md)
/// for a full checklist.
library;

import 'dart:math' as math;

import '../tensor/tensor.dart';
import 'attention/multi_head_attention.dart';
import 'attention/multi_head_cross_attention.dart';
import 'layer_norm.dart';
import 'linear.dart';
import 'module.dart';

/// Config mirroring `WanModel(**kwargs)` in the reference impl.
class SkyReelsV2Config {
  final String modelType; // 't2v' | 'i2v' — only 't2v' supported here.
  final List<int> patchSize;
  final int textLen;
  final int inDim;
  final int dim;
  final int ffnDim;
  final int freqDim;
  final int textDim;
  final int outDim;
  final int numHeads;
  final int numLayers;
  final bool crossAttnNorm;
  final double eps;
  final Device device;
  final int seed;

  const SkyReelsV2Config({
    this.modelType = 't2v',
    this.patchSize = const [1, 2, 2],
    this.textLen = 512,
    this.inDim = 16,
    required this.dim,
    required this.ffnDim,
    this.freqDim = 256,
    this.textDim = 4096,
    this.outDim = 16,
    required this.numHeads,
    required this.numLayers,
    this.crossAttnNorm = true,
    this.eps = 1e-6,
    this.device = Device.CPU,
    this.seed = 0,
  });

  int get headDim => dim ~/ numHeads;
  int get patchProduct => patchSize.fold(1, (a, b) => a * b);

  /// Wan/SkyReels-V2 diffusion-forcing 1.3B config.
  /// `dim=1536, num_layers=30, num_heads=12, ffn_dim=8960` —
  /// standard Wan-1.3B numbers. Confirm against the shipped
  /// `Skywork/SkyReels-V2-DF-1.3B-540P` `config.json` before use.
  static SkyReelsV2Config df1_3B({
    Device device = Device.CPU,
    int seed = 0,
  }) =>
      SkyReelsV2Config(
        dim: 1536,
        ffnDim: 8960,
        numHeads: 12,
        numLayers: 30,
        device: device,
        seed: seed,
      );

  /// Wan/SkyReels-V2 diffusion-forcing 14B config.
  /// `dim=5120, num_layers=40, num_heads=40, ffn_dim=13824` — Wan-14B
  /// numbers. Confirm against
  /// `Skywork/SkyReels-V2-DF-14B-540P` config before use.
  static SkyReelsV2Config df14B({
    Device device = Device.CPU,
    int seed = 0,
  }) =>
      SkyReelsV2Config(
        dim: 5120,
        ffnDim: 13824,
        numHeads: 40,
        numLayers: 40,
        device: device,
        seed: seed,
      );
}

/// GELU with the tanh approximation used by SkyReels-V2 (`nn.GELU(
/// approximate="tanh")`).
Tensor geluTanh(Tensor x) {
  const c = 0.7978845608028654;
  final inner = (x + x.pow(3.0) * 0.044715) * c;
  return x * 0.5 * (inner.tanh() + 1.0);
}

/// SiLU (Swish): `x * sigmoid(x)`.
Tensor silu(Tensor x) => x * x.sigmoid();

/// One transformer block. Layout:
///
///   1.  `e = block.modulation + e0` split into 6 [1, dim] pieces.
///   2.  `h = norm1(x) * (1 + e[1]) + e[0]`
///   3.  `x = x + selfAttn(h) * e[2]`
///   4.  `h = norm3(x)` then `x = x + crossAttn(h, context)`
///   5.  `h = norm2(x) * (1 + e[4]) + e[3]`
///   6.  `x = x + ffn2(gelu(ffn1(h))) * e[5]`
///
/// `norm1` / `norm2` are affine-free LayerNorms in the reference;
/// this port always builds a LayerNorm with weight/bias fields but
/// leaves them at their defaults (1, 0) so no keys are consumed
/// from the checkpoint for those two.
class SkyReelsV2Block extends Module {
  final int dim;
  final int ffnDim;
  final int numHeads;

  final Tensor modulation; // [1, 6, dim]

  final LayerNorm norm1;
  final LayerNorm norm3;
  final LayerNorm norm2;

  final MultiHeadAttention selfAttn;
  final MultiHeadCrossAttention crossAttn;

  final Linear ffn1;
  final Linear ffn2;

  SkyReelsV2Block(
    this.dim,
    this.ffnDim,
    this.numHeads, {
    double eps = 1e-6,
    Device device = Device.CPU,
    int seed = 0,
  })  : modulation = Tensor.fill(
          [1, 6, dim],
          0.0,
          device: device,
        ),
        norm1 = LayerNorm(dim, eps: eps, device: device),
        norm3 = LayerNorm(dim, eps: eps, device: device),
        norm2 = LayerNorm(dim, eps: eps, device: device),
        selfAttn = MultiHeadAttention(
          dim,
          numHeads,
          bias: true,
          device: device,
          seed: seed,
        ),
        crossAttn = MultiHeadCrossAttention(
          dim,
          dim,
          numHeads,
          bias: true,
          device: device,
          seed: seed + 10000,
        ),
        ffn1 = Linear(
          dim,
          ffnDim,
          bias: true,
          device: device,
          seed: seed + 20000,
        ),
        ffn2 = Linear(
          ffnDim,
          dim,
          bias: true,
          device: device,
          seed: seed + 30000,
        );

  /// `x`: `[seq_len, dim]`, `e6`: six `[1, dim]` modulation vectors
  /// (already summed with the block's own `modulation` param),
  /// `context`: `[text_len, dim]`.
  Tensor call(Tensor x, List<Tensor> e6, Tensor context) {
    if (e6.length != 6) {
      throw ArgumentError('SkyReelsV2Block: e6 must have 6 entries');
    }
    var h = norm1(x);
    h = h * (e6[1] + 1.0) + e6[0];
    final a = selfAttn(h);
    var y = x + a * e6[2];

    final c = crossAttn(norm3(y), context);
    y = y + c;

    var m = norm2(y);
    m = m * (e6[4] + 1.0) + e6[3];
    final f = ffn2(geluTanh(ffn1(m)));
    y = y + f * e6[5];
    return y;
  }

  @override
  List<Tensor> parameters() => [
        modulation,
        ...norm1.parameters(),
        ...norm3.parameters(),
        ...norm2.parameters(),
        ...selfAttn.parameters(),
        ...crossAttn.parameters(),
        ...ffn1.parameters(),
        ...ffn2.parameters(),
      ];

  @override
  List<Module> submodules() => [
        norm1,
        norm3,
        norm2,
        selfAttn,
        crossAttn,
        ffn1,
        ffn2,
      ];
}

/// Full SkyReels-V2 DiT backbone. See the library doc for the list
/// of things that are placeholders vs. real.
class SkyReelsV2Model extends Module {
  final SkyReelsV2Config config;

  /// Placeholder for `Conv3d(in_dim, dim, patch_size, patch_size)` —
  /// applied to pre-patched input tokens `[N, in_dim * prod(patch_size)]`.
  final Linear patchEmbed;

  /// `nn.Sequential(Linear(text_dim, dim), GELU(tanh), Linear(dim, dim))`.
  final Linear textEmbed0;
  final Linear textEmbed2;

  /// Time embedding MLP: sinusoidal(freq_dim) → Linear(freq_dim, dim)
  /// → SiLU → Linear(dim, dim), then time_projection: SiLU →
  /// Linear(dim, 6*dim) for AdaLN modulation.
  final Linear timeEmbed0;
  final Linear timeEmbed2;
  final Linear timeProjection;

  final List<SkyReelsV2Block> blocks;

  /// Output head: modulation [1, 2, dim] + LayerNorm (no affine) +
  /// Linear(dim, prod(patch_size) * out_dim).
  final Tensor headModulation;
  final LayerNorm headNorm;
  final Linear headOut;

  SkyReelsV2Model(this.config)
      : patchEmbed = Linear(
          config.inDim * config.patchProduct,
          config.dim,
          bias: true,
          device: config.device,
          seed: config.seed + 1,
        ),
        textEmbed0 = Linear(
          config.textDim,
          config.dim,
          bias: true,
          device: config.device,
          seed: config.seed + 2,
        ),
        textEmbed2 = Linear(
          config.dim,
          config.dim,
          bias: true,
          device: config.device,
          seed: config.seed + 3,
        ),
        timeEmbed0 = Linear(
          config.freqDim,
          config.dim,
          bias: true,
          device: config.device,
          seed: config.seed + 4,
        ),
        timeEmbed2 = Linear(
          config.dim,
          config.dim,
          bias: true,
          device: config.device,
          seed: config.seed + 5,
        ),
        timeProjection = Linear(
          config.dim,
          6 * config.dim,
          bias: true,
          device: config.device,
          seed: config.seed + 6,
        ),
        blocks = <SkyReelsV2Block>[],
        headModulation = Tensor.fill(
          [1, 2, config.dim],
          0.0,
          device: config.device,
        ),
        headNorm = LayerNorm(config.dim, eps: config.eps, device: config.device),
        headOut = Linear(
          config.dim,
          config.patchProduct * config.outDim,
          bias: true,
          device: config.device,
          seed: config.seed + 7,
        ) {
    for (int i = 0; i < config.numLayers; i++) {
      blocks.add(
        SkyReelsV2Block(
          config.dim,
          config.ffnDim,
          config.numHeads,
          eps: config.eps,
          device: config.device,
          seed: config.seed + 100000 + i * 1000,
        ),
      );
    }
  }

  /// Sinusoidal timestep embedding — matches `sinusoidal_embedding_1d`
  /// in the reference. `t` is a rank-0 or rank-1 tensor with the
  /// integer step; here `step` is passed directly as a double.
  Tensor sinusoidalTimestep(double step) {
    final half = config.freqDim ~/ 2;
    final data = List<double>.filled(config.freqDim, 0.0);
    for (int i = 0; i < half; i++) {
      final freq = math.pow(10000.0, -i / half) as double;
      final ang = step * freq;
      data[i] = math.cos(ang);
      data[i + half] = math.sin(ang);
    }
    return Tensor.fromList([1, config.freqDim], data, device: config.device);
  }

  /// Produce the six [1, dim] modulation vectors for a block from a
  /// timestep and the block's own `modulation` parameter.
  List<Tensor> _computeE6ForBlock(Tensor e0, Tensor blockModulation) {
    // e0: [1, 6*dim] flattened; blockModulation: [1, 6, dim].
    // Result: 6 × [1, dim] with block's modulation added.
    final e0List = e0.toList();
    final modList = blockModulation.toList();
    final dim = config.dim;
    final out = <Tensor>[];
    for (int k = 0; k < 6; k++) {
      final slice = List<double>.filled(dim, 0.0);
      for (int i = 0; i < dim; i++) {
        slice[i] = e0List[k * dim + i] + modList[k * dim + i];
      }
      out.add(Tensor.fromList([1, dim], slice, device: config.device));
    }
    return out;
  }

  /// Forward. `patches` is `[num_tokens, in_dim * prod(patch_size)]`
  /// (video already patched by the caller — a real port would apply
  /// Conv3D here). `text` is `[text_len, text_dim]`. `step` is the
  /// diffusion timestep (float).
  Tensor call(Tensor patches, Tensor text, double step) {
    if (patches.shape.length != 2 ||
        patches.shape[1] != config.inDim * config.patchProduct) {
      throw ArgumentError(
        'SkyReelsV2Model: patches must be [N, ${config.inDim * config.patchProduct}]; '
        'got ${patches.shape}',
      );
    }
    if (text.shape.length != 2 || text.shape[1] != config.textDim) {
      throw ArgumentError(
        'SkyReelsV2Model: text must be [textLen, ${config.textDim}]; '
        'got ${text.shape}',
      );
    }

    var x = patchEmbed(patches); // [N, dim]

    final ctx = textEmbed2(geluTanh(textEmbed0(text))); // [textLen, dim]

    final tSin = sinusoidalTimestep(step); // [1, freqDim]
    final tEmb = timeEmbed2(silu(timeEmbed0(tSin))); // [1, dim]
    final e0 = timeProjection(silu(tEmb)); // [1, 6*dim]

    for (int i = 0; i < config.numLayers; i++) {
      final e6 = _computeE6ForBlock(e0, blocks[i].modulation);
      x = blocks[i](x, e6, ctx);
    }

    // Head: modulate + norm + linear.
    // modulation is [1, 2, dim], time gives one [1, dim]; reference
    // does `(mod + t.unsqueeze(1)).chunk(2, dim=1)` with `e` (not e0).
    final modList = headModulation.toList();
    final tList = tEmb.toList();
    final dim = config.dim;
    final shift = List<double>.filled(dim, 0.0);
    final scale = List<double>.filled(dim, 0.0);
    for (int i = 0; i < dim; i++) {
      shift[i] = modList[i] + tList[i];
      scale[i] = modList[dim + i] + tList[i];
    }
    final shiftT = Tensor.fromList([1, dim], shift, device: config.device);
    final scaleT = Tensor.fromList([1, dim], scale, device: config.device);
    final h = headNorm(x) * (scaleT + 1.0) + shiftT;
    return headOut(h); // [N, patchProduct * outDim]
  }

  @override
  List<Tensor> parameters() => [
        ...patchEmbed.parameters(),
        ...textEmbed0.parameters(),
        ...textEmbed2.parameters(),
        ...timeEmbed0.parameters(),
        ...timeEmbed2.parameters(),
        ...timeProjection.parameters(),
        for (final b in blocks) ...b.parameters(),
        headModulation,
        ...headNorm.parameters(),
        ...headOut.parameters(),
      ];

  @override
  List<Module> submodules() => [
        patchEmbed,
        textEmbed0,
        textEmbed2,
        timeEmbed0,
        timeEmbed2,
        timeProjection,
        ...blocks,
        headNorm,
        headOut,
      ];
}
