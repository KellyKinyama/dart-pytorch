/// DINOv2 (Facebook, 2023) ViT backbone. Self-supervised vision
/// embeddings — SOTA general-purpose visual features. Ports
/// `facebook/dinov2-small` (ViT-S/14, 22M params) directly from
/// safetensors.
///
/// Differences from vanilla ViT ([`ViTBackbone`](vit_backbone.dart)):
///
///   * **LayerScale** — a per-channel learnable multiplier
///     `lambda1` after both attention and MLP, applied inside the
///     residual: `x = x + gamma * f(ln(x))`. Facebook initializes at
///     1e-5; the pretrained checkpoint has learned values.
///   * **Patch embed layout** — HF stores the patch projection as a
///     Conv2d weight `[D, 3, P, P]`. Semantically identical to a
///     `Linear(P*P*3, D)` on the flattened patch pixel-vector; the
///     loader reshapes on the fly.
///   * **Position embedding interpolation** — the checkpoint is at
///     `image_size = 518` (37×37 patches). For any other input size
///     we bilinearly resize the spatial part of `position_embeddings`
///     while keeping the CLS-token row unchanged.
///
/// No **register tokens** in `facebook/dinov2-small`; the
/// `dinov2-*-with-registers` variants adds four learnable prefix
/// tokens — that's a separate follow-up.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../../tensor/tensor.dart';
import '../attention/multi_head_attention.dart';
import '../dropout.dart';
import '../layer_norm.dart';
import '../linear.dart';
import '../module.dart';
import 'vision_encoder.dart';

class DinoV2Block extends Module {
  final int embedDim;
  final int numHeads;
  final int ffnDim;
  final LayerNorm norm1;
  final MultiHeadAttention attn;
  final Tensor layerScale1; // [embedDim] learnable per-channel γ
  final LayerNorm norm2;
  final Linear mlp1;
  final Linear mlp2;
  final Tensor layerScale2;
  final Dropout dropout;

  DinoV2Block(
    this.embedDim,
    this.numHeads, {
    int? ffnDim,
    double dropoutP = 0.0,
    Device device = Device.CPU,
    int seed = 0,
  }) : ffnDim = ffnDim ?? embedDim * 4,
       norm1 = LayerNorm(embedDim, device: device),
       attn = MultiHeadAttention(
         embedDim,
         numHeads,
         bias: true, // DINOv2 uses qkv_bias=true
         dropoutP: dropoutP,
         device: device,
         seed: seed,
       ),
       layerScale1 = Tensor.fill(
         [embedDim],
         1.0,
         requiresGrad: true,
         device: device,
       ),
       norm2 = LayerNorm(embedDim, device: device),
       mlp1 = Linear(
         embedDim,
         ffnDim ?? embedDim * 4,
         device: device,
         seed: seed + 4000,
       ),
       mlp2 = Linear(
         ffnDim ?? embedDim * 4,
         embedDim,
         device: device,
         seed: seed + 5000,
       ),
       layerScale2 = Tensor.fill(
         [embedDim],
         1.0,
         requiresGrad: true,
         device: device,
       ),
       dropout = Dropout(dropoutP);

  Tensor call(Tensor x) {
    final normed = norm1(x);
    final attnOut = attn(normed);
    var h = x + _scaleRows(attnOut, layerScale1);
    final mlpNormed = norm2(h);
    var m = mlp1(mlpNormed);
    m = _geluExact(m);
    m = mlp2(m);
    return h + _scaleRows(dropout(m), layerScale2);
  }

  /// Multiply each row of `[S, D]` by the per-channel `scale [D]`.
  static Tensor _scaleRows(Tensor x, Tensor scale) {
    if (x.shape.length != 2) {
      throw ArgumentError(
        'DinoV2Block._scaleRows expects [S, D]; got ${x.shape}',
      );
    }
    // scale is [D]; broadcast row-wise. Reshape to [1, D] and use
    // `*` — which supports row-broadcast on both CPU and GPU for
    // `add` but not `mul`. Fall back to a host scatter so we work
    // consistently on both devices.
    final data = x.toFloat32List();
    final sc = scale.toFloat32List();
    final s = x.shape[0];
    final d = x.shape[1];
    final out = Float32List(s * d);
    for (int i = 0; i < s; i++) {
      for (int j = 0; j < d; j++) {
        out[i * d + j] = data[i * d + j] * sc[j];
      }
    }
    return Tensor.fromFloat32List([s, d], out, device: x.device);
  }

  static Tensor _geluExact(Tensor x) {
    // erf-based GELU: 0.5 * x * (1 + erf(x / sqrt(2))).
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
    ...norm1.parameters(),
    ...attn.parameters(),
    layerScale1,
    ...norm2.parameters(),
    ...mlp1.parameters(),
    ...mlp2.parameters(),
    layerScale2,
  ];

  @override
  List<Module> submodules() => [norm1, attn, norm2, mlp1, mlp2, dropout];
}

class DinoV2Backbone extends Module implements VisionEncoder {
  final int imageSize;
  final int patchSize;
  final int numChannels;
  @override
  final int embedDim;
  final int numLayers;
  final int numHeads;
  @override
  final int numPatches;

  final Linear patchProjection; // Conv-equivalent: [P*P*C -> D], bias=true
  final Tensor clsToken; // [1, D]
  final Tensor positionEmbeddings; // [numPatches + 1, D] — interpolated on load
  final List<DinoV2Block> blocks;
  final LayerNorm norm;

  DinoV2Backbone({
    required this.imageSize,
    required this.patchSize,
    this.numChannels = 3,
    required this.embedDim,
    required this.numLayers,
    required this.numHeads,
    Device device = Device.CPU,
  }) : assert(imageSize % patchSize == 0),
       numPatches = (imageSize ~/ patchSize) * (imageSize ~/ patchSize),
       patchProjection = Linear(
         patchSize * patchSize * numChannels,
         embedDim,
         bias: true,
         device: device,
       ),
       clsToken = Tensor.fill([1, embedDim], 0.0, device: device),
       positionEmbeddings = Tensor.fill(
         [(imageSize ~/ patchSize) * (imageSize ~/ patchSize) + 1, embedDim],
         0.0,
         device: device,
       ),
       blocks = List.generate(
         numLayers,
         (_) => DinoV2Block(embedDim, numHeads, device: device),
       ),
       norm = LayerNorm(embedDim, device: device);

  /// Forward pass on patchified image `[numPatches, P*P*C]`.
  /// Returns `[numPatches + 1, embedDim]` with CLS at row 0.
  @override
  Tensor call(Tensor patchifiedImage) {
    final expected = patchSize * patchSize * numChannels;
    if (patchifiedImage.shape.length != 2 ||
        patchifiedImage.shape[0] != numPatches ||
        patchifiedImage.shape[1] != expected) {
      throw ArgumentError(
        'DinoV2Backbone: expected [$numPatches, $expected]; '
        'got ${patchifiedImage.shape}',
      );
    }
    final patchFeats = patchProjection(patchifiedImage); // [numPatches, D]
    var xSeq = TensorConcat.concat([clsToken, patchFeats], axis: 0);
    xSeq = xSeq + positionEmbeddings;
    for (final b in blocks) {
      xSeq = b(xSeq);
    }
    return norm(xSeq);
  }

  @override
  List<Tensor> parameters() => [
    ...patchProjection.parameters(),
    clsToken,
    positionEmbeddings,
    for (final b in blocks) ...b.parameters(),
    ...norm.parameters(),
  ];

  @override
  List<Module> submodules() => [patchProjection, ...blocks, norm];
}
