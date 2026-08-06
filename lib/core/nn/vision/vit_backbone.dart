/// Vision Transformer (ViT) backbone.
///
/// Takes a batch of pre-patchified images (each patch flattened to a
/// vector) and produces per-token contextualized features via the
/// standard ViT recipe:
///
/// 1. **Patch projection** — `Linear(patchSize * patchSize * numChannels,
///    embedDim)` maps each patch pixel vector to an embedding.
/// 2. **[CLS] token** — a single learnable `[1, embedDim]` vector is
///    prepended to the sequence. Downstream heads read the CLS row of
///    the encoded output as an image-level summary.
/// 3. **Positional embeddings** — a learnable `[numPatches + 1,
///    embedDim]` table is added to the (CLS + patches) sequence.
/// 4. **Transformer encoder** — the standard pre-LN block stack from
///    [TransformerEncoder]. No causal mask (vision is bidirectional).
///
/// This is the "learn from images" analog of a language model: swap
/// the token embedding for a patch projection, keep the transformer
/// stack, tack a task-specific head on top of the CLS output.
///
/// Currently 2D (single image per call): input shape
/// `[numPatches, patchSize * patchSize * numChannels]`, output shape
/// `[numPatches + 1, embedDim]`. Batched image support can be added by
/// switching the concat to axis 1 with `[B, ...]` inputs, once a
/// batched concat kernel lands.
library;

import 'dart:math' as math;

import '../../tensor/tensor.dart';
import '../linear.dart';
import '../module.dart';
import '../transformer_encoder.dart';
import 'vision_encoder.dart';

class ViTBackbone extends Module implements VisionEncoder {
  final int imageSize;
  final int patchSize;
  final int numChannels;
  @override
  final int embedDim;
  final int numLayers;
  final int numHeads;
  @override
  final int numPatches;

  final Linear patchProjection;
  final Tensor clsToken; // [1, embedDim]
  final Tensor posEmbeddings; // [numPatches + 1, embedDim]
  final TransformerEncoder encoder;

  ViTBackbone({
    required this.imageSize,
    required this.patchSize,
    this.numChannels = 3,
    required this.embedDim,
    this.numLayers = 4,
    this.numHeads = 4,
    double dropoutP = 0.0,
    Device device = Device.CPU,
    int seed = 0,
  }) : assert(
         imageSize % patchSize == 0,
         'imageSize must be divisible by patchSize',
       ),
       numPatches = (imageSize ~/ patchSize) * (imageSize ~/ patchSize),
       patchProjection = Linear(
         patchSize * patchSize * numChannels,
         embedDim,
         bias: true,
         device: device,
         seed: seed + 1,
       ),
       clsToken = _initSmallGaussian(
         [1, embedDim],
         scale: 0.02,
         device: device,
         seed: seed + 2,
       ),
       posEmbeddings = _initSmallGaussian(
         [(imageSize ~/ patchSize) * (imageSize ~/ patchSize) + 1, embedDim],
         scale: 0.02,
         device: device,
         seed: seed + 3,
       ),
       encoder = TransformerEncoder(
         numLayers,
         embedDim,
         numHeads,
         dropoutP: dropoutP,
         device: device,
         seed: seed + 100,
       );

  static Tensor _initSmallGaussian(
    List<int> shape, {
    required double scale,
    required Device device,
    required int seed,
  }) {
    final rng = math.Random(seed);
    final n = shape.fold<int>(1, (a, b) => a * b);
    // Box–Muller for a small standard-normal, then scaled.
    final vals = List<double>.generate(n, (_) {
      final u1 = rng.nextDouble().clamp(1e-9, 1.0);
      final u2 = rng.nextDouble();
      final z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2);
      return z * scale;
    });
    return Tensor.fromList(shape, vals, requiresGrad: true, device: device);
  }

  /// Forward pass.
  ///
  /// * `patchifiedImage` — `[numPatches, patchSize * patchSize *
  ///   numChannels]`. Each row is one flattened patch.
  ///
  /// Returns the encoder output `[numPatches + 1, embedDim]` with the
  /// CLS token at row 0.
  @override
  Tensor call(Tensor patchifiedImage) {
    final expectedPixels = patchSize * patchSize * numChannels;
    if (patchifiedImage.shape.length != 2 ||
        patchifiedImage.shape[0] != numPatches ||
        patchifiedImage.shape[1] != expectedPixels) {
      throw ArgumentError(
        'ViTBackbone: expected input [$numPatches, $expectedPixels]; '
        'got ${patchifiedImage.shape}',
      );
    }

    // 1. Project each patch to the model dim: [numPatches, embedDim].
    final xPatches = patchProjection(patchifiedImage);

    // 2. Prepend CLS token: [numPatches + 1, embedDim].
    final xSeq = TensorConcat.concat([clsToken, xPatches], axis: 0);

    // 3. Add positional embeddings.
    final xPos = xSeq + posEmbeddings;

    // 4. Transformer encoder (no causal mask for vision).
    return encoder(xPos);
  }

  @override
  List<Tensor> parameters() => [
    ...patchProjection.parameters(),
    clsToken,
    posEmbeddings,
    ...encoder.parameters(),
  ];

  @override
  List<Module> submodules() => [patchProjection, encoder];
}

/// Take the CLS row (row 0) of a ViT encoder output. Convenience for
/// heads that want a `[1, embedDim]` image-level feature vector.
Tensor vitClsFeature(Tensor encoded) {
  if (encoded.shape.length != 2) {
    throw ArgumentError(
      'vitClsFeature: expected 2D [seqLen, embedDim]; got ${encoded.shape}',
    );
  }
  return TensorAft.sliceTopLeft(encoded, 1, encoded.shape[1]);
}
