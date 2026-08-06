/// OpenAI **CLIP** vision transformer — implementation compatible
/// with HuggingFace `openai/clip-vit-base-patch{16,32}` and
/// `openai/clip-vit-large-patch14` checkpoints.
///
/// Architecture recap (matches HF `CLIPVisionModel`):
///
/// ```
///   patch_embedding    Conv2d(3, D, kernel=P, stride=P, bias=False)
///                        ← equivalent to Linear(3*P*P, D, bias=False)
///                          when we patchify the image ourselves.
///   class_embedding    learnable [D] prepended
///   position_embedding learnable [numPatches + 1, D] added
///   pre_layrnorm       LayerNorm(D)                 ← HF typo preserved
///   encoder            12× TransformerBlock (pre-LN, QuickGELU, attnBias=true)
///   post_layernorm     LayerNorm(D)
/// ```
///
/// Forward:
///
///     x = patchProjection(patchifiedImage)          # [P², D]
///     x = concat([clsToken, x], axis=0)             # [1 + P², D]
///     x = x + posEmbeddings
///     x = preLayerNorm(x)
///     x = encoder(x)                                # includes post_layernorm
///     return x                                      # [1 + P², D]
///
/// This class only implements the *vision* half of CLIP (no text
/// tower, no `visual_projection`). The output is the full
/// `[num_patches + 1, embedDim]` sequence — LLaVA-style adapters
/// consume the patch rows (rows `1..`) as image tokens.
///
/// Input contract (matches the rest of our vision stack):
///
///   patchifiedImage: `[numPatches, patchSize * patchSize * numChannels]`
///     Each row is one flattened patch with channels interleaved
///     per pixel in `(row, col, channel)` order — the layout
///     produced by our patchify helpers (e.g. `bin/llama_image_rag.dart`).
///     The safetensors loader ([ClipHFLoader]) permutes the
///     original Conv2d weight from `(out, C, H, W)` to
///     `(out, H, W, C)` so the numeric result is identical to
///     applying the original CLIP `Conv2d` on `[C, H, W]` pixels.
///
/// Weights are randomly initialised at construction — use
/// [ClipHFLoader.loadFile] to overwrite them with a real CLIP
/// checkpoint. Config presets for B/32, B/16 and L/14 are provided
/// on [ClipHFLoader].
library;

import 'dart:math' as math;

import '../../tensor/tensor.dart';
import '../layer_norm.dart';
import '../linear.dart';
import '../module.dart';
import '../transformer.dart';
import '../transformer_encoder.dart';
import 'vision_encoder.dart';

/// Configuration for a [CLIPVisionModel]. Values map 1:1 onto keys of
/// the HuggingFace CLIP `config.json` (`vision_config` subtree).
class CLIPVisionConfig {
  final int imageSize;
  final int patchSize;
  final int numChannels;
  final int embedDim;
  final int numLayers;
  final int numHeads;
  final int ffnDim;
  final double layerNormEps;
  final Device device;
  final int seed;

  const CLIPVisionConfig({
    this.imageSize = 224,
    this.patchSize = 32,
    this.numChannels = 3,
    this.embedDim = 768,
    this.numLayers = 12,
    this.numHeads = 12,
    this.ffnDim = 3072,
    this.layerNormEps = 1e-5,
    this.device = Device.CPU,
    this.seed = 0,
  });

  int get numPatches =>
      (imageSize ~/ patchSize) * (imageSize ~/ patchSize);

  int get patchPixels => patchSize * patchSize * numChannels;
}

class CLIPVisionModel extends Module implements VisionEncoder {
  final CLIPVisionConfig config;

  final Linear patchProjection; // [D, 3*P*P], no bias
  final Tensor clsToken; // [1, D]
  final Tensor posEmbeddings; // [numPatches + 1, D]
  final LayerNorm preLayerNorm;
  final TransformerEncoder encoder; // includes final post_layernorm

  @override
  int get embedDim => config.embedDim;

  @override
  int get numPatches => config.numPatches;

  CLIPVisionModel(this.config)
    : patchProjection = Linear(
        config.patchPixels,
        config.embedDim,
        bias: false,
        device: config.device,
        seed: config.seed + 1,
      ),
      clsToken = _initSmallGaussian(
        [1, config.embedDim],
        scale: 0.02,
        device: config.device,
        seed: config.seed + 2,
      ),
      posEmbeddings = _initSmallGaussian(
        [config.numPatches + 1, config.embedDim],
        scale: 0.02,
        device: config.device,
        seed: config.seed + 3,
      ),
      preLayerNorm = LayerNorm(
        config.embedDim,
        eps: config.layerNormEps,
        device: config.device,
      ),
      encoder = TransformerEncoder(
        config.numLayers,
        config.embedDim,
        config.numHeads,
        ffnDim: config.ffnDim,
        finalNorm: true,
        attnBias: true,
        activation: Activation.quickGelu,
        device: config.device,
        seed: config.seed + 100,
      );

  static Tensor _initSmallGaussian(
    List<int> shape, {
    required double scale,
    required Device device,
    required int seed,
  }) {
    final rng = math.Random(seed);
    final n = shape.fold<int>(1, (a, b) => a * b);
    final vals = List<double>.generate(n, (_) {
      final u1 = rng.nextDouble().clamp(1e-9, 1.0);
      final u2 = rng.nextDouble();
      final z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2);
      return z * scale;
    });
    return Tensor.fromList(shape, vals, requiresGrad: true, device: device);
  }

  @override
  Tensor call(Tensor patchifiedImage) {
    final expected = config.patchPixels;
    if (patchifiedImage.shape.length != 2 ||
        patchifiedImage.shape[0] != config.numPatches ||
        patchifiedImage.shape[1] != expected) {
      throw ArgumentError(
        'CLIPVisionModel: expected input [${config.numPatches}, $expected]; '
        'got ${patchifiedImage.shape}',
      );
    }

    final xPatches = patchProjection(patchifiedImage); // [P², D]
    final xSeq = TensorConcat.concat([clsToken, xPatches], axis: 0);
    final xPos = xSeq + posEmbeddings;
    final xPre = preLayerNorm(xPos);
    return encoder(xPre); // includes post_layernorm
  }

  @override
  List<Tensor> parameters() => [
    ...patchProjection.parameters(),
    clsToken,
    posEmbeddings,
    ...preLayerNorm.parameters(),
    ...encoder.parameters(),
  ];

  @override
  List<Module> submodules() =>
      [patchProjection, preLayerNorm, encoder];
}
