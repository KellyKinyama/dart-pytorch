/// Small interface shared by vision backbones that plug into
/// [LlamaVision] (and, more generally, anything that treats an image
/// as a sequence of tokens).
///
/// The contract:
///
/// * `call(patchifiedImage)` — takes a rank-2 tensor
///   `[numPatches, patchPixels]` (rows = patches in raster order,
///   channels interleaved per pixel — the same layout our other
///   patchify helpers produce) and returns
///   `[numPatches + 1, embedDim]` where row 0 is the CLS / summary
///   token and rows `1..` are per-patch contextualised features.
///
/// * `embedDim` — the trailing dim of the returned features.
///
/// * `numPatches` — number of image patches per forward, excluding
///   the CLS token. Callers can use `numPatches + 1` as the image
///   token count when computing sequence offsets.
///
/// Implementations so far: [ViTBackbone] (our from-scratch ViT) and
/// [CLIPVisionModel] (loads HuggingFace CLIP checkpoints). A wrapper
/// like [LlamaVision] can accept any [VisionEncoder] without caring
/// which is which.
library;

import '../../tensor/tensor.dart';
import '../module.dart';

/// A vision backbone that produces `[numPatches + 1, embedDim]`
/// features from a `[numPatches, patchPixels]` patchified image.
abstract class VisionEncoder extends Module {
  int get embedDim;
  int get numPatches;

  Tensor call(Tensor patchifiedImage);
}
