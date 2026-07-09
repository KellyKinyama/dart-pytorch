/// Face-embedding head on top of a Vision Transformer backbone.
///
/// Wraps a [ViTBackbone] plus an optional `Linear(embedDim, outputDim)`
/// projection to a fixed embedding size (default 512), and finishes
/// with row-wise L2 normalization so pairwise cosine similarity is
/// just a dot product. Train with a metric-learning loss (triplet,
/// contrastive, ArcFace, etc.) on top of the returned `[1, outputDim]`
/// vector.
///
/// The L2 norm is composed from primitives:
/// `y = x / sqrt(sum(x*x) + eps)`. This is fine for single-row output
/// (batch = 1 face at a time). For batched face embeddings, add a
/// per-row reduction op and use it here instead.
library;

import '../../tensor/tensor.dart';
import '../linear.dart';
import '../module.dart';
import 'vit_backbone.dart';

class ViTFaceEmbedding extends Module {
  final ViTBackbone backbone;
  final Linear? projection;
  final int outputDim;
  final double eps;

  ViTFaceEmbedding({
    required int imageSize,
    required int patchSize,
    int numChannels = 3,
    required int embedDim,
    this.outputDim = 512,
    int numLayers = 4,
    int numHeads = 4,
    double dropoutP = 0.0,
    this.eps = 1e-10,
    Device device = Device.CPU,
    int seed = 0,
  })  : backbone = ViTBackbone(
          imageSize: imageSize,
          patchSize: patchSize,
          numChannels: numChannels,
          embedDim: embedDim,
          numLayers: numLayers,
          numHeads: numHeads,
          dropoutP: dropoutP,
          device: device,
          seed: seed,
        ),
        projection = embedDim == outputDim
            ? null
            : Linear(
                embedDim,
                outputDim,
                bias: false,
                device: device,
                seed: seed + 888888,
              );

  /// `patchifiedImage` — `[numPatches, patchSize * patchSize *
  /// numChannels]`. Returns an L2-normalized `[1, outputDim]` vector.
  Tensor call(Tensor patchifiedImage) {
    final encoded = backbone(patchifiedImage);
    final cls = vitClsFeature(encoded); // [1, embedDim]
    final feat = projection == null ? cls : projection!(cls);

    // Row-wise L2 normalize. `feat` is [1, outputDim] so a global sum
    // is equivalent to a per-row sum.
    final sq = feat * feat;
    final sumSq = sq.sum(); // scalar [1]
    final norm = (sumSq + eps).pow(0.5); // scalar sqrt
    // Broadcast the scalar across the [1, outputDim] vector via /.
    return feat / norm;
  }

  /// Cosine similarity between two `[1, outputDim]` L2-normalized
  /// embeddings, reduced to a single scalar.
  static Tensor cosineSimilarity(Tensor a, Tensor b) {
    if (a.shape.length != 2 || b.shape.length != 2 ||
        a.shape[0] != 1 || b.shape[0] != 1 ||
        a.shape[1] != b.shape[1]) {
      throw ArgumentError(
        'cosineSimilarity: expected two [1, D] tensors; '
        'got ${a.shape} and ${b.shape}',
      );
    }
    return (a * b).sum();
  }

  @override
  List<Tensor> parameters() => [
        ...backbone.parameters(),
        if (projection != null) ...projection!.parameters(),
      ];

  @override
  List<Module> submodules() => [
        backbone,
        if (projection != null) projection!,
      ];
}
