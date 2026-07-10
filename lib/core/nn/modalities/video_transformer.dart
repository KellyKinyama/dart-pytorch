/// Video Transformer — sequence encoder over per-frame feature
/// embeddings (e.g. CNN-extracted features per video frame).
///
/// Input:  `[numFrames, frameFeatureDim]`
/// Output: `[numFrames, embedDim]` (via `call`).
///
/// Recipe mirrors [`AudioTransformer`] — an optional feature
/// projection when `frameFeatureDim != embedDim`, then positional
/// embeddings, then a [TransformerEncoder]. Use [`poolMean`] to
/// collapse the timeline to a single `[1, embedDim]` vector.
library;

import '../../tensor/tensor.dart';
import '../linear.dart';
import '../module.dart';
import '../positional.dart';
import '../transformer_encoder.dart';
import 'audio_transformer.dart' show meanRows;

class VideoTransformer extends Module {
  final int frameFeatureDim;
  final int maxFrames;
  final int embedDim;
  final int numLayers;
  final int numHeads;

  final Linear? frameProjection;
  final LearnedPositionalEmbedding posEmbed;
  final TransformerEncoder encoder;

  VideoTransformer({
    required this.frameFeatureDim,
    required this.maxFrames,
    required this.embedDim,
    this.numLayers = 2,
    this.numHeads = 4,
    double dropoutP = 0.0,
    Device device = Device.CPU,
    int seed = 0,
  }) : frameProjection = frameFeatureDim == embedDim
           ? null
           : Linear(
               frameFeatureDim,
               embedDim,
               bias: true,
               device: device,
               seed: seed + 1,
             ),
       posEmbed = LearnedPositionalEmbedding(
         maxFrames,
         embedDim,
         device: device,
         seed: seed + 2,
       ),
       encoder = TransformerEncoder(
         numLayers,
         embedDim,
         numHeads,
         dropoutP: dropoutP,
         device: device,
         seed: seed + 100,
       );

  /// Encode a variable-length frame sequence.
  ///
  /// `frames` — `[numFrames, frameFeatureDim]`, `numFrames ≤ maxFrames`.
  Tensor call(Tensor frames) {
    if (frames.shape.length != 2 || frames.shape[1] != frameFeatureDim) {
      throw ArgumentError(
        'VideoTransformer: expected [numFrames, $frameFeatureDim]; '
        'got ${frames.shape}',
      );
    }
    final projected = frameProjection == null
        ? frames
        : frameProjection!(frames);
    final withPos = posEmbed(projected);
    return encoder(withPos);
  }

  Tensor poolMean(Tensor frames) => meanRows(this(frames));

  @override
  List<Tensor> parameters() => [
    if (frameProjection != null) ...frameProjection!.parameters(),
    ...posEmbed.parameters(),
    ...encoder.parameters(),
  ];

  @override
  List<Module> submodules() => [
    if (frameProjection != null) frameProjection!,
    posEmbed,
    encoder,
  ];
}
