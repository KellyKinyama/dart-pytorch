/// Audio Transformer — sequence encoder for pre-extracted audio
/// features (e.g. MFCCs / mel-spectrogram frames).
///
/// Input:  `[seqLen, featureDim]`
/// Output: `[seqLen, embedDim]` (via `call`) — the standard encoder
///         output with per-frame contextualized features.
///
/// Recipe:
///   1. `Linear(featureDim, embedDim)` project each frame's features.
///   2. `LearnedPositionalEmbedding` add per-timestep positions.
///   3. `TransformerEncoder` self-attend across the timeline.
///
/// For a classification-style "single embedding per clip" use case,
/// call [poolMean] on the returned sequence — this dispatches to the
/// module-agnostic [meanRows] helper.
library;

import '../../tensor/tensor.dart';
import '../linear.dart';
import '../module.dart';
import '../positional.dart';
import '../transformer_encoder.dart';

class AudioTransformer extends Module {
  final int featureDim;
  final int maxSeqLen;
  final int embedDim;
  final int numLayers;
  final int numHeads;

  final Linear featureProjection;
  final LearnedPositionalEmbedding posEmbed;
  final TransformerEncoder encoder;

  AudioTransformer({
    required this.featureDim,
    required this.maxSeqLen,
    required this.embedDim,
    this.numLayers = 4,
    this.numHeads = 4,
    double dropoutP = 0.0,
    Device device = Device.CPU,
    int seed = 0,
  }) : featureProjection = Linear(
         featureDim,
         embedDim,
         bias: true,
         device: device,
         seed: seed + 1,
       ),
       posEmbed = LearnedPositionalEmbedding(
         maxSeqLen,
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

  /// Encode a variable-length audio-feature sequence.
  ///
  /// `features` — `[seqLen, featureDim]`, `seqLen ≤ maxSeqLen`.
  /// Returns `[seqLen, embedDim]`.
  Tensor call(Tensor features) {
    if (features.shape.length != 2 || features.shape[1] != featureDim) {
      throw ArgumentError(
        'AudioTransformer: expected [seqLen, $featureDim]; '
        'got ${features.shape}',
      );
    }
    final projected = featureProjection(features);
    final withPos = posEmbed(projected);
    return encoder(withPos);
  }

  /// Convenience: full-clip embedding as `[1, embedDim]`.
  Tensor poolMean(Tensor features) => meanRows(this(features));

  @override
  List<Tensor> parameters() => [
    ...featureProjection.parameters(),
    ...posEmbed.parameters(),
    ...encoder.parameters(),
  ];

  @override
  List<Module> submodules() => [featureProjection, posEmbed, encoder];
}

/// Mean over the row axis of a 2D `[N, D]` tensor, returning
/// `[1, D]`. Implemented as `ones([1, N] / N) @ x`, so it's a single
/// matmul that participates in autograd through both operands.
///
/// Handy for global-average pooling a variable-length sequence into
/// a single feature vector.
Tensor meanRows(Tensor x) {
  if (x.shape.length != 2) {
    throw ArgumentError('meanRows: expected 2D [N, D]; got ${x.shape}');
  }
  final n = x.shape[0];
  if (n == 0) {
    throw ArgumentError('meanRows: cannot pool an empty sequence');
  }
  final ones = Tensor.fill([1, n], 1.0 / n, device: x.device);
  return ones.matmul(x);
}
