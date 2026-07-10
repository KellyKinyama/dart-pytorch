/// Multimodal classifier — fuses per-modality features via mean
/// pooling + concatenation + a small MLP head.
///
/// Late-fusion recipe (each modality is encoded independently and
/// then combined at the feature level):
///
///   1. `AudioTransformer(audio)` → mean-pool → `[1, embedA]`
///   2. `VideoTransformer(video)` → mean-pool → `[1, embedV]`
///   3. optional `TextTransformer(tokens)` → mean-pool → `[1, embedT]`
///   4. `concat` on axis 1 → `[1, embedA + embedV + (embedT?)]`
///   5. `Linear(..., numClasses)` → logits `[1, numClasses]`
///
/// The three encoders don't need to share `embedDim` — the fusion
/// head sizes itself from their totals.
library;

import '../../tensor/tensor.dart';
import '../linear.dart';
import '../module.dart';
import 'audio_transformer.dart';
import 'text_transformer.dart';
import 'video_transformer.dart';

class MultiModalClassifier extends Module {
  final AudioTransformer audio;
  final VideoTransformer video;
  final TextTransformer? text;
  final Linear head;
  final int numClasses;

  MultiModalClassifier({
    required this.audio,
    required this.video,
    this.text,
    required this.numClasses,
    Device device = Device.CPU,
    int seed = 0,
  }) : head = Linear(
         audio.embedDim + video.embedDim + (text?.embedDim ?? 0),
         numClasses,
         bias: true,
         device: device,
         seed: seed,
       );

  /// Forward.
  ///
  /// * `audioIn`  — `[seqA, audio.featureDim]`
  /// * `videoIn`  — `[seqV, video.frameFeatureDim]`
  /// * `textIn`   — 1D `[seqT]` token indices; required iff a text
  ///                encoder was configured.
  ///
  /// Returns logits `[1, numClasses]`.
  Tensor call(Tensor audioIn, Tensor videoIn, {Tensor? textIn}) {
    if ((text == null) != (textIn == null)) {
      throw ArgumentError(
        'MultiModalClassifier: text encoder present=${text != null} '
        'but textIn present=${textIn != null}',
      );
    }
    final a = audio.poolMean(audioIn); // [1, embedA]
    final v = video.poolMean(videoIn); // [1, embedV]
    final feats = <Tensor>[a, v];
    if (textIn != null) {
      feats.add(text!.poolMean(textIn)); // [1, embedT]
    }
    final fused = TensorConcat.concat(feats, axis: 1);
    return head(fused);
  }

  @override
  List<Tensor> parameters() => [
    ...audio.parameters(),
    ...video.parameters(),
    if (text != null) ...text!.parameters(),
    ...head.parameters(),
  ];

  @override
  List<Module> submodules() => [audio, video, if (text != null) text!, head];
}
