/// Multimodal encoder — fuses per-modality *sequences* into a joint
/// sequence context, suitable for feeding into a text decoder for
/// tasks like captioning or spoken-Q&A.
///
/// Mid-fusion recipe (modalities interact through self-attention
/// after being tokenized to a common `jointEmbedDim`):
///
///   1. Each unimodal encoder runs independently and produces a
///      sequence of length `seq_i` at width `jointEmbedDim`.
///   2. Sequences are `concat`-ed along the *row* axis into a single
///      `[seqA + seqV + seqT, jointEmbedDim]` block.
///   3. A fusion [TransformerEncoder] self-attends across the merged
///      block so features from different modalities can talk to each
///      other.
///
/// All three unimodal encoders must produce features of the same
/// `jointEmbedDim` — the constructor asserts this. Text is optional.
///
/// The returned tensor is `[seqTotal, jointEmbedDim]`. Wire it into
/// a [TransformerDecoder] (via `EncoderDecoder`) if you want a
/// generative head; feed it into another pooling + MLP layer for
/// classification.
library;

import '../../tensor/tensor.dart';
import '../module.dart';
import '../transformer_encoder.dart';
import 'audio_transformer.dart';
import 'text_transformer.dart';
import 'video_transformer.dart';

class MultiModalEncoder extends Module {
  final AudioTransformer audio;
  final VideoTransformer video;
  final TextTransformer? text;
  final TransformerEncoder fusion;
  final int jointEmbedDim;

  MultiModalEncoder({
    required this.audio,
    required this.video,
    this.text,
    required this.jointEmbedDim,
    int fusionLayers = 2,
    int fusionHeads = 4,
    double dropoutP = 0.0,
    Device device = Device.CPU,
    int seed = 0,
  }) : assert(
         audio.embedDim == jointEmbedDim &&
             video.embedDim == jointEmbedDim &&
             (text == null || text.embedDim == jointEmbedDim),
         'MultiModalEncoder: all unimodal encoders must share '
         'jointEmbedDim=$jointEmbedDim',
       ),
       fusion = TransformerEncoder(
         fusionLayers,
         jointEmbedDim,
         fusionHeads,
         dropoutP: dropoutP,
         device: device,
         seed: seed,
       );

  /// Fuse per-modality sequences into a single sequence of length
  /// `seqA + seqV [+ seqT]` at width `jointEmbedDim`.
  ///
  /// * `audioIn`  — `[seqA, audio.featureDim]`
  /// * `videoIn`  — `[seqV, video.frameFeatureDim]`
  /// * `textIn`   — 1D `[seqT]` token indices (only when a text
  ///                encoder was configured).
  Tensor call(Tensor audioIn, Tensor videoIn, {Tensor? textIn}) {
    if ((text == null) != (textIn == null)) {
      throw ArgumentError(
        'MultiModalEncoder: text encoder present=${text != null} '
        'but textIn present=${textIn != null}',
      );
    }
    final aSeq = audio(audioIn); // [seqA, D]
    final vSeq = video(videoIn); // [seqV, D]
    final parts = <Tensor>[aSeq, vSeq];
    if (textIn != null) {
      parts.add(text!(textIn)); // [seqT, D]
    }
    final joint = TensorConcat.concat(parts, axis: 0);
    return fusion(joint);
  }

  @override
  List<Tensor> parameters() => [
    ...audio.parameters(),
    ...video.parameters(),
    if (text != null) ...text!.parameters(),
    ...fusion.parameters(),
  ];

  @override
  List<Module> submodules() => [audio, video, if (text != null) text!, fusion];
}
