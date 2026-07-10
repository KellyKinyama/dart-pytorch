/// Multimodal generative transformer — the "spits out text" analog of
/// Gemini / VLM architectures. Encodes any subset of
/// `{image, audio, video, text}` into a joint memory sequence, then
/// autoregressively decodes a target text stream via causal self-
/// attention plus cross-attention over the memory.
///
/// Architecture:
///
///   1. **Per-modality encoders** (all optional, all producing
///      `[seq_i, jointEmbedDim]`):
///      * [ViTBackbone]         for `[numPatches, patchPixels]`.
///      * [AudioTransformer]    for `[seqA, audioFeatureDim]`.
///      * [VideoTransformer]    for `[seqV, videoFrameFeatureDim]`.
///      * [TextTransformer]     for `[seqT]` token indices.
///
///   2. **Fusion** — the per-modality sequences are concatenated
///      along the row axis and, optionally, run through a small
///      [TransformerEncoder] so features from different modalities
///      can talk to each other. Output is the joint memory of shape
///      `[seqTotal, jointEmbedDim]`.
///
///   3. **Decoder** — a causal [TransformerDecoder] with cross-
///      attention over the joint memory, plus a `Linear` LM head
///      producing logits `[seqTgt, targetVocabSize]`.
///
/// Any modality left `null` at construction cannot be used at forward
/// time (and vice versa) — the module asserts symmetry. All provided
/// encoders must share `jointEmbedDim`.
library;

import '../../tensor/tensor.dart';
import '../embedding.dart';
import '../linear.dart';
import '../masks.dart';
import '../module.dart';
import '../positional.dart';
import '../transformer_decoder.dart';
import '../transformer_encoder.dart';
import '../vision/vit_backbone.dart';
import 'audio_transformer.dart';
import 'text_transformer.dart';
import 'video_transformer.dart';

class MultiModalGenerator extends Module {
  final ViTBackbone? image;
  final AudioTransformer? audio;
  final VideoTransformer? video;
  final TextTransformer? text;

  final TransformerEncoder? fusion;
  final int jointEmbedDim;

  final int targetVocabSize;
  final int maxTargetLen;

  final Embedding targetEmb;
  final SinusoidalPositionalEncoding tgtPosEnc;
  final TransformerDecoder decoder;
  final Linear head;

  MultiModalGenerator({
    this.image,
    this.audio,
    this.video,
    this.text,
    required this.targetVocabSize,
    required this.maxTargetLen,
    required this.jointEmbedDim,
    int decoderLayers = 2,
    int decoderHeads = 4,
    int? decoderFfnDim,
    int fusionLayers = 1,
    int fusionHeads = 4,
    bool useFusion = true,
    double dropoutP = 0.0,
    Device device = Device.CPU,
    int seed = 0,
  }) : assert(
         image != null || audio != null || video != null || text != null,
         'MultiModalGenerator: at least one modality encoder must be '
         'provided',
       ),
       assert(
         image == null || image.embedDim == jointEmbedDim,
         'MultiModalGenerator: image encoder embedDim must equal '
         'jointEmbedDim=$jointEmbedDim',
       ),
       assert(
         audio == null || audio.embedDim == jointEmbedDim,
         'MultiModalGenerator: audio encoder embedDim must equal '
         'jointEmbedDim=$jointEmbedDim',
       ),
       assert(
         video == null || video.embedDim == jointEmbedDim,
         'MultiModalGenerator: video encoder embedDim must equal '
         'jointEmbedDim=$jointEmbedDim',
       ),
       assert(
         text == null || text.embedDim == jointEmbedDim,
         'MultiModalGenerator: text encoder embedDim must equal '
         'jointEmbedDim=$jointEmbedDim',
       ),
       fusion = useFusion
           ? TransformerEncoder(
               fusionLayers,
               jointEmbedDim,
               fusionHeads,
               dropoutP: dropoutP,
               device: device,
               seed: seed + 200000,
             )
           : null,
       targetEmb = Embedding(
         targetVocabSize,
         jointEmbedDim,
         device: device,
         seed: seed + 300000,
       ),
       tgtPosEnc = SinusoidalPositionalEncoding(jointEmbedDim),
       decoder = TransformerDecoder(
         decoderLayers,
         jointEmbedDim,
         decoderHeads,
         kvEmbedDim: jointEmbedDim,
         ffnDim: decoderFfnDim,
         dropoutP: dropoutP,
         device: device,
         seed: seed + 400000,
       ),
       head = Linear(
         jointEmbedDim,
         targetVocabSize,
         bias: true,
         device: device,
         seed: seed + 500000,
       );

  /// Encode any subset of the four modalities into a joint memory
  /// sequence of shape `[seqTotal, jointEmbedDim]`.
  Tensor encode({
    Tensor? imagePatches,
    Tensor? audioFeatures,
    Tensor? videoFrames,
    Tensor? textIn,
  }) {
    _checkModality('image', image, imagePatches);
    _checkModality('audio', audio, audioFeatures);
    _checkModality('video', video, videoFrames);
    _checkModality('text', text, textIn);

    final parts = <Tensor>[];
    if (imagePatches != null) parts.add(image!(imagePatches));
    if (audioFeatures != null) parts.add(audio!(audioFeatures));
    if (videoFrames != null) parts.add(video!(videoFrames));
    if (textIn != null) parts.add(text!(textIn));

    if (parts.isEmpty) {
      throw ArgumentError(
        'MultiModalGenerator.encode: no modality inputs were provided',
      );
    }

    var joint = parts.length == 1
        ? parts[0]
        : TensorConcat.concat(parts, axis: 0);
    if (fusion != null) joint = fusion!(joint);
    return joint;
  }

  /// Decoder-side forward given a precomputed memory. Applies a
  /// causal mask to the decoder self-attention. Returns logits
  /// `[seqTgt, targetVocabSize]`.
  Tensor decode(Tensor targetTokens, Tensor memory) {
    if (targetTokens.shape.length != 1) {
      throw ArgumentError(
        'MultiModalGenerator.decode: targetTokens must be 1D [seqTgt]; '
        'got ${targetTokens.shape}',
      );
    }
    final n = targetTokens.shape.last;
    if (n > maxTargetLen) {
      throw ArgumentError(
        'MultiModalGenerator.decode: seqLen $n exceeds '
        'maxTargetLen $maxTargetLen',
      );
    }
    var y = targetEmb(targetTokens);
    y = tgtPosEnc(y);
    final mask = causalMask(n, device: y.device);
    y = decoder(y, memory, selfMask: mask);
    return head(y);
  }

  /// Full forward: encode any provided modalities, decode target
  /// text, return logits `[seqTgt, targetVocabSize]`.
  Tensor call({
    Tensor? imagePatches,
    Tensor? audioFeatures,
    Tensor? videoFrames,
    Tensor? textIn,
    required Tensor targetTokens,
  }) {
    final memory = encode(
      imagePatches: imagePatches,
      audioFeatures: audioFeatures,
      videoFrames: videoFrames,
      textIn: textIn,
    );
    return decode(targetTokens, memory);
  }

  /// Greedy autoregressive generation given a subset of modality
  /// inputs and an initial prompt (list of token ids, typically
  /// beginning with a BOS token). Encodes the modalities once, then
  /// re-runs the decoder for each new token. Stops early if `eosId`
  /// is produced.
  List<int> generate({
    Tensor? imagePatches,
    Tensor? audioFeatures,
    Tensor? videoFrames,
    Tensor? textIn,
    required List<int> prompt,
    required int maxNewTokens,
    int? eosId,
    Device device = Device.CPU,
  }) {
    if (prompt.isEmpty) {
      throw ArgumentError(
        'MultiModalGenerator.generate: prompt must have at least one '
        'token (typically a BOS)',
      );
    }
    final memory = encode(
      imagePatches: imagePatches,
      audioFeatures: audioFeatures,
      videoFrames: videoFrames,
      textIn: textIn,
    );
    final out = List<int>.of(prompt);
    for (int step = 0; step < maxNewTokens; step++) {
      if (out.length >= maxTargetLen) break;
      final ctx = Tensor.fromList(
        [out.length],
        out.map((i) => i.toDouble()).toList(),
        device: device,
      );
      final logits = decode(ctx, memory);
      final flat = logits.toList();
      final base = (out.length - 1) * targetVocabSize;
      int best = 0;
      double bestVal = flat[base];
      for (int i = 1; i < targetVocabSize; i++) {
        if (flat[base + i] > bestVal) {
          bestVal = flat[base + i];
          best = i;
        }
      }
      out.add(best);
      if (eosId != null && best == eosId) break;
    }
    return out;
  }

  void _checkModality(String name, Module? enc, Object? input) {
    if ((enc == null) != (input == null)) {
      throw ArgumentError(
        'MultiModalGenerator: $name encoder present=${enc != null} '
        'but $name input present=${input != null}',
      );
    }
  }

  @override
  List<Tensor> parameters() => [
    if (image != null) ...image!.parameters(),
    if (audio != null) ...audio!.parameters(),
    if (video != null) ...video!.parameters(),
    if (text != null) ...text!.parameters(),
    if (fusion != null) ...fusion!.parameters(),
    ...targetEmb.parameters(),
    ...tgtPosEnc.parameters(),
    ...decoder.parameters(),
    ...head.parameters(),
  ];

  @override
  List<Module> submodules() => [
    if (image != null) image!,
    if (audio != null) audio!,
    if (video != null) video!,
    if (text != null) text!,
    if (fusion != null) fusion!,
    targetEmb,
    tgtPosEnc,
    decoder,
    head,
  ];
}
