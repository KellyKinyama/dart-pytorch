/// Decoder-only multimodal language model — the "packed causal
/// stream" architecture used by Gemini / GPT-4V / Fuyu.
///
/// Unlike [MultiModalGenerator] (encoder-decoder with cross-attention
/// between fused modality memory and a separate text decoder), this
/// model packs every modality into a **single causal sequence**:
///
///   `[image_toks ‖ audio_toks ‖ video_toks ‖ prompt_toks ‖ target_toks]`
///
/// and runs it through one causal self-attention stack (a
/// [TransformerEncoder] with a causal mask — i.e. GPT). Only the
/// target-position outputs feed the LM head, so the loss only sees
/// the text portion, but the model *attends* to every modality
/// through the same self-attention weights.
///
/// Modality tokenization is deliberately thin (a `Linear` projection
/// per modality plus a learnable `[1, embedDim]` type embedding),
/// with a shared [Embedding] for prompt + target tokens and a global
/// [SinusoidalPositionalEncoding] applied to the concatenated stream.
/// This mirrors Fuyu's design: no per-modality transformer, all
/// cross-modal fusion happens inside the unified causal stack.
///
/// Autoregressive generation is implemented by [`generate`], which
/// tokenizes the modality context once per new token (no KV cache
/// yet) and appends the argmax of the last row until `maxNewTokens`
/// or `eosId` is produced.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../../tensor/tensor.dart';
import '../embedding.dart';
import '../linear.dart';
import '../masks.dart';
import '../module.dart';
import '../positional.dart';
import '../transformer_encoder.dart';

class MultiModalLM extends Module {
  // ---------------------- Modality tokenizers ----------------------
  final int? imagePatchPixels;
  final int? audioFeatureDim;
  final int? videoFrameFeatureDim;
  final bool useTextPrompt;

  final Linear? imageProj;
  final Linear? audioProj;
  final Linear? videoProj;

  final int vocabSize;
  final Embedding tokenEmbed;

  // Learnable per-modality type embeddings ([1, D] each), added to
  // every token of that modality so the causal stack can distinguish
  // them.
  final Tensor? imageTypeEmb;
  final Tensor? audioTypeEmb;
  final Tensor? videoTypeEmb;
  final Tensor? textPromptTypeEmb;
  final Tensor targetTypeEmb; // target is always present

  // ---------------------- Causal decoder ---------------------------
  final int embedDim;
  final int maxTotalSeqLen;
  final SinusoidalPositionalEncoding posEnc;
  final TransformerEncoder decoder; // causal via mask
  final Linear head;

  MultiModalLM({
    int? imagePatchPixels,
    int? audioFeatureDim,
    int? videoFrameFeatureDim,
    bool useTextPrompt = true,
    required this.vocabSize,
    required this.embedDim,
    required this.maxTotalSeqLen,
    int numLayers = 4,
    int numHeads = 4,
    int? ffnDim,
    double dropoutP = 0.0,
    Device device = Device.CPU,
    int seed = 0,
  }) : imagePatchPixels = imagePatchPixels,
       audioFeatureDim = audioFeatureDim,
       videoFrameFeatureDim = videoFrameFeatureDim,
       useTextPrompt = useTextPrompt,
       imageProj = imagePatchPixels == null
           ? null
           : Linear(
               imagePatchPixels,
               embedDim,
               bias: true,
               device: device,
               seed: seed + 1,
             ),
       audioProj = audioFeatureDim == null
           ? null
           : Linear(
               audioFeatureDim,
               embedDim,
               bias: true,
               device: device,
               seed: seed + 2,
             ),
       videoProj = videoFrameFeatureDim == null
           ? null
           : Linear(
               videoFrameFeatureDim,
               embedDim,
               bias: true,
               device: device,
               seed: seed + 3,
             ),
       tokenEmbed = Embedding(
         vocabSize,
         embedDim,
         device: device,
         seed: seed + 4,
       ),
       imageTypeEmb = imagePatchPixels == null
           ? null
           : _initTypeEmbedding(embedDim, device: device, seed: seed + 11),
       audioTypeEmb = audioFeatureDim == null
           ? null
           : _initTypeEmbedding(embedDim, device: device, seed: seed + 12),
       videoTypeEmb = videoFrameFeatureDim == null
           ? null
           : _initTypeEmbedding(embedDim, device: device, seed: seed + 13),
       textPromptTypeEmb = useTextPrompt
           ? _initTypeEmbedding(embedDim, device: device, seed: seed + 14)
           : null,
       targetTypeEmb = _initTypeEmbedding(
         embedDim,
         device: device,
         seed: seed + 15,
       ),
       posEnc = SinusoidalPositionalEncoding(embedDim),
       decoder = TransformerEncoder(
         numLayers,
         embedDim,
         numHeads,
         ffnDim: ffnDim,
         dropoutP: dropoutP,
         device: device,
         seed: seed + 100000,
       ),
       head = Linear(
         embedDim,
         vocabSize,
         bias: true,
         device: device,
         seed: seed + 200000,
       );

  /// Small-Gaussian `[1, embedDim]` trainable initializer, matching
  /// the CLS-token / pos-embed convention in [ViTBackbone].
  static Tensor _initTypeEmbedding(
    int embedDim, {
    required Device device,
    required int seed,
    double scale = 0.02,
  }) {
    final rng = math.Random(seed);
    final vals = Float32List(embedDim);
    for (int i = 0; i < embedDim; i++) {
      final u1 = rng.nextDouble().clamp(1e-9, 1.0);
      final u2 = rng.nextDouble();
      final z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2);
      vals[i] = z * scale;
    }
    return Tensor.fromList(
      [1, embedDim],
      vals,
      requiresGrad: true,
      device: device,
    );
  }

  // ---------------------- Tokenization helpers ---------------------

  Tensor _tokenizeContinuous(Tensor input, Linear proj, Tensor typeEmb) {
    return proj(input) + typeEmb; // [N, D] + [1, D] broadcast
  }

  Tensor _tokenizeText(Tensor tokens, Tensor typeEmb) {
    return tokenEmbed(tokens) + typeEmb;
  }

  // ---------------------- Forward paths ----------------------------

  /// Forward that returns logits over the **entire** concatenated
  /// stream plus the row range that corresponds to the target
  /// tokens. Useful when the caller wants both target logits (for
  /// loss) and other diagnostics.
  ({Tensor logits, int targetStart, int totalLen}) forwardAll({
    Tensor? imagePatches,
    Tensor? audioFeatures,
    Tensor? videoFrames,
    Tensor? textPrompt,
    required Tensor targetInputTokens,
  }) {
    _checkModality('image', imagePatchPixels, imagePatches);
    _checkModality('audio', audioFeatureDim, audioFeatures);
    _checkModality('video', videoFrameFeatureDim, videoFrames);
    if (useTextPrompt && textPrompt == null) {
      throw ArgumentError(
        'MultiModalLM: useTextPrompt=true but textPrompt input is null',
      );
    }
    if (!useTextPrompt && textPrompt != null) {
      throw ArgumentError(
        'MultiModalLM: useTextPrompt=false but textPrompt input was given',
      );
    }

    final parts = <Tensor>[];
    if (imagePatches != null) {
      parts.add(_tokenizeContinuous(imagePatches, imageProj!, imageTypeEmb!));
    }
    if (audioFeatures != null) {
      parts.add(_tokenizeContinuous(audioFeatures, audioProj!, audioTypeEmb!));
    }
    if (videoFrames != null) {
      parts.add(_tokenizeContinuous(videoFrames, videoProj!, videoTypeEmb!));
    }
    if (textPrompt != null) {
      parts.add(_tokenizeText(textPrompt, textPromptTypeEmb!));
    }
    final tgt = _tokenizeText(targetInputTokens, targetTypeEmb);
    parts.add(tgt);

    var x = parts.length == 1 ? parts[0] : TensorConcat.concat(parts, axis: 0);
    final totalLen = x.shape[0];
    if (totalLen > maxTotalSeqLen) {
      throw ArgumentError(
        'MultiModalLM.forwardAll: total sequence length $totalLen exceeds '
        'maxTotalSeqLen $maxTotalSeqLen',
      );
    }
    x = posEnc(x);
    final mask = causalMask(totalLen, device: x.device);
    x = decoder(x, mask: mask);
    final logits = head(x); // [totalLen, vocab]
    final targetLen = targetInputTokens.shape[0];
    return (
      logits: logits,
      targetStart: totalLen - targetLen,
      totalLen: totalLen,
    );
  }

  /// Convenience: forward and return only the target-position logits
  /// `[targetLen, vocabSize]`. This is what you feed into
  /// `crossEntropy(targetLabels)`.
  Tensor call({
    Tensor? imagePatches,
    Tensor? audioFeatures,
    Tensor? videoFrames,
    Tensor? textPrompt,
    required Tensor targetInputTokens,
  }) {
    final f = forwardAll(
      imagePatches: imagePatches,
      audioFeatures: audioFeatures,
      videoFrames: videoFrames,
      textPrompt: textPrompt,
      targetInputTokens: targetInputTokens,
    );
    return _sliceTargetLogits(f.logits, f.targetStart, f.totalLen);
  }

  /// Select rows `[targetStart, totalLen)` of `logits` using the
  /// differentiable `embedding` op (which treats its receiver as a
  /// `[rows, cols]` lookup table). Gradient flows back only to the
  /// selected rows — exactly what we want since the loss is only on
  /// the target portion.
  Tensor _sliceTargetLogits(Tensor logits, int targetStart, int totalLen) {
    final targetLen = totalLen - targetStart;
    if (targetStart == 0) return logits;
    final idxData = Float32List(targetLen);
    for (int i = 0; i < targetLen; i++) {
      idxData[i] = (targetStart + i).toDouble();
    }
    final indices = Tensor.fromList(
      [targetLen],
      idxData,
      device: logits.device,
    );
    return logits.embedding(indices);
  }

  // ---------------------- Generation -------------------------------

  /// Greedy autoregressive generation. Tokenizes the modality
  /// context once per new token (no KV cache yet) and appends the
  /// argmax of the last-position logits until `maxNewTokens` or
  /// `eosId` is produced.
  List<int> generate({
    Tensor? imagePatches,
    Tensor? audioFeatures,
    Tensor? videoFrames,
    Tensor? textPrompt,
    required List<int> prompt,
    required int maxNewTokens,
    int? eosId,
    Device device = Device.CPU,
  }) {
    if (prompt.isEmpty) {
      throw ArgumentError(
        'MultiModalLM.generate: prompt must have at least one token '
        '(typically a BOS)',
      );
    }
    final out = List<int>.of(prompt);
    for (int step = 0; step < maxNewTokens; step++) {
      final ctx = Tensor.fromList(
        [out.length],
        out.map((i) => i.toDouble()).toList(),
        device: device,
      );
      final tgtLogits = this(
        imagePatches: imagePatches,
        audioFeatures: audioFeatures,
        videoFrames: videoFrames,
        textPrompt: textPrompt,
        targetInputTokens: ctx,
      );
      final flat = tgtLogits.toList();
      final base = (out.length - 1) * vocabSize;
      int best = 0;
      double bestVal = flat[base];
      for (int i = 1; i < vocabSize; i++) {
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

  void _checkModality(String name, int? dim, Object? input) {
    if ((dim == null) != (input == null)) {
      throw ArgumentError(
        'MultiModalLM: $name enabled=${dim != null} but $name input '
        'present=${input != null}',
      );
    }
  }

  @override
  List<Tensor> parameters() => [
    if (imageProj != null) ...imageProj!.parameters(),
    if (audioProj != null) ...audioProj!.parameters(),
    if (videoProj != null) ...videoProj!.parameters(),
    ...tokenEmbed.parameters(),
    if (imageTypeEmb != null) imageTypeEmb!,
    if (audioTypeEmb != null) audioTypeEmb!,
    if (videoTypeEmb != null) videoTypeEmb!,
    if (textPromptTypeEmb != null) textPromptTypeEmb!,
    targetTypeEmb,
    ...posEnc.parameters(),
    ...decoder.parameters(),
    ...head.parameters(),
  ];

  @override
  List<Module> submodules() => [
    if (imageProj != null) imageProj!,
    if (audioProj != null) audioProj!,
    if (videoProj != null) videoProj!,
    tokenEmbed,
    posEnc,
    decoder,
    head,
  ];
}
