/// Vision-conditioned Llama — from-scratch multi-modal wrapper.
///
/// Wires together three pieces:
///
/// 1. A [ViTBackbone] that produces per-patch features
///    (`[numPatches + 1, vitEmbedDim]`) from a patchified image.
/// 2. A [VisionProjector] that maps those features into Llama's
///    embedding space (`[numPatches + 1, llamaEmbedDim]`).
/// 3. A [Llama] decoder that consumes `[image_tokens; text_tokens]`
///    as a single causal sequence.
///
/// The image tokens sit *before* the text tokens in the sequence, so
/// the causal mask lets every text token attend to all image tokens,
/// but not vice versa. This matches the LLaVA / IDEFICS convention.
///
/// This is a "from-scratch demo" wiring:
///
/// * ViT weights are randomly initialised — this class does not
///   pretend to load CLIP or any pretrained vision encoder.
/// * Llama weights can be either randomly initialised (fast smoke
///   demo) or loaded via `loadLlamaSafetensorsInto`.
/// * The projector is always initialised fresh and is what a training
///   loop actually optimises (LLaVA-style stage-1 alignment).
///
/// See `bin/train_llama_vision_demo.dart` for the accompanying
/// tiny-caption training loop.
library;

import 'dart:math' as math;

import '../tensor/tensor.dart';
import 'llama.dart';
import 'module.dart';
import 'vision/vit_backbone.dart';
import 'vision_projector.dart';

class LlamaVision extends Module {
  final ViTBackbone vit;
  final VisionProjector projector;
  final Llama llama;

  /// Number of image tokens produced per image (patches + CLS). Cached
  /// for callers that need to know the offset into the sequence.
  int get numImageTokens => vit.numPatches + 1;

  LlamaVision({required this.vit, required this.projector, required this.llama})
    : assert(
        projector.outDim == llama.config.embedDim,
        'projector.outDim (${projector.outDim}) must equal '
        "llama.config.embedDim (${llama.config.embedDim})",
      ),
      assert(
        projector.inDim == vit.embedDim,
        'projector.inDim (${projector.inDim}) must equal '
        "vit.embedDim (${vit.embedDim})",
      );

  /// Convenience constructor that builds a vision projector matching
  /// the two backbones. All three pieces must already be on the same
  /// device.
  factory LlamaVision.build({
    required ViTBackbone vit,
    required Llama llama,
    int? projectorHiddenDim,
    int seed = 0,
  }) {
    final proj = VisionProjector(
      vit.embedDim,
      llama.config.embedDim,
      hiddenDim: projectorHiddenDim,
      device: llama.config.device,
      seed: seed,
    );
    return LlamaVision(vit: vit, projector: proj, llama: llama);
  }

  /// Encode `patchifiedImage` into image tokens ready to prepend to a
  /// text sequence: `[numImageTokens, llama.config.embedDim]`.
  Tensor encodeImage(Tensor patchifiedImage) {
    final vitOut = vit(patchifiedImage); // [P+1, vitEmbedDim]
    return projector(vitOut); // [P+1, llamaEmbedDim]
  }

  /// Full teacher-forcing forward: image + text tokens → logits over
  /// the full `[numImageTokens + T, vocab]` sequence.
  ///
  /// `textTokens` is a 1D `[T]` tensor of token ids as float32 (same
  /// convention as [Llama.call]).
  Tensor call(Tensor patchifiedImage, Tensor textTokens) {
    if (textTokens.shape.length != 1) {
      throw ArgumentError(
        'LlamaVision: textTokens must be 1D [T]; got ${textTokens.shape}',
      );
    }
    final imageTokens = encodeImage(patchifiedImage); // [P+1, D]
    final textEmb = llama.embedIn(textTokens); // [T, D]
    final full = TensorConcat.concat([imageTokens, textEmb], axis: 0);
    if (full.shape[0] > llama.config.maxCtx) {
      throw ArgumentError(
        'LlamaVision: total sequence length ${full.shape[0]} '
        'exceeds llama.maxCtx=${llama.config.maxCtx}',
      );
    }
    return llama.forwardFromEmbeddings(full);
  }

  /// Autoregressive greedy / top-K sampling conditioned on a single
  /// image and a text prompt.
  ///
  /// Returns the *text-side* token id list: prompt followed by
  /// generated tokens. The image tokens are consumed internally and
  /// never appear in the returned list.
  ///
  /// Same sampling semantics as [Llama.generate]. Currently
  /// re-encodes the full `[image + text]` sequence every step
  /// (cache-less); for a tiny demo model this is fine, and it keeps
  /// the wiring obvious. If speed becomes a problem, use
  /// [Llama.forwardFromEmbeddings] with an [EncoderCache] populated
  /// on the first call.
  List<double> generate(
    Tensor patchifiedImage,
    List<double> textPrompt, {
    required int maxNewTokens,
    double temperature = 1.0,
    int? topK,
    math.Random? rng,
    int? stopId,
  }) {
    if (textPrompt.isEmpty) {
      throw ArgumentError('LlamaVision.generate: prompt must be non-empty');
    }
    final r = rng ?? math.Random();
    final wasTraining = training;
    eval();
    try {
      return Tensor.noGrad(
        () => _generateNoCache(
          patchifiedImage,
          textPrompt,
          maxNewTokens,
          temperature,
          topK,
          r,
          stopId,
        ),
      );
    } finally {
      if (wasTraining) train();
    }
  }

  List<double> _generateNoCache(
    Tensor patchifiedImage,
    List<double> prompt,
    int maxNewTokens,
    double temperature,
    int? topK,
    math.Random rng,
    int? stopId,
  ) {
    final v = llama.config.vocabSize;
    final imageTokens = encodeImage(patchifiedImage); // [P+1, D]
    final numImg = imageTokens.shape[0];
    final out = List<double>.of(prompt);

    for (int step = 0; step < maxNewTokens; step++) {
      final maxText = llama.config.maxCtx - numImg;
      if (maxText <= 0) break;
      final start = out.length > maxText ? out.length - maxText : 0;
      final ctxList = out.sublist(start);
      final ctx = Tensor.fromList(
        [ctxList.length],
        ctxList,
        device: llama.config.device,
      );
      final textEmb = llama.embedIn(ctx);
      final full = TensorConcat.concat([imageTokens, textEmb], axis: 0);
      final logits = llama.forwardFromEmbeddings(full).toList();
      final lastBase = (full.shape[0] - 1) * v;
      final row = List<double>.generate(v, (i) => logits[lastBase + i]);
      final next = _sampleFromLogits(row, temperature, topK, rng);
      out.add(next.toDouble());
      if (stopId != null && next == stopId) break;
    }
    return out;
  }

  int _sampleFromLogits(
    List<double> logits,
    double temperature,
    int? topK,
    math.Random rng,
  ) {
    final v = logits.length;
    final greedy = temperature <= 0.0 || (topK != null && topK <= 1);
    if (greedy) {
      var bestI = 0;
      var bestV = logits[0];
      for (int i = 1; i < v; i++) {
        if (logits[i] > bestV) {
          bestV = logits[i];
          bestI = i;
        }
      }
      return bestI;
    }
    final scaled = List<double>.generate(v, (i) => logits[i] / temperature);
    if (topK != null && topK < v) {
      final sorted = List<int>.generate(v, (i) => i)
        ..sort((a, b) => scaled[b].compareTo(scaled[a]));
      final keep = sorted.take(topK).toSet();
      for (int i = 0; i < v; i++) {
        if (!keep.contains(i)) scaled[i] = double.negativeInfinity;
      }
    }
    var maxV = scaled[0];
    for (int i = 1; i < v; i++) {
      if (scaled[i] > maxV) maxV = scaled[i];
    }
    final exps = List<double>.generate(
      v,
      (i) => scaled[i] == double.negativeInfinity
          ? 0.0
          : math.exp(scaled[i] - maxV),
    );
    var sum = 0.0;
    for (final e in exps) {
      sum += e;
    }
    final u = rng.nextDouble() * sum;
    var acc = 0.0;
    for (int i = 0; i < v; i++) {
      acc += exps[i];
      if (u <= acc) return i;
    }
    return v - 1;
  }

  @override
  List<Tensor> parameters() => [
    ...vit.parameters(),
    ...projector.parameters(),
    ...llama.parameters(),
  ];

  @override
  List<Module> submodules() => [vit, projector, llama];
}
