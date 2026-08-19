/// OpenAI **CLIP** text transformer — the text half of `CLIPModel`.
/// Compatible with HuggingFace `openai/clip-vit-{base,large}-patch*`
/// checkpoints (12-layer causal transformer over 49 408-BPE text).
///
/// Architecture (matches HF `CLIPTextModel`):
///
/// ```
///   token_embedding       [V, D]  learnable
///   position_embedding    [maxCtx, D]  learnable
///   encoder               N × TransformerBlock (pre-LN, QuickGELU,
///                         attnBias=true, **causal mask**)
///   final_layer_norm      LayerNorm(D)
/// ```
///
/// Forward: 1-D `[seqLen]` token indices → `[seqLen, D]` hidden states.
///
/// The **pooled text embedding** used for zero-shot classification /
/// image retrieval is the hidden state at the position of the last
/// non-pad token (typically the `<|endoftext|>` — HF calls this
/// `eos_token_id = 49407`). Use [pooledEmbedding] to fetch it.
///
/// Weights are randomly initialised at construction; load a real
/// checkpoint via `ClipHFLoader.loadTextFile(...)`.
library;

import 'dart:math' as math;

import '../../tensor/tensor.dart';
import '../embedding.dart';
import '../layer_norm.dart';
import '../masks.dart';
import '../module.dart';
import '../transformer.dart';
import '../transformer_encoder.dart';

/// Config for [CLIPTextModel]. Values map onto HF `text_config`
/// entries in the CLIP JSON.
class CLIPTextConfig {
  final int vocabSize;
  final int maxCtx;
  final int embedDim;
  final int numLayers;
  final int numHeads;
  final int ffnDim;
  final double layerNormEps;
  final Device device;
  final int seed;

  const CLIPTextConfig({
    this.vocabSize = 49408,
    this.maxCtx = 77,
    this.embedDim = 512,
    this.numLayers = 12,
    this.numHeads = 8,
    this.ffnDim = 2048,
    this.layerNormEps = 1e-5,
    this.device = Device.CPU,
    this.seed = 0,
  });
}

class CLIPTextModel extends Module {
  final CLIPTextConfig config;

  final Embedding tokenEmbedding;
  final Tensor positionEmbedding; // [maxCtx, D]
  final TransformerEncoder encoder;
  final LayerNorm finalLayerNorm;

  CLIPTextModel(this.config)
      : tokenEmbedding = Embedding(
          config.vocabSize,
          config.embedDim,
          device: config.device,
          seed: config.seed + 1,
        ),
        positionEmbedding = _initSmallGaussian(
          [config.maxCtx, config.embedDim],
          scale: 0.02,
          device: config.device,
          seed: config.seed + 2,
        ),
        encoder = TransformerEncoder(
          config.numLayers,
          config.embedDim,
          config.numHeads,
          ffnDim: config.ffnDim,
          finalNorm: false,
          attnBias: true,
          activation: Activation.quickGelu,
          device: config.device,
          seed: config.seed + 100,
        ),
        finalLayerNorm = LayerNorm(
          config.embedDim,
          eps: config.layerNormEps,
          device: config.device,
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

  /// Forward pass. `tokens` is a 1D `[seqLen]` tensor of BPE ids
  /// encoded as floats. `seqLen ≤ maxCtx`.
  Tensor call(Tensor tokens) {
    if (tokens.shape.length != 1) {
      throw ArgumentError(
        'CLIPTextModel: expected 1D [seqLen] tokens; got ${tokens.shape}',
      );
    }
    final n = tokens.shape[0];
    if (n == 0) {
      throw ArgumentError('CLIPTextModel: empty sequence');
    }
    if (n > config.maxCtx) {
      throw ArgumentError(
        'CLIPTextModel: seqLen $n exceeds maxCtx ${config.maxCtx}',
      );
    }
    final tokEmb = tokenEmbedding(tokens); // [N, D]
    // Slice the first N rows of the position table via host — the
    // shared `Tensor.sliceRows` is CPU-only and we need this to work
    // on GPU too. Grad flow doesn't matter here at inference; at
    // training the position table is small enough that the extra
    // host copy is negligible.
    final posSlice = _slicePosOnDevice(positionEmbedding, n, tokEmb.device);
    var x = tokEmb + posSlice;
    final mask = causalMask(n, device: x.device);
    x = encoder(x, mask: mask);
    return finalLayerNorm(x);
  }

  static Tensor _slicePosOnDevice(Tensor pos, int n, Device device) {
    final all = Tensor.noGrad(() => pos.toList());
    final d = pos.shape[1];
    final out = List<double>.filled(n * d, 0);
    for (int i = 0; i < n * d; i++) {
      out[i] = all[i];
    }
    return Tensor.fromList([n, d], out, device: device);
  }

  /// Pooled sentence embedding: the hidden state at position
  /// [eotPosition] (typically the argmax of the token id list, which
  /// for the CLIP BPE tokenizer is the `<|endoftext|>` position).
  /// Returns a `[embedDim]` vector.
  Tensor pooledEmbedding(Tensor tokens, {int? eotPosition}) {
    final h = call(tokens); // [N, D]
    final n = h.shape[0];
    int pos;
    if (eotPosition != null) {
      pos = eotPosition;
    } else {
      final ids = tokens.toList();
      int argmax = 0;
      double best = ids[0];
      for (int i = 1; i < ids.length; i++) {
        if (ids[i] > best) {
          best = ids[i];
          argmax = i;
        }
      }
      pos = argmax;
    }
    if (pos < 0 || pos >= n) {
      throw ArgumentError(
        'CLIPTextModel.pooledEmbedding: pos $pos out of range [0, $n)',
      );
    }
    // Host-side row-slice so the result works on GPU too.
    final flat = Tensor.noGrad(() => h.toList());
    final d = config.embedDim;
    final row = List<double>.filled(d, 0);
    for (int j = 0; j < d; j++) {
      row[j] = flat[pos * d + j];
    }
    return Tensor.fromList([d], row, device: h.device);
  }

  @override
  List<Tensor> parameters() => [
        ...tokenEmbedding.parameters(),
        positionEmbedding,
        ...encoder.parameters(),
        ...finalLayerNorm.parameters(),
      ];

  @override
  List<Module> submodules() => [tokenEmbedding, encoder, finalLayerNorm];
}
