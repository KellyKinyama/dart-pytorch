/// SAM mask decoder — turns (image embedding + prompt embeddings) into
/// segmentation masks + IoU predictions.
///
/// Ports `segment_anything.modeling.mask_decoder.MaskDecoder` and its
/// `TwoWayTransformer` (two-way cross-attention between prompt tokens
/// and image features). Given the outputs of [SamImageEncoder] and
/// [SamPromptEncoder], produces:
///
///   * `masks` — `[num_masks, H_out, W_out]` where `H_out = W_out =
///     imageEmbedH · 4` (256×256 for SAM ViT-B). These are raw mask
///     **logits**; downstream code thresholds them at 0.
///   * `iou_predictions` — `[num_masks]` scalar quality estimates for
///     each mask.
///
/// SAM produces 4 mask tokens per forward: index 0 is the "no-mask"
/// / single-output token, indices 1..3 are the multi-mask candidates.
/// Call [MaskDecoderOutput.select] on the result to grab either the
/// single mask (`multimaskOutput: false`) or the three candidates
/// (`multimaskOutput: true`).
///
/// Modules built here (all specific to SAM):
///
///   * [SamAttention] — MHA with an optional `downsample_rate` that
///     reduces the internal Q/K/V dim before projecting back to
///     `embedDim`. SAM's cross-attention layers use `downsample_rate
///     = 2` (embedDim 256 → internal 128) to save compute.
///   * [SamMlpBlock] — two-layer MLP with GELU between, matching
///     SAM's `MLPBlock`.
///   * [SamMlp] — thin MLP wrapper: `Linear → ReLU → … → Linear`
///     with configurable depth (used for the mask-hypernetwork and
///     IoU heads).
///   * [SamTwoWayAttentionBlock] — one block of the two-way
///     transformer: self-attn on queries + q→k cross-attn + MLP +
///     k→q cross-attn.
///   * [SamTwoWayTransformer] — 2 × [SamTwoWayAttentionBlock] + a
///     final token-to-image attention + LayerNorm.
///   * [SamMaskDecoder] — the top-level module: prepends the IoU +
///     mask tokens to the sparse prompt embeddings, runs the two-way
///     transformer, upsamples the image features with two
///     [ConvTranspose2d]s, applies per-mask hypernet MLPs, and dot-
///     products with the upsampled features to produce mask logits.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../../tensor/tensor.dart';
import '../conv_transpose_2d.dart';
import '../layer_norm.dart';
import '../linear.dart';
import '../module.dart';
import 'sam_image_encoder.dart' show LayerNorm2d;

// ---------------------------------------------------------------------------
// SamAttention — MHA with SAM's downsample_rate knob.
// ---------------------------------------------------------------------------

class SamAttention extends Module {
  final int embedDim;
  final int numHeads;
  final int internalDim;
  final int headDim;
  final Linear qProj;
  final Linear kProj;
  final Linear vProj;
  final Linear outProj;

  SamAttention({
    required this.embedDim,
    required this.numHeads,
    int downsampleRate = 1,
    Device device = Device.CPU,
    int seed = 0,
  }) : internalDim = embedDim ~/ downsampleRate,
       headDim = (embedDim ~/ downsampleRate) ~/ numHeads,
       qProj = Linear(
         embedDim,
         embedDim ~/ downsampleRate,
         bias: true,
         device: device,
         seed: seed,
       ),
       kProj = Linear(
         embedDim,
         embedDim ~/ downsampleRate,
         bias: true,
         device: device,
         seed: seed + 1000,
       ),
       vProj = Linear(
         embedDim,
         embedDim ~/ downsampleRate,
         bias: true,
         device: device,
         seed: seed + 2000,
       ),
       outProj = Linear(
         embedDim ~/ downsampleRate,
         embedDim,
         bias: true,
         device: device,
         seed: seed + 3000,
       );

  /// `q, k, v: [N, embedDim]` — single-batch convention. Returns
  /// `[N_q, embedDim]`.
  Tensor call(Tensor q, Tensor k, Tensor v) {
    if (q.shape.length != 2 || k.shape.length != 2 || v.shape.length != 2) {
      throw ArgumentError(
        'SamAttention: expected 2D q/k/v; got q=${q.shape} k=${k.shape} '
        'v=${v.shape}',
      );
    }
    if (q.shape[1] != embedDim ||
        k.shape[1] != embedDim ||
        v.shape[1] != embedDim) {
      throw ArgumentError(
        'SamAttention: last dim must be $embedDim; got q=${q.shape} '
        'k=${k.shape} v=${v.shape}',
      );
    }
    // Project to internal dim.
    final qi = qProj(q); // [Nq, internal]
    final ki = kProj(k); // [Nk, internal]
    final vi = vProj(v); // [Nk, internal]

    // Per-head split on host — internal is contiguous per head.
    final nq = qi.shape[0];
    final nk = ki.shape[0];
    final qData = qi.toFloat32List();
    final kData = ki.toFloat32List();
    final vData = vi.toFloat32List();
    final out = Float32List(nq * internalDim);
    final scale = 1.0 / math.sqrt(headDim);

    for (int h = 0; h < numHeads; h++) {
      final headStart = h * headDim;
      // Compute scores [Nq, Nk] for this head.
      final scores = Float32List(nq * nk);
      for (int i = 0; i < nq; i++) {
        for (int j = 0; j < nk; j++) {
          double dot = 0;
          for (int d = 0; d < headDim; d++) {
            dot +=
                qData[i * internalDim + headStart + d] *
                kData[j * internalDim + headStart + d];
          }
          scores[i * nk + j] = dot * scale;
        }
      }
      // Softmax over the k axis.
      for (int i = 0; i < nq; i++) {
        var maxVal = scores[i * nk];
        for (int j = 1; j < nk; j++) {
          if (scores[i * nk + j] > maxVal) maxVal = scores[i * nk + j];
        }
        double sum = 0;
        for (int j = 0; j < nk; j++) {
          final e = math.exp(scores[i * nk + j] - maxVal);
          scores[i * nk + j] = e;
          sum += e;
        }
        for (int j = 0; j < nk; j++) {
          scores[i * nk + j] /= sum;
        }
      }
      // Weighted sum -> outHead [Nq, headDim].
      for (int i = 0; i < nq; i++) {
        for (int d = 0; d < headDim; d++) {
          double acc = 0;
          for (int j = 0; j < nk; j++) {
            acc += scores[i * nk + j] * vData[j * internalDim + headStart + d];
          }
          out[i * internalDim + headStart + d] = acc;
        }
      }
    }
    final concat = Tensor.fromFloat32List(
      [nq, internalDim],
      out,
      device: q.device,
    );
    return outProj(concat);
  }

  @override
  List<Tensor> parameters() => [
    ...qProj.parameters(),
    ...kProj.parameters(),
    ...vProj.parameters(),
    ...outProj.parameters(),
  ];

  @override
  List<Module> submodules() => [qProj, kProj, vProj, outProj];
}

// ---------------------------------------------------------------------------
// SamMlpBlock — two-layer MLP with GELU.
// ---------------------------------------------------------------------------

class SamMlpBlock extends Module {
  final Linear fc1;
  final Linear fc2;

  SamMlpBlock({
    required int embedDim,
    required int mlpDim,
    Device device = Device.CPU,
    int seed = 0,
  }) : fc1 = Linear(embedDim, mlpDim, bias: true, device: device, seed: seed),
       fc2 = Linear(
         mlpDim,
         embedDim,
         bias: true,
         device: device,
         seed: seed + 1,
       );

  Tensor call(Tensor x) => fc2(_gelu(fc1(x)));

  static Tensor _gelu(Tensor x) {
    const invSqrt2 = 0.7071067811865475;
    final data = x.toFloat32List();
    final out = Float32List(data.length);
    for (int i = 0; i < data.length; i++) {
      out[i] = 0.5 * data[i] * (1.0 + _erf(data[i] * invSqrt2));
    }
    return Tensor.fromFloat32List(x.shape, out, device: x.device);
  }

  static double _erf(double x) {
    final sign = x < 0 ? -1.0 : 1.0;
    x = x.abs();
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;
    final t = 1.0 / (1.0 + p * x);
    final y =
        1.0 -
        (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x);
    return sign * y;
  }

  @override
  List<Tensor> parameters() => [...fc1.parameters(), ...fc2.parameters()];

  @override
  List<Module> submodules() => [fc1, fc2];
}

// ---------------------------------------------------------------------------
// SamMlp — variable-depth Linear + ReLU stack used by the heads.
// ---------------------------------------------------------------------------

/// SAM's `MLP` helper: `numLayers` stacked Linear layers with ReLU
/// between (final layer has no activation). Optional `sigmoidOutput`.
class SamMlp extends Module {
  final List<Linear> layers;
  final bool sigmoidOutput;

  SamMlp({
    required int inputDim,
    required int hiddenDim,
    required int outputDim,
    required int numLayers,
    this.sigmoidOutput = false,
    Device device = Device.CPU,
    int seed = 0,
  }) : layers = <Linear>[] {
    for (int i = 0; i < numLayers; i++) {
      final inD = i == 0 ? inputDim : hiddenDim;
      final outD = i == numLayers - 1 ? outputDim : hiddenDim;
      layers.add(
        Linear(inD, outD, bias: true, device: device, seed: seed + i * 1000),
      );
    }
  }

  Tensor call(Tensor x) {
    var h = x;
    for (int i = 0; i < layers.length; i++) {
      h = layers[i](h);
      if (i < layers.length - 1) h = h.relu();
    }
    if (sigmoidOutput) h = h.sigmoid();
    return h;
  }

  @override
  List<Tensor> parameters() => [for (final l in layers) ...l.parameters()];

  @override
  List<Module> submodules() => [...layers];
}

// ---------------------------------------------------------------------------
// SamTwoWayAttentionBlock — one block of the two-way transformer.
// ---------------------------------------------------------------------------

/// The two-way attention block from SAM's mask decoder:
///
///   1. self-attn(queries)
///   2. cross-attn: queries attend to keys (image features)
///   3. MLP on queries
///   4. cross-attn: keys attend to queries
///
/// [skipFirstLayerPe] controls whether the first sub-block adds the
/// query positional embedding before the self-attn (SAM's first
/// block sets this to true to preserve the raw prompt tokens).
class SamTwoWayAttentionBlock extends Module {
  final int embedDim;
  final SamAttention selfAttn;
  final LayerNorm norm1;
  final SamAttention crossAttnTokenToImage;
  final LayerNorm norm2;
  final SamMlpBlock mlp;
  final LayerNorm norm3;
  final SamAttention crossAttnImageToToken;
  final LayerNorm norm4;
  final bool skipFirstLayerPe;

  SamTwoWayAttentionBlock({
    required this.embedDim,
    required int numHeads,
    required int mlpDim,
    int attentionDownsampleRate = 2,
    this.skipFirstLayerPe = false,
    Device device = Device.CPU,
    int seed = 0,
  }) : selfAttn = SamAttention(
         embedDim: embedDim,
         numHeads: numHeads,
         device: device,
         seed: seed,
       ),
       norm1 = LayerNorm(embedDim, eps: 1e-6, device: device),
       crossAttnTokenToImage = SamAttention(
         embedDim: embedDim,
         numHeads: numHeads,
         downsampleRate: attentionDownsampleRate,
         device: device,
         seed: seed + 10_000,
       ),
       norm2 = LayerNorm(embedDim, eps: 1e-6, device: device),
       mlp = SamMlpBlock(
         embedDim: embedDim,
         mlpDim: mlpDim,
         device: device,
         seed: seed + 20_000,
       ),
       norm3 = LayerNorm(embedDim, eps: 1e-6, device: device),
       crossAttnImageToToken = SamAttention(
         embedDim: embedDim,
         numHeads: numHeads,
         downsampleRate: attentionDownsampleRate,
         device: device,
         seed: seed + 30_000,
       ),
       norm4 = LayerNorm(embedDim, eps: 1e-6, device: device);

  /// Returns updated `(queries, keys)`. All tensors are 2-D `[N, D]`.
  ({Tensor queries, Tensor keys}) call({
    required Tensor queries,
    required Tensor keys,
    required Tensor queryPe,
    required Tensor keyPe,
  }) {
    // Self-attention on queries.
    Tensor q1;
    if (skipFirstLayerPe) {
      q1 = selfAttn(queries, queries, queries);
    } else {
      final q = queries + queryPe;
      final attnOut = selfAttn(q, q, queries);
      q1 = queries + attnOut;
    }
    final q2 = norm1(q1);

    // Cross-attn: queries attend to keys (image features).
    final qWithPe = q2 + queryPe;
    final kWithPe = keys + keyPe;
    final attnOut2 = crossAttnTokenToImage(qWithPe, kWithPe, keys);
    final q3 = q2 + attnOut2;
    final q4 = norm2(q3);

    // MLP on queries.
    final mlpOut = mlp(q4);
    final q5 = q4 + mlpOut;
    final q6 = norm3(q5);

    // Cross-attn: keys attend to queries.
    final q6Pe = q6 + queryPe;
    final kPe = keys + keyPe;
    final attnOut3 = crossAttnImageToToken(kPe, q6Pe, q6);
    final k1 = keys + attnOut3;
    final k2 = norm4(k1);

    return (queries: q6, keys: k2);
  }

  @override
  List<Tensor> parameters() => [
    ...selfAttn.parameters(),
    ...norm1.parameters(),
    ...crossAttnTokenToImage.parameters(),
    ...norm2.parameters(),
    ...mlp.parameters(),
    ...norm3.parameters(),
    ...crossAttnImageToToken.parameters(),
    ...norm4.parameters(),
  ];

  @override
  List<Module> submodules() => [
    selfAttn,
    norm1,
    crossAttnTokenToImage,
    norm2,
    mlp,
    norm3,
    crossAttnImageToToken,
    norm4,
  ];
}

// ---------------------------------------------------------------------------
// SamTwoWayTransformer — 2 blocks + final token-to-image attn.
// ---------------------------------------------------------------------------

class SamTwoWayTransformer extends Module {
  final int depth;
  final int embedDim;
  final int numHeads;
  final int mlpDim;
  final int attentionDownsampleRate;

  final List<SamTwoWayAttentionBlock> blocks;
  final SamAttention finalAttnTokenToImage;
  final LayerNorm normFinal;

  SamTwoWayTransformer({
    required this.depth,
    required this.embedDim,
    required this.numHeads,
    required this.mlpDim,
    this.attentionDownsampleRate = 2,
    Device device = Device.CPU,
    int seed = 0,
  }) : blocks = <SamTwoWayAttentionBlock>[],
       finalAttnTokenToImage = SamAttention(
         embedDim: embedDim,
         numHeads: numHeads,
         downsampleRate: attentionDownsampleRate,
         device: device,
         seed: seed + 900_000,
       ),
       normFinal = LayerNorm(embedDim, eps: 1e-6, device: device) {
    for (int i = 0; i < depth; i++) {
      blocks.add(
        SamTwoWayAttentionBlock(
          embedDim: embedDim,
          numHeads: numHeads,
          mlpDim: mlpDim,
          attentionDownsampleRate: attentionDownsampleRate,
          skipFirstLayerPe: i == 0,
          device: device,
          seed: seed + i * 100_000,
        ),
      );
    }
  }

  /// `imageEmbedding: [1, embedDim, H, W]`
  /// `imagePe:        [embedDim, H, W]`
  /// `pointEmbedding: [num_tokens, embedDim]`
  ///
  /// Returns `(queries, keys)`:
  ///   queries: `[num_tokens, embedDim]` — updated prompt tokens
  ///   keys:    `[H·W, embedDim]` — updated flat image tokens
  ({Tensor queries, Tensor keys}) call({
    required Tensor imageEmbedding,
    required Tensor imagePe,
    required Tensor pointEmbedding,
  }) {
    if (imageEmbedding.shape.length != 4 ||
        imageEmbedding.shape[0] != 1 ||
        imageEmbedding.shape[1] != embedDim) {
      throw ArgumentError(
        'SamTwoWayTransformer: imageEmbedding must be [1, $embedDim, H, W]; '
        'got ${imageEmbedding.shape}',
      );
    }
    if (imagePe.shape.length != 3 || imagePe.shape[0] != embedDim) {
      throw ArgumentError(
        'SamTwoWayTransformer: imagePe must be [$embedDim, H, W]; '
        'got ${imagePe.shape}',
      );
    }
    final h = imageEmbedding.shape[2];
    final w = imageEmbedding.shape[3];

    // Flatten [1, C, H, W] -> [H·W, C] and [C, H, W] -> [H·W, C].
    final imageFlat = _permuteNCHWtoTokens(imageEmbedding);
    final imagePeFlat = _permuteCHWtoTokens(imagePe);

    var queries = pointEmbedding;
    var keys = imageFlat;

    for (final block in blocks) {
      final r = block(
        queries: queries,
        keys: keys,
        queryPe: pointEmbedding,
        keyPe: imagePeFlat,
      );
      queries = r.queries;
      keys = r.keys;
    }

    // Final token-to-image attention.
    final qWithPe = queries + pointEmbedding;
    final kWithPe = keys + imagePeFlat;
    final attnOut = finalAttnTokenToImage(qWithPe, kWithPe, keys);
    queries = normFinal(queries + attnOut);

    (h, w); // silence unused
    return (queries: queries, keys: keys);
  }

  static Tensor _permuteNCHWtoTokens(Tensor t) {
    final n = t.shape[0];
    final c = t.shape[1];
    final h = t.shape[2];
    final w = t.shape[3];
    if (n != 1) {
      throw StateError('SamTwoWayTransformer: batch>1 not supported');
    }
    final src = t.toFloat32List();
    final out = Float32List(h * w * c);
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        for (int ci = 0; ci < c; ci++) {
          out[(y * w + x) * c + ci] = src[(ci * h + y) * w + x];
        }
      }
    }
    return Tensor.fromFloat32List([h * w, c], out, device: t.device);
  }

  static Tensor _permuteCHWtoTokens(Tensor t) {
    final c = t.shape[0];
    final h = t.shape[1];
    final w = t.shape[2];
    final src = t.toFloat32List();
    final out = Float32List(h * w * c);
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        for (int ci = 0; ci < c; ci++) {
          out[(y * w + x) * c + ci] = src[(ci * h + y) * w + x];
        }
      }
    }
    return Tensor.fromFloat32List([h * w, c], out, device: t.device);
  }

  @override
  List<Tensor> parameters() => [
    for (final b in blocks) ...b.parameters(),
    ...finalAttnTokenToImage.parameters(),
    ...normFinal.parameters(),
  ];

  @override
  List<Module> submodules() => [...blocks, finalAttnTokenToImage, normFinal];
}

// ---------------------------------------------------------------------------
// SamMaskDecoder — end-to-end mask + IoU prediction.
// ---------------------------------------------------------------------------

/// Config for [SamMaskDecoder]. Defaults match `facebook/sam-vit-*`.
class SamMaskDecoderConfig {
  final int embedDim;
  final int numHeads;
  final int mlpDim;
  final int transformerDepth;
  final int numMultimaskOutputs; // typically 3
  final int iouHeadDepth;
  final int iouHeadHiddenDim;
  final Device device;
  final int seed;

  const SamMaskDecoderConfig({
    this.embedDim = 256,
    this.numHeads = 8,
    this.mlpDim = 2048,
    this.transformerDepth = 2,
    this.numMultimaskOutputs = 3,
    this.iouHeadDepth = 3,
    this.iouHeadHiddenDim = 256,
    this.device = Device.CPU,
    this.seed = 0,
  });

  /// Total number of tokens the decoder prepends to the sparse prompt
  /// embeddings: 1 IoU token + `1 + numMultimaskOutputs` mask tokens.
  int get numMaskTokens => 1 + numMultimaskOutputs;

  /// Total prepended token count = 1 IoU + [numMaskTokens] mask tokens.
  int get numOutputTokens => 1 + numMaskTokens;
}

/// Output of [SamMaskDecoder.call].
class MaskDecoderOutput {
  /// All `[numMaskTokens, H_out, W_out]` mask logits (1 no-mask +
  /// `numMultimaskOutputs` candidates).
  final Tensor masks;

  /// `[numMaskTokens]` per-mask IoU predictions.
  final Tensor iouPredictions;

  const MaskDecoderOutput({required this.masks, required this.iouPredictions});

  /// Select the single output (index 0) when `multimask == false`,
  /// or the `numMultimaskOutputs` candidate masks (indices 1..)
  /// otherwise. Returns a `(masks, iou)` record with sliced tensors.
  ({Tensor masks, Tensor iouPredictions}) select({required bool multimask}) {
    if (multimask) {
      return (
        masks: _sliceFirstAxis(masks, 1, masks.shape[0]),
        iouPredictions: _slice1D(iouPredictions, 1, iouPredictions.shape[0]),
      );
    } else {
      return (
        masks: _sliceFirstAxis(masks, 0, 1),
        iouPredictions: _slice1D(iouPredictions, 0, 1),
      );
    }
  }

  static Tensor _slice1D(Tensor t, int start, int end) {
    if (t.shape.length != 1) {
      throw ArgumentError('_slice1D: expected rank 1, got ${t.shape}');
    }
    final data = t.toList();
    final out = List<double>.filled(end - start, 0);
    for (int i = 0; i < end - start; i++) {
      out[i] = data[start + i];
    }
    return Tensor.fromList([end - start], out, device: t.device);
  }

  /// Slice a 3-D `[N, H, W]` tensor along its first axis on host. Used
  /// by [select] since [Tensor.sliceRows] is 2-D only.
  static Tensor _sliceFirstAxis(Tensor t, int start, int end) {
    if (t.shape.length != 3) {
      throw ArgumentError('_sliceFirstAxis: expected rank 3, got ${t.shape}');
    }
    final n = t.shape[0];
    final h = t.shape[1];
    final w = t.shape[2];
    if (start < 0 || end > n || start >= end) {
      throw ArgumentError(
        '_sliceFirstAxis: [$start, $end) out of range for $n',
      );
    }
    final data = t.toList();
    final rowStride = h * w;
    final outLen = (end - start) * rowStride;
    final out = List<double>.filled(outLen, 0);
    for (int i = 0; i < end - start; i++) {
      for (int j = 0; j < rowStride; j++) {
        out[i * rowStride + j] = data[(start + i) * rowStride + j];
      }
    }
    return Tensor.fromList([end - start, h, w], out, device: t.device);
  }
}

class SamMaskDecoder extends Module {
  final SamMaskDecoderConfig config;

  /// `[1 + numMultimaskOutputs, embedDim]` — 1 IoU token + M mask tokens.
  final Tensor tokenEmbeddings;

  final SamTwoWayTransformer transformer;

  // Output upscaling: ConvTranspose2d -> LN2d -> GELU -> ConvTranspose2d -> GELU
  final ConvTranspose2d output_upscaling_1;
  final LayerNorm2d output_upscaling_ln;
  final ConvTranspose2d output_upscaling_2;

  /// Per-mask hypernetwork MLPs — one per mask token. Each maps a
  /// mask token `[embedDim]` to `[upscaledChannels]` where
  /// upscaledChannels is `embedDim // 8` (32 for SAM ViT-B).
  final List<SamMlp> outputHypernetworksMlps;

  /// IoU prediction head: maps the updated IoU token to
  /// `[numMaskTokens]` scalar IoU estimates.
  final SamMlp iouPredictionHead;

  SamMaskDecoder(this.config)
    : tokenEmbeddings = _initTokens(
        config.numOutputTokens,
        config.embedDim,
        config.seed,
        config.device,
      ),
      transformer = SamTwoWayTransformer(
        depth: config.transformerDepth,
        embedDim: config.embedDim,
        numHeads: config.numHeads,
        mlpDim: config.mlpDim,
        device: config.device,
        seed: config.seed + 1_000,
      ),
      output_upscaling_1 = ConvTranspose2d(
        config.embedDim,
        config.embedDim ~/ 4,
        kernel: 2,
        stride: 2,
        padding: 0,
        bias: true,
        device: config.device,
        seed: config.seed + 700_000,
      ),
      output_upscaling_ln = LayerNorm2d(
        config.embedDim ~/ 4,
        device: config.device,
      ),
      output_upscaling_2 = ConvTranspose2d(
        config.embedDim ~/ 4,
        config.embedDim ~/ 8,
        kernel: 2,
        stride: 2,
        padding: 0,
        bias: true,
        device: config.device,
        seed: config.seed + 800_000,
      ),
      outputHypernetworksMlps = List<SamMlp>.generate(
        config.numMaskTokens,
        (i) => SamMlp(
          inputDim: config.embedDim,
          hiddenDim: config.embedDim,
          outputDim: config.embedDim ~/ 8,
          numLayers: 3,
          device: config.device,
          seed: config.seed + 500_000 + i * 1_000,
        ),
      ),
      iouPredictionHead = SamMlp(
        inputDim: config.embedDim,
        hiddenDim: config.iouHeadHiddenDim,
        outputDim: config.numMaskTokens,
        numLayers: config.iouHeadDepth,
        device: config.device,
        seed: config.seed + 600_000,
      );

  static Tensor _initTokens(int n, int d, int seed, Device device) {
    final rng = math.Random(seed);
    final vals = List<double>.generate(n * d, (_) {
      final u1 = rng.nextDouble().clamp(1e-9, 1.0);
      final u2 = rng.nextDouble();
      final z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2);
      return z * 0.02;
    });
    return Tensor.fromList([n, d], vals, requiresGrad: true, device: device);
  }

  /// Forward pass.
  ///
  /// `imageEmbedding: [1, embedDim, H, W]` — from [SamImageEncoder].
  /// `imagePe:        [embedDim, H, W]` — from
  ///                   [SamPromptEncoder.imagePositionEmbedding].
  /// `sparsePrompts:  [N_sparse, embedDim]` — from
  ///                   [SamPromptEncoder.encodeSparse].
  /// `densePrompts:   [embedDim, H, W]` — from
  ///                   [SamPromptEncoder.encodeDense].
  MaskDecoderOutput call({
    required Tensor imageEmbedding,
    required Tensor imagePe,
    required Tensor sparsePrompts,
    required Tensor densePrompts,
  }) {
    // Prepend our IoU + mask tokens to the sparse prompt embeddings.
    final tokens = _concatRows(tokenEmbeddings, sparsePrompts);

    // Add the dense prompt embeddings elementwise to the image
    // embedding (both are `[embedDim, H, W]`; imageEmbedding has a
    // leading batch axis).
    final srcNCHW = _addDenseToImage(imageEmbedding, densePrompts);

    // Run the two-way transformer.
    final tr = transformer(
      imageEmbedding: srcNCHW,
      imagePe: imagePe,
      pointEmbedding: tokens,
    );
    final hs = tr.queries; // [N_tokens, embedDim]
    final srcTokens = tr.keys; // [H·W, embedDim]

    final iouTokenOut = _rowSlice(hs, 0, 1); // [1, embedDim]
    final maskTokensOut = _rowSlice(hs, 1, 1 + config.numMaskTokens);
    // Actually: prepended = tokenEmbeddings ([1 + M, D]) + sparse.
    // Layout: index 0 is IoU, indices 1..1+M-1 are mask tokens.
    // But note tokenEmbeddings is [numMaskTokens=1+M, D], so slice
    // 1..numMaskTokens excludes the IoU and includes all mask tokens.

    // Upscale the image features.
    final srcNchwOut = _tokensToNchw(
      srcTokens,
      srcNCHW.shape[2],
      srcNCHW.shape[3],
      config.embedDim,
      srcNCHW.device,
    );
    var up = output_upscaling_1(srcNchwOut);
    up = output_upscaling_ln(up);
    up = _gelu(up);
    up = output_upscaling_2(up);
    up = _gelu(up);
    // up: [1, embedDim//8, H·4, W·4]

    // Per-mask hypernet MLPs → [numMaskTokens, embedDim // 8].
    final hyperRows = <Tensor>[];
    for (int i = 0; i < config.numMaskTokens; i++) {
      final tok = _rowSlice(maskTokensOut, i, i + 1); // [1, embedDim]
      hyperRows.add(outputHypernetworksMlps[i](tok));
    }
    final hyper = _concatRowsList(hyperRows); // [numMaskTokens, embedDim//8]

    // Masks: for each mask token, dot-product its hypernet feature
    // against the upscaled feature map.
    final upChannels = config.embedDim ~/ 8;
    final hOut = up.shape[2];
    final wOut = up.shape[3];
    final upData = up.toFloat32List();
    final hyperData = hyper.toList();
    final maskData = Float32List(config.numMaskTokens * hOut * wOut);
    for (int i = 0; i < config.numMaskTokens; i++) {
      for (int y = 0; y < hOut; y++) {
        for (int x = 0; x < wOut; x++) {
          double acc = 0;
          for (int c = 0; c < upChannels; c++) {
            acc +=
                hyperData[i * upChannels + c] *
                upData[(c * hOut + y) * wOut + x];
          }
          maskData[(i * hOut + y) * wOut + x] = acc;
        }
      }
    }
    final masks = Tensor.fromFloat32List(
      [config.numMaskTokens, hOut, wOut],
      maskData,
      device: up.device,
    );

    // IoU predictions.
    final iou = iouPredictionHead(iouTokenOut).reshape([config.numMaskTokens]);
    return MaskDecoderOutput(masks: masks, iouPredictions: iou);
  }

  /// Concatenate row-wise: `[A, D]` + `[B, D]` → `[A+B, D]`. Handles
  /// empty `[0, D]` correctly.
  Tensor _concatRows(Tensor a, Tensor b) {
    if (a.shape[0] == 0) return b;
    if (b.shape[0] == 0) return a;
    return TensorConcat.concat([a, b], axis: 0);
  }

  Tensor _concatRowsList(List<Tensor> rows) {
    return TensorConcat.concat(rows, axis: 0);
  }

  Tensor _rowSlice(Tensor t, int start, int end) => t.sliceRows(start, end);

  /// Add `[C, H, W]` dense prompt to `[1, C, H, W]` image embedding.
  Tensor _addDenseToImage(Tensor image, Tensor dense) {
    final c = image.shape[1];
    final h = image.shape[2];
    final w = image.shape[3];
    if (dense.shape.length != 3 ||
        dense.shape[0] != c ||
        dense.shape[1] != h ||
        dense.shape[2] != w) {
      throw ArgumentError(
        'SamMaskDecoder: dense prompt shape ${dense.shape} does not '
        'match image [$c, $h, $w]',
      );
    }
    final iData = image.toFloat32List();
    final dData = dense.toFloat32List();
    final out = Float32List(iData.length);
    for (int i = 0; i < iData.length; i++) {
      out[i] = iData[i] + dData[i];
    }
    return Tensor.fromFloat32List([1, c, h, w], out, device: image.device);
  }

  static Tensor _tokensToNchw(
    Tensor tokens,
    int h,
    int w,
    int c,
    Device device,
  ) {
    final src = tokens.toFloat32List();
    final out = Float32List(c * h * w);
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        for (int ci = 0; ci < c; ci++) {
          out[(ci * h + y) * w + x] = src[(y * w + x) * c + ci];
        }
      }
    }
    return Tensor.fromFloat32List([1, c, h, w], out, device: device);
  }

  static Tensor _gelu(Tensor x) {
    const invSqrt2 = 0.7071067811865475;
    final data = x.toFloat32List();
    final out = Float32List(data.length);
    for (int i = 0; i < data.length; i++) {
      out[i] = 0.5 * data[i] * (1.0 + _erf(data[i] * invSqrt2));
    }
    return Tensor.fromFloat32List(x.shape, out, device: x.device);
  }

  static double _erf(double x) {
    final sign = x < 0 ? -1.0 : 1.0;
    x = x.abs();
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;
    final t = 1.0 / (1.0 + p * x);
    final y =
        1.0 -
        (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x);
    return sign * y;
  }

  @override
  List<Tensor> parameters() => [
    tokenEmbeddings,
    ...transformer.parameters(),
    ...output_upscaling_1.parameters(),
    ...output_upscaling_ln.parameters(),
    ...output_upscaling_2.parameters(),
    for (final m in outputHypernetworksMlps) ...m.parameters(),
    ...iouPredictionHead.parameters(),
  ];

  @override
  List<Module> submodules() => [
    transformer,
    output_upscaling_1,
    output_upscaling_ln,
    output_upscaling_2,
    ...outputHypernetworksMlps,
    iouPredictionHead,
  ];
}
