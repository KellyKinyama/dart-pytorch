/// SAM prompt encoder — sparse (points + boxes) + dense (masks) prompt
/// embeddings for the SAM mask decoder.
///
/// Ports `segment_anything.modeling.prompt_encoder.PromptEncoder`. Given
/// user prompts — 2-D point coords with `positive/negative` labels,
/// axis-aligned boxes, and optional low-resolution masks — produces the
/// two embedding streams the mask decoder consumes:
///
///   * **Sparse embeddings** `[N_sparse, embedDim]` — one row per
///     point or box-corner prompt. Each row is a Fourier positional
///     encoding of the (x, y) coordinate + a learned "point-type"
///     embedding (positive click, negative click, top-left box corner,
///     bottom-right box corner).
///   * **Dense embeddings** `[embedDim, H, W]` — one activation map
///     matching the image encoder's grid. Comes from the mask
///     downsampling CNN when a mask is provided, or the "no-mask"
///     learned embedding broadcast over the grid.
///
/// This module also carries the **image position encoding**
/// `[embedDim, H, W]` that the mask decoder additively injects into
/// its query image features — see [imagePositionEmbedding].
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../../tensor/tensor.dart';
import '../conv2d.dart';
import '../module.dart';
import 'sam_image_encoder.dart' show LayerNorm2d;

/// Enum of prompt-token types matching SAM's original labels.
/// Values map onto rows of [SamPromptEncoder.pointEmbeddings].
enum SamPointType {
  negative, // background click
  positive, // foreground click
  boxTopLeft, // upper-left corner of a box prompt
  boxBottomRight, // lower-right corner of a box prompt
}

/// Random-Fourier-features positional encoding. Ports SAM's
/// `PositionEmbeddingRandom` (Tancik et al. 2020).
///
/// Given a 2-D coordinate `(x, y)` normalised to `[0, 1]`, produces a
/// `[2 · numPosFeats]` feature vector:
///
///     f(x, y) = concat( sin(2π · [x, y] @ B),
///                       cos(2π · [x, y] @ B) )
///
/// where `B ∈ ℝ^{2 × numPosFeats}` is a matrix of frozen Gaussian
/// samples. SAM uses `scale = 1.0` by default; the encoding is
/// therefore **not learned** — it's a constant map from coord to
/// feature vector.
class SamPositionEmbeddingRandom extends Module {
  final int numPosFeats;
  final double scale;

  /// `[2, numPosFeats]` — frozen at construction; not a trainable
  /// parameter (SAM's original uses `nn.Parameter(..., requires_grad=False)`
  /// with a `torch.Generator` set to the fixed seed 3141592).
  final Tensor gaussianMatrix;

  SamPositionEmbeddingRandom({
    required this.numPosFeats,
    this.scale = 1.0,
    int seed = 3141592,
    Device device = Device.CPU,
  }) : gaussianMatrix = _initGaussian(numPosFeats, scale, seed, device);

  static Tensor _initGaussian(int nf, double scale, int seed, Device device) {
    final rng = math.Random(seed);
    final vals = List<double>.generate(2 * nf, (_) {
      final u1 = rng.nextDouble().clamp(1e-12, 1.0);
      final u2 = rng.nextDouble();
      final z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2);
      return z * scale;
    });
    return Tensor.fromList([2, nf], vals, device: device);
  }

  /// Encode a single `[N, 2]` batch of coordinates in `[0, 1]`.
  /// Returns `[N, 2 · numPosFeats]`.
  Tensor encodeCoords(Tensor xy) {
    if (xy.shape.length != 2 || xy.shape[1] != 2) {
      throw ArgumentError(
        'SamPositionEmbeddingRandom.encodeCoords: expected [N, 2]; '
        'got ${xy.shape}',
      );
    }
    // 2π · [N, 2] · [2, F] = [N, F].
    // Do the matmul on host — the input is small (< 100 rows).
    final coords = xy.toFloat32List();
    final gm = gaussianMatrix.toFloat32List();
    final n = xy.shape[0];
    final f = numPosFeats;
    final proj = Float32List(n * f);
    for (int i = 0; i < n; i++) {
      final xi = coords[i * 2];
      final yi = coords[i * 2 + 1];
      for (int j = 0; j < f; j++) {
        proj[i * f + j] = 2 * math.pi * (xi * gm[j] + yi * gm[f + j]);
      }
    }
    final out = Float32List(n * 2 * f);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < f; j++) {
        final theta = proj[i * f + j];
        out[i * 2 * f + j] = math.sin(theta);
        out[i * 2 * f + f + j] = math.cos(theta);
      }
    }
    return Tensor.fromFloat32List([n, 2 * f], out, device: xy.device);
  }

  /// Build the dense positional embedding for an `[H, W]` grid — one
  /// `[2·numPosFeats]` vector per pixel, arranged as `[2·numPosFeats,
  /// H, W]`. Pixels use the mid-cell coordinate `(x + 0.5)/W`.
  Tensor encodeGrid(int h, int w) {
    final coords = Float32List(h * w * 2);
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        coords[(y * w + x) * 2] = (x + 0.5) / w;
        coords[(y * w + x) * 2 + 1] = (y + 0.5) / h;
      }
    }
    final xy = Tensor.fromFloat32List(
      [h * w, 2],
      coords,
      device: gaussianMatrix.device,
    );
    final enc = encodeCoords(xy); // [H*W, 2·F]
    // Permute [H*W, D] -> [D, H, W].
    final d = 2 * numPosFeats;
    final flat = enc.toList();
    final out = Float32List(d * h * w);
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        for (int ci = 0; ci < d; ci++) {
          out[(ci * h + y) * w + x] = flat[(y * w + x) * d + ci];
        }
      }
    }
    return Tensor.fromFloat32List(
      [d, h, w],
      out,
      device: gaussianMatrix.device,
    );
  }

  @override
  List<Tensor> parameters() => const [];
}

/// SAM prompt encoder.
class SamPromptEncoder extends Module {
  final int embedDim; // 256 for SAM ViT-B
  final int imageEmbedH; // 64 for 1024/16
  final int imageEmbedW;
  final int imageSize; // input image side (1024 for SAM)
  final int maskInSize; // low-res mask input, typically 4·imageEmbedH = 256
  final int maskInputChannels; // typically 16 for SAM

  final SamPositionEmbeddingRandom posEmbed;

  /// `[4, embedDim]` — one row per [SamPointType] value. Point/box
  /// prompts get their type-embedding added to the Fourier-encoded
  /// coordinate.
  final Tensor pointEmbeddings;

  /// `[embedDim]` — "no mask" fallback embedding, broadcast over the
  /// grid when no mask prompt is provided.
  final Tensor noMaskEmbedding;

  // Mask downsampling CNN: [1, mask_in, mask_in] -> [embedDim, H, W]
  final Conv2d maskConv1; // 1 -> maskInputChannels ~/ 4, stride 2
  final LayerNorm2d maskLn1;
  final Conv2d maskConv2; // -> maskInputChannels, stride 2
  final LayerNorm2d maskLn2;
  final Conv2d maskConv3; // -> embedDim, stride 1

  SamPromptEncoder({
    this.embedDim = 256,
    this.imageEmbedH = 64,
    this.imageEmbedW = 64,
    this.imageSize = 1024,
    this.maskInSize = 256,
    this.maskInputChannels = 16,
    Device device = Device.CPU,
    int seed = 0,
  }) : posEmbed = SamPositionEmbeddingRandom(
         numPosFeats: embedDim ~/ 2,
         seed: 3141592,
         device: device,
       ),
       pointEmbeddings = _initSmallGaussian(
         [4, embedDim],
         scale: 1.0, // SAM uses default nn.Embedding init.
         seed: seed + 1,
         device: device,
         requiresGrad: true,
       ),
       noMaskEmbedding = _initSmallGaussian(
         [embedDim],
         scale: 1.0,
         seed: seed + 2,
         device: device,
         requiresGrad: true,
       ),
       maskConv1 = Conv2d(
         1,
         maskInputChannels ~/ 4,
         kernel: 2,
         stride: 2,
         padding: 0,
         bias: true,
         device: device,
         seed: seed + 10,
       ),
       maskLn1 = LayerNorm2d(maskInputChannels ~/ 4, device: device),
       maskConv2 = Conv2d(
         maskInputChannels ~/ 4,
         maskInputChannels,
         kernel: 2,
         stride: 2,
         padding: 0,
         bias: true,
         device: device,
         seed: seed + 11,
       ),
       maskLn2 = LayerNorm2d(maskInputChannels, device: device),
       maskConv3 = Conv2d(
         maskInputChannels,
         embedDim,
         kernel: 1,
         stride: 1,
         padding: 0,
         bias: true,
         device: device,
         seed: seed + 12,
       );

  static Tensor _initSmallGaussian(
    List<int> shape, {
    required double scale,
    required int seed,
    required Device device,
    bool requiresGrad = false,
  }) {
    final rng = math.Random(seed);
    final n = shape.fold<int>(1, (a, b) => a * b);
    final vals = List<double>.generate(n, (_) {
      final u1 = rng.nextDouble().clamp(1e-12, 1.0);
      final u2 = rng.nextDouble();
      final z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2);
      return z * scale;
    });
    return Tensor.fromList(
      shape,
      vals,
      requiresGrad: requiresGrad,
      device: device,
    );
  }

  /// Encode a batch of point + box prompts.
  ///
  /// `pointsXY` — `[N_p, 2]` coordinates in image-pixel space `[0,
  /// imageSize)`. May be `null` when only boxes are used.
  ///
  /// `pointLabels` — `[N_p]` of [SamPointType]. Must have the same
  /// length as `pointsXY`.
  ///
  /// `boxesXYXY` — `[N_b, 4]` axis-aligned boxes as `(x1, y1, x2, y2)`
  /// in image-pixel space. Each box contributes 2 sparse tokens
  /// (top-left and bottom-right corners). May be `null`.
  ///
  /// Returns `[N_p + 2·N_b, embedDim]`. Row order: points first, then
  /// boxes' top-left, then boxes' bottom-right.
  Tensor encodeSparse({
    Tensor? pointsXY,
    List<SamPointType>? pointLabels,
    Tensor? boxesXYXY,
  }) {
    final rows = <List<double>>[];

    // Points.
    if (pointsXY != null) {
      if (pointLabels == null || pointLabels.length != pointsXY.shape[0]) {
        throw ArgumentError(
          'SamPromptEncoder.encodeSparse: pointLabels length must match '
          'pointsXY rows',
        );
      }
      final norm = _normalisePointCoords(pointsXY);
      final enc = posEmbed.encodeCoords(norm); // [N_p, embedDim]
      final encVals = enc.toList();
      final typeVals = pointEmbeddings.toList();
      for (int i = 0; i < pointLabels.length; i++) {
        final t = pointLabels[i].index;
        final base = i * embedDim;
        final row = List<double>.filled(embedDim, 0);
        for (int j = 0; j < embedDim; j++) {
          row[j] = encVals[base + j] + typeVals[t * embedDim + j];
        }
        rows.add(row);
      }
    }

    // Boxes — decomposed into (top-left, bottom-right) corner tokens.
    if (boxesXYXY != null) {
      if (boxesXYXY.shape.length != 2 || boxesXYXY.shape[1] != 4) {
        throw ArgumentError(
          'SamPromptEncoder.encodeSparse: boxes must be [N, 4] '
          '(x1, y1, x2, y2); got ${boxesXYXY.shape}',
        );
      }
      final n = boxesXYXY.shape[0];
      final boxVals = boxesXYXY.toList();
      // Corners in the same 0..1 normalised space.
      final coords = Float32List(n * 2 * 2);
      for (int i = 0; i < n; i++) {
        coords[(i * 2) * 2 + 0] = boxVals[i * 4 + 0] / imageSize;
        coords[(i * 2) * 2 + 1] = boxVals[i * 4 + 1] / imageSize;
        coords[(i * 2 + 1) * 2 + 0] = boxVals[i * 4 + 2] / imageSize;
        coords[(i * 2 + 1) * 2 + 1] = boxVals[i * 4 + 3] / imageSize;
      }
      final xy = Tensor.fromFloat32List(
        [n * 2, 2],
        coords,
        device: boxesXYXY.device,
      );
      final enc = posEmbed.encodeCoords(xy).toList();
      final typeVals = pointEmbeddings.toList();
      final tlBase = SamPointType.boxTopLeft.index * embedDim;
      final brBase = SamPointType.boxBottomRight.index * embedDim;
      for (int i = 0; i < n; i++) {
        // Emit ALL top-lefts first, then all bottom-rights? SAM emits
        // (tl, br) per box, interleaved. We match interleaved layout.
        final tlRow = List<double>.filled(embedDim, 0);
        final brRow = List<double>.filled(embedDim, 0);
        for (int j = 0; j < embedDim; j++) {
          tlRow[j] = enc[(i * 2) * embedDim + j] + typeVals[tlBase + j];
          brRow[j] = enc[(i * 2 + 1) * embedDim + j] + typeVals[brBase + j];
        }
        rows.add(tlRow);
        rows.add(brRow);
      }
    }

    if (rows.isEmpty) {
      // SAM's default: empty sparse embedding.
      return Tensor.fromList([0, embedDim], const <double>[]);
    }
    final flat = <double>[];
    for (final r in rows) {
      flat.addAll(r);
    }
    return Tensor.fromList([rows.length, embedDim], flat);
  }

  /// Encode a dense mask prompt (`[1, maskInSize, maskInSize]`
  /// low-res mask logits) into `[embedDim, imageEmbedH, imageEmbedW]`.
  /// Pass `null` to fall back to the "no-mask" broadcast.
  Tensor encodeDense({Tensor? mask}) {
    if (mask != null) {
      if (mask.shape.length != 4 ||
          mask.shape[0] != 1 ||
          mask.shape[1] != 1 ||
          mask.shape[2] != maskInSize ||
          mask.shape[3] != maskInSize) {
        throw ArgumentError(
          'SamPromptEncoder.encodeDense: expected mask [1, 1, '
          '$maskInSize, $maskInSize]; got ${mask.shape}',
        );
      }
      var h = maskConv1(mask);
      h = maskLn1(h);
      h = _gelu(h);
      h = maskConv2(h);
      h = maskLn2(h);
      h = _gelu(h);
      h = maskConv3(h);
      // Result: [1, embedDim, imageEmbedH, imageEmbedW].
      return h.reshape([embedDim, imageEmbedH, imageEmbedW]);
    }
    // No-mask fallback: broadcast the learned [embedDim] vector.
    final vals = noMaskEmbedding.toList();
    final out = Float32List(embedDim * imageEmbedH * imageEmbedW);
    for (int c = 0; c < embedDim; c++) {
      final v = vals[c];
      final base = c * imageEmbedH * imageEmbedW;
      for (int i = 0; i < imageEmbedH * imageEmbedW; i++) {
        out[base + i] = v;
      }
    }
    return Tensor.fromFloat32List(
      [embedDim, imageEmbedH, imageEmbedW],
      out,
      device: noMaskEmbedding.device,
    );
  }

  /// Positional embedding for the image feature grid. The mask
  /// decoder adds this to the image features once per forward pass.
  Tensor imagePositionEmbedding() {
    return posEmbed.encodeGrid(imageEmbedH, imageEmbedW);
  }

  Tensor _normalisePointCoords(Tensor xy) {
    final data = xy.toList();
    final n = xy.shape[0];
    final buf = Float32List(n * 2);
    for (int i = 0; i < n; i++) {
      buf[i * 2] = data[i * 2] / imageSize;
      buf[i * 2 + 1] = data[i * 2 + 1] / imageSize;
    }
    return Tensor.fromFloat32List([n, 2], buf, device: xy.device);
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
    pointEmbeddings,
    noMaskEmbedding,
    ...maskConv1.parameters(),
    ...maskLn1.parameters(),
    ...maskConv2.parameters(),
    ...maskLn2.parameters(),
    ...maskConv3.parameters(),
  ];

  @override
  List<Module> submodules() => [
    maskConv1,
    maskLn1,
    maskConv2,
    maskLn2,
    maskConv3,
  ];
}
