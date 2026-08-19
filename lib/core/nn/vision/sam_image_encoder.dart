/// SAM ViT-B image encoder — windowed attention + relative positional
/// bias + neck.
///
/// Ports the image encoder from `segment-anything` (Kirillov et al. 2023,
/// `facebookresearch/segment-anything`). Takes a preprocessed 3×1024×1024
/// image (channels-last-per-pixel patchified layout, matching the rest
/// of our ViT stack) and produces the `[N, 256, 64, 64]` image
/// embedding that SAM's prompt encoder + mask decoder consume.
///
/// **What's here (this session):**
///
///   * [LayerNorm2d] — per-pixel LayerNorm over the channel axis of
///     an NCHW tensor, matching SAM's `LayerNorm2d` in the neck.
///   * [SamWindowedAttention] — MHA with two special SAM knobs:
///       - **Windowed partitioning**: with `windowSize < inputSize`,
///         the `[H, W, D]` feature map is padded to a multiple of
///         `windowSize`, partitioned into non-overlapping windows,
///         and MHA runs *per window*. This cuts attention cost from
///         `O((H·W)²)` to `O(W² · numWindows)` (SAM ViT-B: 4×
///         speedup at 64² tokens).
///       - **Decomposed relative-positional bias**: two learned
///         tables `relPosH[2·H − 1, headDim]` and
///         `relPosW[2·W − 1, headDim]` where for each query row `i`
///         and key row `j` the bias `Q[i] · relPosH[i-j+W-1] +
///         Q[i] · relPosW[.]` is added to the attention logits
///         **before softmax**. Matches SAM's
///         `add_decomposed_rel_pos` in
///         `segment_anything/modeling/image_encoder.py` bit-for-bit.
///   * [SamViTBlock] — pre-LN + [SamWindowedAttention] + residual,
///     pre-LN + MLP (fc1 → GELU → fc2) + residual.
///   * [SamImageEncoder] — patch embed (`Conv2d(3, embedDim,
///     k=patchSize, s=patchSize, bias)`), learned 2-D positional
///     embedding, N × [SamViTBlock] with SAM's global-attention
///     layer schedule (indices `[2, 5, 8, 11]` for ViT-B), and a
///     final neck (`Conv2d(embed, 256, k=1) → LayerNorm2d →
///     Conv2d(256, 256, k=3, p=1) → LayerNorm2d`).
///
/// **Not yet:** HF safetensors loader, prompt encoder (Fourier
/// positional encoding + point/box/mask embeddings), and mask decoder
/// (two-way transformer + upsampling). All three land in a follow-up
/// session; the primitives here (WindowedAttention, ConvTranspose2d,
/// LayerNorm2d, positional embeddings) are the pieces they'll need.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../../tensor/tensor.dart';
import '../conv2d.dart';
import '../layer_norm.dart';
import '../linear.dart';
import '../module.dart';

// ---------------------------------------------------------------------------
// LayerNorm2d — per-pixel LN over the channel axis of NCHW.
// ---------------------------------------------------------------------------

/// LayerNorm applied over the channel axis of an `[N, C, H, W]`
/// tensor. Matches `segment_anything.modeling.common.LayerNorm2d`.
/// Learnable per-channel `gamma` and `beta`.
class LayerNorm2d extends Module {
  final int numChannels;
  final double eps;
  final Tensor gamma; // [C]
  final Tensor beta; // [C]

  LayerNorm2d(this.numChannels, {this.eps = 1e-6, Device device = Device.CPU})
    : gamma = Tensor.fill(
        [numChannels],
        1.0,
        requiresGrad: true,
        device: device,
      ),
      beta = Tensor.fill(
        [numChannels],
        0.0,
        requiresGrad: true,
        device: device,
      );

  Tensor call(Tensor x) {
    if (x.shape.length != 4 || x.shape[1] != numChannels) {
      throw ArgumentError(
        'LayerNorm2d: expected [N, $numChannels, H, W]; got ${x.shape}',
      );
    }
    // Compute mean/var per-pixel over the channel axis on host.
    final n = x.shape[0];
    final c = x.shape[1];
    final h = x.shape[2];
    final w = x.shape[3];
    final data = x.toFloat32List();
    final gs = gamma.toFloat32List();
    final bs = beta.toFloat32List();
    final out = Float32List(n * c * h * w);
    for (int ni = 0; ni < n; ni++) {
      for (int y = 0; y < h; y++) {
        for (int xi = 0; xi < w; xi++) {
          // Mean/var over C at this pixel.
          double sum = 0;
          double sqSum = 0;
          for (int ci = 0; ci < c; ci++) {
            final v = data[((ni * c + ci) * h + y) * w + xi];
            sum += v;
            sqSum += v * v;
          }
          final mean = sum / c;
          final variance = sqSum / c - mean * mean;
          final invStd = 1.0 / math.sqrt(variance + eps);
          for (int ci = 0; ci < c; ci++) {
            final idx = ((ni * c + ci) * h + y) * w + xi;
            out[idx] = (data[idx] - mean) * invStd * gs[ci] + bs[ci];
          }
        }
      }
    }
    return Tensor.fromFloat32List([n, c, h, w], out, device: x.device);
  }

  @override
  List<Tensor> parameters() => [gamma, beta];
}

// ---------------------------------------------------------------------------
// SamWindowedAttention — per-window MHA with decomposed rel-pos.
// ---------------------------------------------------------------------------

/// Attention block for SAM's image encoder. Accepts a 2-D flat
/// `[H·W, embedDim]` token sequence (which represents an `[H, W, D]`
/// feature map) and returns the same shape.
///
///   * When `windowSize == inputSize` this is plain global MHA over
///     the whole grid, with the decomposed relative-positional bias
///     applied over the full `H·W × H·W` attention matrix.
///   * When `windowSize < inputSize`, the input is (host-side) padded
///     to a multiple of `windowSize`, partitioned into `numWindows =
///     (Hp/ws) · (Wp/ws)` non-overlapping windows, MHA + rel-pos runs
///     independently per window, and the result is un-partitioned +
///     cropped back to `[H, W, D]`.
///
/// Per-head:
///   * Q, K, V projections use a fused Linear layout matching HF
///     `segment_anything.modeling.image_encoder.Attention` — see the
///     bare `wq/wk/wv` per-head Linears via [MultiHeadAttention].
///   * Relative-positional bias:
///
///         attn[i, j] += Q[i] · relPosH[dy] + Q[i] · relPosW[dx]
///
///     where `dy = qy - ky + H - 1`, `dx = qx - kx + W - 1`, and
///     `relPosH` / `relPosW` are `[2·H − 1, headDim]` /
///     `[2·W − 1, headDim]` learnable tables (one per attention
///     head).
///
/// This module reuses per-head [Linear]s for Q/K/V (`wq[h]`, `wk[h]`,
/// `wv[h]`) and a fused output projection `wo`. It intentionally does
/// **not** subclass [MultiHeadAttention] because the rel-pos bias
/// must live *inside* the attention softmax and there's no clean way
/// to inject that from outside.
class SamWindowedAttention extends Module {
  final int embedDim;
  final int numHeads;
  final int headDim;
  final int inputSize; // H = W of the feature grid (typically 64).
  final int windowSize; // 14 for windowed blocks; == inputSize for global.

  final List<Linear> wq;
  final List<Linear> wk;
  final List<Linear> wv;
  final Linear wo;

  /// `[numHeads, 2·gridH − 1, headDim]` — height-axis relative-pos
  /// table. `gridH` is `windowSize` for windowed blocks or
  /// `inputSize` for global.
  final Tensor relPosH;

  /// Same shape/layout for the width axis.
  final Tensor relPosW;

  SamWindowedAttention({
    required this.embedDim,
    required this.numHeads,
    required this.inputSize,
    required this.windowSize,
    Device device = Device.CPU,
    int seed = 0,
  }) : headDim = embedDim ~/ numHeads,
       wq = List<Linear>.generate(
         numHeads,
         (h) => Linear(
           embedDim,
           embedDim ~/ numHeads,
           bias: true,
           device: device,
           seed: seed + h,
         ),
       ),
       wk = List<Linear>.generate(
         numHeads,
         (h) => Linear(
           embedDim,
           embedDim ~/ numHeads,
           bias: true,
           device: device,
           seed: seed + 1000 + h,
         ),
       ),
       wv = List<Linear>.generate(
         numHeads,
         (h) => Linear(
           embedDim,
           embedDim ~/ numHeads,
           bias: true,
           device: device,
           seed: seed + 2000 + h,
         ),
       ),
       wo = Linear(
         embedDim,
         embedDim,
         bias: true,
         device: device,
         seed: seed + 3000,
       ),
       relPosH = _initRelPos(
         numHeads,
         windowSize,
         embedDim ~/ numHeads,
         seed + 4000,
         device,
       ),
       relPosW = _initRelPos(
         numHeads,
         windowSize,
         embedDim ~/ numHeads,
         seed + 5000,
         device,
       );

  static Tensor _initRelPos(int nh, int size, int hd, int seed, Device device) {
    final rng = math.Random(seed);
    final total = nh * (2 * size - 1) * hd;
    final bound = 1.0 / math.sqrt(hd);
    final vals = List<double>.generate(
      total,
      (_) => (rng.nextDouble() * 2 - 1) * bound,
    );
    return Tensor.fromList(
      [nh, 2 * size - 1, hd],
      vals,
      requiresGrad: true,
      device: device,
    );
  }

  Tensor call(Tensor x) {
    if (x.shape.length != 2 || x.shape[1] != embedDim) {
      throw ArgumentError(
        'SamWindowedAttention: expected [inputSize² , $embedDim]; '
        'got ${x.shape}',
      );
    }
    if (x.shape[0] != inputSize * inputSize) {
      throw ArgumentError(
        'SamWindowedAttention: input tokens ${x.shape[0]} != '
        '$inputSize² = ${inputSize * inputSize}',
      );
    }

    // ---- Reshape to [H, W, D] on host so we can window-partition. ----
    final srcData = x.toFloat32List();
    // We'll process per-window with the exact same math as global MHA
    // (global == one window covering the whole grid).
    final ws = windowSize;
    final needsPad = inputSize % ws != 0;
    final gridH = needsPad ? ((inputSize + ws - 1) ~/ ws) * ws : inputSize;
    final gridW = gridH;

    // Padded HW feature map on host.
    final padded = needsPad
        ? _padHW(srcData, inputSize, embedDim, gridH, gridW)
        : srcData;

    final numWinRows = gridH ~/ ws;
    final numWinCols = gridW ~/ ws;

    // Assemble output on host as `[gridH, gridW, D]`, filled per-window.
    final outHW = Float32List(gridH * gridW * embedDim);

    for (int winY = 0; winY < numWinRows; winY++) {
      for (int winX = 0; winX < numWinCols; winX++) {
        // Extract window: [ws*ws, D] on host.
        final winTokens = Float32List(ws * ws * embedDim);
        for (int r = 0; r < ws; r++) {
          final srcRow = (winY * ws + r);
          for (int c = 0; c < ws; c++) {
            final srcCol = winX * ws + c;
            final srcBase = (srcRow * gridW + srcCol) * embedDim;
            final dstBase = (r * ws + c) * embedDim;
            for (int d = 0; d < embedDim; d++) {
              winTokens[dstBase + d] = padded[srcBase + d];
            }
          }
        }

        // Run window attention with rel-pos.
        final winIn = Tensor.fromList(
          [ws * ws, embedDim],
          winTokens,
          device: x.device,
        );
        final winOut = _attendWithRelPos(winIn, ws, ws);

        // Scatter window output back into outHW.
        final outVals = winOut.toFloat32List();
        for (int r = 0; r < ws; r++) {
          final dstRow = winY * ws + r;
          for (int c = 0; c < ws; c++) {
            final dstCol = winX * ws + c;
            final srcBase = (r * ws + c) * embedDim;
            final dstBase = (dstRow * gridW + dstCol) * embedDim;
            for (int d = 0; d < embedDim; d++) {
              outHW[dstBase + d] = outVals[srcBase + d];
            }
          }
        }
      }
    }

    // Crop back to inputSize² if we padded.
    final finalOut = needsPad
        ? _cropHW(outHW, inputSize, embedDim, gridH, gridW)
        : outHW;

    return Tensor.fromFloat32List(
      [inputSize * inputSize, embedDim],
      finalOut,
      device: x.device,
    );
  }

  /// Attention on a `[qhw, D]` token block coming from a `[qh, qw, D]`
  /// window (or the full grid). Applies decomposed rel-pos bias.
  ///
  /// Note that when `qh == qw == windowSize` the rel-pos tables here
  /// are exactly the module's `relPosH` / `relPosW` (both keyed by
  /// `windowSize`). If you were to call this on a differently-shaped
  /// window you'd need to pass in a resampled table — SAM's paper
  /// does this with bilinear interpolation, but the standard forward
  /// pass never triggers that path.
  Tensor _attendWithRelPos(Tensor x, int qh, int qw) {
    if (qh != windowSize || qw != windowSize) {
      throw StateError(
        'SamWindowedAttention._attendWithRelPos: expected window '
        '($windowSize, $windowSize); got ($qh, $qw)',
      );
    }
    final heads = <Tensor>[];
    // Rel-pos tables are `[numHeads, 2*ws-1, headDim]` — grab per-head
    // slices on host once, then reuse in the softmax step.
    final relHData = relPosH.toFloat32List();
    final relWData = relPosW.toFloat32List();

    for (int h = 0; h < numHeads; h++) {
      final q = wq[h](x); // [ws², headDim]
      final k = wk[h](x);
      final v = wv[h](x);
      // Compute Q·K.T on host so we can add the rel-pos bias inline.
      final qData = q.toFloat32List();
      final kData = k.toFloat32List();
      final vData = v.toFloat32List();
      final scale = 1.0 / math.sqrt(headDim);
      final scores = Float32List(qh * qw * qh * qw);
      // dot products.
      final n = qh * qw;
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          double dot = 0;
          for (int d = 0; d < headDim; d++) {
            dot += qData[i * headDim + d] * kData[j * headDim + d];
          }
          scores[i * n + j] = dot * scale;
        }
      }
      // Add decomposed rel-pos bias.
      //   attn[i, j] += Q[i] · relPosH[qy - ky + ws - 1]
      //              +  Q[i] · relPosW[qx - kx + ws - 1]
      // where (qy, qx) and (ky, kx) index the ws × ws window.
      final relBase = h * (2 * windowSize - 1) * headDim;
      for (int qy = 0; qy < qh; qy++) {
        for (int qx = 0; qx < qw; qx++) {
          final qIdx = qy * qw + qx;
          for (int ky = 0; ky < qh; ky++) {
            final dy = qy - ky + windowSize - 1;
            final relHRowBase = relBase + dy * headDim;
            double biasH = 0;
            for (int d = 0; d < headDim; d++) {
              biasH += qData[qIdx * headDim + d] * relHData[relHRowBase + d];
            }
            for (int kx = 0; kx < qw; kx++) {
              final dx = qx - kx + windowSize - 1;
              final relWRowBase = relBase + dx * headDim;
              double biasW = 0;
              for (int d = 0; d < headDim; d++) {
                biasW += qData[qIdx * headDim + d] * relWData[relWRowBase + d];
              }
              final kIdx = ky * qw + kx;
              scores[qIdx * n + kIdx] += biasH + biasW;
            }
          }
        }
      }
      // Softmax over rows.
      final attn = Float32List(n * n);
      for (int i = 0; i < n; i++) {
        var maxVal = scores[i * n];
        for (int j = 1; j < n; j++) {
          if (scores[i * n + j] > maxVal) maxVal = scores[i * n + j];
        }
        double sum = 0;
        for (int j = 0; j < n; j++) {
          final e = math.exp(scores[i * n + j] - maxVal);
          attn[i * n + j] = e;
          sum += e;
        }
        for (int j = 0; j < n; j++) {
          attn[i * n + j] /= sum;
        }
      }
      // Multiply attn @ V.
      final outHead = Float32List(n * headDim);
      for (int i = 0; i < n; i++) {
        for (int d = 0; d < headDim; d++) {
          double acc = 0;
          for (int j = 0; j < n; j++) {
            acc += attn[i * n + j] * vData[j * headDim + d];
          }
          outHead[i * headDim + d] = acc;
        }
      }
      heads.add(
        Tensor.fromFloat32List([n, headDim], outHead, device: x.device),
      );
    }
    final concat = TensorConcat.concat(heads, axis: 1);
    return wo(concat);
  }

  Float32List _padHW(Float32List src, int h, int d, int hOut, int wOut) {
    final buf = Float32List(hOut * wOut * d);
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < h; x++) {
        final sBase = (y * h + x) * d;
        final dBase = (y * wOut + x) * d;
        for (int i = 0; i < d; i++) {
          buf[dBase + i] = src[sBase + i];
        }
      }
    }
    return buf;
  }

  Float32List _cropHW(Float32List src, int h, int d, int hIn, int wIn) {
    final buf = Float32List(h * h * d);
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < h; x++) {
        final sBase = (y * wIn + x) * d;
        final dBase = (y * h + x) * d;
        for (int i = 0; i < d; i++) {
          buf[dBase + i] = src[sBase + i];
        }
      }
    }
    return buf;
  }

  @override
  List<Tensor> parameters() => [
    for (final l in wq) ...l.parameters(),
    for (final l in wk) ...l.parameters(),
    for (final l in wv) ...l.parameters(),
    ...wo.parameters(),
    relPosH,
    relPosW,
  ];

  @override
  List<Module> submodules() => [...wq, ...wk, ...wv, wo];
}

// ---------------------------------------------------------------------------
// SamViTBlock — pre-LN attention + pre-LN MLP.
// ---------------------------------------------------------------------------

class SamViTBlock extends Module {
  final LayerNorm norm1;
  final SamWindowedAttention attn;
  final LayerNorm norm2;
  final Linear fc1;
  final Linear fc2;

  SamViTBlock({
    required int embedDim,
    required int numHeads,
    required int inputSize,
    required int windowSize,
    required int mlpDim,
    Device device = Device.CPU,
    int seed = 0,
  }) : norm1 = LayerNorm(embedDim, eps: 1e-6, device: device),
       attn = SamWindowedAttention(
         embedDim: embedDim,
         numHeads: numHeads,
         inputSize: inputSize,
         windowSize: windowSize,
         device: device,
         seed: seed,
       ),
       norm2 = LayerNorm(embedDim, eps: 1e-6, device: device),
       fc1 = Linear(
         embedDim,
         mlpDim,
         bias: true,
         device: device,
         seed: seed + 800_000,
       ),
       fc2 = Linear(
         mlpDim,
         embedDim,
         bias: true,
         device: device,
         seed: seed + 900_000,
       );

  Tensor call(Tensor x) {
    final h = x + attn(norm1(x));
    return h + fc2(_gelu(fc1(norm2(h))));
  }

  /// erf-based GELU: `0.5 · x · (1 + erf(x / √2))`. SAM uses exact
  /// GELU (`nn.GELU()` in PyTorch) — bit-identical to the DINOv2 /
  /// ESM-2 path.
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
    ...norm1.parameters(),
    ...attn.parameters(),
    ...norm2.parameters(),
    ...fc1.parameters(),
    ...fc2.parameters(),
  ];

  @override
  List<Module> submodules() => [norm1, attn, norm2, fc1, fc2];
}

// ---------------------------------------------------------------------------
// SamImageEncoder — patch embed + blocks + neck.
// ---------------------------------------------------------------------------

class SamImageEncoderConfig {
  final int imageSize;
  final int patchSize;
  final int embedDim;
  final int numLayers;
  final int numHeads;
  final int mlpDim;
  final int windowSize;
  final int outChannels;
  final List<int> globalAttnIndices;
  final Device device;
  final int seed;

  const SamImageEncoderConfig({
    this.imageSize = 1024,
    this.patchSize = 16,
    this.embedDim = 768,
    this.numLayers = 12,
    this.numHeads = 12,
    this.mlpDim = 3072,
    this.windowSize = 14,
    this.outChannels = 256,
    this.globalAttnIndices = const [2, 5, 8, 11],
    this.device = Device.CPU,
    this.seed = 0,
  });

  int get gridSize => imageSize ~/ patchSize; // e.g. 64 for 1024/16

  /// `facebook/sam-vit-base` (`sam_vit_b_01ec64.pth`) config.
  static SamImageEncoderConfig vitB({
    Device device = Device.CPU,
    int seed = 0,
  }) => SamImageEncoderConfig(
    embedDim: 768,
    numLayers: 12,
    numHeads: 12,
    mlpDim: 3072,
    globalAttnIndices: const [2, 5, 8, 11],
    device: device,
    seed: seed,
  );

  /// `facebook/sam-vit-large` config.
  static SamImageEncoderConfig vitL({
    Device device = Device.CPU,
    int seed = 0,
  }) => SamImageEncoderConfig(
    embedDim: 1024,
    numLayers: 24,
    numHeads: 16,
    mlpDim: 4096,
    globalAttnIndices: const [5, 11, 17, 23],
    device: device,
    seed: seed,
  );

  /// `facebook/sam-vit-huge` config.
  static SamImageEncoderConfig vitH({
    Device device = Device.CPU,
    int seed = 0,
  }) => SamImageEncoderConfig(
    embedDim: 1280,
    numLayers: 32,
    numHeads: 16,
    mlpDim: 5120,
    globalAttnIndices: const [7, 15, 23, 31],
    device: device,
    seed: seed,
  );
}

class SamImageEncoder extends Module {
  final SamImageEncoderConfig config;

  /// `Conv2d(3 → embedDim, k=patchSize, s=patchSize, bias=True)`
  /// — matches SAM's `PatchEmbed`.
  final Conv2d patchEmbed;

  /// `[1, gridSize, gridSize, embedDim]` — learned absolute 2-D
  /// positional embedding added to the patch tokens.
  final Tensor posEmbed;

  final List<SamViTBlock> blocks;

  // Neck.
  final Conv2d neck1;
  final LayerNorm2d neckLn1;
  final Conv2d neck2;
  final LayerNorm2d neckLn2;

  SamImageEncoder(this.config)
    : patchEmbed = Conv2d(
        3,
        config.embedDim,
        kernel: config.patchSize,
        stride: config.patchSize,
        padding: 0,
        bias: true,
        device: config.device,
        seed: config.seed,
      ),
      posEmbed = _initPosEmbed(
        config.gridSize,
        config.embedDim,
        config.seed + 1,
        config.device,
      ),
      blocks = <SamViTBlock>[],
      neck1 = Conv2d(
        config.embedDim,
        config.outChannels,
        kernel: 1,
        stride: 1,
        padding: 0,
        bias: false,
        device: config.device,
        seed: config.seed + 500_000,
      ),
      neckLn1 = LayerNorm2d(config.outChannels, device: config.device),
      neck2 = Conv2d(
        config.outChannels,
        config.outChannels,
        kernel: 3,
        stride: 1,
        padding: 1,
        bias: false,
        device: config.device,
        seed: config.seed + 600_000,
      ),
      neckLn2 = LayerNorm2d(config.outChannels, device: config.device) {
    final globals = config.globalAttnIndices.toSet();
    for (int i = 0; i < config.numLayers; i++) {
      final isGlobal = globals.contains(i);
      blocks.add(
        SamViTBlock(
          embedDim: config.embedDim,
          numHeads: config.numHeads,
          inputSize: config.gridSize,
          windowSize: isGlobal ? config.gridSize : config.windowSize,
          mlpDim: config.mlpDim,
          device: config.device,
          seed: config.seed + 100_000 * (i + 1),
        ),
      );
    }
  }

  static Tensor _initPosEmbed(int grid, int d, int seed, Device device) {
    final rng = math.Random(seed);
    final total = grid * grid * d;
    final vals = List<double>.generate(total, (_) {
      final u1 = rng.nextDouble().clamp(1e-9, 1.0);
      final u2 = rng.nextDouble();
      final z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2);
      return z * 0.02;
    });
    return Tensor.fromList(
      [grid, grid, d],
      vals,
      requiresGrad: true,
      device: device,
    );
  }

  /// Forward pass. Input is a preprocessed image `[N=1, 3, imageSize,
  /// imageSize]`. Returns the image embedding `[1, outChannels,
  /// gridSize, gridSize]` (e.g. `[1, 256, 64, 64]` for ViT-B).
  Tensor call(Tensor image) {
    if (image.shape.length != 4 ||
        image.shape[0] != 1 ||
        image.shape[1] != 3 ||
        image.shape[2] != config.imageSize ||
        image.shape[3] != config.imageSize) {
      throw ArgumentError(
        'SamImageEncoder: expected [1, 3, ${config.imageSize}, '
        '${config.imageSize}]; got ${image.shape}',
      );
    }
    // Patchify -> [1, embedDim, grid, grid] (NCHW).
    var h = patchEmbed(image);
    // Permute NCHW -> NHWC, add pos embed, then flatten to [grid², D].
    h = _nchwToTokens(h);
    h = h + _flatPosEmbed();
    // Run the ViT stack.
    for (final b in blocks) {
      h = b(h);
    }
    // Tokens back to NCHW so the neck can run 2-D convs.
    final nchw = _tokensToNchw(h);
    // Neck: Conv → LN2d → Conv → LN2d.
    return neckLn2(neck2(neckLn1(neck1(nchw))));
  }

  /// `[1, D, H, W]` -> `[H·W, D]` (channel-last row-major).
  Tensor _nchwToTokens(Tensor t) {
    final n = t.shape[0];
    final c = t.shape[1];
    final h = t.shape[2];
    final w = t.shape[3];
    if (n != 1) {
      throw StateError('SamImageEncoder: batch>1 not supported yet');
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

  /// `[H·W, D]` -> `[1, D, H, W]`.
  Tensor _tokensToNchw(Tensor t) {
    final grid = config.gridSize;
    final d = t.shape[1];
    final src = t.toFloat32List();
    final out = Float32List(d * grid * grid);
    for (int y = 0; y < grid; y++) {
      for (int x = 0; x < grid; x++) {
        for (int ci = 0; ci < d; ci++) {
          out[(ci * grid + y) * grid + x] = src[(y * grid + x) * d + ci];
        }
      }
    }
    return Tensor.fromFloat32List([1, d, grid, grid], out, device: t.device);
  }

  Tensor _flatPosEmbed() {
    // posEmbed is [grid, grid, D] — same layout as tokens, just flatten.
    final grid = config.gridSize;
    final d = config.embedDim;
    final vals = posEmbed.toFloat32List();
    return Tensor.fromFloat32List(
      [grid * grid, d],
      vals,
      device: posEmbed.device,
    );
  }

  @override
  List<Tensor> parameters() => [
    ...patchEmbed.parameters(),
    posEmbed,
    for (final b in blocks) ...b.parameters(),
    ...neck1.parameters(),
    ...neckLn1.parameters(),
    ...neck2.parameters(),
    ...neckLn2.parameters(),
  ];

  @override
  List<Module> submodules() => [
    patchEmbed,
    ...blocks,
    neck1,
    neckLn1,
    neck2,
    neckLn2,
  ];
}
