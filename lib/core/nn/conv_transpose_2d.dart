/// 2D transposed convolution — inference-oriented, matmul + col2im scatter.
///
/// Also known as "deconvolution" or "fractionally-strided convolution".
/// Takes NCHW input `[N, Cin, H, W]` and produces `[N, Cout, Hout, Wout]`
/// where
///
///   Hout = (H - 1) * stride - 2 * padding + kernelH + outputPadding
///
/// (analogous for Wout). Same PyTorch `nn.ConvTranspose2d` semantics —
/// with the same weight layout, `[Cin, Cout, Kh, Kw]` (**inverse** of
/// `Conv2d`, which uses `[Cout, Cin, Kh, Kw]`).
///
/// Used by U-Net-style upsamplers (DDPM), SAM's mask decoder, and TTS
/// vocoders (HiFi-GAN, BigVGAN). Together with `Conv2d`, this is the
/// full 2-D convolution surface needed for those models.
///
/// Implementation:
///
///   1. Permute input NCHW → `[N*H*W, Cin]`.
///   2. Matmul against `weight.reshape([Cin, Cout*Kh*Kw])`, so the
///      dense compute runs on `weight.device` (CPU **or** GPU).
///   3. Host-side col2im scatter of the `[N*H*W, Cout*Kh*Kw]` result
///      into an `[N, Cout, Hout, Wout]` NCHW buffer, adding
///      overlapping-stride contributions.
///   4. Add optional per-channel bias `[Cout]`.
///
/// Padding is subtracted from the scatter destination (unlike `Conv2d`
/// where it pads the input) — that mirrors PyTorch's convention that
/// `ConvTranspose2d(padding=p)` removes `p` rows/cols from each side of
/// the raw output. `outputPadding` **adds** rows/cols to the right/bottom
/// only (never involved in the scatter loop bounds — it just enlarges
/// the destination), matching PyTorch.
///
/// `groups` and `dilation` are not supported.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../tensor/tensor.dart';
import 'module.dart';

class ConvTranspose2d extends Module {
  final int inChannels;
  final int outChannels;
  final int kernelH;
  final int kernelW;
  final int stride;
  final int paddingH;
  final int paddingW;
  final int outputPaddingH;
  final int outputPaddingW;

  /// Weight shape: `[Cin, Cout, Kh, Kw]` — same as PyTorch
  /// `nn.ConvTranspose2d.weight`.
  final Tensor weight;

  /// Optional per-output-channel bias, shape `[Cout]`.
  final Tensor? bias;

  ConvTranspose2d(
    this.inChannels,
    this.outChannels, {
    int kernel = 3,
    int? kernelH,
    int? kernelW,
    this.stride = 1,
    int padding = 0,
    int? paddingH,
    int? paddingW,
    int outputPadding = 0,
    int? outputPaddingH,
    int? outputPaddingW,
    bool bias = true,
    Device device = Device.CPU,
    int seed = 0,
  }) : kernelH = kernelH ?? kernel,
       kernelW = kernelW ?? kernel,
       paddingH = paddingH ?? padding,
       paddingW = paddingW ?? padding,
       outputPaddingH = outputPaddingH ?? outputPadding,
       outputPaddingW = outputPaddingW ?? outputPadding,
       weight = _initWeight(
         inChannels,
         outChannels,
         kernelH ?? kernel,
         kernelW ?? kernel,
         device,
         seed,
       ),
       bias = bias
           ? Tensor.fill([outChannels], 0.0, requiresGrad: true, device: device)
           : null {
    final opH = outputPaddingH ?? outputPadding;
    final opW = outputPaddingW ?? outputPadding;
    if (opH >= stride || opW >= stride) {
      throw ArgumentError(
        'ConvTranspose2d: outputPadding ($opH,$opW) must be < stride '
        '($stride) — PyTorch enforces the same constraint.',
      );
    }
  }

  static Tensor _initWeight(
    int inC,
    int outC,
    int kh,
    int kw,
    Device device,
    int seed,
  ) {
    final rng = math.Random(seed);
    // PyTorch's default init: uniform in [-k, k] with
    // k = sqrt(1 / (Cout * Kh * Kw)).
    final fanIn = outC * kh * kw;
    final bound = 1.0 / math.sqrt(fanIn);
    final vals = List<double>.generate(
      inC * outC * kh * kw,
      (_) => (rng.nextDouble() * 2 - 1) * bound,
    );
    return Tensor.fromList(
      [inC, outC, kh, kw],
      vals,
      requiresGrad: true,
      device: device,
    );
  }

  int outputHeight(int h) =>
      (h - 1) * stride - 2 * paddingH + kernelH + outputPaddingH;

  int outputWidth(int w) =>
      (w - 1) * stride - 2 * paddingW + kernelW + outputPaddingW;

  Tensor call(Tensor x) {
    if (x.shape.length != 4) {
      throw ArgumentError(
        'ConvTranspose2d: expected [N, Cin, H, W]; got ${x.shape}',
      );
    }
    final n = x.shape[0];
    final cin = x.shape[1];
    final h = x.shape[2];
    final w = x.shape[3];
    if (cin != inChannels) {
      throw ArgumentError(
        'ConvTranspose2d: input channels $cin != declared inChannels '
        '$inChannels',
      );
    }
    final hOut = outputHeight(h);
    final wOut = outputWidth(w);
    if (hOut <= 0 || wOut <= 0) {
      throw ArgumentError(
        'ConvTranspose2d: non-positive output size '
        '${[n, outChannels, hOut, wOut]} for input ${x.shape}, kernel '
        '${[kernelH, kernelW]}, stride $stride, padding '
        '${[paddingH, paddingW]}, outputPadding '
        '${[outputPaddingH, outputPaddingW]}',
      );
    }

    // 1. Permute NCHW -> NHWC and reshape to [N*H*W, Cin] on the
    //    weight's device (so the matmul runs there).
    final rowsIn = _permuteNCHWtoRows(x, n, cin, h, w, weight.device);

    // 2. Reshape weight [Cin, Cout, Kh, Kw] -> [Cin, Cout*Kh*Kw] for
    //    the matmul.
    final wFlat = weight.reshape([inChannels, outChannels * kernelH * kernelW]);
    final scattered = rowsIn.matmul(wFlat);

    // 3. Host-side col2im: scatter [N*H*W, Cout*Kh*Kw] into
    //    [N, Cout, Hout, Wout] with overlapping-stride accumulation.
    final biasList = bias?.toList();
    final out = _col2im(
      scattered,
      n: n,
      h: h,
      w: w,
      hOut: hOut,
      wOut: wOut,
      biasList: biasList,
    );
    return Tensor.fromFloat32List(
      [n, outChannels, hOut, wOut],
      out,
      device: weight.device,
    );
  }

  Tensor _permuteNCHWtoRows(
    Tensor x,
    int n,
    int cin,
    int h,
    int w,
    Device targetDevice,
  ) {
    final data = Tensor.noGrad(() => x.toList());
    final rows = n * h * w;
    final buf = Float32List(rows * cin);
    for (int ni = 0; ni < n; ni++) {
      for (int c = 0; c < cin; c++) {
        final srcBase = (ni * cin + c) * h * w;
        for (int y = 0; y < h; y++) {
          for (int xi = 0; xi < w; xi++) {
            final row = (ni * h + y) * w + xi;
            buf[row * cin + c] = data[srcBase + y * w + xi];
          }
        }
      }
    }
    return Tensor.fromList([rows, cin], buf, device: targetDevice);
  }

  Float32List _col2im(
    Tensor scattered, {
    required int n,
    required int h,
    required int w,
    required int hOut,
    required int wOut,
    List<double>? biasList,
  }) {
    final data = Tensor.noGrad(() => scattered.toList());
    final out = Float32List(n * outChannels * hOut * wOut);
    // Seed with bias if present (host-side broadcast).
    if (biasList != null) {
      for (int ni = 0; ni < n; ni++) {
        for (int oc = 0; oc < outChannels; oc++) {
          final base = (ni * outChannels + oc) * hOut * wOut;
          final b = biasList[oc];
          for (int i = 0; i < hOut * wOut; i++) {
            out[base + i] = b;
          }
        }
      }
    }
    final kSize = kernelH * kernelW;
    final colStride = outChannels * kSize;
    for (int ni = 0; ni < n; ni++) {
      final nOutBase = ni * outChannels * hOut * wOut;
      for (int ih = 0; ih < h; ih++) {
        for (int iw = 0; iw < w; iw++) {
          final rowBase = ((ni * h + ih) * w + iw) * colStride;
          for (int oc = 0; oc < outChannels; oc++) {
            final ocBase = rowBase + oc * kSize;
            final dstOcBase = nOutBase + oc * hOut * wOut;
            for (int kh = 0; kh < kernelH; kh++) {
              final oh = ih * stride - paddingH + kh;
              if (oh < 0 || oh >= hOut) continue;
              for (int kw = 0; kw < kernelW; kw++) {
                final ow = iw * stride - paddingW + kw;
                if (ow < 0 || ow >= wOut) continue;
                out[dstOcBase + oh * wOut + ow] +=
                    data[ocBase + kh * kernelW + kw];
              }
            }
          }
        }
      }
    }
    return out;
  }

  @override
  List<Tensor> parameters() => [weight, if (bias != null) bias!];
}
