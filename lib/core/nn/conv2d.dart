/// 2D convolution — inference-only, im2col + matmul.
///
/// Takes NCHW input `[N, Cin, H, W]` and produces
/// `[N, Cout, Hout, Wout]` where
/// `Hout = (H + 2*pad - Kh) / stride + 1` and analogously for Wout.
///
/// The forward pass unfolds every output position into a
/// `Cin * Kh * Kw` column and stacks all such columns row-wise to
/// get a `[N*Hout*Wout, Cin*Kh*Kw]` matrix. That is multiplied by
/// the pre-transposed weight `[Cin*Kh*Kw, Cout]` and reshaped back to
/// NCHW. The result stays connected to the autograd tape through the
/// underlying matmul, but Conv2d weights are meant to be loaded from
/// pretrained checkpoints (e.g. LC0), not trained in this repo.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../tensor/tensor.dart';
import 'module.dart';

class Conv2d extends Module {
  final int inChannels;
  final int outChannels;
  final int kernelH;
  final int kernelW;
  final int stride;
  final int padding;

  /// Weight shape: `[Cout, Cin, Kh, Kw]` — same as PyTorch nn.Conv2d.
  final Tensor weight;

  /// Optional per-output-channel bias, shape `[Cout]`.
  final Tensor? bias;

  Conv2d(
    this.inChannels,
    this.outChannels, {
    int kernel = 3,
    int? kernelH,
    int? kernelW,
    this.stride = 1,
    this.padding = 0,
    bool bias = true,
    Device device = Device.CPU,
    int seed = 0,
  }) : kernelH = kernelH ?? kernel,
       kernelW = kernelW ?? kernel,
       weight = _initWeight(
         outChannels,
         inChannels,
         kernelH ?? kernel,
         kernelW ?? kernel,
         device,
         seed,
       ),
       bias = bias
           ? Tensor.fill(
               [outChannels],
               0.0,
               requiresGrad: true,
               device: device,
             )
           : null;

  static Tensor _initWeight(
    int outC,
    int inC,
    int kh,
    int kw,
    Device device,
    int seed,
  ) {
    final rng = math.Random(seed);
    final fanIn = inC * kh * kw;
    final bound = 1.0 / math.sqrt(fanIn);
    final vals = List<double>.generate(
      outC * inC * kh * kw,
      (_) => (rng.nextDouble() * 2 - 1) * bound,
    );
    return Tensor.fromList(
      [outC, inC, kh, kw],
      vals,
      requiresGrad: true,
      device: device,
    );
  }

  Tensor call(Tensor x) {
    if (x.shape.length != 4) {
      throw ArgumentError(
        'Conv2d: expected [N, Cin, H, W]; got ${x.shape}',
      );
    }
    final n = x.shape[0];
    final cin = x.shape[1];
    final h = x.shape[2];
    final w = x.shape[3];
    if (cin != inChannels) {
      throw ArgumentError(
        'Conv2d: input channels $cin != declared inChannels $inChannels',
      );
    }
    final hOut = (h + 2 * padding - kernelH) ~/ stride + 1;
    final wOut = (w + 2 * padding - kernelW) ~/ stride + 1;
    if (hOut <= 0 || wOut <= 0) {
      throw ArgumentError(
        'Conv2d: non-positive output size ${[n, outChannels, hOut, wOut]} '
        'for input ${x.shape}, kernel ${[kernelH, kernelW]}, '
        'stride $stride, padding $padding',
      );
    }
    if (x.device != weight.device) {
      throw ArgumentError(
        'Conv2d: input on ${x.device}, weight on ${weight.device}',
      );
    }

    // im2col: expand [N, Cin, H, W] -> [N*Hout*Wout, Cin*Kh*Kw].
    final cols = _im2colCpu(x, n, cin, h, w, hOut, wOut);

    // Weight [Cout, Cin, Kh, Kw] -> [Cout, Cin*Kh*Kw] -> transpose to
    // [Cin*Kh*Kw, Cout] for matmul.
    final wFlat = weight.reshape([outChannels, inChannels * kernelH * kernelW]);
    final wT = wFlat.transpose();

    // [N*Hout*Wout, Cin*Kh*Kw] @ [Cin*Kh*Kw, Cout] -> [N*Hout*Wout, Cout].
    var out = cols.matmul(wT);

    if (bias != null) {
      // Broadcast bias [Cout] across every spatial position.
      final bRow = bias!.reshape([1, outChannels]);
      out = out + bRow;
    }

    // Reshape to [N, Hout, Wout, Cout] then permute to [N, Cout, Hout, Wout].
    final nhwc = out.reshape([n, hOut, wOut, outChannels]);
    return _permuteNHWCtoNCHW(nhwc);
  }

  Tensor _im2colCpu(
    Tensor x,
    int n,
    int cin,
    int h,
    int w,
    int hOut,
    int wOut,
  ) {
    // Only CPU inference supported for now.
    if (x.device != Device.CPU) {
      throw UnsupportedError('Conv2d.im2col: GPU input not supported yet');
    }
    final data = Tensor.noGrad(() => x.toList());
    final kh = kernelH;
    final kw = kernelW;
    final rows = n * hOut * wOut;
    final cols = cin * kh * kw;
    final out = Float32List(rows * cols);
    var rowOff = 0;
    for (int ni = 0; ni < n; ni++) {
      final nBase = ni * cin * h * w;
      for (int y = 0; y < hOut; y++) {
        final yIn0 = y * stride - padding;
        for (int xo = 0; xo < wOut; xo++) {
          final xIn0 = xo * stride - padding;
          var colOff = rowOff;
          for (int c = 0; c < cin; c++) {
            final cBase = nBase + c * h * w;
            for (int ky = 0; ky < kh; ky++) {
              final yIn = yIn0 + ky;
              if (yIn < 0 || yIn >= h) {
                colOff += kw; // zeros
                continue;
              }
              final rowBase = cBase + yIn * w;
              for (int kx = 0; kx < kw; kx++) {
                final xIn = xIn0 + kx;
                if (xIn >= 0 && xIn < w) {
                  out[colOff] = data[rowBase + xIn];
                }
                colOff++;
              }
            }
          }
          rowOff += cols;
        }
      }
    }
    return Tensor.fromList([rows, cols], out, device: Device.CPU);
  }

  Tensor _permuteNHWCtoNCHW(Tensor t) {
    // t: [N, H, W, C] -> [N, C, H, W] via a host-side gather.
    final n = t.shape[0];
    final h = t.shape[1];
    final w = t.shape[2];
    final c = t.shape[3];
    final src = Tensor.noGrad(() => t.toList());
    final out = Float32List(n * c * h * w);
    for (int ni = 0; ni < n; ni++) {
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          final srcBase = ((ni * h + y) * w + x) * c;
          for (int ci = 0; ci < c; ci++) {
            out[((ni * c + ci) * h + y) * w + x] = src[srcBase + ci];
          }
        }
      }
    }
    return Tensor.fromList([n, c, h, w], out, device: t.device);
  }

  @override
  List<Tensor> parameters() => [weight, if (bias != null) bias!];
}
