/// 1-D transposed convolution — inference-oriented, matmul + col2im
/// scatter. The 1-D twin of `ConvTranspose2d`.
///
/// Takes NCL input `[N, Cin, L]` and produces `[N, Cout, Lout]` where
///
///   Lout = (L - 1) * stride - 2 * padding + kernel + outputPadding
///
/// Same PyTorch `nn.ConvTranspose1d` semantics — with the same weight
/// layout `[Cin, Cout, K]`.
///
/// Used by TTS vocoders (HiFi-GAN, BigVGAN) whose upsampling stack is
/// a sequence of transposed 1-D convolutions on a `[N, C, T]` mel-like
/// tensor.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../tensor/tensor.dart';
import 'module.dart';

class ConvTranspose1d extends Module {
  final int inChannels;
  final int outChannels;
  final int kernelSize;
  final int stride;
  final int padding;
  final int outputPadding;

  /// Weight shape `[Cin, Cout, K]` — matches PyTorch
  /// `nn.ConvTranspose1d.weight`.
  final Tensor weight;

  /// Optional per-output-channel bias, shape `[Cout]`.
  final Tensor? bias;

  ConvTranspose1d(
    this.inChannels,
    this.outChannels, {
    required this.kernelSize,
    this.stride = 1,
    this.padding = 0,
    this.outputPadding = 0,
    bool bias = true,
    Device device = Device.CPU,
    int seed = 0,
  }) : weight = _initWeight(inChannels, outChannels, kernelSize, device, seed),
       bias = bias
           ? Tensor.fill([outChannels], 0.0, requiresGrad: true, device: device)
           : null {
    if (outputPadding >= stride) {
      throw ArgumentError(
        'ConvTranspose1d: outputPadding ($outputPadding) must be < stride '
        '($stride) — PyTorch enforces the same constraint.',
      );
    }
  }

  static Tensor _initWeight(int inC, int outC, int k, Device device, int seed) {
    final rng = math.Random(seed);
    // PyTorch's default init: uniform in [-k, k] with
    // k = sqrt(1 / (Cout * K)).
    final fanIn = outC * k;
    final bound = 1.0 / math.sqrt(fanIn);
    final vals = List<double>.generate(
      inC * outC * k,
      (_) => (rng.nextDouble() * 2 - 1) * bound,
    );
    return Tensor.fromList(
      [inC, outC, k],
      vals,
      requiresGrad: true,
      device: device,
    );
  }

  int outputLength(int l) =>
      (l - 1) * stride - 2 * padding + kernelSize + outputPadding;

  Tensor call(Tensor x) {
    if (x.shape.length != 3) {
      throw ArgumentError(
        'ConvTranspose1d: expected [N, Cin, L]; got ${x.shape}',
      );
    }
    final n = x.shape[0];
    final cin = x.shape[1];
    final l = x.shape[2];
    if (cin != inChannels) {
      throw ArgumentError(
        'ConvTranspose1d: input channels $cin != declared inChannels '
        '$inChannels',
      );
    }
    final lOut = outputLength(l);
    if (lOut <= 0) {
      throw ArgumentError(
        'ConvTranspose1d: non-positive output length $lOut for input '
        '${x.shape}, kernel $kernelSize, stride $stride, padding '
        '$padding, outputPadding $outputPadding',
      );
    }

    // 1. Permute [N, Cin, L] -> rows [N*L, Cin] on the weight's device.
    final rows = _permuteToRows(x, n, cin, l, weight.device);

    // 2. Reshape weight [Cin, Cout, K] -> [Cin, Cout*K] and matmul.
    final wFlat = weight.reshape([inChannels, outChannels * kernelSize]);
    final scattered = rows.matmul(wFlat);

    // 3. Host-side col2im scatter into [N, Cout, Lout].
    final biasList = bias?.toList();
    final out = _col2im(scattered, n: n, l: l, lOut: lOut, biasList: biasList);
    return Tensor.fromFloat32List(
      [n, outChannels, lOut],
      out,
      device: weight.device,
    );
  }

  Tensor _permuteToRows(Tensor x, int n, int cin, int l, Device targetDevice) {
    final data = Tensor.noGrad(() => x.toList());
    final rows = n * l;
    final buf = Float32List(rows * cin);
    for (int ni = 0; ni < n; ni++) {
      for (int c = 0; c < cin; c++) {
        final srcBase = (ni * cin + c) * l;
        for (int li = 0; li < l; li++) {
          buf[(ni * l + li) * cin + c] = data[srcBase + li];
        }
      }
    }
    return Tensor.fromList([rows, cin], buf, device: targetDevice);
  }

  Float32List _col2im(
    Tensor scattered, {
    required int n,
    required int l,
    required int lOut,
    List<double>? biasList,
  }) {
    final data = Tensor.noGrad(() => scattered.toList());
    final out = Float32List(n * outChannels * lOut);
    if (biasList != null) {
      for (int ni = 0; ni < n; ni++) {
        for (int oc = 0; oc < outChannels; oc++) {
          final base = (ni * outChannels + oc) * lOut;
          final b = biasList[oc];
          for (int i = 0; i < lOut; i++) {
            out[base + i] = b;
          }
        }
      }
    }
    final colStride = outChannels * kernelSize;
    for (int ni = 0; ni < n; ni++) {
      final nOutBase = ni * outChannels * lOut;
      for (int li = 0; li < l; li++) {
        final rowBase = (ni * l + li) * colStride;
        for (int oc = 0; oc < outChannels; oc++) {
          final ocBase = rowBase + oc * kernelSize;
          final dstOcBase = nOutBase + oc * lOut;
          for (int kx = 0; kx < kernelSize; kx++) {
            final ox = li * stride - padding + kx;
            if (ox < 0 || ox >= lOut) continue;
            out[dstOcBase + ox] += data[ocBase + kx];
          }
        }
      }
    }
    return out;
  }

  @override
  List<Tensor> parameters() => [weight, if (bias != null) bias!];
}
