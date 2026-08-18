/// 1-D convolution module for `dart_pytorch`.
///
/// Uses the standard im2col + matmul reduction: reshape the input
/// `[batch, Cin, L]` into columns `[batch * Lout, Cin * K]`, multiply
/// by the transposed weight `[Cin*K, Cout]`, and reshape back to
/// `[batch, Cout, Lout]`.
///
/// CPU-only. Naive Dart loops for im2col; fine for VAD-sized models
/// (kernel ≤ 256, batch=1, L ≤ 512).
///
/// **Autograd caveat.** The im2col and BCL↔BLC permutes bypass the
/// tape (they use `.toFloat32List()` + `Tensor.fromFloat32List`), so
/// gradients don't flow back through them by default. This module
/// is production-ready for **inference**. For training you need to
/// call [call] inside a `Tensor.noGrad(() {...})` block for the
/// permute portions and drive gradients through the matmul only —
/// or replace the permutes with a proper autograd-aware `.permute`
/// once it's added to `dart_pytorch`.
library;

import 'dart:typed_data';

import '../tensor/tensor.dart';
import 'module.dart';

class Conv1d extends Module {
  final int inChannels;
  final int outChannels;
  final int kernelSize;
  final int stride;
  final int padding;
  final Device device;

  /// Kernel weights, stored as `[Cin*K, Cout]` (already transposed
  /// for the matmul fast path). External loaders that receive the
  /// PyTorch `[Cout, Cin, K]` layout should call [loadFromPytorch]
  /// instead of assigning directly.
  late Tensor weightMat;

  /// Optional bias, shape `[1, Cout]`. `null` when `bias: false`.
  late Tensor? biasVec;

  Conv1d({
    required this.inChannels,
    required this.outChannels,
    required this.kernelSize,
    this.stride = 1,
    this.padding = 0,
    bool bias = true,
    this.device = Device.CPU,
  }) {
    weightMat = Tensor.fill(
      [inChannels * kernelSize, outChannels],
      0.0,
      device: device,
      requiresGrad: true,
    );
    biasVec = bias
        ? Tensor.fill([1, outChannels], 0.0, device: device, requiresGrad: true)
        : null;
  }

  /// Load a PyTorch-layout kernel `[Cout, Cin, K]` and (optional) bias
  /// `[Cout]` into this module. Transposes the kernel into the
  /// `[Cin*K, Cout]` matmul layout on the fly, keeping storage on
  /// this module's [device].
  void loadFromPytorch(Float32List kernel, Float32List? bias) {
    final expected = outChannels * inChannels * kernelSize;
    if (kernel.length != expected) {
      throw ArgumentError(
        'Conv1d.loadFromPytorch: expected $expected weight values, '
        'got ${kernel.length}',
      );
    }
    final out = Float32List(inChannels * kernelSize * outChannels);
    for (int o = 0; o < outChannels; o++) {
      for (int c = 0; c < inChannels; c++) {
        for (int k = 0; k < kernelSize; k++) {
          out[(c * kernelSize + k) * outChannels + o] =
              kernel[(o * inChannels + c) * kernelSize + k];
        }
      }
    }
    weightMat = Tensor.fromFloat32List(
      [inChannels * kernelSize, outChannels],
      out,
      device: device,
      requiresGrad: weightMat.requiresGrad,
    );
    if (bias != null) {
      if (biasVec == null) {
        throw StateError('Conv1d: bias provided but ctor was bias:false');
      }
      if (bias.length != outChannels) {
        throw ArgumentError(
          'Conv1d.loadFromPytorch: expected $outChannels bias values, '
          'got ${bias.length}',
        );
      }
      biasVec = Tensor.fromFloat32List(
        [1, outChannels],
        Float32List.fromList(bias),
        device: device,
        requiresGrad: biasVec!.requiresGrad,
      );
    }
  }

  /// Forward pass. Accepts a 3-D tensor `[batch, Cin, L]` (any
  /// device — im2col always runs on CPU, then the packed columns
  /// are uploaded to the weight's device for the matmul). Returns
  /// `[batch, Cout, Lout]` on the same device as the weights.
  Tensor call(Tensor input) {
    if (input.shape.length != 3) {
      throw ArgumentError('Conv1d expects [B, Cin, L]; got ${input.shape}');
    }
    final b = input.shape[0];
    final cin = input.shape[1];
    if (cin != inChannels) {
      throw ArgumentError(
        'Conv1d: expected $inChannels input channels, got $cin',
      );
    }
    final l = input.shape[2];
    final lOut = (l + 2 * padding - kernelSize) ~/ stride + 1;
    if (lOut <= 0) {
      throw ArgumentError(
        'Conv1d: kernel/padding produce non-positive output length',
      );
    }

    final cols = _im2col1d(input, b, l, lOut);
    var out = cols.matmul(weightMat); // [B*Lout, Cout] on `device`
    if (biasVec != null) out = out + biasVec!;

    // Reshape [B*Lout, Cout] -> [B, Cout, Lout] via [B, Lout, Cout] transpose.
    final reshaped = out.reshape([b, lOut, outChannels]);
    return _permuteBLCtoBCL(reshaped, b, lOut, outChannels);
  }

  Tensor _im2col1d(Tensor input, int b, int l, int lOut) {
    final flat = input.toFloat32List();
    final cols = Float32List(b * lOut * inChannels * kernelSize);
    var write = 0;
    for (int bi = 0; bi < b; bi++) {
      final batchBase = bi * inChannels * l;
      for (int t = 0; t < lOut; t++) {
        final start = t * stride - padding;
        for (int c = 0; c < inChannels; c++) {
          final channelBase = batchBase + c * l;
          for (int k = 0; k < kernelSize; k++) {
            final idx = start + k;
            cols[write++] = (idx >= 0 && idx < l)
                ? flat[channelBase + idx]
                : 0.0;
          }
        }
      }
    }
    return Tensor.fromFloat32List(
      [b * lOut, inChannels * kernelSize],
      cols,
      device: device,
    );
  }

  Tensor _permuteBLCtoBCL(Tensor blc, int b, int l, int c) {
    final data = blc.toFloat32List();
    final out = Float32List(b * c * l);
    for (int bi = 0; bi < b; bi++) {
      for (int li = 0; li < l; li++) {
        for (int ci = 0; ci < c; ci++) {
          out[bi * c * l + ci * l + li] = data[bi * l * c + li * c + ci];
        }
      }
    }
    return Tensor.fromFloat32List([b, c, l], out, device: device);
  }

  @override
  List<Tensor> parameters() =>
      biasVec == null ? [weightMat] : [weightMat, biasVec!];
}
