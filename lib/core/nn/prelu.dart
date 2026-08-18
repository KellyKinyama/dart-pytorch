/// Parametric ReLU: `y = max(x, 0) + slope[c] * min(x, 0)`, with a
/// learnable per-channel slope. Used by MTCNN's P/R/O nets (all
/// convs are Conv2d + PReLU + optional pool).
///
/// Implemented as a host-side scatter so it works on CPU or GPU input
/// (round-trips through Float32List and returns on the same device).
library;

import 'dart:typed_data';

import '../tensor/tensor.dart';
import 'module.dart';

class PReLU extends Module {
  final int numChannels;
  final Tensor weight; // per-channel slope, shape [numChannels]

  PReLU(this.numChannels, {Device device = Device.CPU})
    : weight = Tensor.fill(
        [numChannels],
        0.25,
        requiresGrad: true,
        device: device,
      );

  Tensor call(Tensor x) {
    if (x.shape.isEmpty) {
      throw ArgumentError('PReLU: got empty shape');
    }
    // Accept [N, C], [N, C, L], [N, C, H, W].
    final data = x.toFloat32List();
    final slopes = weight.toFloat32List();
    final out = Float32List(data.length);
    final c = x.shape.length >= 2 ? x.shape[1] : numChannels;
    if (c != numChannels) {
      throw ArgumentError(
        'PReLU: channel mismatch — got $c, expected $numChannels',
      );
    }
    // Spatial size = total / (N * C). Batch size unused directly — the
    // loop only needs a per-channel slope index.
    final spatial = x.shape.length <= 2
        ? 1
        : x.shape.sublist(2).fold<int>(1, (a, b) => a * b);
    final chStride = spatial;
    for (int idx = 0; idx < data.length; idx++) {
      final v = data[idx];
      if (v >= 0) {
        out[idx] = v;
      } else {
        final ci = (idx ~/ chStride) % numChannels;
        out[idx] = slopes[ci] * v;
      }
    }
    return Tensor.fromFloat32List(x.shape, out, device: x.device);
  }

  @override
  List<Tensor> parameters() => [weight];
}
