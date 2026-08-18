/// 2-D pooling for NCHW tensors — inference-friendly, autograd-lite.
///
/// Two ops we need for Inception-ResNet-V1:
///
///   * `MaxPool2d(kernel, stride, padding)` — used once in the FaceNet
///     stem (k=3, s=2, no padding).
///   * `AdaptiveAvgPool2d((1, 1))` — global average pool that collapses
///     `[N, C, H, W]` to `[N, C, 1, 1]` (we return `[N, C]` for the
///     downstream Linear).
///
/// Both are pure functions (no learnable state), so they're free
/// helpers rather than `Module`s. Both run on the host; the input is
/// downloaded via `.toFloat32List()` and the output is pushed back to
/// the same device with `Tensor.fromFloat32List`. That's fine at
/// inference-time and for training the head, since we don't need to
/// propagate gradients through them.
library;

import 'dart:typed_data';

import '../../tensor/tensor.dart';

/// Max-pool `[N, C, H, W]` with kernel `k`, `stride`, `padding` (all
/// spatially symmetric). Output shape `[N, C, Hout, Wout]` where
/// `Hout = (H + 2*p - k) / s + 1` (or ceil-divided if `ceilMode`).
Tensor maxPool2d(
  Tensor x, {
  required int kernel,
  required int stride,
  int padding = 0,
  bool ceilMode = false,
}) {
  if (x.shape.length != 4) {
    throw ArgumentError('maxPool2d: expected [N, C, H, W]; got ${x.shape}');
  }
  final n = x.shape[0];
  final c = x.shape[1];
  final h = x.shape[2];
  final w = x.shape[3];
  int outDim(int inSize) {
    final num = inSize + 2 * padding - kernel;
    if (num < 0) return 0;
    if (ceilMode) {
      return (num + stride - 1) ~/ stride + 1;
    }
    return num ~/ stride + 1;
  }

  final hOut = outDim(h);
  final wOut = outDim(w);
  if (hOut <= 0 || wOut <= 0) {
    throw ArgumentError('maxPool2d: non-positive output ${[n, c, hOut, wOut]}');
  }
  final data = x.toFloat32List();
  final out = Float32List(n * c * hOut * wOut);
  for (int ni = 0; ni < n; ni++) {
    for (int ci = 0; ci < c; ci++) {
      final srcBase = (ni * c + ci) * h * w;
      final dstBase = (ni * c + ci) * hOut * wOut;
      for (int oy = 0; oy < hOut; oy++) {
        final iy0 = oy * stride - padding;
        for (int ox = 0; ox < wOut; ox++) {
          final ix0 = ox * stride - padding;
          double best = -double.infinity;
          for (int ky = 0; ky < kernel; ky++) {
            final iy = iy0 + ky;
            if (iy < 0 || iy >= h) continue;
            for (int kx = 0; kx < kernel; kx++) {
              final ix = ix0 + kx;
              if (ix < 0 || ix >= w) continue;
              final v = data[srcBase + iy * w + ix];
              if (v > best) best = v;
            }
          }
          if (best == -double.infinity) best = 0.0;
          out[dstBase + oy * wOut + ox] = best;
        }
      }
    }
  }
  return Tensor.fromFloat32List([n, c, hOut, wOut], out, device: x.device);
}

/// Global average pool over the spatial axes: `[N, C, H, W] -> [N, C]`.
Tensor globalAvgPool2d(Tensor x) {
  if (x.shape.length != 4) {
    throw ArgumentError(
      'globalAvgPool2d: expected [N, C, H, W]; got ${x.shape}',
    );
  }
  final n = x.shape[0];
  final c = x.shape[1];
  final h = x.shape[2];
  final w = x.shape[3];
  final spatial = h * w;
  final data = x.toFloat32List();
  final out = Float32List(n * c);
  for (int ni = 0; ni < n; ni++) {
    for (int ci = 0; ci < c; ci++) {
      final base = (ni * c + ci) * spatial;
      double s = 0.0;
      for (int k = 0; k < spatial; k++) {
        s += data[base + k];
      }
      out[ni * c + ci] = s / spatial;
    }
  }
  return Tensor.fromFloat32List([n, c], out, device: x.device);
}
