/// NCHW utilities for CNN backbones (FaceNet, future ResNet ports).
///
///   * `catChannels([t1, t2, ...])` — concat `[N, C_i, H, W]` tensors
///     along the channel axis. Inception blocks fan out into 2-4
///     parallel branches and rejoin here.
///   * `l2NormalizeRows(x, eps)` — row-wise L2 normalize a `[B, D]`
///     tensor. Composed from device-native ops (`.pow`, matmul with
///     a summing vector, `.sqrt`), so it stays on the input's device
///     and preserves autograd.
library;

import 'dart:typed_data';

import '../../tensor/tensor.dart';

/// Concatenate `[N, C_i, H, W]` tensors along the channel axis.
/// All inputs must share `N`, `H`, `W` and device.
Tensor catChannels(List<Tensor> parts) {
  if (parts.isEmpty) {
    throw ArgumentError('catChannels: empty input list');
  }
  final first = parts.first;
  if (first.shape.length != 4) {
    throw ArgumentError(
      'catChannels: expected [N, C, H, W]; got ${first.shape}',
    );
  }
  final n = first.shape[0];
  final h = first.shape[2];
  final w = first.shape[3];
  final device = first.device;
  var cTotal = 0;
  for (final p in parts) {
    if (p.shape.length != 4 ||
        p.shape[0] != n ||
        p.shape[2] != h ||
        p.shape[3] != w) {
      throw ArgumentError(
        'catChannels: shape mismatch — got ${p.shape}, expected [N=$n, *, H=$h, W=$w]',
      );
    }
    if (p.device != device) {
      throw ArgumentError(
        'catChannels: device mismatch — got ${p.device}, expected $device',
      );
    }
    cTotal += p.shape[1];
  }
  final srcs = <Float32List>[];
  final cs = <int>[];
  for (final p in parts) {
    srcs.add(p.toFloat32List());
    cs.add(p.shape[1]);
  }
  final out = Float32List(n * cTotal * h * w);
  final spatial = h * w;
  for (int ni = 0; ni < n; ni++) {
    int cOff = 0;
    for (int i = 0; i < parts.length; i++) {
      final c = cs[i];
      final srcBase = ni * c * spatial;
      final dstBase = (ni * cTotal + cOff) * spatial;
      out.setRange(dstBase, dstBase + c * spatial, srcs[i], srcBase);
      cOff += c;
    }
  }
  return Tensor.fromFloat32List([n, cTotal, h, w], out, device: device);
}

/// Row-wise L2 normalize a `[B, D]` tensor:
///   `y[i] = x[i] / sqrt(sum(x[i]²) + eps)`.
///
/// Composed from device-native ops so the whole path stays on
/// `x.device` and remains autograd-friendly (matmul + pow + sqrt +
/// div all propagate gradients).
Tensor l2NormalizeRows(Tensor x, {double eps = 1e-10}) {
  if (x.shape.length != 2) {
    throw ArgumentError('l2NormalizeRows: expected [B, D]; got ${x.shape}');
  }
  final d = x.shape[1];
  // sum(x²) per row = x² @ 1_D  → [B, 1].
  final sq = x.pow(2);
  final ones = Tensor.fill([d, 1], 1.0, device: x.device);
  final sumSq = sq.matmul(ones); // [B, 1]
  final norm = (sumSq + eps).pow(0.5); // [B, 1]
  // Broadcast [B, 1] over [B, D] via matmul on a 1×D ones-row.
  // Simpler: element-wise div supports [B, D] / [B, 1] via row-broadcast?
  // We don't rely on that: instead, expand norm to [B, D] via matmul.
  final onesRow = Tensor.fill([1, d], 1.0, device: x.device);
  final normRow = norm.matmul(onesRow); // [B, D]
  return x / normRow;
}

/// Convenience for the single-face case: `[D] -> [1, D]`, normalize,
/// return `[1, D]`. Same code path as [l2NormalizeRows] but skips the
/// caller-side reshape gymnastics.
Tensor l2NormalizeVector(Tensor v, {double eps = 1e-10}) {
  if (v.shape.length == 1) {
    return l2NormalizeRows(v.reshape([1, v.shape[0]]), eps: eps);
  }
  return l2NormalizeRows(v, eps: eps);
}
