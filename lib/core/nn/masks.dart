/// Attention masks — helpers that build the additive `[N, N]` masks
/// consumed by `scaledDotProductAttention`.
///
/// The convention across the library is *additive* masks: `0` for
/// allowed positions and a large negative value (default `-1e9`) for
/// blocked positions. The mask is added to the pre-softmax scores.
library;

import 'dart:typed_data';

import '../tensor/tensor.dart';

/// Upper-triangular (strict) causal mask of shape `[n, n]`. Position
/// `i` is allowed to attend to positions `j <= i`; entries with `j > i`
/// are set to `blockValue` (default `-1e9`). Non-trainable.
Tensor causalMask(
  int n, {
  double blockValue = -1e9,
  Device device = Device.CPU,
}) {
  final data = Float32List(n * n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      data[i * n + j] = j > i ? blockValue : 0.0;
    }
  }
  return Tensor.fromList([n, n], data, device: device);
}
