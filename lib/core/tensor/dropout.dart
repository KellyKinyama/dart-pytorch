part of 'tensor.dart';

/// Inverted dropout.
///
/// During training, each element is independently kept with probability
/// `1 - p` and scaled by `1 / (1 - p)` so activation magnitudes are
/// preserved in expectation. In eval mode (or when `p == 0`), returns
/// `this` unchanged.
///
/// Backward is automatic: the op is `x * mask`, where `mask` is a
/// constant tensor (`requiresGrad = false`), so `dX = dOut * mask` via
/// the existing multiply backward — no custom gradient needed.
extension TensorDropout on Tensor {
  Tensor dropout(double p, {bool training = true, math.Random? rng}) {
    if (p < 0.0 || p >= 1.0) {
      throw ArgumentError('dropout: p must be in [0, 1); got $p');
    }
    if (!training || p == 0.0) return this;

    final r = rng ?? math.Random();
    final scale = 1.0 / (1.0 - p);
    final maskVals = Float32List(length);
    for (int i = 0; i < length; i++) {
      maskVals[i] = r.nextDouble() < p ? 0.0 : scale;
    }
    final mask = device == Device.CPU
        ? Tensor._cpu(shape, maskVals)
        : Tensor._gpu(shape, Tensor._uploadToGpu(shape, maskVals));
    return this * mask;
  }
}
