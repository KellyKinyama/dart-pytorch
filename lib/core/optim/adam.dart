/// Adam optimizer (Kingma & Ba, 2015) with bias correction.
///
/// Update rule (per parameter, per step `t`, starting at `t = 1`):
///
///     m <- beta1 * m + (1 - beta1) * g
///     v <- beta2 * v + (1 - beta2) * g * g
///     m_hat <- m / (1 - beta1^t)
///     v_hat <- v / (1 - beta2^t)
///     p <- p - lr * m_hat / (sqrt(v_hat) + eps)
///
/// `weightDecay > 0` adds a decoupled L2 term (`lr * weightDecay * p`,
/// AdamW-style — kept out of the moment estimates).
library;

import '../tensor/tensor.dart';
import 'optimizer.dart';

class Adam extends Optimizer {
  @override
  double lr;
  final double beta1;
  final double beta2;
  final double eps;
  final double weightDecay;

  int _step = 0;
  final List<Tensor?> _m;
  final List<Tensor?> _v;

  Adam(
    super.parameters, {
    this.lr = 1e-3,
    this.beta1 = 0.9,
    this.beta2 = 0.999,
    this.eps = 1e-8,
    this.weightDecay = 0.0,
  }) : _m = List<Tensor?>.filled(parameters.length, null),
       _v = List<Tensor?>.filled(parameters.length, null);

  @override
  void step() {
    _step += 1;
    final biasCorr1 = 1.0 - _pow(beta1, _step);
    final biasCorr2 = 1.0 - _pow(beta2, _step);

    for (int i = 0; i < parameters.length; i++) {
      final p = parameters[i];
      final g = p.grad;
      if (g == null) continue;

      // Lazily allocate moment buffers to match parameter shape/device.
      var m = _m[i];
      var v = _v[i];
      if (m == null) {
        m = Tensor.fill(p.shape, 0.0, device: p.device);
        _m[i] = m;
      }
      if (v == null) {
        v = Tensor.fill(p.shape, 0.0, device: p.device);
        _v[i] = v;
      }

      m.assign(m * beta1 + g * (1.0 - beta1));
      v.assign(v * beta2 + (g * g) * (1.0 - beta2));

      final mHat = m / biasCorr1;
      final vHat = v / biasCorr2;

      // sqrt(vHat) via pow(0.5), then +eps in denom.
      final denom = vHat.pow(0.5) + eps;
      var update = (mHat / denom) * lr;

      if (weightDecay != 0.0) {
        update = update + p.detach() * (lr * weightDecay);
      }

      p.assign(p.detach() - update);
    }
  }

  static double _pow(double base, int exp) {
    var r = 1.0;
    var b = base;
    var e = exp;
    while (e > 0) {
      if ((e & 1) == 1) r *= b;
      b *= b;
      e >>= 1;
    }
    return r;
  }
}
