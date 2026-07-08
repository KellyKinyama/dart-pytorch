/// Stochastic gradient descent with optional heavy-ball momentum.
///
/// Update rule (per parameter, when `momentum > 0`):
///
///     v <- momentum * v + grad
///     p <- p - lr * v
///
/// When `momentum == 0`, the velocity buffer is skipped and the update
/// reduces to `p <- p - lr * grad`. `weightDecay > 0` adds an L2
/// penalty `weightDecay * p` to the gradient before the velocity step.
library;

import '../tensor/tensor.dart';
import 'optimizer.dart';

class SGD extends Optimizer {
  final double lr;
  final double momentum;
  final double weightDecay;

  /// Per-parameter velocity buffers, allocated lazily on the first step
  /// that sees a non-null grad (so parameters that never receive a
  /// gradient cost nothing).
  final List<Tensor?> _velocity;

  SGD(
    super.parameters, {
    required this.lr,
    this.momentum = 0.0,
    this.weightDecay = 0.0,
  }) : _velocity = List<Tensor?>.filled(parameters.length, null);

  @override
  void step() {
    for (int i = 0; i < parameters.length; i++) {
      final p = parameters[i];
      final g = p.grad;
      if (g == null) continue;

      // Effective gradient with optional weight decay.
      Tensor eff = g;
      if (weightDecay != 0.0) {
        eff = g + p.detach() * weightDecay;
      }

      Tensor update;
      if (momentum != 0.0) {
        var v = _velocity[i];
        if (v == null) {
          v = eff.clone();
          _velocity[i] = v;
        } else {
          v.assign(v * momentum + eff);
        }
        update = v * lr;
      } else {
        update = eff * lr;
      }

      p.assign(p.detach() - update);
    }
  }
}
