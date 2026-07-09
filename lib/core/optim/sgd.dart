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
  @override
  double lr;
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

      // Effective gradient with optional weight decay. Track whether
      // we allocated a fresh `eff` so we can dispose it after use
      // (otherwise `p.detach() * weightDecay` leaks a GPU handle per
      // param per step → OOM in a few dozen steps for mid-size models).
      Tensor eff = g;
      Tensor? effOwned;
      if (weightDecay != 0.0) {
        final pDet = p.detach();
        final wd = pDet * weightDecay;
        pDet.dispose();
        eff = g + wd;
        wd.dispose();
        effOwned = eff;
      }

      Tensor update;
      bool updateOwned = true;
      if (momentum != 0.0) {
        var v = _velocity[i];
        if (v == null) {
          v = eff.clone();
          _velocity[i] = v;
        } else {
          final vMom = v * momentum;
          final vNew = vMom + eff;
          vMom.dispose();
          v.assign(vNew);
        }
        update = v * lr;
      } else {
        update = eff * lr;
      }
      // `eff` is either `g` (do not dispose — owned by param.grad) or
      // a fresh tensor we allocated above.
      effOwned?.dispose();

      final pDet = p.detach();
      final newP = pDet - update;
      pDet.dispose();
      if (updateOwned) update.dispose();
      p.assign(newP);
    }
  }
}
