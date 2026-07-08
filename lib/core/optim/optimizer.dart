/// Base class for parameter optimizers.
///
/// An optimizer owns references to the trainable [Tensor]s of a model
/// and updates them in place via [step] using their accumulated grads.
/// It never participates in the autograd graph: updates are computed
/// with detached tensors (so no `_backward` closures are attached) and
/// then swapped into each parameter via `Tensor.assign`.
library;

import '../tensor/tensor.dart';

abstract class Optimizer {
  /// The parameters this optimizer updates. Order is stable.
  final List<Tensor> parameters;

  Optimizer(this.parameters);

  /// Apply one update step using each parameter's current `.grad`.
  /// Parameters with `grad == null` are skipped.
  void step();

  /// Zero the gradient buffer of every registered parameter.
  void zeroGrad() {
    for (final p in parameters) {
      p.zeroGrad();
    }
  }
}
