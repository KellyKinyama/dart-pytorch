/// `nn.Dropout` — stochastic zeroing of activations during training.
///
/// Wraps [Tensor.dropout] with a `training` flag inherited from
/// [Module], so `dropout.eval()` disables it (identity pass-through).
library;

import 'dart:math' as math;

import '../tensor/tensor.dart';
import 'module.dart';

class Dropout extends Module {
  /// Fraction of activations to zero out. Must be in `[0, 1)`.
  final double p;

  /// Optional RNG for reproducible masks. When `null`, uses a fresh
  /// `math.Random()` each forward.
  final math.Random? rng;

  Dropout(this.p, {this.rng}) {
    if (p < 0.0 || p >= 1.0) {
      throw ArgumentError('Dropout: p must be in [0, 1); got $p');
    }
  }

  Tensor call(Tensor x) => x.dropout(p, training: training, rng: rng);

  @override
  List<Tensor> parameters() => const [];
}
