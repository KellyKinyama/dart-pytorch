/// Embedding module — trainable table of shape `[V, D]`.
library;

import 'dart:math' as math;

import '../tensor/tensor.dart';
import 'module.dart';

class Embedding extends Module {
  final int numEmbeddings;
  final int embeddingDim;
  final Tensor weight;

  /// Constructs a table of shape `[numEmbeddings, embeddingDim]`
  /// initialized with a small-scale normal (std = 1/sqrt(dim)) — this
  /// keeps output activations roughly unit-scaled before the first
  /// forward pass through the rest of the model.
  Embedding(
    this.numEmbeddings,
    this.embeddingDim, {
    Device device = Device.CPU,
    int seed = 0,
  }) : weight = _initWeight(numEmbeddings, embeddingDim, device, seed);

  static Tensor _initWeight(int v, int d, Device device, int seed) {
    final rng = math.Random(seed);
    final std = 1.0 / math.sqrt(d);
    final vals = List<double>.generate(v * d, (_) => _gauss(rng) * std);
    return Tensor.fromList([v, d], vals, requiresGrad: true, device: device);
  }

  static double _gauss(math.Random rng) {
    // Box-Muller — cheap and stateless.
    final u1 = rng.nextDouble().clamp(1e-9, 1.0);
    final u2 = rng.nextDouble();
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2);
  }

  Tensor call(Tensor indices) => weight.embedding(indices);

  @override
  List<Tensor> parameters() => [weight];
}
