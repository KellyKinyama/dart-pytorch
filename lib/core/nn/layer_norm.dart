/// LayerNorm module — a trainable wrapper around `Tensor.layerNorm`.
library;

import '../tensor/tensor.dart';
import 'module.dart';

class LayerNorm extends Module {
  final int dim;
  final double eps;
  final Tensor gamma;
  final Tensor beta;

  LayerNorm(this.dim, {this.eps = 1e-5, Device device = Device.CPU})
    : gamma = Tensor.fill([dim], 1.0, requiresGrad: true, device: device),
      beta = Tensor.fill([dim], 0.0, requiresGrad: true, device: device);

  Tensor call(Tensor x) => x.layerNorm(gamma, beta, eps: eps);

  @override
  List<Tensor> parameters() => [gamma, beta];
}
