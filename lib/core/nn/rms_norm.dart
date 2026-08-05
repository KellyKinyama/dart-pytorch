/// RMSNorm module — a trainable wrapper around `Tensor.rmsNorm`.
///
/// LayerNorm without the mean-subtract and without the beta bias.
/// Used by Llama, Mistral, Qwen and Phi-3.
library;

import '../tensor/tensor.dart';
import 'module.dart';

class RMSNorm extends Module {
  final int dim;
  final double eps;
  final Tensor gamma;

  RMSNorm(this.dim, {this.eps = 1e-6, Device device = Device.CPU})
    : gamma = Tensor.fill([dim], 1.0, requiresGrad: true, device: device);

  Tensor call(Tensor x) => x.rmsNorm(gamma, eps: eps);

  @override
  List<Tensor> parameters() => [gamma];
}
