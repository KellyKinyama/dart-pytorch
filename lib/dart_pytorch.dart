/// Public entry point for the `dart_pytorch` package.
///
/// Re-exports the GPU-backed [Tensor] and its `matmul` extension.
/// The CUDA library `libmat_mul.so` is loaded lazily by the first
/// tensor factory call; see the README for build instructions.
library;

export 'core/tensor/tensor.dart';
export 'core/nn/module.dart';
export 'core/nn/layer_norm.dart';
export 'core/nn/embedding.dart';
export 'core/nn/linear.dart';
export 'core/nn/dropout.dart';
export 'core/nn/multi_head_attention.dart';
export 'core/nn/transformer.dart';
export 'core/nn/positional.dart';
export 'core/nn/masks.dart';
export 'core/nn/transformer_encoder.dart';
export 'core/nn/transformer_lm.dart';
export 'core/optim/optimizer.dart';
export 'core/optim/sgd.dart';
export 'core/optim/adam.dart';
export 'core/optim/grad_utils.dart';

/// Trivial sanity function used by the smoke test.
int calculate() {
  return 6 * 7;
}
