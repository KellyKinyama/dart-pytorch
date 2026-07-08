/// Base class for stateful neural-network modules.
///
/// Mirrors PyTorch's `nn.Module`: a module bundles trainable
/// parameters and defines a forward pass via `call`. Subclasses expose
/// their trainable tensors via [parameters], which optimizers iterate.
library;

import '../tensor/tensor.dart';

abstract class Module {
  /// Whether this module is in training mode. Layers that behave
  /// differently between training and inference (e.g. `Dropout`) read
  /// this flag in their `call` method. Defaults to training mode.
  bool training = true;

  /// Put this module (and any registered submodules) into training mode.
  void train() {
    training = true;
    for (final m in submodules()) {
      m.train();
    }
  }

  /// Put this module (and any registered submodules) into evaluation mode.
  void eval() {
    training = false;
    for (final m in submodules()) {
      m.eval();
    }
  }

  /// Submodules owned by this module. Subclasses that compose other
  /// modules should override this so `train()` / `eval()` propagate.
  /// Default: empty.
  List<Module> submodules() => const [];

  /// Trainable tensors owned by this module (and its submodules).
  List<Tensor> parameters();

  /// Zero every parameter's gradient. Safe to call before each `backward`.
  void zeroGrad() {
    for (final p in parameters()) {
      p.zeroGrad();
    }
  }
}
