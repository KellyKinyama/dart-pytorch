part of 'tensor.dart';

/// Scratch space passed between an [AutogradFunction]'s forward and
/// backward passes — the analog of PyTorch's `ctx`.
///
/// Use [saveForBackward] to stash the input/output tensors the backward
/// pass needs, and [setData] / [getData] for any non-tensor state (e.g.
/// a scalar scale, a reduction axis).
class AutogradContext {
  List<Tensor> _saved = const [];
  final Map<Object, Object?> _data = {};

  /// Remember `tensors` for use in [AutogradFunction.backward].
  void saveForBackward(List<Tensor> tensors) {
    _saved = List<Tensor>.unmodifiable(tensors);
  }

  /// Tensors previously stashed via [saveForBackward].
  List<Tensor> get savedTensors => _saved;

  /// Store an arbitrary non-tensor value for the backward pass.
  void setData(Object key, Object? value) => _data[key] = value;

  /// Retrieve a value previously stored with [setData].
  Object? getData(Object key) => _data[key];
}

/// Base class for a user-defined differentiable op — the Dart analog of
/// `torch.autograd.Function`.
///
/// Subclass and override [forward] (compute outputs from inputs) and
/// [backward] (map output gradients to input gradients), then call
/// [apply] to run the op and splice it into the autograd tape as a
/// single opaque node.
///
/// The ops used inside [forward] run detached (no internal graph is
/// built), so [backward] is the sole definition of the gradient. This
/// lets a function fuse several primitives yet expose one backward.
///
/// Current limitation: exactly one output tensor is supported.
abstract class AutogradFunction {
  /// Compute the output tensor(s) from `inputs`. Stash anything the
  /// backward pass needs on `ctx`.
  List<Tensor> forward(AutogradContext ctx, List<Tensor> inputs);

  /// Given the gradient flowing into each output, return the gradient
  /// for each input (same order/length as the `inputs` passed to
  /// [apply]; use `null` for inputs that take no gradient).
  List<Tensor?> backward(AutogradContext ctx, List<Tensor> gradOutputs);

  /// Run [forward] and, when any input requires grad, wire [backward]
  /// into the tape so the op participates in `backward()` / `grads()`.
  List<Tensor> apply(List<Tensor> inputs) {
    final ctx = AutogradContext();
    // Detach the forward internals: the custom backward is authoritative.
    final outputs = Tensor.noGrad(() => forward(ctx, inputs));

    final anyReq = inputs.any((t) => t.requiresGrad);
    if (!anyReq || Tensor._noGradDepth > 0) return outputs;

    if (outputs.length != 1) {
      throw UnsupportedError(
        'AutogradFunction.apply currently supports a single output; '
        'got ${outputs.length}.',
      );
    }
    final out = outputs[0];
    out._setBackward(inputs, () {
      final grads = backward(ctx, [out._grad!]);
      if (grads.length != inputs.length) {
        throw StateError(
          'AutogradFunction.backward returned ${grads.length} grads for '
          '${inputs.length} inputs',
        );
      }
      for (int i = 0; i < inputs.length; i++) {
        final gi = grads[i];
        if (gi != null && inputs[i].requiresGrad) {
          inputs[i]._accumulateGrad(
            Tensor._reduceForBroadcast(gi, inputs[i].shape),
          );
        }
      }
    });
    return outputs;
  }
}
