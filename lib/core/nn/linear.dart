/// Linear (affine) layer — `y = x @ W.T + b`.
///
/// Weight shape `[outFeatures, inFeatures]` (same convention as
/// PyTorch's `nn.Linear`). Bias is optional — if `bias == false`, the
/// module trains `W` only.
///
/// Currently uses row-broadcast add for the bias, which restricts GPU
/// support to what's already implemented in `add_tensor_row_broadcast`.
/// For CPU (default for small tensors), all shapes are supported.
library;

import 'dart:math' as math;

import '../tensor/tensor.dart';
import 'module.dart';

class Linear extends Module {
  final int inFeatures;
  final int outFeatures;
  final Tensor weight;
  final Tensor? bias;

  /// Weight init: Kaiming-uniform with `a = sqrt(5)` (matches PyTorch's
  /// default). Bias, when present, uses uniform in `[-1/sqrt(fan_in),
  /// +1/sqrt(fan_in)]`.
  Linear(
    this.inFeatures,
    this.outFeatures, {
    bool bias = true,
    Device device = Device.CPU,
    int seed = 0,
  }) : weight = _initWeight(inFeatures, outFeatures, device, seed),
       bias = bias
           ? _initBias(inFeatures, outFeatures, device, seed + 1)
           : null;

  static Tensor _initWeight(int inF, int outF, Device device, int seed) {
    final rng = math.Random(seed);
    // Kaiming-uniform with a = sqrt(5): bound = sqrt(6 / ((1+5) * fan_in))
    //   = sqrt(1 / fan_in).
    final bound = 1.0 / math.sqrt(inF);
    final vals = List<double>.generate(
      outF * inF,
      (_) => (rng.nextDouble() * 2 - 1) * bound,
    );
    return Tensor.fromList(
      [outF, inF],
      vals,
      requiresGrad: true,
      device: device,
    );
  }

  static Tensor _initBias(int inF, int outF, Device device, int seed) {
    final rng = math.Random(seed);
    final bound = 1.0 / math.sqrt(inF);
    final vals = List<double>.generate(
      outF,
      (_) => (rng.nextDouble() * 2 - 1) * bound,
    );
    return Tensor.fromList([1, outF], vals, requiresGrad: true, device: device);
  }

  /// Forward: `x @ W.T + b`. `x` shape `[..., inFeatures]`, returns
  /// `[..., outFeatures]`. Rank-3 or higher input is handled by
  /// reshaping the leading dims into a single row axis and back.
  Tensor call(Tensor x) {
    if (x.shape.isEmpty || x.shape.last != inFeatures) {
      throw ArgumentError(
        'Linear: expected input [..., $inFeatures]; got ${x.shape}',
      );
    }
    if (x.shape.length > 2) {
      final rows = x.length ~/ inFeatures;
      final original = List<int>.of(x.shape);
      final flat = x.reshape([rows, inFeatures]);
      var y = flat.matmul(weight.transpose());
      if (bias != null) {
        y = y + bias!;
      }
      return y.reshape([...original.sublist(0, original.length - 1), outFeatures]);
    }
    if (x.shape.length != 2) {
      throw ArgumentError(
        'Linear: expected input [..., $inFeatures]; got ${x.shape}',
      );
    }
    var y = x.matmul(weight.transpose());
    if (bias != null) {
      y = y + bias!;
    }
    return y;
  }

  @override
  List<Tensor> parameters() => bias == null ? [weight] : [weight, bias!];
}
