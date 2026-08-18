/// Single-layer LSTM cell.
///
/// Standard PyTorch semantics:
///
///   gates = x @ W_ih^T + b_ih + h @ W_hh^T + b_hh          # [B, 4H]
///   i, f, g, o = split(gates, 4)                           # each [B, H]
///   i_t = sigmoid(i), f_t = sigmoid(f)
///   g_t = tanh(g),    o_t = sigmoid(o)
///   c_t = f_t * c_{t-1} + i_t * g_t
///   h_t = o_t * tanh(c_t)
///
/// Gate ordering (i, f, g, o) matches `torch.nn.LSTMCell`. Weights
/// are stored in PyTorch's `[4*hidden, input]` and `[4*hidden, hidden]`
/// layouts, transposed on load into the `[input, 4*hidden]` and
/// `[hidden, 4*hidden]` matmul layouts.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../tensor/tensor.dart';
import 'module.dart';

class LSTMCell extends Module {
  final int inputSize;
  final int hiddenSize;

  /// `[inputSize, 4*hiddenSize]` — already transposed from PyTorch's
  /// `weight_ih` shape `[4*hidden, input]`.
  late Tensor weightIhT;

  /// `[hiddenSize, 4*hiddenSize]` — transposed from `weight_hh`.
  late Tensor weightHhT;

  /// `[1, 4*hiddenSize]` — sum of PyTorch's `bias_ih + bias_hh` for a
  /// small speed win (they're always added together in the gate sum).
  late Tensor biasSum;

  LSTMCell(this.inputSize, this.hiddenSize) {
    final g = 4 * hiddenSize;
    weightIhT = Tensor.fill([inputSize, g], 0.0, requiresGrad: true);
    weightHhT = Tensor.fill([hiddenSize, g], 0.0, requiresGrad: true);
    biasSum = Tensor.fill([1, g], 0.0, requiresGrad: true);
  }

  /// Load PyTorch-style weights: `weight_ih` `[4H, I]`, `weight_hh`
  /// `[4H, H]`, `bias_ih`/`bias_hh` each `[4H]`.
  void loadFromPytorch({
    required Float32List weightIh,
    required Float32List weightHh,
    required Float32List biasIh,
    required Float32List biasHh,
  }) {
    final g = 4 * hiddenSize;
    if (weightIh.length != g * inputSize) {
      throw ArgumentError(
        'weightIh length ${weightIh.length} != $g × $inputSize',
      );
    }
    if (weightHh.length != g * hiddenSize) {
      throw ArgumentError(
        'weightHh length ${weightHh.length} != $g × $hiddenSize',
      );
    }
    if (biasIh.length != g || biasHh.length != g) {
      throw ArgumentError('biases must each have length $g');
    }

    final wIhT = Float32List(inputSize * g);
    for (int o = 0; o < g; o++) {
      for (int i = 0; i < inputSize; i++) {
        wIhT[i * g + o] = weightIh[o * inputSize + i];
      }
    }
    final wHhT = Float32List(hiddenSize * g);
    for (int o = 0; o < g; o++) {
      for (int h = 0; h < hiddenSize; h++) {
        wHhT[h * g + o] = weightHh[o * hiddenSize + h];
      }
    }
    final bSum = Float32List(g);
    for (int i = 0; i < g; i++) {
      bSum[i] = biasIh[i] + biasHh[i];
    }

    weightIhT = Tensor.fromFloat32List(
      [inputSize, g],
      wIhT,
      device: Device.CPU,
      requiresGrad: weightIhT.requiresGrad,
    );
    weightHhT = Tensor.fromFloat32List(
      [hiddenSize, g],
      wHhT,
      device: Device.CPU,
      requiresGrad: weightHhT.requiresGrad,
    );
    biasSum = Tensor.fromFloat32List(
      [1, g],
      bSum,
      device: Device.CPU,
      requiresGrad: biasSum.requiresGrad,
    );
  }

  /// Forward one timestep.
  ///
  /// Inputs:
  ///   x: `[batch, inputSize]`
  ///   h: `[batch, hiddenSize]`
  ///   c: `[batch, hiddenSize]`
  /// Returns `(hNext, cNext)` both `[batch, hiddenSize]`.
  ({Tensor h, Tensor c}) call(Tensor x, Tensor h, Tensor c) {
    final b = x.shape[0];
    if (x.shape[1] != inputSize) {
      throw ArgumentError('x last dim != inputSize');
    }
    if (h.shape[0] != b || c.shape[0] != b) {
      throw ArgumentError('h/c batch dim mismatch');
    }
    if (h.shape[1] != hiddenSize || c.shape[1] != hiddenSize) {
      throw ArgumentError('h/c last dim != hiddenSize');
    }

    final gates = x.matmul(weightIhT) + h.matmul(weightHhT) + biasSum;
    final gatesData = gates.toFloat32List();
    final g = 4 * hiddenSize;

    final iOut = Float32List(b * hiddenSize);
    final fOut = Float32List(b * hiddenSize);
    final gOut = Float32List(b * hiddenSize);
    final oOut = Float32List(b * hiddenSize);
    for (int bi = 0; bi < b; bi++) {
      final rowBase = bi * g;
      for (int j = 0; j < hiddenSize; j++) {
        iOut[bi * hiddenSize + j] = _sigmoid(gatesData[rowBase + j]);
        fOut[bi * hiddenSize + j] = _sigmoid(
          gatesData[rowBase + hiddenSize + j],
        );
        gOut[bi * hiddenSize + j] = _tanh(
          gatesData[rowBase + 2 * hiddenSize + j],
        );
        oOut[bi * hiddenSize + j] = _sigmoid(
          gatesData[rowBase + 3 * hiddenSize + j],
        );
      }
    }

    final cPrev = c.toFloat32List();
    final cNextData = Float32List(b * hiddenSize);
    final hNextData = Float32List(b * hiddenSize);
    for (int i = 0; i < b * hiddenSize; i++) {
      cNextData[i] = fOut[i] * cPrev[i] + iOut[i] * gOut[i];
      hNextData[i] = oOut[i] * _tanh(cNextData[i]);
    }

    final cNext = Tensor.fromFloat32List(
      [b, hiddenSize],
      cNextData,
      device: Device.CPU,
    );
    final hNext = Tensor.fromFloat32List(
      [b, hiddenSize],
      hNextData,
      device: Device.CPU,
    );
    return (h: hNext, c: cNext);
  }

  static double _sigmoid(double x) {
    if (x >= 0) {
      final z = math.exp(-x);
      return 1.0 / (1.0 + z);
    }
    final z = math.exp(x);
    return z / (1.0 + z);
  }

  static double _tanh(double x) {
    if (x > 20) return 1.0;
    if (x < -20) return -1.0;
    final e2 = math.exp(2 * x);
    return (e2 - 1) / (e2 + 1);
  }

  @override
  List<Tensor> parameters() => [weightIhT, weightHhT, biasSum];
}
