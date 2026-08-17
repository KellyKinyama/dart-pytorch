/// LC0 classical CNN — inference-only.
///
/// Assembles the network described by an [Lc0Weights] value (from
/// [Lc0Reader]) into a forward pass:
///
///   input conv (3x3, 112 -> F filters) + BN + ReLU
///   -> N residual blocks:
///        (3x3 F -> F + BN + ReLU) + (3x3 F -> F + BN + skip + ReLU)
///   -> policy head:  3x3 F -> F + BN + ReLU, then 3x3 F -> P (raw logits)
///   -> value head:   1x1 F -> Vf + BN + ReLU, flatten,
///                    FC (Vf*64 -> valueFCUnits) + ReLU,
///                    FC (valueFCUnits -> W)
///
/// The output is `(policyLogits [1, P, 8, 8], valueOrWdl [1, W])`.
/// If `wdl == 3`, the value tensor is (win, draw, loss) softmax
/// probabilities; if `wdl == 1`, a single scalar in [-1, 1].
///
/// BatchNorm at inference: `y = (x - mean) / std * gamma + beta`,
/// where `mean`, `std`, `gamma`, `beta` are per-channel. LC0's
/// `bn_stddivs` field stores the standard deviation as computed
/// during training (not its inverse), so we divide.
///
/// Everything runs on CPU under `Tensor.noGrad`; forward on a
/// starting position takes a few seconds on a laptop CPU because
/// each 3x3 conv is a im2col + matmul executed in pure Dart.
library;

import 'dart:typed_data';

import '../tensor/tensor.dart';
import 'conv2d.dart';
import 'lc0_proto.dart';
import 'module.dart';

class Lc0Output {
  /// Raw policy logits, shape `[1, policyOutputPlanes, 8, 8]`.
  final Tensor policyLogits;

  /// WDL probs `[1, 3]` (win, draw, loss) if `wdl == 3`, or a
  /// scalar `[1, 1]` value in [-1, 1] if `wdl == 1`.
  final Tensor value;

  const Lc0Output(this.policyLogits, this.value);

  /// Convenience: scalar centipawn-style estimate. For WDL heads,
  /// `p_win - p_loss`; for scalar heads, the raw output.
  double get scalarValue {
    final v = value.toList();
    if (v.length == 3) return v[0] - v[2];
    return v[0];
  }
}

class Lc0Net extends Module {
  final Lc0Weights w;

  final Conv2d inputConv;
  final List<Conv2d> resConvA;
  final List<Conv2d> resConvB;
  final Conv2d policy1;
  final Conv2d policyOut;
  final Conv2d valueConv;

  Lc0Net(this.w)
    : inputConv = _mkConv(w.input, 112, w.filters, 3, 1),
      resConvA = [
        for (final r in w.residual) _mkConv(r.conv1, w.filters, w.filters, 3, 1),
      ],
      resConvB = [
        for (final r in w.residual) _mkConv(r.conv2, w.filters, w.filters, 3, 1),
      ],
      policy1 = _mkConv(w.policy1, w.filters, w.filters, 3, 1),
      policyOut = _mkConv(
        w.policyOut,
        w.filters,
        w.policyOutputPlanes,
        3,
        1,
      ),
      valueConv = _mkConv(w.valueConv, w.filters, w.valueFilters, 1, 0);

  static Conv2d _mkConv(
    Lc0ConvBlock cb,
    int cin,
    int cout,
    int k,
    int pad,
  ) {
    final c = Conv2d(cin, cout, kernel: k, padding: pad, bias: false);
    c.weight.assign(cb.weights);
    // BN below applies the per-channel scale/shift AND folds in any
    // conv bias, so Conv2d itself carries no bias.
    return c;
  }

  /// Runs the whole network on a `[1, 112, 8, 8]` input tensor.
  Lc0Output call(Tensor input) {
    if (input.shape.length != 4 ||
        input.shape[0] != 1 ||
        input.shape[1] != 112 ||
        input.shape[2] != 8 ||
        input.shape[3] != 8) {
      throw ArgumentError(
        'Lc0Net: expected input [1, 112, 8, 8]; got ${input.shape}',
      );
    }

    return Tensor.noGrad(() {
      var h = _bnRelu(inputConv(input), w.input);
      for (int i = 0; i < w.residual.length; i++) {
        final skip = h;
        h = _bnRelu(resConvA[i](h), w.residual[i].conv1);
        h = _applyBn(resConvB[i](h), w.residual[i].conv2);
        h = _addAndRelu(h, skip);
      }
      // Policy head.
      final p1 = _bnRelu(policy1(h), w.policy1);
      final policy = _applyBn(policyOut(p1), w.policyOut);

      // Value head.
      var v = _bnRelu(valueConv(h), w.valueConv);
      // Flatten [1, Vf, 8, 8] -> [1, Vf*64].
      final vf = w.valueFilters;
      v = v.reshape([1, vf * 64]);
      // FC1 with ReLU. LC0 stores W as [out, in], so matmul with W.T.
      v = (v.matmul(w.ip1ValW.transpose()) + w.ip1ValB.reshape([1, w.valueFCUnits]))
          .relu();
      // FC2 -> WDL or scalar.
      v = v.matmul(w.ip2ValW.transpose()) +
          w.ip2ValB.reshape([1, w.wdl]);
      if (w.wdl == 3) v = v.softmax();
      // wdl == 1: leave as raw scalar; caller can .tanh() if it wants.

      return Lc0Output(policy, v);
    });
  }

  Tensor _bnRelu(Tensor x, Lc0ConvBlock cb) {
    return _applyBn(x, cb).relu();
  }

  Tensor _addAndRelu(Tensor a, Tensor b) {
    return (a + b).relu();
  }

  Tensor _applyBn(Tensor x, Lc0ConvBlock cb) {
    // x: [N, C, H, W]; cb.bnMeans/bnStddivs/bnGammas/bnBetas: [C].
    final n = x.shape[0];
    final c = x.shape[1];
    final h = x.shape[2];
    final wSp = x.shape[3];
    if (c != cb.bnMeans.length) {
      throw ArgumentError(
        'BN: channel mismatch — tensor $c, BN vector ${cb.bnMeans.length}',
      );
    }
    final src = x.toList();
    final mean = cb.bnMeans.toList();
    final std = cb.bnStddivs.toList();
    final gamma = cb.bnGammas.toList();
    final beta = cb.bnBetas.toList();
    final biases = cb.biases.toList();
    final out = Float32List(src.length);
    for (int ni = 0; ni < n; ni++) {
      for (int ci = 0; ci < c; ci++) {
        final m = mean[ci];
        final s = std[ci];
        final g = gamma[ci];
        final bt = beta[ci];
        final bi = biases[ci];
        final scale = s == 0 ? 0.0 : (g / s);
        final shift = bt - (m - bi) * scale;
        final base = ((ni * c) + ci) * h * wSp;
        for (int p = 0; p < h * wSp; p++) {
          out[base + p] = src[base + p] * scale + shift;
        }
      }
    }
    return Tensor.fromList([n, c, h, wSp], out, device: Device.CPU);
  }

  @override
  List<Tensor> parameters() => const [];
}
