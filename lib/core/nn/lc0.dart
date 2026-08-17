/// LC0 classical CNN (with optional SE units) — inference-only.
///
/// Assembles the network described by an [Lc0Weights] value into a
/// forward pass. BN is folded into the preceding conv's weights and
/// biases at construction time, exactly matching
/// lc0/src/neural/network_legacy.cc:
///
///   new_gamma[o] = gamma[o] / sqrt(var[o] + 1e-5)
///   new_weight[o, ...] = weight[o, ...] * new_gamma[o]
///   new_bias[o]        = -new_gamma[o] * (mean[o] - old_bias[o]) + beta[o]
///
/// where the misleadingly-named `bn_stddivs` field stores per-channel
/// VARIANCE. Residual block with SE follows se_unit.cc :: ApplySEUnit.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../tensor/tensor.dart';
import 'conv2d.dart';
import 'lc0_proto.dart';
import 'module.dart';

class Lc0Output {
  final Tensor policyLogits;
  final Tensor value;

  const Lc0Output(this.policyLogits, this.value);

  double get scalarValue {
    final v = value.toList();
    if (v.length == 3) return v[0] - v[2];
    return v[0];
  }
}

class _FoldedConv {
  final Conv2d conv;
  final Float32List bias;
  const _FoldedConv(this.conv, this.bias);
}

class Lc0Net extends Module {
  final Lc0Weights w;

  final _FoldedConv inputConv;
  final List<_FoldedConv> resConvA;
  final List<_FoldedConv> resConvB;
  final _FoldedConv policy1;
  final _FoldedConv policyOut;
  final _FoldedConv valueConv;

  Lc0Net(this.w)
    : inputConv = _fold(w.input, 112, w.filters, 3, 1),
      resConvA = [
        for (final r in w.residual) _fold(r.conv1, w.filters, w.filters, 3, 1),
      ],
      resConvB = [
        for (final r in w.residual) _fold(r.conv2, w.filters, w.filters, 3, 1),
      ],
      policy1 = _fold(w.policy1, w.filters, w.filters, 3, 1),
      policyOut = _fold(w.policyOut, w.filters, w.policyOutputPlanes, 3, 1),
      valueConv = _fold(w.valueConv, w.filters, w.valueFilters, 1, 0);

  static _FoldedConv _fold(Lc0ConvBlock cb, int cin, int cout, int k, int pad) {
    const eps = 1e-5;
    final rawW = cb.weights.toList();
    final rawB = cb.biases.toList();
    final variance = cb.bnStddivs.toList();
    final mean = cb.bnMeans.toList();
    final gamma = cb.bnGammas.toList();
    final beta = cb.bnBetas.toList();

    final foldedW = Float32List(rawW.length);
    final foldedB = Float32List(cout);
    final inputsPerOutput = rawW.length ~/ cout;
    for (int o = 0; o < cout; o++) {
      final newGamma = gamma[o] / math.sqrt(variance[o] + eps);
      final base = o * inputsPerOutput;
      for (int i = 0; i < inputsPerOutput; i++) {
        foldedW[base + i] = rawW[base + i] * newGamma;
      }
      foldedB[o] = -newGamma * (mean[o] - rawB[o]) + beta[o];
    }

    final conv = Conv2d(
      cin,
      cout,
      kernel: k,
      padding: pad,
      bias: false,
      device: Device.GPU,
    );
    conv.weight.assign(
      Tensor.fromList([cout, cin, k, k], foldedW.toList(), device: Device.GPU),
    );
    return _FoldedConv(conv, foldedB);
  }

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
      var h = _applyBiasRelu(inputConv.conv(input), inputConv.bias);
      for (int i = 0; i < w.residual.length; i++) {
        h = _residualForward(h, i);
      }
      final p1 = _applyBiasRelu(policy1.conv(h), policy1.bias);
      final policy = _applyBias(policyOut.conv(p1), policyOut.bias);

      var v = _applyBiasRelu(valueConv.conv(h), valueConv.bias);
      final vf = w.valueFilters;
      v = v.reshape([1, vf * 64]);
      v =
          (v.matmul(w.ip1ValW.transpose()) +
                  w.ip1ValB.reshape([1, w.valueFCUnits]))
              .relu();
      v = v.matmul(w.ip2ValW.transpose()) + w.ip2ValB.reshape([1, w.wdl]);
      if (w.wdl == 3) v = v.softmax();
      return Lc0Output(policy, v);
    });
  }

  Tensor _residualForward(Tensor skip, int i) {
    var h = _applyBiasRelu(resConvA[i].conv(skip), resConvA[i].bias);
    final h2 = resConvB[i].conv(h);
    final conv2Bias = resConvB[i].bias;

    final se = w.residual[i].se;
    if (se == null) {
      return _addBiasAndSkipRelu(h2, conv2Bias, skip);
    }
    return _seForward(h2, conv2Bias, skip, se);
  }

  // Port of lc0/src/neural/backends/blas/se_unit.cc :: ApplySEUnit,
  // with the "activation" fixed to ReLU (classical net default).
  Tensor _seForward(
    Tensor h2,
    Float32List conv2Bias,
    Tensor skip,
    Lc0SEUnit se,
  ) {
    final c = h2.shape[1];
    const hw = 64;
    final hData = h2.toList();
    final sData = skip.toList();

    final pool = Float32List(c);
    for (int ch = 0; ch < c; ch++) {
      double acc = 0;
      final base = ch * hw;
      for (int p = 0; p < hw; p++) {
        acc += hData[base + p];
      }
      pool[ch] = acc / hw + conv2Bias[ch];
    }

    final w1 = se.w1.toList();
    final b1 = se.b1.toList();
    final seHidden = se.w1.shape[1];
    final fc1 = Float32List(seHidden);
    for (int j = 0; j < seHidden; j++) {
      double acc = b1[j];
      for (int ch = 0; ch < c; ch++) {
        acc += pool[ch] * w1[ch * seHidden + j];
      }
      fc1[j] = acc > 0 ? acc : 0;
    }

    final w2 = se.w2.toList();
    final b2 = se.b2.toList();
    final twoC = 2 * c;
    final fc2 = Float32List(twoC);
    for (int j = 0; j < twoC; j++) {
      double acc = b2[j];
      for (int m = 0; m < seHidden; m++) {
        acc += fc1[m] * w2[m * twoC + j];
      }
      fc2[j] = acc;
    }

    final out = Float32List(c * hw);
    for (int ch = 0; ch < c; ch++) {
      final gamma = 1.0 / (1.0 + math.exp(-fc2[ch]));
      final beta = fc2[ch + c] + gamma * conv2Bias[ch];
      final base = ch * hw;
      for (int p = 0; p < hw; p++) {
        final v = gamma * hData[base + p] + beta + sData[base + p];
        out[base + p] = v > 0 ? v : 0;
      }
    }
    return Tensor.fromList([1, c, 8, 8], out.toList(), device: Device.CPU);
  }

  Tensor _applyBias(Tensor x, Float32List bias) {
    final c = x.shape[1];
    const hw = 64;
    final src = x.toList();
    final out = Float32List(src.length);
    for (int ch = 0; ch < c; ch++) {
      final base = ch * hw;
      final b = bias[ch];
      for (int p = 0; p < hw; p++) {
        out[base + p] = src[base + p] + b;
      }
    }
    return Tensor.fromList(x.shape, out.toList(), device: Device.CPU);
  }

  Tensor _applyBiasRelu(Tensor x, Float32List bias) {
    final c = x.shape[1];
    const hw = 64;
    final src = x.toList();
    final out = Float32List(src.length);
    for (int ch = 0; ch < c; ch++) {
      final base = ch * hw;
      final b = bias[ch];
      for (int p = 0; p < hw; p++) {
        final v = src[base + p] + b;
        out[base + p] = v > 0 ? v : 0;
      }
    }
    return Tensor.fromList(x.shape, out.toList(), device: Device.CPU);
  }

  Tensor _addBiasAndSkipRelu(Tensor x, Float32List bias, Tensor skip) {
    final c = x.shape[1];
    const hw = 64;
    final src = x.toList();
    final sk = skip.toList();
    final out = Float32List(src.length);
    for (int ch = 0; ch < c; ch++) {
      final base = ch * hw;
      final b = bias[ch];
      for (int p = 0; p < hw; p++) {
        final v = src[base + p] + b + sk[base + p];
        out[base + p] = v > 0 ? v : 0;
      }
    }
    return Tensor.fromList(x.shape, out.toList(), device: Device.CPU);
  }

  @override
  List<Tensor> parameters() => const [];
}
