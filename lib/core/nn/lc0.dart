/// LC0 classical CNN (with optional SE units) — GPU inference.
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
/// Runtime layout: NHWC-flat `[H*W, C]` throughout the tower. This
/// shape lets bias-add, skip-add, and SE's per-channel gate use the
/// existing GPU row-broadcast (`[H*W, C] + [1, C]`), and it means
/// Conv2d's matmul output can flow straight into the next layer
/// without a host-side NCHW permute.
///
/// Reference: lc0/src/neural/backends/blas/{network_blas.cc, se_unit.cc}.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../tensor/tensor.dart';
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

class _ConvGpu {
  final int cin;
  final int cout;
  final int k;
  final int pad;
  final Tensor wT; // [Cin*K*K, Cout] on GPU, pre-transposed for `cols @ wT`.
  final Tensor bias; // [1, Cout] on GPU.
  const _ConvGpu({
    required this.cin,
    required this.cout,
    required this.k,
    required this.pad,
    required this.wT,
    required this.bias,
  });
}

class _SEGpu {
  final Tensor w1; // [C, seHidden]
  final Tensor b1; // [1, seHidden]
  final Tensor w2Gamma; // [seHidden, C]
  final Tensor w2Beta; // [seHidden, C]
  final Tensor b2Gamma; // [1, C]
  final Tensor b2Beta; // [1, C]
  const _SEGpu({
    required this.w1,
    required this.b1,
    required this.w2Gamma,
    required this.w2Beta,
    required this.b2Gamma,
    required this.b2Beta,
  });
}

class Lc0Net extends Module {
  final Lc0Weights w;
  final Device device;

  final _ConvGpu inputConv;
  final List<_ConvGpu> resA;
  final List<_ConvGpu> resB;
  final List<_SEGpu?> se;
  final _ConvGpu policy1;
  final _ConvGpu policyOut;
  final _ConvGpu valueConv;

  final Tensor ip1ValWT;
  final Tensor ip1ValB;
  final Tensor ip2ValWT;
  final Tensor ip2ValB;

  final Tensor _ones64x1;
  final Tensor _avgWeights;

  Lc0Net(this.w, {Device device = Device.GPU})
    : device = device,
      inputConv = _makeConv(w.input, 112, w.filters, 3, 1, device),
      resA = [
        for (final r in w.residual)
          _makeConv(r.conv1, w.filters, w.filters, 3, 1, device),
      ],
      resB = [
        for (final r in w.residual)
          _makeConv(r.conv2, w.filters, w.filters, 3, 1, device),
      ],
      se = [
        for (final r in w.residual)
          r.se == null ? null : _makeSE(r.se!, w.filters, device),
      ],
      policy1 = _makeConv(w.policy1, w.filters, w.filters, 3, 1, device),
      policyOut = _makeConv(
        w.policyOut,
        w.filters,
        w.policyOutputPlanes,
        3,
        1,
        device,
      ),
      valueConv = _makeConv(
        w.valueConv,
        w.filters,
        w.valueFilters,
        1,
        0,
        device,
      ),
      ip1ValWT = Tensor.fromList(
        [w.valueFilters * 64, w.valueFCUnits],
        _ip1ValWNhwc(
          w.ip1ValW.toList(),
          w.valueFCUnits,
          w.valueFilters,
        ),
        device: device,
      ),
      ip1ValB = Tensor.fromList(
        [1, w.valueFCUnits],
        w.ip1ValB.toList(),
        device: device,
      ),
      ip2ValWT = Tensor.fromList(
        [w.valueFCUnits, w.wdl],
        _transpose2d(w.ip2ValW.toList(), w.wdl, w.valueFCUnits),
        device: device,
      ),
      ip2ValB = Tensor.fromList([1, w.wdl], w.ip2ValB.toList(), device: device),
      _ones64x1 = Tensor.fill([64, 1], 1.0, device: device),
      _avgWeights = Tensor.fill([1, 64], 1.0 / 64.0, device: device);

  static _ConvGpu _makeConv(
    Lc0ConvBlock cb,
    int cin,
    int cout,
    int k,
    int pad,
    Device device,
  ) {
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

    final wT = _transpose2d(foldedW, cout, inputsPerOutput);

    return _ConvGpu(
      cin: cin,
      cout: cout,
      k: k,
      pad: pad,
      wT: Tensor.fromList([inputsPerOutput, cout], wT.toList(), device: device),
      bias: Tensor.fromList([1, cout], foldedB.toList(), device: device),
    );
  }

  static _SEGpu _makeSE(Lc0SEUnit u, int cout, Device device) {
    final w2Host = u.w2.toList();
    final seHidden = u.w2.shape[0];
    final twoC = 2 * cout;
    final wG = Float32List(seHidden * cout);
    final wB = Float32List(seHidden * cout);
    for (int m = 0; m < seHidden; m++) {
      for (int c = 0; c < cout; c++) {
        wG[m * cout + c] = w2Host[m * twoC + c];
        wB[m * cout + c] = w2Host[m * twoC + cout + c];
      }
    }
    final b2Host = u.b2.toList();
    final bG = Float32List(cout);
    final bB = Float32List(cout);
    for (int c = 0; c < cout; c++) {
      bG[c] = b2Host[c];
      bB[c] = b2Host[cout + c];
    }
    return _SEGpu(
      w1: Tensor.fromList(u.w1.shape, u.w1.toList(), device: device),
      b1: Tensor.fromList([1, seHidden], u.b1.toList(), device: device),
      w2Gamma: Tensor.fromList([seHidden, cout], wG.toList(), device: device),
      w2Beta: Tensor.fromList([seHidden, cout], wB.toList(), device: device),
      b2Gamma: Tensor.fromList([1, cout], bG.toList(), device: device),
      b2Beta: Tensor.fromList([1, cout], bB.toList(), device: device),
    );
  }

  static Float32List _transpose2d(List<double> src, int rows, int cols) {
    final out = Float32List(rows * cols);
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        out[c * rows + r] = src[r * cols + c];
      }
    }
    return out;
  }

  // Re-order ip1_val_w so a natural row-major flatten of `[64, Vf]`
  // NHWC-flat activations lines up with the weight without an
  // extra transpose in the forward. LC0 stores the weight as
  //   `[FCUnits, Vf*64]` in `(c, s)` inner order.
  // After reordering to `(s, c)` and transposing we get
  //   `[64*Vf, FCUnits]` — directly matmul-friendly for both
  // single- and multi-batch input.
  static Float32List _ip1ValWNhwc(List<double> src, int fcUnits, int vf) {
    final out = Float32List(fcUnits * 64 * vf);
    for (int o = 0; o < fcUnits; o++) {
      for (int c = 0; c < vf; c++) {
        for (int s = 0; s < 64; s++) {
          // src is [fcUnits, vf * 64] with inner (c, s).
          // wT wants (s, c) transposed to [64*Vf, fcUnits].
          out[(s * vf + c) * fcUnits + o] = src[o * vf * 64 + c * 64 + s];
        }
      }
    }
    return out;
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
      final nhwc = _initialNHWC(input);
      var h = (_convForward(nhwc, inputConv) + inputConv.bias).relu();

      for (int i = 0; i < w.residual.length; i++) {
        h = _residual(h, i);
      }

      final p1 = (_convForward(h, policy1) + policy1.bias).relu();
      final policyFlat = _convForward(p1, policyOut) + policyOut.bias;
      // NHWC-flat [64, P] -> NCHW [1, P, 8, 8] for the caller (matches
      // the layout LC0's policy-map table expects).
      final policyNCHW = policyFlat.transpose().reshape([
        1,
        w.policyOutputPlanes,
        8,
        8,
      ]);

      var v = (_convForward(h, valueConv) + valueConv.bias).relu();
      // v is [64, Vf] NHWC-flat; row-major flatten to [1, 64*Vf]
      // gives (s, c) inner ordering — matches ip1ValWT's re-ordered
      // layout, so no transpose needed.
      v = v.reshape([1, w.valueFilters * 64]);
      v = (v.matmul(ip1ValWT) + ip1ValB).relu();
      v = v.matmul(ip2ValWT) + ip2ValB;
      if (w.wdl == 3) v = v.softmax();

      return Lc0Output(policyNCHW, v);
    });
  }

  Tensor _residual(Tensor h, int i) {
    final skip = h;
    final hA = (_convForward(h, resA[i]) + resA[i].bias).relu();
    final hB = _convForward(hA, resB[i]);
    final seU = se[i];
    if (seU == null) {
      return ((hB + resB[i].bias) + skip).relu();
    }
    return _seForward(hB, resB[i], seU, skip);
  }

  Tensor _seForward(Tensor hB, _ConvGpu convB, _SEGpu seU, Tensor skip) {
    // Global-average pool + per-channel bias: [1, 64] @ [64, C] + [1, C].
    final pool = _avgWeights.matmul(hB) + convB.bias;
    // FC1 + ReLU.
    final fc1 = (pool.matmul(seU.w1) + seU.b1).relu();
    // FC2 split into gamma / beta halves at load time.
    final gammaPre = fc1.matmul(seU.w2Gamma) + seU.b2Gamma;
    final betaPre = fc1.matmul(seU.w2Beta) + seU.b2Beta;
    // gamma = sigmoid(pre); beta = betaPre + gamma * conv2Bias
    //   (both `[1, C]`, so plain elementwise mul, no broadcast).
    final gamma = gammaPre.sigmoid();
    final beta = betaPre + gamma * convB.bias;
    // `[64, C] * [1, C]` isn't a wired GPU row-broadcast for `*`, so
    // materialise gamma at full spatial width via a broadcast matmul:
    // `[64, 1] @ [1, C] = [64, C]`.
    final gammaFull = _ones64x1.matmul(gamma);
    return ((hB * gammaFull + beta) + skip).relu();
  }

  Tensor _convForward(Tensor input, _ConvGpu conv) {
    final cols = _im2colNHWC(input, conv.cin, conv.k, conv.pad);
    return cols.matmul(conv.wT);
  }

  Tensor _im2colNHWC(Tensor input, int cin, int k, int pad) {
    final data = input.toList();
    const w = 8;
    const h = 8;
    final rows = h * w;
    final colsWidth = cin * k * k;
    final out = Float32List(rows * colsWidth);
    var rowOff = 0;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        var colOff = rowOff;
        for (int c = 0; c < cin; c++) {
          for (int ky = 0; ky < k; ky++) {
            final yin = y + ky - pad;
            for (int kx = 0; kx < k; kx++) {
              final xin = x + kx - pad;
              if (yin >= 0 && yin < h && xin >= 0 && xin < w) {
                out[colOff] = data[(yin * w + xin) * cin + c];
              }
              colOff++;
            }
          }
        }
        rowOff += colsWidth;
      }
    }
    return Tensor.fromList([rows, colsWidth], out.toList(), device: device);
  }

  // Take [1, 112, 8, 8] NCHW input and emit [64, 112] NHWC-flat on
  // GPU (one host-side permute, then a single upload).
  Tensor _initialNHWC(Tensor input) {
    final data = input.toList();
    const c = 112, hw = 64;
    final out = Float32List(hw * c);
    for (int ci = 0; ci < c; ci++) {
      for (int p = 0; p < hw; p++) {
        out[p * c + ci] = data[ci * hw + p];
      }
    }
    return Tensor.fromList([hw, c], out.toList(), device: device);
  }

  @override
  List<Tensor> parameters() => const [];
}
