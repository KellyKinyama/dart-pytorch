/// Fold a `Conv2d(bias=false) → BatchNorm2d` pair into a single
/// `Conv2d(bias=true)` at load time.
///
/// Given (from HuggingFace-style safetensors, PyTorch layout):
///
///   convW      shape `[Cout, Cin, Kh, Kw]`
///   convB      shape `[Cout]`               (optional; zero if absent)
///   bnGamma    shape `[Cout]`               (`weight`)
///   bnBeta     shape `[Cout]`               (`bias`)
///   bnMean     shape `[Cout]`               (`running_mean`)
///   bnVar      shape `[Cout]`               (`running_var`)
///   bnEps      scalar (facenet-pytorch: 1e-3, PyTorch default: 1e-5)
///
/// Produces:
///
///   W' = (γ / √(σ² + ε)) · W          — per-output-channel scale
///   b' = (γ / √(σ² + ε)) · (b − μ) + β
///
/// So that a single `Conv2d.loadFromPytorch(W', b')` reproduces
///
///   γ · (conv(x) − μ) / √(σ² + ε) + β
///
/// exactly (inference-time).
///
/// The output weights carry `requiresGrad: true` — you can fine-tune
/// the folded module as if BN never existed. Grads still flow into
/// `W'` and `b'` via [Conv2d]'s matmul path. (See the caveat in
/// `doc/facenet.md` about backpropagating *through* Conv2d.)
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../../tensor/tensor.dart';
import '../conv2d.dart';

class ConvBnFoldResult {
  /// Folded weight `[Cout, Cin, Kh, Kw]`.
  final Float32List weight;

  /// Folded bias `[Cout]`.
  final Float32List bias;
  const ConvBnFoldResult(this.weight, this.bias);
}

/// Fold the given Conv+BN parameters into a single Conv weight+bias.
ConvBnFoldResult foldConvBn({
  required Float32List convW,
  required int outChannels,
  required int inChannels,
  required int kernelH,
  required int kernelW,
  Float32List? convB,
  required Float32List bnGamma,
  required Float32List bnBeta,
  required Float32List bnMean,
  required Float32List bnVar,
  double bnEps = 1e-3,
}) {
  final expectedW = outChannels * inChannels * kernelH * kernelW;
  if (convW.length != expectedW) {
    throw ArgumentError(
      'foldConvBn: convW length ${convW.length} != $expectedW',
    );
  }
  for (final name in ['gamma', 'beta', 'mean', 'var']) {
    final v = {
      'gamma': bnGamma,
      'beta': bnBeta,
      'mean': bnMean,
      'var': bnVar,
    }[name]!;
    if (v.length != outChannels) {
      throw ArgumentError(
        'foldConvBn: bn$name length ${v.length} != $outChannels',
      );
    }
  }
  if (convB != null && convB.length != outChannels) {
    throw ArgumentError(
      'foldConvBn: convB length ${convB.length} != $outChannels',
    );
  }

  final inputsPerOut = inChannels * kernelH * kernelW;
  final wOut = Float32List(expectedW);
  final bOut = Float32List(outChannels);
  for (int o = 0; o < outChannels; o++) {
    final scale = bnGamma[o] / math.sqrt(bnVar[o] + bnEps);
    final base = o * inputsPerOut;
    for (int i = 0; i < inputsPerOut; i++) {
      wOut[base + i] = convW[base + i] * scale;
    }
    final b = convB == null ? 0.0 : convB[o];
    bOut[o] = scale * (b - bnMean[o]) + bnBeta[o];
  }
  return ConvBnFoldResult(wOut, bOut);
}

/// Convenience wrapper: fold + assign into an existing [Conv2d] module
/// (which must already have matching `outChannels`, `inChannels`,
/// `kernelH`, `kernelW`, and `bias: true`).
void loadConvBnFolded(
  Conv2d conv, {
  required Float32List convW,
  Float32List? convB,
  required Float32List bnGamma,
  required Float32List bnBeta,
  required Float32List bnMean,
  required Float32List bnVar,
  double bnEps = 1e-3,
}) {
  final folded = foldConvBn(
    convW: convW,
    outChannels: conv.outChannels,
    inChannels: conv.inChannels,
    kernelH: conv.kernelH,
    kernelW: conv.kernelW,
    convB: convB,
    bnGamma: bnGamma,
    bnBeta: bnBeta,
    bnMean: bnMean,
    bnVar: bnVar,
    bnEps: bnEps,
  );
  // Push the folded [Cout, Cin, Kh, Kw] weight into Conv2d's parameter
  // slot. Conv2d stores the raw NCHW layout; the im2col transpose
  // happens per-forward-pass.
  final wT = Tensor.fromFloat32List(
    conv.weight.shape,
    folded.weight,
    device: conv.weight.device,
    requiresGrad: conv.weight.requiresGrad,
  );
  conv.weight.assign(wT);
  if (conv.bias == null) {
    throw StateError(
      'loadConvBnFolded: destination Conv2d has no bias slot — construct '
      'it with `bias: true` when folding a BN in.',
    );
  }
  final bT = Tensor.fromFloat32List(
    conv.bias!.shape,
    folded.bias,
    device: conv.bias!.device,
    requiresGrad: conv.bias!.requiresGrad,
  );
  conv.bias!.assign(bT);
}
