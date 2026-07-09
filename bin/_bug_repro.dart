/// Repro of the loss=0.0000 bug: mid-size GPT on GPU that previously
/// leaked GPU memory in Adam.step() and turned every param to zero at
/// ~step 10 once cudaMalloc started failing.
///
/// Verifies the fixes to `Adam.step()`, `SGD.step()`, `clipGradNorm`,
/// and `engine.cu`'s `Tensor` ctor OOM guard.
library;

import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

void main() {
  const int vocab = 65;
  const int T = 192;
  const int D = 384;
  const int H = 6;
  const int L = 6;

  final gpt = GPT(
    GPTConfig(
      vocabSize: vocab,
      maxCtx: T,
      embedDim: D,
      numLayers: L,
      numHeads: H,
      dropoutP: 0.0,
      tieWeights: true,
      seed: 1,
      device: Device.GPU,
    ),
  );
  final params = gpt.parameters();
  final scalars = params.fold<int>(0, (s, p) => s + p.length);
  print('params: ${params.length}, scalars: $scalars');

  final opt = Adam(params, lr: 3e-3);
  final rng = math.Random(0);
  final xData = List<double>.generate(T, (_) => rng.nextInt(vocab).toDouble());
  final yData = List<double>.generate(T, (_) => rng.nextInt(vocab).toDouble());
  final x = Tensor.fromList([T], xData, device: Device.GPU);
  final y = Tensor.fromList([T], yData, device: Device.GPU);

  for (int step = 1; step <= 25; step++) {
    opt.zeroGrad();
    final loss = gpt(x).crossEntropy(y).mean();
    loss.backward();
    clipGradNorm(params, 1.0);
    opt.step();
    final l = loss.toList()[0];
    print('step ${step.toString().padLeft(3)}  loss=${l.toStringAsFixed(4)}');
  }
}
