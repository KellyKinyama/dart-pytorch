/// Benchmark AFT GPU vs CPU at a realistic-ish size.
/// One forward + backward per step, warmup steps excluded.
library;

import 'package:dart_pytorch/dart_pytorch.dart';

const int _vocab = 65;
const int _seq = 128;
const int _embed = 128;
const int _layers = 2;
const int _warmup = 3;
const int _steps = 10;

double _bench(Device device) {
  final lm = AFTLanguageModel(
    vocabSize: _vocab,
    embedDim: _embed,
    numLayers: _layers,
    maxLen: _seq,
    device: device,
    seed: 1,
  );
  final params = lm.parameters();
  final opt = Adam(params, lr: 3e-3);

  final ids = Tensor.fromList(
    [_seq],
    List<double>.generate(_seq, (i) => (i % _vocab).toDouble()),
    device: device,
  );
  final tgt = Tensor.fromList(
    [_seq],
    List<double>.generate(_seq, (i) => ((i + 1) % _vocab).toDouble()),
    device: device,
  );

  // Warmup.
  for (int i = 0; i < _warmup; i++) {
    opt.zeroGrad();
    lm(ids).crossEntropy(tgt).mean().backward();
    opt.step();
  }
  final sw = Stopwatch()..start();
  for (int i = 0; i < _steps; i++) {
    opt.zeroGrad();
    lm(ids).crossEntropy(tgt).mean().backward();
    opt.step();
  }
  sw.stop();
  return sw.elapsedMilliseconds / _steps;
}

void main() {
  print(
    'AFT bench: vocab=$_vocab seq=$_seq embed=$_embed layers=$_layers '
    'warmup=$_warmup steps=$_steps',
  );
  final cpuMs = _bench(Device.CPU);
  final gpuMs = _bench(Device.GPU);
  print('  CPU : ${cpuMs.toStringAsFixed(1)} ms/step');
  print('  GPU : ${gpuMs.toStringAsFixed(1)} ms/step');
  print('  speedup = ${(cpuMs / gpuMs).toStringAsFixed(2)}x');
}
