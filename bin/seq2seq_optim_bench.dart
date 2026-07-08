/// Seq2seq encoder-decoder demo with an optimizer speed benchmark.
///
/// Trains a small `EncoderDecoderTransformer` on a synthetic
/// **reverse-copy** task (source = a random sequence of tokens,
/// target = that sequence reversed) with several optimizer
/// configurations side by side. Reports:
///
///   * wall-clock time per optimizer step (ms),
///   * total training time,
///   * initial and final loss,
///   * a greedy decode of the trained model on a fresh source.
///
/// The comparison uses the same random seed for the model and the same
/// batch schedule for every optimizer, so wall-clock differences
/// reflect the optimizer's own per-step overhead (Adam maintains two
/// running moments per parameter and does more per-step math than
/// plain SGD).
///
/// Run with:
///
///     dart run bin/seq2seq_optim_bench.dart
library;

import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

// ------------------ Task / data ------------------

const int _vocab = 12; // reserve 0 = <bos>
const int _bos = 0;
const int _seq = 6; // source sequence length
const int _tgt = _seq; // target = reversed source (same length)

/// Build one (src, tgtIn, tgtOut) triple.
///
/// * `src`      = `[t1, t2, ..., tS]`
/// * `tgtIn`    = `[BOS, t_S, t_{S-1}, ..., t_2]`      (teacher forcing)
/// * `tgtOut`   = `[t_S, t_{S-1}, ..., t_1]`           (labels)
({List<int> src, List<int> tgtIn, List<int> tgtOut}) _sample(math.Random rng) {
  final src = List<int>.generate(_seq, (_) => 1 + rng.nextInt(_vocab - 1));
  final rev = src.reversed.toList();
  final tgtIn = [_bos, ...rev.sublist(0, _seq - 1)];
  final tgtOut = rev;
  return (src: src, tgtIn: tgtIn, tgtOut: tgtOut);
}

/// Stack `batchSize` samples into three `[B, S]` tensors (as float32).
({Tensor src, Tensor tgtIn, Tensor tgtOut}) _batch(
  int batchSize,
  math.Random rng,
) {
  final srcBuf = List<double>.filled(batchSize * _seq, 0.0);
  final inBuf = List<double>.filled(batchSize * _tgt, 0.0);
  final outBuf = List<double>.filled(batchSize * _tgt, 0.0);
  for (int b = 0; b < batchSize; b++) {
    final s = _sample(rng);
    for (int i = 0; i < _seq; i++) {
      srcBuf[b * _seq + i] = s.src[i].toDouble();
    }
    for (int i = 0; i < _tgt; i++) {
      inBuf[b * _tgt + i] = s.tgtIn[i].toDouble();
      outBuf[b * _tgt + i] = s.tgtOut[i].toDouble();
    }
  }
  return (
    src: Tensor.fromList([batchSize, _seq], srcBuf),
    tgtIn: Tensor.fromList([batchSize, _tgt], inBuf),
    tgtOut: Tensor.fromList([batchSize, _tgt], outBuf),
  );
}

// ------------------ Model factory ------------------

EncoderDecoderTransformer _newModel({int seed = 42}) => EncoderDecoderTransformer(
      sourceVocabSize: _vocab,
      targetVocabSize: _vocab,
      embedDim: 32,
      numLayers: 2,
      numHeads: 4,
      maxSourceLen: _seq,
      maxTargetLen: _tgt,
      seed: seed,
    );

// ------------------ Benchmark ------------------

class _RunResult {
  final String name;
  final double lossInit;
  final double lossFinal;
  final int totalMs;
  final double msPerStep;
  final String greedyDecode;
  _RunResult(this.name, this.lossInit, this.lossFinal, this.totalMs,
      this.msPerStep, this.greedyDecode);
}

_RunResult _runOne({
  required String name,
  required Optimizer Function(List<Tensor>) optFactory,
  required int steps,
  required int batchSize,
  required int modelSeed,
  required int dataSeed,
  required List<int> probeSrc,
}) {
  final model = _newModel(seed: modelSeed);
  final opt = optFactory(model.parameters());

  final rng = math.Random(dataSeed);
  final b0 = _batch(batchSize, rng);
  final l0 = model(b0.src, b0.tgtIn).crossEntropy(b0.tgtOut).mean().toList()[0];

  final sw = Stopwatch()..start();
  double lastLoss = l0;
  final trainRng = math.Random(dataSeed + 1);
  for (int step = 0; step < steps; step++) {
    opt.zeroGrad();
    final b = _batch(batchSize, trainRng);
    final loss = model(b.src, b.tgtIn).crossEntropy(b.tgtOut).mean();
    loss.backward();
    clipGradNorm(model.parameters(), 1.0);
    opt.step();
    lastLoss = loss.toList()[0];
  }
  sw.stop();

  // Greedy decode: encode `probeSrc`, then autoregressively pick argmax.
  model.eval();
  final srcT = Tensor.fromList(
    [probeSrc.length],
    probeSrc.map((i) => i.toDouble()).toList(),
  );
  final memory = model.encode(srcT);
  final produced = <int>[];
  for (int t = 0; t < _tgt; t++) {
    final prefix = [_bos, ...produced];
    final tgtT = Tensor.fromList(
      [prefix.length],
      prefix.map((i) => i.toDouble()).toList(),
    );
    final logits = model.decode(tgtT, memory).toList();
    // Last row of logits: prefix.length - 1 offset, width = _vocab.
    final off = (prefix.length - 1) * _vocab;
    int best = 0;
    double bestVal = double.negativeInfinity;
    for (int v = 0; v < _vocab; v++) {
      if (logits[off + v] > bestVal) {
        bestVal = logits[off + v];
        best = v;
      }
    }
    produced.add(best);
  }
  final decode = produced.join(',');

  return _RunResult(
    name,
    l0,
    lastLoss,
    sw.elapsedMilliseconds,
    sw.elapsedMilliseconds / steps,
    decode,
  );
}

void main() {
  print('=== seq2seq optimizer speed benchmark ===');
  print(
    'task            : reverse-copy (vocab=$_vocab, seq=$_seq, tgt=$_tgt)',
  );
  print('model           : EncoderDecoderTransformer(embed=32, layers=2, heads=4)');
  const steps = 100;
  const batchSize = 16;
  print('training config : steps=$steps  batchSize=$batchSize\n');

  final probeRng = math.Random(999);
  final probeSample = _sample(probeRng);
  final probeSrc = probeSample.src;
  final probeExpected = probeSample.tgtOut;
  print('probe source    : ${probeSrc.join(',')}');
  print('probe expected  : ${probeExpected.join(',')}\n');

  final configs = <(String, Optimizer Function(List<Tensor>))>[
    ('SGD lr=0.05',          (p) => SGD(p, lr: 0.05)),
    ('SGD+mom lr=0.05 m=.9', (p) => SGD(p, lr: 0.05, momentum: 0.9)),
    ('Adam lr=1e-3',         (p) => Adam(p, lr: 1e-3)),
    ('AdamW lr=1e-3 wd=.01', (p) => Adam(p, lr: 1e-3, weightDecay: 0.01)),
  ];

  final results = <_RunResult>[];
  for (final (name, factory) in configs) {
    print('running $name ...');
    final r = _runOne(
      name: name,
      optFactory: factory,
      steps: steps,
      batchSize: batchSize,
      modelSeed: 42, // same weights for every optimizer
      dataSeed: 7,   // same batch schedule
      probeSrc: probeSrc,
    );
    results.add(r);
    print(
      '  loss ${r.lossInit.toStringAsFixed(3)} -> '
      '${r.lossFinal.toStringAsFixed(3)}   '
      '${r.totalMs} ms total   '
      '${r.msPerStep.toStringAsFixed(2)} ms/step',
    );
  }

  // ------------------ Summary table ------------------

  print('\n${'-' * 78}');
  print(
    '${'optimizer'.padRight(22)}  '
    '${'loss0'.padLeft(7)}  '
    '${'lossF'.padLeft(7)}  '
    '${'totalMs'.padLeft(8)}  '
    '${'ms/step'.padLeft(8)}  '
    'greedy',
  );
  print('-' * 78);
  final expectedStr = probeExpected.join(',');
  for (final r in results) {
    final match = r.greedyDecode == expectedStr ? '  [MATCH]' : '';
    print(
      '${r.name.padRight(22)}  '
      '${r.lossInit.toStringAsFixed(3).padLeft(7)}  '
      '${r.lossFinal.toStringAsFixed(3).padLeft(7)}  '
      '${r.totalMs.toString().padLeft(8)}  '
      '${r.msPerStep.toStringAsFixed(2).padLeft(8)}  '
      '${r.greedyDecode}$match',
    );
  }
  print('-' * 78);

  final fastest =
      results.reduce((a, b) => a.msPerStep <= b.msPerStep ? a : b);
  final lowest =
      results.reduce((a, b) => a.lossFinal <= b.lossFinal ? a : b);
  print(
    '\nfastest per step : ${fastest.name} '
    '(${fastest.msPerStep.toStringAsFixed(2)} ms)',
  );
  print(
    'lowest final loss: ${lowest.name} '
    '(${lowest.lossFinal.toStringAsFixed(3)})',
  );
}
