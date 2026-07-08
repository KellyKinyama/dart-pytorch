/// Attention-Free Transformer (AFT) vs. standard multi-head attention:
/// side-by-side training and speed comparison.
///
/// Both models overfit the same tiny next-token sequence (identical
/// vocabulary, seed schedule, optimizer, and number of steps). The
/// AFT variant uses `AFTLanguageModel`; the baseline uses
/// `TransformerLM`. We report:
///
///   * initial + final loss,
///   * total wall-clock time,
///   * ms/step,
///   * greedy next-token decode from a short prompt.
///
/// Not a fair "big-scale" throughput benchmark — the two models don't
/// have quite the same parameter count and AFT here is a pure-Dart
/// CPU implementation (`Tensor.aftFull`), while standard attention
/// leans on the batched matmul path.
///
/// Run with:
///
///     dart run bin/aft_demo.dart
library;

import 'package:dart_pytorch/dart_pytorch.dart';

// A short "sequence-to-continue" task: the model must predict the next
// integer in the mod-`vocab` counting sequence.
const int _vocab = 12;
const int _seq = 16;

({Tensor x, Tensor y}) _makeSequence() {
  final tokens = List<int>.generate(_seq + 1, (i) => i % _vocab);
  return (
    x: Tensor.fromList([
      _seq,
    ], tokens.sublist(0, _seq).map((i) => i.toDouble()).toList()),
    y: Tensor.fromList([
      _seq,
    ], tokens.sublist(1).map((i) => i.toDouble()).toList()),
  );
}

class _Result {
  final String name;
  final int paramCount;
  final double lossInit;
  final double lossFinal;
  final int totalMs;
  final double msPerStep;
  final String decode;
  _Result(
    this.name,
    this.paramCount,
    this.lossInit,
    this.lossFinal,
    this.totalMs,
    this.msPerStep,
    this.decode,
  );
}

_Result _run(
  String name,
  Tensor Function(Tensor) forward,
  List<Tensor> params,
  Tensor x,
  Tensor y, {
  required int steps,
  required double lr,
  required List<int> promptIds,
  required int decodeSteps,
}) {
  final opt = Adam(params, lr: lr);
  final scalars = params.fold<int>(0, (a, p) => a + p.length);
  final l0 = forward(x).crossEntropy(y).mean().toList()[0];
  final sw = Stopwatch()..start();
  double lastLoss = l0;
  for (int step = 0; step < steps; step++) {
    opt.zeroGrad();
    final loss = forward(x).crossEntropy(y).mean();
    loss.backward();
    clipGradNorm(params, 1.0);
    opt.step();
    lastLoss = loss.toList()[0];
  }
  sw.stop();

  // Greedy decode from `promptIds`.
  final buf = List<int>.from(promptIds);
  for (int i = 0; i < decodeSteps; i++) {
    final ids = Tensor.fromList([
      buf.length,
    ], buf.map((v) => v.toDouble()).toList());
    final logits = forward(ids).toList();
    final off = (buf.length - 1) * _vocab;
    int best = 0;
    double bestV = double.negativeInfinity;
    for (int v = 0; v < _vocab; v++) {
      if (logits[off + v] > bestV) {
        bestV = logits[off + v];
        best = v;
      }
    }
    buf.add(best);
  }
  return _Result(
    name,
    scalars,
    l0,
    lastLoss,
    sw.elapsedMilliseconds,
    sw.elapsedMilliseconds / steps,
    buf.join(','),
  );
}

void main() {
  print('=== AFT vs standard attention ===');
  print('task     : next-token on mod-$_vocab counting (seq=$_seq)');
  const embedDim = 16;
  const numLayers = 2;
  const steps = 100;
  const lr = 3e-3;
  print(
    'config   : embedDim=$embedDim, layers=$numLayers, steps=$steps, '
    'lr=${lr.toStringAsExponential(0)}\n',
  );

  final data = _makeSequence();

  // AFT model.
  final aft = AFTLanguageModel(
    vocabSize: _vocab,
    embedDim: embedDim,
    numLayers: numLayers,
    maxLen: _seq,
    seed: 1,
  );

  // Standard-attention model (matched to same embedDim / layers).
  final mha = TransformerLM(
    vocabSize: _vocab,
    embedDim: embedDim,
    numLayers: numLayers,
    numHeads: 2,
    maxLen: _seq,
    seed: 1,
  );

  final promptIds = [0, 1, 2, 3];
  print(
    'prompt   : ${promptIds.join(',')} '
    '(expected continuation: 4,5,6,7,...)\n',
  );

  final resAft = _run(
    'AFTLanguageModel',
    (t) => aft(t),
    aft.parameters(),
    data.x,
    data.y,
    steps: steps,
    lr: lr,
    promptIds: promptIds,
    decodeSteps: 8,
  );
  print(
    'AFT   done: loss ${resAft.lossInit.toStringAsFixed(3)} -> '
    '${resAft.lossFinal.toStringAsFixed(3)}   '
    '${resAft.totalMs} ms   '
    '${resAft.msPerStep.toStringAsFixed(2)} ms/step',
  );

  final resMha = _run(
    'TransformerLM(MHA)',
    (t) => mha(t),
    mha.parameters(),
    data.x,
    data.y,
    steps: steps,
    lr: lr,
    promptIds: promptIds,
    decodeSteps: 8,
  );
  print(
    'MHA   done: loss ${resMha.lossInit.toStringAsFixed(3)} -> '
    '${resMha.lossFinal.toStringAsFixed(3)}   '
    '${resMha.totalMs} ms   '
    '${resMha.msPerStep.toStringAsFixed(2)} ms/step\n',
  );

  // Summary.
  print('-' * 80);
  print(
    '${'model'.padRight(22)}  '
    '${'params'.padLeft(8)}  '
    '${'loss0'.padLeft(7)}  '
    '${'lossF'.padLeft(7)}  '
    '${'totalMs'.padLeft(8)}  '
    '${'ms/step'.padLeft(8)}  greedy decode',
  );
  print('-' * 80);
  for (final r in [resAft, resMha]) {
    print(
      '${r.name.padRight(22)}  '
      '${r.paramCount.toString().padLeft(8)}  '
      '${r.lossInit.toStringAsFixed(3).padLeft(7)}  '
      '${r.lossFinal.toStringAsFixed(3).padLeft(7)}  '
      '${r.totalMs.toString().padLeft(8)}  '
      '${r.msPerStep.toStringAsFixed(2).padLeft(8)}  '
      '${r.decode}',
    );
  }
  print('-' * 80);

  final fast = resAft.msPerStep <= resMha.msPerStep ? resAft : resMha;
  final low = resAft.lossFinal <= resMha.lossFinal ? resAft : resMha;
  print(
    '\nfaster per step  : ${fast.name} '
    '(${fast.msPerStep.toStringAsFixed(2)} ms/step)',
  );
  print(
    'lower final loss : ${low.name} '
    '(${low.lossFinal.toStringAsFixed(3)})',
  );
  print(
    '\nnote: AFT here is a pure-Dart CPU op; standard attention uses the '
    'batched matmul path.',
  );
}
