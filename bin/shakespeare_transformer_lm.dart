/// Tiny-Shakespeare character-level demo using the sinusoidal
/// [TransformerLM] baseline (no weight tying, learned pos embed
/// disabled, standard multi-head attention).
///
/// Mirrors `bin/shakespeare_gpt.dart` for easy comparison. Runs on
/// CPU only.
///
/// Run from the repository root:
///
///     dart run bin/shakespeare_transformer_lm.dart
library;

import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

import 'shakespeare_util.dart';

const int _corpusChars = 50000;
const int _blockSize = 32;
const int _embedDim = 64;
const int _numLayers = 2;
const int _numHeads = 4;
const int _trainSteps = 500;
const double _lr = 3e-3;
const int _sampleTokens = 200;

void main() {
  print('=== shakespeare_transformer_lm : sinusoidal Transformer LM ===');
  final text = loadCorpus(maxChars: _corpusChars);
  final tok = CharTokenizer.fromText(text);
  final ids = tok.encode(text);
  print(
    'corpus: ${text.length} chars, vocab: ${tok.vocabSize}, tokens: ${ids.length}',
  );

  final lm = TransformerLM(
    vocabSize: tok.vocabSize,
    embedDim: _embedDim,
    numLayers: _numLayers,
    numHeads: _numHeads,
    maxLen: _blockSize,
    seed: 1,
  );
  final params = lm.parameters();
  print(
    'model : TransformerLM ${_numLayers}L x ${_embedDim}d x ${_numHeads}h  '
    '${paramScalarCount(params)} scalars',
  );

  final opt = Adam(params, lr: _lr);
  final rng = math.Random(0);
  final sw = Stopwatch()..start();
  print('training $_trainSteps steps (blockSize=$_blockSize, lr=$_lr)...');
  double lastLoss = double.nan;
  for (int step = 1; step <= _trainSteps; step++) {
    opt.zeroGrad();
    final (x, y) = getWindow(ids, _blockSize, rng);
    final loss = lm(x).crossEntropy(y).mean();
    loss.backward();
    clipGradNorm(params, 1.0);
    opt.step();
    lastLoss = loss.toList()[0];
    if (step == 1 || step % 100 == 0 || step == _trainSteps) {
      print(
        '  step ${step.toString().padLeft(4)}  '
        'loss=${lastLoss.toStringAsFixed(4)}',
      );
    }
  }
  sw.stop();
  print(
    'training done in ${sw.elapsedMilliseconds} ms '
    '(${(sw.elapsedMilliseconds / _trainSteps).toStringAsFixed(1)} ms/step)',
  );

  const promptStr = 'ROMEO:';
  final prompt = tok.encode(promptStr);
  print('\nprompt: "$promptStr"');
  lm.eval();

  List<double> stepFn(List<double> ctx) {
    final x = Tensor.fromList([ctx.length], ctx);
    return lastRowLogits(lm(x), tok.vocabSize);
  }

  final greedy = generateText(
    prompt,
    maxNewTokens: _sampleTokens,
    maxCtx: _blockSize,
    vocabSize: tok.vocabSize,
    stepFn: stepFn,
    temperature: 0.0,
  );
  print('--- greedy ---');
  print(tok.decode(greedy));

  final sampled = generateText(
    prompt,
    maxNewTokens: _sampleTokens,
    maxCtx: _blockSize,
    vocabSize: tok.vocabSize,
    stepFn: stepFn,
    temperature: 0.8,
    topK: 20,
    rng: math.Random(1),
  );
  print('\n--- T=0.8 top-k=20 ---');
  print(tok.decode(sampled));
}
