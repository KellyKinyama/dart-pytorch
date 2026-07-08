/// Tiny-Shakespeare character-level GPT demo.
///
/// Trains a small weight-tied [GPT] on the leading slice of
/// `data/tiny_shakespeare.txt`, then samples continuations with the
/// KV-cache-accelerated [GPT.generate].
///
/// This is a short (~1 min on CPU) demo, not a serious training run.
/// Increase `_trainSteps`, `_blockSize`, and the model dims for
/// something closer to the reference `example/bin/shakespear.dart`.
///
/// Run from the repository root:
///
///     dart run bin/shakespeare_gpt.dart
library;

import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

import 'shakespeare_util.dart';

const int _corpusChars = 50000; // ~5% of full corpus for a fast demo
const int _blockSize = 32;
const int _embedDim = 64;
const int _numLayers = 2;
const int _numHeads = 4;
const int _trainSteps = 500;
const double _lr = 3e-3;

void main() {
  print('=== shakespeare_gpt : char-level GPT demo ===');
  final text = loadCorpus(maxChars: _corpusChars);
  final tok = CharTokenizer.fromText(text);
  final ids = tok.encode(text);
  print(
    'corpus: ${text.length} chars, vocab: ${tok.vocabSize}, tokens: ${ids.length}',
  );

  final gpt = GPT(
    GPTConfig(
      vocabSize: tok.vocabSize,
      maxCtx: _blockSize,
      embedDim: _embedDim,
      numLayers: _numLayers,
      numHeads: _numHeads,
      dropoutP: 0.0,
      tieWeights: true,
      seed: 1,
    ),
  );
  final params = gpt.parameters();
  print(
    'model : GPT (weight-tied) '
    '${_numLayers}L x ${_embedDim}d x ${_numHeads}h  '
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
    final loss = gpt(x).crossEntropy(y).mean();
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
  final prompt = tok.encode(promptStr).map((i) => i.toDouble()).toList();
  print('\nprompt: "$promptStr"');
  gpt.eval();
  final greedy = gpt.generate(
    prompt,
    maxNewTokens: 200,
    temperature: 0.0,
    useCache: true,
  );
  print('--- greedy ---');
  print(tok.decode(greedy.map((v) => v.toInt()).toList()));

  final sampled = gpt.generate(
    prompt,
    maxNewTokens: 200,
    temperature: 0.8,
    topK: 20,
    rng: math.Random(1),
    useCache: true,
  );
  print('\n--- T=0.8 top-k=20 ---');
  print(tok.decode(sampled.map((v) => v.toInt()).toList()));
}
