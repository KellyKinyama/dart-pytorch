/// Tiny-Shakespeare character-level GPT demo — **large GPU config**.
///
/// This is the "make my 6GB GPU beat CPU" version of
/// `shakespeare_gpt.dart`: same code path, ~168x more scalars
/// (~10.7M vs ~64k), longer context, and the whole model pinned to
/// [Device.GPU]. Trains for a few hundred steps and samples a
/// continuation, so it's a real (if short) training run rather than a
/// smoke test.
///
/// Sizing knobs are constants at the top so you can tune to your card.
/// Defaults target a 6 GB VRAM budget with headroom.
///
/// Run from the repository root:
///
///     dart run bin/shakespeare_gpt_big.dart
library;

import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

import 'shakespeare_util.dart';

// ~10.7M-scalar weight-tied GPT. Fits well inside 6 GB with room for
// forward activations, backward closures, and Adam moments.
const int _corpusChars = 200000;
const int _blockSize = 192;
const int _embedDim = 384;
const int _numLayers = 6;
const int _numHeads = 6;
const int _trainSteps = 500;
const int _logEvery = 10;
const double _lr = 3e-4;

void main() {
  print('=== shakespeare_gpt_big : large char-level GPT (GPU) ===');
  final text = loadCorpus(maxChars: _corpusChars);
  final tok = CharTokenizer.fromText(text);
  final ids = tok.encode(text);
  print(
    'corpus: ${text.length} chars, vocab: ${tok.vocabSize}, '
    'tokens: ${ids.length}',
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
      device: Device.GPU,
      seed: 1,
    ),
  );
  final params = gpt.parameters();
  print(
    'model : GPT (weight-tied, GPU) '
    '${_numLayers}L x ${_embedDim}d x ${_numHeads}h  '
    '${paramScalarCount(params)} scalars',
  );

  final opt = Adam(params, lr: _lr);
  final rng = math.Random(0);
  final sw = Stopwatch()..start();
  print(
    'training $_trainSteps steps '
    '(blockSize=$_blockSize, lr=$_lr, device=GPU)...',
  );
  double lastLoss = double.nan;
  for (int step = 1; step <= _trainSteps; step++) {
    opt.zeroGrad();
    final (x, y) = getWindow(ids, _blockSize, rng, device: Device.GPU);
    final loss = gpt(x).crossEntropy(y).mean();
    loss.backward();
    clipGradNorm(params, 1.0);
    opt.step();
    lastLoss = loss.toList()[0];
    if (step == 1 || step % _logEvery == 0 || step == _trainSteps) {
      final elapsed = sw.elapsedMilliseconds;
      final msPerStep = elapsed / step;
      final flag = lastLoss.isNaN
          ? ' NaN'
          : lastLoss.isInfinite
          ? ' inf'
          : lastLoss == 0.0
          ? ' ZERO'
          : '';
      print(
        '  step ${step.toString().padLeft(4)}  '
        'loss=$lastLoss$flag  '
        '(${msPerStep.toStringAsFixed(1)} ms/step)',
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
    maxNewTokens: 300,
    temperature: 0.0,
    useCache: true,
  );
  print('--- greedy ---');
  print(tok.decode(greedy.map((v) => v.toInt()).toList()));

  final sampled = gpt.generate(
    prompt,
    maxNewTokens: 300,
    temperature: 0.8,
    topK: 20,
    rng: math.Random(1),
    useCache: true,
  );
  print('\n--- T=0.8 top-k=20 ---');
  print(tok.decode(sampled.map((v) => v.toInt()).toList()));
}
