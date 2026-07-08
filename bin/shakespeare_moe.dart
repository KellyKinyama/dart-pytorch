/// Tiny-Shakespeare character-level demo using the
/// [MoELanguageModel] — mixture-of-experts causal LM with
/// DeepSeek-V3-style top-K routing and aux-loss-free load balancing.
///
/// Each block replaces the dense FFN with `numRoutedExperts` sparse
/// experts (top-K selected per token) + `numSharedExperts` always-on
/// shared experts. `updateRoutingBias` is called every
/// `_biasUpdateEvery` steps to keep expert load balanced without an
/// auxiliary loss term.
///
/// Run from the repository root:
///
///     dart run bin/shakespeare_moe.dart
library;

import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

import 'shakespeare_util.dart';

const int _corpusChars = 50000;
const int _blockSize = 32;
const int _embedDim = 64;
const int _numLayers = 2;
const int _numHeads = 4;
const int _numRoutedExperts = 4;
const int _numSharedExperts = 1;
const int _topK = 2;
const int _expertHiddenDim = 128;
const int _trainSteps = 500;
const int _biasUpdateEvery = 50;
const double _lr = 3e-3;
const int _sampleTokens = 200;

void main() {
  print('=== shakespeare_moe : MoE Transformer LM ===');
  final text = loadCorpus(maxChars: _corpusChars);
  final tok = CharTokenizer.fromText(text);
  final ids = tok.encode(text);
  print(
    'corpus: ${text.length} chars, vocab: ${tok.vocabSize}, tokens: ${ids.length}',
  );

  final lm = MoELanguageModel(
    vocabSize: tok.vocabSize,
    embedDim: _embedDim,
    numLayers: _numLayers,
    numHeads: _numHeads,
    maxLen: _blockSize,
    numRoutedExperts: _numRoutedExperts,
    numSharedExperts: _numSharedExperts,
    topK: _topK,
    expertHiddenDim: _expertHiddenDim,
    biasUpdateRate: 0.01,
    seed: 1,
  );
  final params = lm.parameters();
  print(
    'model : MoE LM ${_numLayers}L x ${_embedDim}d x ${_numHeads}h  '
    '(routed=$_numRoutedExperts shared=$_numSharedExperts topK=$_topK '
    'hidden=$_expertHiddenDim)  ${paramScalarCount(params)} scalars',
  );

  final opt = Adam(params, lr: _lr);
  final rng = math.Random(0);
  final sw = Stopwatch()..start();
  print(
    'training $_trainSteps steps (blockSize=$_blockSize, lr=$_lr, '
    'bias-update every $_biasUpdateEvery steps)...',
  );
  double lastLoss = double.nan;
  for (int step = 1; step <= _trainSteps; step++) {
    opt.zeroGrad();
    final (x, y) = getWindow(ids, _blockSize, rng);
    final loss = lm(x).crossEntropy(y).mean();
    loss.backward();
    clipGradNorm(params, 1.0);
    opt.step();
    lastLoss = loss.toList()[0];
    if (step % _biasUpdateEvery == 0) {
      lm.updateRoutingBias();
    }
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

  // Report routing bias distribution after training — should have
  // spread out from zero as the balancer nudged loads.
  print('\nfinal routing bias per block:');
  for (int b = 0; b < lm.blocks.length; b++) {
    final biases = lm.blocks[b].moe.routingBias;
    final formatted = biases.map((v) => v.toStringAsFixed(3)).join(', ');
    print('  block $b: [$formatted]');
  }

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
