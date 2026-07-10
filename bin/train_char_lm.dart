/// Real character-level LM training on [TextTokenDataset].
///
/// End-to-end LM training pipeline exercising the new data loaders:
///
///  1. Writes a small public-domain corpus (three complete
///     19th/early-20th century poems by Blake, Dickinson, and
///     Sandburg — all long out of copyright) to a temp text file.
///     Repeated enough times to give ~4 KB of training material.
///
///  2. Builds a [CharTokenizer] over the raw text, then loads it
///     with [TextTokenDataset.fromFile] — exercises the real
///     file-decoding path. A separate val split is carved off the
///     tail 15 % of the token stream.
///
///  3. Trains a [TransformerLM] with next-token cross-entropy over
///     batches yielded by [DataLoader]. This is the exact code path
///     a user would follow to train on their own text corpus.
///
///  4. Reports train + val loss per epoch and generates a short
///     sample from a seed prompt via greedy autoregressive decoding.
///
/// Runs on CPU by default; pass `--gpu` for CUDA.
///
/// Run:
///
///     dart run bin/train_char_lm.dart          # CPU
///     dart run bin/train_char_lm.dart --gpu    # GPU
library;

import 'dart:io';
import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

import 'shakespeare_util.dart'
    show paramScalarCount, generateText, lastRowLogits;

const int _blockSize = 32;
const int _batchSize = 16;
const int _embedDim = 64;
const int _numLayers = 2;
const int _numHeads = 4;
const int _epochs = 4;
const int _stepsPerEpoch = 60;
const double _lr = 3e-3;
const int _sampleTokens = 120;

/// Public-domain corpus: three complete poems by William Blake
/// (1794), Emily Dickinson (~1861), and Carl Sandburg (1916). All
/// three predate 1929, comfortably in the US public domain.
const String _basePoems = '''
Tyger Tyger, burning bright,
In the forests of the night;
What immortal hand or eye,
Could frame thy fearful symmetry?

In what distant deeps or skies.
Burnt the fire of thine eyes?
On what wings dare he aspire?
What the hand, dare seize the fire?

And what shoulder, & what art,
Could twist the sinews of thy heart?
And when thy heart began to beat.
What dread hand? & what dread feet?

"Hope" is the thing with feathers -
That perches in the soul -
And sings the tune without the words -
And never stops - at all -

And sweetest - in the Gale - is heard -
And sore must be the storm -
That could abash the little Bird
That kept so many warm -

I've heard it in the chillest land -
And on the strangest Sea -
Yet - never - in Extremity,
It asked a crumb - of me.

The fog comes
on little cat feet.

It sits looking
over harbor and city
on silent haunches
and then moves on.
''';

/// Materialize a ~4 KB training corpus on disk (repeated poems).
File _writeCorpus() {
  final buf = StringBuffer();
  for (int r = 0; r < 4; r++) {
    buf.write(_basePoems);
    buf.write('\n');
  }
  final f = File(
    '${Directory.systemTemp.createTempSync('dp_lm_').path}/corpus.txt',
  );
  f.writeAsStringSync(buf.toString());
  return f;
}

/// Average cross-entropy loss over the val dataset, no grad.
double _evalValLoss(TransformerLM lm, TextTokenDataset val, Device device) {
  return Tensor.noGrad(() {
    double lossSum = 0;
    int n = 0;
    // Small val — iterate one sample at a time to avoid huge activations.
    for (int i = 0; i < val.length; i++) {
      final s = val[i];
      final loss = lm(s.input).crossEntropy(s.target).mean();
      lossSum += loss.toList()[0];
      n++;
    }
    return n == 0 ? double.nan : lossSum / n;
  });
}

/// Stack a batch of `LmSample` into `[B, blockSize]` input and target
/// tensors. Uses `.toList()` on inputs (they don't require grad).
(Tensor input, Tensor target) _stackBatch(
  List<LmSample> batch,
  Device device,
) {
  final b = batch.length;
  final n = batch[0].input.length;
  final xs = List<double>.filled(b * n, 0);
  final ys = List<double>.filled(b * n, 0);
  for (int i = 0; i < b; i++) {
    final xi = batch[i].input.toList();
    final yi = batch[i].target.toList();
    for (int k = 0; k < n; k++) {
      xs[i * n + k] = xi[k];
      ys[i * n + k] = yi[k];
    }
  }
  return (
    Tensor.fromList([b, n], xs, device: device),
    Tensor.fromList([b, n], ys, device: device),
  );
}

void main(List<String> args) {
  final device = args.contains('--gpu') ? Device.GPU : Device.CPU;
  print('=== train_char_lm (${device.name}) ===');

  // 1. Write real text corpus to disk.
  final corpusFile = _writeCorpus();
  try {
    final rawText = corpusFile.readAsStringSync();
    print('corpus:    ${corpusFile.path}');
    print('           ${rawText.length} chars');

    // 2. Build tokenizer + dataset.
    final tok = CharTokenizer.fromText(rawText);
    print('tokenizer: char, vocab=${tok.vocabSize}');

    // Load whole dataset first (to know token count), then re-split
    // by carving val off the tail.
    final allIds = tok.encode(rawText);
    final splitIdx = (allIds.length * 0.85).floor();
    // val gets `blockSize` leading tokens as prompt context.
    final trainTokens = allIds.sublist(0, splitIdx);
    final valTokens = allIds.sublist(splitIdx - _blockSize);

    final train = TextTokenDataset.fromTokens(
      trainTokens,
      blockSize: _blockSize,
      device: device,
    );
    final val = TextTokenDataset.fromTokens(
      valTokens,
      blockSize: _blockSize,
      device: device,
    );
    print('tokens:    ${allIds.length}');
    print('train/val: ${train.length} / ${val.length} sliding windows');

    // 3. Build model + optimizer.
    final lm = TransformerLM(
      vocabSize: tok.vocabSize,
      embedDim: _embedDim,
      numLayers: _numLayers,
      numHeads: _numHeads,
      maxLen: _blockSize,
      device: device,
      seed: 0,
    );
    final params = lm.parameters();
    print('params:    ${paramScalarCount(params)} scalars');

    final opt = Adam(params, lr: _lr);

    // 4. Baseline val loss.
    final baselineVal = _evalValLoss(lm, val, device);
    print('\nBEFORE  val CE = ${baselineVal.toStringAsFixed(4)}  '
        '(uniform ≈ ${math.log(tok.vocabSize).toStringAsFixed(4)})');

    // 5. Train.
    print('\ntraining $_epochs epochs × $_stepsPerEpoch steps '
        '(batch=$_batchSize, blockSize=$_blockSize, lr=$_lr)...');
    final sw = Stopwatch()..start();
    for (int epoch = 1; epoch <= _epochs; epoch++) {
      final loader = DataLoader<LmSample>(
        train,
        batchSize: _batchSize,
        shuffle: true,
        seed: epoch,
      );
      double lossSum = 0;
      int steps = 0;
      for (final batch in loader.batches()) {
        if (steps >= _stepsPerEpoch) break;
        opt.zeroGrad();
        final (xB, yB) = _stackBatch(batch, device);
        final loss = lm(xB).crossEntropy(yB).mean();
        loss.backward();
        clipGradNorm(params, 1.0);
        opt.step();
        lossSum += loss.toList()[0];
        steps++;
      }
      final trainAvg = steps == 0 ? double.nan : lossSum / steps;
      lm.eval();
      final valLoss = _evalValLoss(lm, val, device);
      lm.train();
      final ms = sw.elapsedMilliseconds;
      print('  epoch $epoch  '
          'train=${trainAvg.toStringAsFixed(4)}  '
          'val=${valLoss.toStringAsFixed(4)}  '
          '(${(ms / epoch).toStringAsFixed(0)} ms/epoch cumulative)');
    }
    sw.stop();

    // 6. Sample text from trained model.
    lm.eval();
    const promptStr = 'The fog';
    final prompt = tok.encode(promptStr);

    List<double> stepFn(List<double> ctx) {
      final x = Tensor.fromList([ctx.length], ctx, device: device);
      return Tensor.noGrad(() => lastRowLogits(lm(x), tok.vocabSize));
    }

    final greedy = generateText(
      prompt,
      maxNewTokens: _sampleTokens,
      maxCtx: _blockSize,
      vocabSize: tok.vocabSize,
      stepFn: stepFn,
      temperature: 0.0,
    );
    print('\n--- greedy from "$promptStr" ---');
    print(tok.decode(greedy));

    final sampled = generateText(
      prompt,
      maxNewTokens: _sampleTokens,
      maxCtx: _blockSize,
      vocabSize: tok.vocabSize,
      stepFn: stepFn,
      temperature: 0.8,
      topK: 10,
      rng: math.Random(1),
    );
    print('\n--- T=0.8 top-k=10 from "$promptStr" ---');
    print(tok.decode(sampled));
  } finally {
    corpusFile.parent.deleteSync(recursive: true);
  }
}
