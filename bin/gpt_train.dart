/// End-to-end tiny training pipeline that composes everything the
/// library provides so far:
///
///   * a small hard-coded prose corpus,
///   * a byte-level BPE tokenizer trained on that corpus,
///   * a small `GPT` model,
///   * `Adam` with `LinearWarmupCosineDecay`,
///   * real `[B, S]` batched forward + backward,
///   * checkpoint save + reload,
///   * `GPT.generate` from a seed prompt with the KV cache.
///
/// The intent is a fast-running demo (< ~30 s on CPU) that exercises
/// every piece together — not to produce quality samples.
///
/// Run with:
///
///     dart run bin/gpt_train.dart
library;

import 'dart:io';
import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

// A short chunk of public-domain text (excerpted / paraphrased from
// the King James Bible book of Ecclesiastes). ~700 chars — enough for
// BPE to find meaningful merges without slowing the demo down.
const _corpus =
    'to every thing there is a season and a time to every purpose under the heaven. '
    'a time to be born and a time to die. a time to plant and a time to pluck up '
    'that which is planted. a time to kill and a time to heal. a time to break down '
    'and a time to build up. a time to weep and a time to laugh. a time to mourn '
    'and a time to dance. a time to cast away stones and a time to gather stones '
    'together. a time to embrace and a time to refrain from embracing. a time to '
    'get and a time to lose. a time to keep and a time to cast away. a time to '
    'rend and a time to sew. a time to keep silence and a time to speak. ';

void main() {
  print('=== dart-pytorch : end-to-end training demo ===\n');
  print('corpus: ${_corpus.length} chars\n');

  // ---------------- BPE tokenizer ----------------
  final tok = BpeTokenizer.train(_corpus, targetVocabSize: 320);
  print(
    'BPE vocab size    : ${tok.vocabSize} '
    '(${tok.merges.length} merges over 256 bytes)',
  );
  final ids = tok.encode(_corpus);
  print(
    'encoded length    : ${ids.length} tokens '
    '(compression x${(_corpus.length / ids.length).toStringAsFixed(2)})',
  );
  final roundTrip = tok.decode(ids);
  if (roundTrip != _corpus) {
    stderr.writeln('!! BPE round-trip mismatch');
    exit(1);
  }
  print('BPE round-trip    : OK\n');

  // ---------------- Model ----------------
  const maxCtx = 32;
  final gpt = GPT(
    GPTConfig(
      vocabSize: tok.vocabSize,
      maxCtx: maxCtx,
      embedDim: 32,
      numLayers: 2,
      numHeads: 4,
      dropoutP: 0.0,
      tieWeights: true,
      seed: 1,
    ),
  );
  final numScalars = gpt.parameters().fold<int>(0, (a, p) => a + p.length);
  print(
    'model params      : ${gpt.parameters().length} tensors, '
    '$numScalars scalars (weight-tied)',
  );
  print('context length    : $maxCtx\n');

  // ---------------- Optimizer + scheduler ----------------
  const totalSteps = 200;
  const warmupSteps = 20;
  const batchSize = 4; // real [B, S] batched forward per opt step
  final opt = Adam(gpt.parameters(), lr: 0.0);
  final sched = LinearWarmupCosineDecay(
    opt,
    warmupSteps: warmupSteps,
    totalSteps: totalSteps,
    maxLr: 3e-3,
    minLr: 3e-4,
  );
  print(
    'optimizer         : Adam + LinearWarmupCosineDecay '
    '(warmup=$warmupSteps, total=$totalSteps, maxLr=3e-3, minLr=3e-4)',
  );
  print('batch size        : $batchSize (real [B, S] batched forward)\n');

  // ---------------- Training ----------------
  final rng = math.Random(0);
  final trainable = ids.length - maxCtx - 1;
  if (trainable <= 0) {
    stderr.writeln('!! corpus too short for maxCtx=$maxCtx');
    exit(1);
  }
  final sw = Stopwatch()..start();

  print('training ...');
  final xBuf = List<double>.filled(batchSize * maxCtx, 0.0);
  final yBuf = List<double>.filled(batchSize * maxCtx, 0.0);
  for (int step = 1; step <= totalSteps; step++) {
    opt.zeroGrad();

    // Build a [B, S] batch of random windows.
    for (int b = 0; b < batchSize; b++) {
      final start = rng.nextInt(trainable);
      for (int t = 0; t < maxCtx; t++) {
        xBuf[b * maxCtx + t] = ids[start + t].toDouble();
        yBuf[b * maxCtx + t] = ids[start + t + 1].toDouble();
      }
    }
    final x = Tensor.fromList([batchSize, maxCtx], List<double>.from(xBuf));
    final y = Tensor.fromList([batchSize, maxCtx], List<double>.from(yBuf));

    final loss = gpt(x).crossEntropy(y).mean();
    loss.backward();
    final avg = loss.toList()[0];

    clipGradNorm(gpt.parameters(), 1.0);
    opt.step();
    sched.step();

    if (step == 1 || step % 20 == 0 || step == totalSteps) {
      print(
        '  step ${step.toString().padLeft(3)}  '
        'loss=${avg.toStringAsFixed(4)}  '
        'lr=${opt.lr.toStringAsExponential(2)}',
      );
    }
  }
  sw.stop();
  print('training done in ${sw.elapsedMilliseconds} ms\n');

  // ---------------- Checkpoint ----------------
  final ckptDir = Directory.systemTemp.createTempSync('gpt_train_ckpt_');
  final ckptPath = '${ckptDir.path}/gpt.dptc';
  final tokPath = '${ckptDir.path}/tok.json';
  Checkpoint.saveFile(gpt, ckptPath);
  tok.saveFile(tokPath);
  final ckptSize = File(ckptPath).lengthSync();
  final tokSize = File(tokPath).lengthSync();
  print('checkpoint        : $ckptPath ($ckptSize bytes)');
  print('tokenizer         : $tokPath ($tokSize bytes)\n');

  // ---------------- Reload + sample ----------------
  final gpt2 = GPT(gpt.config); // fresh (untrained) instance
  final tok2 = BpeTokenizer.loadFile(tokPath);
  print('reloaded fresh instance...');
  print('encoder-parity    : ${_encoderParity(tok, tok2) ? "OK" : "FAIL"}');
  Checkpoint.loadIntoFile(gpt2, ckptPath);

  const promptStr = 'a time to ';
  final prompt = tok2.encode(promptStr).map((i) => i.toDouble()).toList();
  gpt2.eval();
  final generatedGreedy = gpt2.generate(
    prompt,
    maxNewTokens: 40,
    temperature: 0.0,
    useCache: true,
  );
  final generatedSampled = gpt2.generate(
    prompt,
    maxNewTokens: 40,
    temperature: 0.8,
    topK: 8,
    rng: math.Random(1),
    useCache: true,
  );

  String render(List<double> ids) =>
      tok2.decode(ids.map((v) => v.toInt()).toList()).replaceAll('\n', ' ');

  print('\nprompt   : "$promptStr"');
  print('greedy   : "${render(generatedGreedy)}"');
  print('T=0.8 k=8: "${render(generatedSampled)}"');

  // Cleanup
  ckptDir.deleteSync(recursive: true);
  print('\ndone.');
}

bool _encoderParity(BpeTokenizer a, BpeTokenizer b) {
  const probes = ['a time to', 'season and a time', 'the heaven'];
  for (final p in probes) {
    if (a.encode(p).toString() != b.encode(p).toString()) return false;
  }
  return true;
}
