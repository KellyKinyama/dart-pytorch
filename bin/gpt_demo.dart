/// Minimal GPT-style character-level demo.
///
/// Trains a tiny weight-tied `GPT` on a short refrain and shows off
/// [GPT.generate] in three modes: greedy, temperature sampling, and
/// top-k sampling.
///
/// Run with:
///
///     dart run bin/gpt_demo.dart
library;

import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

const _corpus = 'to be or not to be that is the question ';

void main() {
  final chars = _corpus.split('').toSet().toList()..sort();
  final ch2id = <String, int>{for (int i = 0; i < chars.length; i++) chars[i]: i};
  final id2ch = chars;
  print('vocab (${chars.length}): ${chars.map((c) => c == ' ' ? '_' : c).join()}');

  final ids = _corpus.split('').map((c) => ch2id[c]!.toDouble()).toList();
  final input = ids.sublist(0, ids.length - 1);
  final target = ids.sublist(1);
  final n = input.length;
  print('sequence length: $n');

  final gpt = GPT(GPTConfig(
    vocabSize: chars.length,
    maxCtx: 64,
    embedDim: 32,
    numLayers: 2,
    numHeads: 4,
    dropoutP: 0.0,
    tieWeights: true,
    seed: 1,
  ));
  final opt = Adam(gpt.parameters(), lr: 5e-3);
  print('parameters: ${gpt.parameters().length} tensors, '
      '${gpt.parameters().fold<int>(0, (a, p) => a + p.length)} scalars '
      '(weight-tied)');

  final x = Tensor.fromList([n], input);
  final y = Tensor.fromList([n], target);

  const steps = 300;
  for (int step = 1; step <= steps; step++) {
    opt.zeroGrad();
    final loss = gpt(x).crossEntropy(y).mean();
    loss.backward();
    clipGradNorm(gpt.parameters(), 1.0);
    opt.step();
    if (step == 1 || step % 50 == 0) {
      print('step $step  loss=${loss.toList()[0].toStringAsFixed(4)}');
    }
  }

  String decode(List<double> ids) =>
      ids.map((v) => id2ch[v.toInt()]).join().replaceAll(' ', '_');

  final promptStr = 'to be ';
  final prompt = promptStr.split('').map((c) => ch2id[c]!.toDouble()).toList();
  print('\nprompt: "${promptStr.replaceAll(' ', '_')}"');

  final greedy = gpt.generate(prompt, maxNewTokens: 30, temperature: 0.0);
  print('greedy       : "${decode(greedy)}"');

  final rng = math.Random(0);
  final sampled = gpt.generate(
    prompt,
    maxNewTokens: 30,
    temperature: 0.8,
    rng: rng,
  );
  print('T=0.8        : "${decode(sampled)}"');

  final topk = gpt.generate(
    prompt,
    maxNewTokens: 30,
    temperature: 1.0,
    topK: 3,
    rng: math.Random(0),
  );
  print('T=1.0 top-k=3: "${decode(topk)}"');
}
