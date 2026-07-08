/// Minimal character-level language-model demo.
///
/// Trains a tiny 2-layer `TransformerLM` to overfit a short refrain by
/// predicting the next character, then samples greedily to reproduce
/// the memorized continuation.
///
/// Run with:
///
///     dart run bin/lm_demo.dart
///
/// This is a smoke test / example, not a benchmark. Expect the loss to
/// drop from ~2.5 down to well below 0.5 in a couple hundred steps on
/// CPU (the whole model is a handful of KiB of parameters).
library;

import 'package:dart_pytorch/dart_pytorch.dart';

const _corpus = 'hello world hello dart hello ';

void main() {
  // Build a tiny closed-vocabulary alphabet from the corpus.
  final chars = _corpus.split('').toSet().toList()..sort();
  final ch2id = <String, int>{
    for (int i = 0; i < chars.length; i++) chars[i]: i,
  };
  final id2ch = chars;
  print(
    'vocab (${chars.length}): ${chars.map((c) => c == ' ' ? '_' : c).join()}',
  );

  // Encode the corpus and split into input / target (shifted by one).
  final ids = _corpus.split('').map((c) => ch2id[c]!.toDouble()).toList();
  final input = ids.sublist(0, ids.length - 1);
  final target = ids.sublist(1);
  final n = input.length;
  print('sequence length: $n');

  // Small model.
  final lm = TransformerLM(
    vocabSize: chars.length,
    embedDim: 32,
    numLayers: 2,
    numHeads: 4,
    maxLen: 128,
    dropoutP: 0.0,
    seed: 1,
  );
  final opt = Adam(lm.parameters(), lr: 5e-3);
  print(
    'parameters: ${lm.parameters().length} tensors, '
    '${lm.parameters().fold<int>(0, (a, p) => a + p.length)} scalars',
  );

  final x = Tensor.fromList([n], input);
  final y = Tensor.fromList([n], target);

  const steps = 300;
  for (int step = 1; step <= steps; step++) {
    opt.zeroGrad();
    final loss = lm(x).crossEntropy(y).mean();
    loss.backward();
    // Modest clip — keeps training stable when the LM starts to
    // overfit and gradients spike on rare tokens.
    clipGradNorm(lm.parameters(), 1.0);
    opt.step();
    if (step == 1 || step % 50 == 0) {
      print('step $step  loss=${loss.toList()[0].toStringAsFixed(4)}');
    }
  }

  // Greedy per-position argmax on the training sequence — should match
  // the target essentially perfectly after overfitting.
  lm.eval();
  final logits = lm(x).toList();
  final preds = <String>[];
  for (int pos = 0; pos < n; pos++) {
    var best = 0;
    var bestVal = logits[pos * chars.length];
    for (int v = 1; v < chars.length; v++) {
      if (logits[pos * chars.length + v] > bestVal) {
        bestVal = logits[pos * chars.length + v];
        best = v;
      }
    }
    preds.add(id2ch[best]);
  }
  final inputStr = input.map((v) => id2ch[v.toInt()]).join();
  final predStr = preds.join();
  final targetStr = target.map((v) => id2ch[v.toInt()]).join();
  print('input   : "$inputStr"');
  print('target  : "$targetStr"');
  print('predicted: "$predStr"');

  final matches = List<int>.generate(
    n,
    (i) => preds[i] == id2ch[target[i].toInt()] ? 1 : 0,
  ).reduce((a, b) => a + b);
  print(
    'accuracy: $matches / $n = '
    '${(100.0 * matches / n).toStringAsFixed(1)}%',
  );
}
