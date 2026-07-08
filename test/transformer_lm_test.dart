import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

void main() {
  group('causalMask', () {
    test('lower-triangular structure with -1e9 above diagonal', () {
      final m = causalMask(3).toList();
      // Expected 3x3:
      //   0     -1e9  -1e9
      //   0     0     -1e9
      //   0     0     0
      expect(m[0], 0.0);
      expect(m[1] < -1e8, isTrue);
      expect(m[2] < -1e8, isTrue);
      expect(m[3], 0.0);
      expect(m[4], 0.0);
      expect(m[5] < -1e8, isTrue);
      expect(m[6], 0.0);
      expect(m[7], 0.0);
      expect(m[8], 0.0);
    });

    test('respects custom blockValue and device', () {
      final m = causalMask(2, blockValue: -100.0).toList();
      expect(m, [0.0, -100.0, 0.0, 0.0]);

      final g = causalMask(2, device: Device.GPU);
      expect(g.device, Device.GPU);
      expect(g.toList(), [0.0, -1e9, 0.0, 0.0]);
      g.dispose();
    });
  });

  group('TransformerEncoder', () {
    test('output shape equals input shape', () {
      final enc = TransformerEncoder(3, 8, 2, seed: 5);
      final x = Tensor.fromList([
        4,
        8,
      ], List<double>.generate(32, (i) => math.sin(i * 0.3)));
      final y = enc(x);
      expect(y.shape, [4, 8]);
    });

    test('parameters scale with numLayers (+ final LN)', () {
      // Each block has 15 params (see TransformerBlock test).
      // 2 layers + final LN (2 params) = 32.
      final enc = TransformerEncoder(2, 8, 2, seed: 1);
      expect(enc.parameters().length, 2 * 15 + 2);
    });

    test('finalNorm: false skips the trailing LayerNorm', () {
      final enc = TransformerEncoder(1, 8, 2, seed: 2, finalNorm: false);
      expect(enc.parameters().length, 15); // just the one block
      expect(enc.finalNorm, isNull);
    });

    test('train()/eval() propagates to every submodule (dropouts)', () {
      final enc = TransformerEncoder(2, 4, 2, dropoutP: 0.5, seed: 3);
      expect(enc.training, isTrue);
      for (final b in enc.blocks) {
        expect(b.training, isTrue);
        expect(b.dropout.training, isTrue);
      }
      enc.eval();
      expect(enc.training, isFalse);
      for (final b in enc.blocks) {
        expect(b.training, isFalse);
        expect(b.dropout.training, isFalse);
      }
    });

    test('accepts an optional mask and every parameter gets a gradient', () {
      final enc = TransformerEncoder(2, 4, 2, seed: 4);
      final x = Tensor.fromList(
        [3, 4],
        [0.1, 0.2, 0.3, 0.4, -0.1, 0.0, 0.1, 0.2, 0.3, -0.2, 0.4, -0.4],
      );
      final mask = causalMask(3);
      enc(x, mask: mask).sum().backward();
      for (final p in enc.parameters()) {
        expect(p.grad, isNotNull, reason: 'missing grad on ${p.shape}');
      }
    });
  });

  group('TransformerLM', () {
    test('rejects rank>2 tokens and out-of-range seqLen', () {
      final lm = TransformerLM(
        vocabSize: 8,
        embedDim: 4,
        numLayers: 1,
        numHeads: 2,
        maxLen: 4,
        seed: 5,
      );
      expect(
        () => lm(Tensor.fromList([1, 2, 2], [0, 1, 2, 3])),
        throwsArgumentError,
      );
      expect(
        () => lm(Tensor.fromList([5], [0, 1, 2, 3, 4])),
        throwsArgumentError,
      );
      expect(
        () => lm(Tensor.fromList([2, 5], List<double>.filled(10, 0.0))),
        throwsArgumentError,
      );
    });

    test('forward produces [seqLen, vocabSize] logits', () {
      final lm = TransformerLM(
        vocabSize: 16,
        embedDim: 8,
        numLayers: 2,
        numHeads: 2,
        maxLen: 32,
        seed: 6,
      );
      final tokens = Tensor.fromList([5], [1.0, 3.0, 5.0, 7.0, 9.0]);
      final logits = lm(tokens);
      expect(logits.shape, [5, 16]);
    });

    test('parameters include token embed + encoder + head', () {
      final lm = TransformerLM(
        vocabSize: 8,
        embedDim: 4,
        numLayers: 1,
        numHeads: 2,
        maxLen: 8,
        seed: 7,
      );
      // Token emb: 1
      // Encoder: 15 (block) + 2 (final LN) = 17
      // Head Linear: 2
      // Total: 20
      expect(lm.parameters().length, 1 + 17 + 2);
    });

    test('overfits a tiny sequence — loss drops to near zero', () {
      // Toy next-token task on a fixed 8-token sequence over vocab=5.
      final lm = TransformerLM(
        vocabSize: 5,
        embedDim: 8,
        numLayers: 2,
        numHeads: 2,
        maxLen: 16,
        seed: 42,
      );
      final opt = Adam(lm.parameters(), lr: 0.05);

      final inputSeq = <double>[0, 1, 2, 3, 4, 0, 1, 2];
      final targetSeq = <double>[1, 2, 3, 4, 0, 1, 2, 3]; // shift by 1
      final x = Tensor.fromList([inputSeq.length], inputSeq);
      final y = Tensor.fromList([targetSeq.length], targetSeq);

      double lossVal() => lm(x).crossEntropy(y).mean().toList()[0];

      final initial = lossVal();
      for (int i = 0; i < 200; i++) {
        opt.zeroGrad();
        lm(x).crossEntropy(y).mean().backward();
        opt.step();
      }
      final finalL = lossVal();
      expect(
        finalL < 0.1,
        isTrue,
        reason: 'LM did not overfit: initial=$initial final=$finalL',
      );
    });

    test('greedy generation reproduces the memorized next-token map', () {
      // After overfitting `[i] -> [i+1 mod V]`, argmax at every position
      // in the training-length context should equal the corresponding
      // target. (We do NOT test rolling generation beyond the training
      // length — the model has only seen contexts of that length.)
      final lm = TransformerLM(
        vocabSize: 4,
        embedDim: 8,
        numLayers: 2,
        numHeads: 2,
        maxLen: 16,
        seed: 11,
      );
      final opt = Adam(lm.parameters(), lr: 0.05);
      final inputSeq = <double>[0, 1, 2, 3, 0, 1];
      final targetSeq = <double>[1, 2, 3, 0, 1, 2];
      final x = Tensor.fromList([inputSeq.length], inputSeq);
      final y = Tensor.fromList([targetSeq.length], targetSeq);
      for (int i = 0; i < 300; i++) {
        opt.zeroGrad();
        lm(x).crossEntropy(y).mean().backward();
        opt.step();
      }

      lm.eval();
      final logits = lm(x).toList();
      const vocab = 4;
      final preds = <double>[];
      for (int pos = 0; pos < inputSeq.length; pos++) {
        var best = 0;
        var bestVal = logits[pos * vocab];
        for (int v = 1; v < vocab; v++) {
          if (logits[pos * vocab + v] > bestVal) {
            bestVal = logits[pos * vocab + v];
            best = v;
          }
        }
        preds.add(best.toDouble());
      }
      expect(
        preds,
        targetSeq,
        reason: 'per-position argmax should match target',
      );
    });
  });
}
