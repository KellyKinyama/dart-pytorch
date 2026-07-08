import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

void main() {
  group('GPT (tied)', () {
    test('forward shape [seqLen, vocabSize]', () {
      final m = GPT(
        GPTConfig(
          vocabSize: 12,
          maxCtx: 16,
          embedDim: 8,
          numLayers: 2,
          numHeads: 2,
          seed: 1,
        ),
      );
      final tokens = Tensor.fromList([5], [0, 1, 2, 3, 4]);
      final logits = m(tokens);
      expect(logits.shape, [5, 12]);
    });

    test('rejects non-1D tokens and seqLen > maxCtx', () {
      final m = GPT(
        GPTConfig(
          vocabSize: 8,
          maxCtx: 4,
          embedDim: 4,
          numLayers: 1,
          numHeads: 2,
          seed: 2,
        ),
      );
      expect(
        () => m(Tensor.fromList([2, 2], [0, 1, 2, 3])),
        throwsArgumentError,
      );
      expect(
        () => m(Tensor.fromList([5], [0, 1, 2, 3, 4])),
        throwsArgumentError,
      );
    });

    test(
      'weight tying: no separate head, tokenEmb grad accumulates via both paths',
      () {
        final m = GPT(
          GPTConfig(
            vocabSize: 6,
            maxCtx: 8,
            embedDim: 4,
            numLayers: 1,
            numHeads: 2,
            seed: 3,
          ),
        );
        expect(m.untiedHead, isNull);

        // Params: tokenEmb (1) + posEmb.table (1) + one block (15) + finalLN (2) = 19
        expect(m.parameters().length, 1 + 1 + 15 + 2);

        final tokens = Tensor.fromList([3], [1.0, 2.0, 3.0]);
        final targets = Tensor.fromList([3], [2.0, 3.0, 4.0]);
        m(tokens).crossEntropy(targets).mean().backward();

        final wg = m.tokenEmb.weight.grad;
        expect(wg, isNotNull);
        // At least one nonzero — both the embedding lookup rows and the
        // head projection contribute, so no row can be exactly zero.
        final gVals = wg!.toList();
        expect(gVals.any((v) => v.abs() > 1e-8), isTrue);
      },
    );

    test(
      'untied variant has a separate head Linear with (V*D) extra params',
      () {
        final m = GPT(
          GPTConfig(
            vocabSize: 6,
            maxCtx: 8,
            embedDim: 4,
            numLayers: 1,
            numHeads: 2,
            tieWeights: false,
            seed: 4,
          ),
        );
        expect(m.untiedHead, isNotNull);
        // 19 shared + 1 head weight (bias=false) = 20.
        expect(m.parameters().length, 1 + 1 + 15 + 2 + 1);
      },
    );

    test('train/eval propagates to embed dropout and every block dropout', () {
      final m = GPT(
        GPTConfig(
          vocabSize: 6,
          maxCtx: 8,
          embedDim: 4,
          numLayers: 2,
          numHeads: 2,
          dropoutP: 0.5,
          seed: 5,
        ),
      );
      expect(m.embedDrop.training, isTrue);
      for (final b in m.encoder.blocks) {
        expect(b.dropout.training, isTrue);
      }
      m.eval();
      expect(m.embedDrop.training, isFalse);
      for (final b in m.encoder.blocks) {
        expect(b.dropout.training, isFalse);
      }
    });

    test('overfits a tiny sequence — loss drops to near zero', () {
      final m = GPT(
        GPTConfig(
          vocabSize: 5,
          maxCtx: 16,
          embedDim: 8,
          numLayers: 2,
          numHeads: 2,
          seed: 42,
        ),
      );
      final opt = Adam(m.parameters(), lr: 0.05);
      final x = Tensor.fromList([8], [0, 1, 2, 3, 4, 0, 1, 2]);
      final y = Tensor.fromList([8], [1, 2, 3, 4, 0, 1, 2, 3]);
      double loss() => m(x).crossEntropy(y).mean().toList()[0];
      final initial = loss();
      for (int i = 0; i < 250; i++) {
        opt.zeroGrad();
        m(x).crossEntropy(y).mean().backward();
        opt.step();
      }
      final finalL = loss();
      expect(
        finalL < 0.1,
        isTrue,
        reason: 'GPT did not overfit: initial=$initial final=$finalL',
      );
    });
  });

  group('GPT.generate', () {
    test('greedy generation extends prompt and length matches', () {
      final m = GPT(
        GPTConfig(
          vocabSize: 5,
          maxCtx: 16,
          embedDim: 8,
          numLayers: 2,
          numHeads: 2,
          seed: 7,
        ),
      );
      final out = m.generate([0.0, 1.0], maxNewTokens: 4, temperature: 0.0);
      expect(out.length, 6);
      expect(out.sublist(0, 2), [0.0, 1.0]);
      for (final t in out.sublist(2)) {
        expect(t >= 0 && t < 5, isTrue);
      }
    });

    test('greedy generation is deterministic without an RNG', () {
      final m = GPT(
        GPTConfig(
          vocabSize: 5,
          maxCtx: 16,
          embedDim: 8,
          numLayers: 2,
          numHeads: 2,
          seed: 8,
        ),
      );
      final a = m.generate([0.0, 1.0, 2.0], maxNewTokens: 5, temperature: 0.0);
      final b = m.generate([0.0, 1.0, 2.0], maxNewTokens: 5, temperature: 0.0);
      expect(a, b);
    });

    test('topK=1 collapses to argmax regardless of temperature', () {
      final m = GPT(
        GPTConfig(
          vocabSize: 5,
          maxCtx: 16,
          embedDim: 8,
          numLayers: 1,
          numHeads: 2,
          seed: 9,
        ),
      );
      final greedy = m.generate([1.0], maxNewTokens: 6, temperature: 0.0);
      final k1 = m.generate(
        [1.0],
        maxNewTokens: 6,
        temperature: 2.0,
        topK: 1,
        rng: math.Random(0),
      );
      expect(greedy, k1);
    });

    test('sampled generation with fixed RNG is reproducible and in-vocab', () {
      final m = GPT(
        GPTConfig(
          vocabSize: 7,
          maxCtx: 12,
          embedDim: 8,
          numLayers: 1,
          numHeads: 2,
          seed: 10,
        ),
      );
      final a = m.generate(
        [0.0, 1.0],
        maxNewTokens: 8,
        temperature: 1.0,
        rng: math.Random(1234),
      );
      final b = m.generate(
        [0.0, 1.0],
        maxNewTokens: 8,
        temperature: 1.0,
        rng: math.Random(1234),
      );
      expect(a, b);
      for (final t in a) {
        expect(t >= 0 && t < 7, isTrue);
      }
    });

    test(
      'useCache=false slides context when prompt+generated exceeds maxCtx',
      () {
        final m = GPT(
          GPTConfig(
            vocabSize: 5,
            maxCtx: 4,
            embedDim: 8,
            numLayers: 1,
            numHeads: 2,
            seed: 11,
          ),
        );
        // Cache-mode can't slide (see next test); disable it and each
        // step re-runs on the last `maxCtx` tokens.
        final out = m.generate(
          [0.0, 1.0, 2.0, 3.0],
          maxNewTokens: 6,
          temperature: 0.0,
          useCache: false,
        );
        expect(out.length, 10);
      },
    );

    test('useCache=true stops when the cache is full', () {
      final m = GPT(
        GPTConfig(
          vocabSize: 5,
          maxCtx: 4,
          embedDim: 8,
          numLayers: 1,
          numHeads: 2,
          seed: 11,
        ),
      );
      // Prompt already fills the cache — only one additional token is
      // sampled (from the prompt-fill logits), then the loop halts.
      final out = m.generate(
        [0.0, 1.0, 2.0, 3.0],
        maxNewTokens: 6,
        temperature: 0.0,
      );
      expect(out.length, 5);
    });

    test('useCache=true rejects a prompt longer than maxCtx', () {
      final m = GPT(
        GPTConfig(
          vocabSize: 5,
          maxCtx: 3,
          embedDim: 8,
          numLayers: 1,
          numHeads: 2,
          seed: 11,
        ),
      );
      expect(
        () =>
            m.generate([0.0, 1.0, 2.0, 3.0], maxNewTokens: 2, temperature: 0.0),
        throwsArgumentError,
      );
    });

    test('after generate() the training flag is restored', () {
      final m = GPT(
        GPTConfig(
          vocabSize: 5,
          maxCtx: 8,
          embedDim: 8,
          numLayers: 1,
          numHeads: 2,
          seed: 12,
        ),
      );
      expect(m.training, isTrue);
      m.generate([0.0], maxNewTokens: 2, temperature: 0.0);
      expect(m.training, isTrue);
      m.eval();
      m.generate([0.0], maxNewTokens: 2, temperature: 0.0);
      expect(m.training, isFalse);
    });
  });
}
