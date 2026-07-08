/// Tests for the Mixture-of-Experts feed-forward block and MoE
/// language model.
library;

import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

void main() {
  group('Expert', () {
    test('forward preserves [T, dim] shape', () {
      final e = Expert(4, 8, seed: 1);
      final x = Tensor.fromList([3, 4], List<double>.generate(12, (i) => i * 0.1));
      final y = e(x);
      expect(y.shape, [3, 4]);
    });

    test('parameters flatten w1 + w2', () {
      final e = Expert(4, 8, seed: 1);
      // Linear has weight + bias by default => 4 tensors total.
      expect(e.parameters().length, 4);
    });
  });

  group('MoEFeedForward', () {
    test('constructor rejects out-of-range topK', () {
      expect(
        () => MoEFeedForward(
          embedDim: 4,
          numRoutedExperts: 3,
          numSharedExperts: 0,
          topK: 0,
          expertHiddenDim: 8,
        ),
        throwsArgumentError,
      );
      expect(
        () => MoEFeedForward(
          embedDim: 4,
          numRoutedExperts: 3,
          numSharedExperts: 0,
          topK: 4,
          expertHiddenDim: 8,
        ),
        throwsArgumentError,
      );
    });

    test('forward preserves [T, embedDim]', () {
      final moe = MoEFeedForward(
        embedDim: 8,
        numRoutedExperts: 4,
        numSharedExperts: 1,
        topK: 2,
        expertHiddenDim: 16,
        seed: 7,
      );
      final x = Tensor.fromList(
        [6, 8],
        List<double>.generate(48, (i) => (i % 7) * 0.05),
      );
      final y = moe(x);
      expect(y.shape, [6, 8]);
    });

    test('routes each token to exactly topK experts', () {
      const t = 5;
      const e = 4;
      const k = 2;
      final moe = MoEFeedForward(
        embedDim: 4,
        numRoutedExperts: e,
        numSharedExperts: 0,
        topK: k,
        expertHiddenDim: 8,
        seed: 3,
      );
      final x = Tensor.fromList(
        [t, 4],
        List<double>.generate(t * 4, (i) => (i % 5) * 0.1),
      );
      moe(x);
      final total = moe.expertLoad.fold<int>(0, (a, b) => a + b);
      expect(total, t * k);
    });

    test('gradients flow to gateW and to expert weights', () {
      final moe = MoEFeedForward(
        embedDim: 4,
        numRoutedExperts: 3,
        numSharedExperts: 1,
        topK: 2,
        expertHiddenDim: 6,
        seed: 11,
      );
      final x = Tensor.fromList(
        [3, 4],
        List<double>.generate(12, (i) => (i % 4) * 0.1 - 0.15),
      );
      final y = moe(x);
      final loss = (y * y).sum() * 0.5;
      loss.backward();

      // gateW should have a non-null, non-zero gradient.
      expect(moe.gateW.grad, isNotNull);
      final gGate = moe.gateW.grad!.toList();
      final anyNonZero = gGate.any((v) => v.abs() > 1e-8);
      expect(anyNonZero, isTrue, reason: 'gateW gradient is all zero');

      // At least one routed expert must have a non-zero grad on its
      // first-layer weight (top-K guarantees at least topK experts
      // active across the batch).
      var routedAnyGrad = false;
      for (final e in moe.routedExperts) {
        final gw = e.w1.weight.grad;
        if (gw == null) continue;
        if (gw.toList().any((v) => v.abs() > 1e-8)) {
          routedAnyGrad = true;
          break;
        }
      }
      expect(routedAnyGrad, isTrue);

      // Shared expert always active => should always have gradient.
      final sharedG = moe.sharedExperts.first.w1.weight.grad;
      expect(sharedG, isNotNull);
      expect(sharedG!.toList().any((v) => v.abs() > 1e-8), isTrue);
    });

    test('updateRoutingBias equalises after biased routing', () {
      const e = 4;
      final moe = MoEFeedForward(
        embedDim: 4,
        numRoutedExperts: e,
        numSharedExperts: 0,
        topK: 1,
        expertHiddenDim: 4,
        biasUpdateRate: 0.5,
        seed: 5,
      );
      // Fake severely-lopsided load by directly poking the internal
      // counter through a synthesised forward: we run a batch that
      // routes largely to a single expert by leaving gateW random and
      // just observe that updateRoutingBias moves biases in the right
      // direction.
      final x = Tensor.fromList(
        [8, 4],
        List<double>.generate(32, (i) => (i % 3) * 0.1),
      );
      moe(x);
      final loadBefore = moe.expertLoad;
      final maxIdx = _argmax(loadBefore);
      final minIdx = _argmin(loadBefore);
      final biasesBefore = List<double>.of(moe.routingBias);
      moe.updateRoutingBias();
      // Under-utilised expert bias must not have decreased.
      expect(
        moe.routingBias[minIdx],
        greaterThanOrEqualTo(biasesBefore[minIdx]),
      );
      // Over-utilised expert bias must not have increased.
      expect(
        moe.routingBias[maxIdx],
        lessThanOrEqualTo(biasesBefore[maxIdx]),
      );
      // Counters reset.
      expect(moe.expertLoad, List<int>.filled(e, 0));
    });
  });

  group('MoELanguageModel', () {
    test('forward returns [seqLen, vocabSize] for 1D tokens', () {
      final lm = MoELanguageModel(
        vocabSize: 12,
        embedDim: 8,
        numLayers: 2,
        numHeads: 2,
        maxLen: 16,
        numRoutedExperts: 3,
        numSharedExperts: 1,
        topK: 2,
        expertHiddenDim: 16,
        seed: 1,
      );
      final tokens = Tensor.fromList(
        [7],
        List<double>.generate(7, (i) => (i % 12).toDouble()),
      );
      final logits = lm(tokens);
      expect(logits.shape, [7, 12]);
    });

    test('rejects non-1D input and oversize sequences', () {
      final lm = MoELanguageModel(
        vocabSize: 6,
        embedDim: 4,
        numLayers: 1,
        numHeads: 2,
        maxLen: 4,
        numRoutedExperts: 2,
        numSharedExperts: 0,
        topK: 1,
        expertHiddenDim: 4,
        seed: 0,
      );
      expect(
        () => lm(Tensor.fromList([2, 3], List<double>.filled(6, 0.0))),
        throwsArgumentError,
      );
      expect(
        () => lm(Tensor.fromList([5], List<double>.filled(5, 0.0))),
        throwsArgumentError,
      );
    });

    test('overfits a tiny next-token map', () {
      const vocab = 6;
      const seqLen = 8;
      final rng = math.Random(0);
      final tokens = List<double>.generate(
        seqLen,
        (_) => rng.nextInt(vocab).toDouble(),
      );
      // targets = tokens shifted by +1 mod vocab.
      final targets = List<double>.generate(
        seqLen,
        (i) => ((tokens[i].toInt() + 1) % vocab).toDouble(),
      );

      final lm = MoELanguageModel(
        vocabSize: vocab,
        embedDim: 8,
        numLayers: 2,
        numHeads: 2,
        maxLen: seqLen,
        numRoutedExperts: 3,
        numSharedExperts: 1,
        topK: 2,
        expertHiddenDim: 16,
        seed: 3,
      );
      final opt = Adam(lm.parameters(), lr: 5e-3);
      final x = Tensor.fromList([seqLen], tokens);
      final y = Tensor.fromList([seqLen], targets);

      double last = double.infinity;
      for (int step = 0; step < 200; step++) {
        opt.zeroGrad();
        final loss = lm(x).crossEntropy(y).mean();
        loss.backward();
        opt.step();
        last = loss.toList()[0];
      }
      expect(last, lessThan(0.2), reason: 'final loss = $last');
    });

    test('updateRoutingBias fans out across all blocks', () {
      final lm = MoELanguageModel(
        vocabSize: 8,
        embedDim: 4,
        numLayers: 2,
        numHeads: 2,
        maxLen: 8,
        numRoutedExperts: 3,
        numSharedExperts: 0,
        topK: 1,
        expertHiddenDim: 4,
        seed: 0,
      );
      final tokens = Tensor.fromList(
        [4],
        List<double>.generate(4, (i) => i.toDouble()),
      );
      lm(tokens);
      // Every block should now have non-zero load counters.
      for (final b in lm.blocks) {
        expect(b.moe.expertLoad.any((c) => c > 0), isTrue);
      }
      lm.updateRoutingBias();
      // And they should have been reset.
      for (final b in lm.blocks) {
        expect(b.moe.expertLoad.every((c) => c == 0), isTrue);
      }
    });
  });
}

int _argmax(List<int> xs) {
  var bi = 0;
  for (int i = 1; i < xs.length; i++) {
    if (xs[i] > xs[bi]) bi = i;
  }
  return bi;
}

int _argmin(List<int> xs) {
  var bi = 0;
  for (int i = 1; i < xs.length; i++) {
    if (xs[i] < xs[bi]) bi = i;
  }
  return bi;
}
