import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

Tensor _unitRows(List<List<double>> rows) {
  final n = rows.length;
  final d = rows[0].length;
  final vals = <double>[];
  for (final r in rows) {
    var s = 0.0;
    for (final v in r) {
      s += v * v;
    }
    final norm = math.sqrt(s);
    for (final v in r) {
      vals.add(v / norm);
    }
  }
  return Tensor.fromList([n, d], vals);
}

void main() {
  group('multipleNegativesRankingLoss', () {
    test('identical anchor/positive pairs give low loss', () {
      // Perfectly aligned pairs — the diagonal of A @ P.T dominates,
      // CE loss shrinks with scale.
      final a = _unitRows([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]);
      final loss = SentenceLosses.multipleNegativesRankingLoss(a, a);
      expect(loss.toList()[0], lessThan(0.01));
    });

    test('scrambled positives give higher loss than aligned', () {
      final a = _unitRows([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]);
      final aligned = SentenceLosses.multipleNegativesRankingLoss(
        a,
        a,
      ).toList()[0];
      final shuffled = _unitRows([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
      ]);
      final misaligned = SentenceLosses.multipleNegativesRankingLoss(
        a,
        shuffled,
      ).toList()[0];
      expect(misaligned, greaterThan(aligned));
    });

    test('rejects shape mismatch', () {
      final a = Tensor.fill([2, 4], 0.0);
      final p = Tensor.fill([2, 3], 0.0);
      expect(
        () => SentenceLosses.multipleNegativesRankingLoss(a, p),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('one training step drives an untrained encoder toward its pair', () {
      final enc = SentenceEncoder(
        vocabSize: 16,
        maxSeqLen: 6,
        embedDim: 8,
        numLayers: 1,
        numHeads: 2,
        seed: 21,
      );
      final opt = SGD(enc.parameters(), lr: 0.5);
      final anchors = [
        Tensor.fromList([3], [1, 2, 3]),
        Tensor.fromList([3], [4, 5, 6]),
      ];
      final positives = [
        Tensor.fromList([3], [7, 2, 1]),
        Tensor.fromList([3], [4, 8, 6]),
      ];

      Tensor step() {
        opt.zeroGrad();
        final a = enc.encodeBatch(anchors);
        final p = enc.encodeBatch(positives);
        final loss = SentenceLosses.multipleNegativesRankingLoss(a, p);
        loss.backward();
        opt.step();
        return loss;
      }

      final before = step().toList()[0];
      double after = before;
      for (int i = 0; i < 15; i++) {
        after = step().toList()[0];
      }
      expect(after, lessThan(before));
    });
  });

  group('cosineSimilarityLoss', () {
    test('perfect match with target=1 gives ~0 loss', () {
      final a = _unitRows([
        [1, 0],
        [0, 1],
      ]);
      final targets = Tensor.fromList([2], [1.0, 1.0]);
      final loss = SentenceLosses.cosineSimilarityLoss(a, a, targets);
      expect(loss.toList()[0], closeTo(0.0, 1e-6));
    });

    test('opposite vectors with target=1 give loss = 4', () {
      // cos = -1, target = 1, diff = -2, mean of squares = 4.
      final a = _unitRows([
        [1, 0],
      ]);
      final b = _unitRows([
        [-1, 0],
      ]);
      final loss = SentenceLosses.cosineSimilarityLoss(
        a,
        b,
        Tensor.fromList([1], [1.0]),
      );
      expect(loss.toList()[0], closeTo(4.0, 1e-4));
    });

    test('rejects mismatched targets length', () {
      final a = Tensor.fill([3, 4], 0.0);
      final b = Tensor.fill([3, 4], 0.0);
      final t = Tensor.fill([2], 0.0);
      expect(
        () => SentenceLosses.cosineSimilarityLoss(a, b, t),
        throwsA(isA<ArgumentError>()),
      );
    });
  });

  group('tripletLoss', () {
    test('when positive is nearer than negative by > margin, loss = 0', () {
      final a = _unitRows([
        [1, 0, 0],
      ]);
      final p = _unitRows([
        [1, 0, 0],
      ]);
      final n = _unitRows([
        [0, 1, 0],
      ]);
      // cosSim(a,p) = 1, cosSim(a,n) = 0. Gap = 0 - 1 + 0.5 = -0.5 → relu → 0.
      final loss = SentenceLosses.tripletLoss(a, p, n);
      expect(loss.toList()[0], closeTo(0.0, 1e-6));
    });

    test('when negative is nearer than positive, loss > 0', () {
      final a = _unitRows([
        [1, 0, 0],
      ]);
      final p = _unitRows([
        [0, 1, 0],
      ]);
      final n = _unitRows([
        [1, 0, 0],
      ]);
      // cosSim(a,p) = 0, cosSim(a,n) = 1. Gap = 1 - 0 + 0.5 = 1.5.
      final loss = SentenceLosses.tripletLoss(a, p, n);
      expect(loss.toList()[0], closeTo(1.5, 1e-4));
    });

    test('margin controls the loss floor', () {
      final a = _unitRows([
        [1, 0, 0],
      ]);
      final p = _unitRows([
        [1, 0, 0],
      ]);
      final n = _unitRows([
        [1, 0, 0],
      ]);
      // cos(a,p) = cos(a,n) = 1. Gap = 0 + margin.
      final loss = SentenceLosses.tripletLoss(a, p, n, margin: 0.3);
      expect(loss.toList()[0], closeTo(0.3, 1e-5));
    });

    test('rejects triplet shape mismatch', () {
      final a = Tensor.fill([2, 3], 0.0);
      final p = Tensor.fill([2, 3], 0.0);
      final n = Tensor.fill([3, 3], 0.0);
      expect(
        () => SentenceLosses.tripletLoss(a, p, n),
        throwsA(isA<ArgumentError>()),
      );
    });
  });
}
