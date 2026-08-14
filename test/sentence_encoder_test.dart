import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

Tensor _tokens(List<int> ids) =>
    Tensor.fromList([ids.length], ids.map((i) => i.toDouble()).toList());

void main() {
  group('poolTokens', () {
    test('mean pool averages rows to [1, D]', () {
      final x = Tensor.fromList(
        [3, 2],
        [
          1, 2, //
          3, 4, //
          5, 6, //
        ],
      );
      final y = poolTokens(x, PoolingMode.mean);
      expect(y.shape, [1, 2]);
      expect(y.toList(), [closeTo(3.0, 1e-6), closeTo(4.0, 1e-6)]);
    });

    test('cls pool selects row 0', () {
      final x = Tensor.fromList([3, 2], [1, 2, 3, 4, 5, 6]);
      final y = poolTokens(x, PoolingMode.cls);
      expect(y.shape, [1, 2]);
      expect(y.toList(), [1, 2]);
    });

    test('rejects empty sequence', () {
      final x = Tensor.fill([0, 4], 0.0);
      expect(
        () => poolTokens(x, PoolingMode.mean),
        throwsA(isA<ArgumentError>()),
      );
    });
  });

  group('l2NormalizeRow', () {
    test('unit-normalizes a [1, D] vector', () {
      final x = Tensor.fromList([1, 3], [3, 0, 4]);
      final y = l2NormalizeRow(x);
      final vals = y.toList();
      expect(vals[0], closeTo(0.6, 1e-5));
      expect(vals[1], closeTo(0.0, 1e-5));
      expect(vals[2], closeTo(0.8, 1e-5));
      final normSq = vals[0] * vals[0] + vals[1] * vals[1] + vals[2] * vals[2];
      expect(normSq, closeTo(1.0, 1e-4));
    });

    test('rejects non-[1, D] input', () {
      final x = Tensor.fill([2, 3], 0.0);
      expect(() => l2NormalizeRow(x), throwsA(isA<ArgumentError>()));
    });
  });

  group('SentenceEncoder', () {
    test('produces a [1, embedDim] embedding', () {
      final enc = SentenceEncoder(
        vocabSize: 32,
        maxSeqLen: 8,
        embedDim: 12,
        numLayers: 1,
        numHeads: 2,
        seed: 1,
      );
      final emb = enc(_tokens([1, 5, 9, 3]));
      expect(emb.shape, [1, 12]);
    });

    test('normalize=true yields a unit vector', () {
      final enc = SentenceEncoder(
        vocabSize: 32,
        maxSeqLen: 8,
        embedDim: 16,
        numLayers: 1,
        numHeads: 2,
        seed: 2,
      );
      final emb = enc(_tokens([1, 2, 3])).toList();
      var s = 0.0;
      for (final v in emb) {
        s += v * v;
      }
      expect(s, closeTo(1.0, 1e-3));
    });

    test('normalize=false skips the projection to the unit sphere', () {
      final enc = SentenceEncoder(
        vocabSize: 32,
        maxSeqLen: 8,
        embedDim: 16,
        numLayers: 1,
        numHeads: 2,
        normalize: false,
        seed: 3,
      );
      final emb = enc(_tokens([1, 2, 3])).toList();
      var s = 0.0;
      for (final v in emb) {
        s += v * v;
      }
      expect(s, isNot(closeTo(1.0, 1e-2)));
    });

    test('cls pool matches selecting the first token feature', () {
      final enc = SentenceEncoder(
        vocabSize: 32,
        maxSeqLen: 8,
        embedDim: 12,
        numLayers: 1,
        numHeads: 2,
        pooling: PoolingMode.cls,
        normalize: false,
        seed: 4,
      );
      final tokens = _tokens([7, 2, 4]);
      final features = enc.backbone(tokens);
      final firstRow = features.sliceRows(0, 1).toList();
      final pooled = enc(tokens).toList();
      for (int i = 0; i < firstRow.length; i++) {
        expect(pooled[i], closeTo(firstRow[i], 1e-5));
      }
    });

    test('encodeBatch stacks per-sentence embeddings into [N, D]', () {
      final enc = SentenceEncoder(
        vocabSize: 32,
        maxSeqLen: 8,
        embedDim: 8,
        numLayers: 1,
        numHeads: 2,
        seed: 5,
      );
      final batch = enc.encodeBatch([
        _tokens([1, 2, 3]),
        _tokens([4, 5]),
        _tokens([7, 6, 5, 4]),
      ]);
      expect(batch.shape, [3, 8]);
    });

    test('SentenceEncoder.wrap reuses an existing TextTransformer', () {
      final backbone = TextTransformer(
        vocabSize: 32,
        maxSeqLen: 8,
        embedDim: 10,
        numLayers: 1,
        numHeads: 2,
        seed: 6,
      );
      final enc = SentenceEncoder.wrap(backbone, normalize: true);
      final emb = enc(_tokens([1, 2, 3]));
      expect(emb.shape, [1, 10]);
      expect(enc.parameters(), backbone.parameters());
    });

    test('trains: MSE loss on a canary target decreases', () {
      final enc = SentenceEncoder(
        vocabSize: 16,
        maxSeqLen: 6,
        embedDim: 8,
        numLayers: 1,
        numHeads: 2,
        normalize: false,
        seed: 7,
      );
      final opt = SGD(enc.parameters(), lr: 0.1);
      final tokens = _tokens([1, 2, 3, 4]);
      final target = Tensor.fromList([1, 8], List<double>.filled(8, 0.5));
      Tensor step() {
        opt.zeroGrad();
        final y = enc(tokens);
        final diff = y - target;
        final loss = (diff * diff).mean();
        loss.backward();
        opt.step();
        return loss;
      }

      final before = step().toList()[0];
      double after = before;
      for (int i = 0; i < 30; i++) {
        after = step().toList()[0];
      }
      expect(after, lessThan(before));
    });
  });

  group('CrossEncoder', () {
    test('produces [1, numLabels] logits', () {
      final ce = CrossEncoder(
        vocabSize: 32,
        maxSeqLen: 8,
        embedDim: 12,
        numLayers: 1,
        numHeads: 2,
        seed: 11,
      );
      final logits = ce(_tokens([1, 2, 3, 4, 5]));
      expect(logits.shape, [1, 1]);
    });

    test('multi-label head returns [1, K]', () {
      final ce = CrossEncoder(
        vocabSize: 32,
        maxSeqLen: 8,
        embedDim: 12,
        numLayers: 1,
        numHeads: 2,
        numLabels: 3,
        seed: 12,
      );
      final logits = ce(_tokens([1, 2, 3]));
      expect(logits.shape, [1, 3]);
    });

    test('score() runs under noGrad and returns a probability in (0, 1)', () {
      final ce = CrossEncoder(
        vocabSize: 32,
        maxSeqLen: 8,
        embedDim: 12,
        numLayers: 1,
        numHeads: 2,
        seed: 13,
      );
      final s = ce.score(_tokens([1, 2, 3, 4]));
      expect(s, greaterThan(0.0));
      expect(s, lessThan(1.0));
    });
  });

  group('Similarity utilities', () {
    test('cosSim of a vector with itself is 1', () {
      final v = l2NormalizeRow(Tensor.fromList([1, 4], [1, 2, 3, 4]));
      expect(Similarity.cosSim(v, v), closeTo(1.0, 1e-4));
    });

    test('cosSim of orthogonal unit vectors is 0', () {
      final a = Tensor.fromList([1, 3], [1, 0, 0]);
      final b = Tensor.fromList([1, 3], [0, 1, 0]);
      expect(Similarity.cosSim(a, b), closeTo(0.0, 1e-6));
    });

    test('pairwise = A @ B^T', () {
      final a = Tensor.fromList([2, 3], [1, 0, 0, 0, 1, 0]);
      final b = Tensor.fromList([2, 3], [1, 0, 0, 0, 0, 1]);
      final m = Similarity.pairwise(a, b);
      expect(m.shape, [2, 2]);
      expect(m.toList(), [1, 0, 0, 0]);
    });

    test('semanticSearch returns top-k sorted by score descending', () {
      final corpus = Tensor.fromList(
        [4, 2],
        [
          1, 0, //
          0, 1, //
          1, 1, //
          -1, 0, //
        ],
      );
      final query = Tensor.fromList([1, 2], [1, 0]);
      final hits = Similarity.semanticSearch(query, corpus, topK: 3);
      expect(hits, hasLength(1));
      expect(hits[0], hasLength(3));
      // Row 0 = (1,0) → sim 1; row 2 = (1,1) → sim 1; row 1 = (0,1) → 0.
      // Row 3 = (-1,0) → -1. Top 3 should exclude row 3.
      final ids = hits[0].map((h) => h.corpusIndex).toSet();
      expect(ids.contains(3), isFalse);
      // Best hit's score must be >= second's.
      expect(hits[0][0].score, greaterThanOrEqualTo(hits[0][1].score));
    });
  });
}
