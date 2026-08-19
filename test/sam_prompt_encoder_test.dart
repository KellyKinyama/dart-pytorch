@Timeout(Duration(minutes: 2))
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

Tensor _fake(List<int> shape, {int seed = 0}) {
  final rng = math.Random(seed);
  var n = 1;
  for (final d in shape) {
    n *= d;
  }
  final v = Float32List(n);
  for (int i = 0; i < n; i++) {
    v[i] = rng.nextDouble() - 0.5;
  }
  return Tensor.fromFloat32List(shape, v);
}

void main() {
  group('SamPositionEmbeddingRandom', () {
    test('encodeCoords output shape [N, 2·numPosFeats]', () {
      final pe = SamPositionEmbeddingRandom(numPosFeats: 8);
      final xy = Tensor.fromList([3, 2], [0.1, 0.2, 0.5, 0.5, 0.9, 0.7]);
      expect(pe.encodeCoords(xy).shape, equals([3, 16]));
    });

    test('encodeGrid produces [2·F, H, W]', () {
      final pe = SamPositionEmbeddingRandom(numPosFeats: 8);
      final g = pe.encodeGrid(4, 4);
      expect(g.shape, equals([16, 4, 4]));
    });

    test('deterministic across construction — seed 3141592 fixed', () {
      final a = SamPositionEmbeddingRandom(numPosFeats: 4);
      final b = SamPositionEmbeddingRandom(numPosFeats: 4);
      final aVals = a.gaussianMatrix.toList();
      final bVals = b.gaussianMatrix.toList();
      expect(aVals, equals(bVals));
    });

    test('different coords produce different features', () {
      final pe = SamPositionEmbeddingRandom(numPosFeats: 8);
      final xy = Tensor.fromList([2, 2], [0.1, 0.2, 0.9, 0.7]);
      final enc = pe.encodeCoords(xy).toList();
      double maxDiff = 0;
      for (int j = 0; j < 16; j++) {
        final d = (enc[j] - enc[16 + j]).abs();
        if (d > maxDiff) maxDiff = d;
      }
      expect(maxDiff > 0, isTrue);
    });

    test('rejects wrong-shape input', () {
      final pe = SamPositionEmbeddingRandom(numPosFeats: 8);
      expect(
        () => pe.encodeCoords(Tensor.fromList([3], [1, 2, 3])),
        throwsArgumentError,
      );
      expect(
        () => pe.encodeCoords(
          Tensor.fromList([3, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ),
        throwsArgumentError,
      );
    });
  });

  group('SamPromptEncoder — sparse (points)', () {
    test('positive click yields [1, embedDim] with type-embedding added', () {
      final enc = SamPromptEncoder(
        embedDim: 32,
        imageEmbedH: 4,
        imageEmbedW: 4,
        imageSize: 32,
        maskInSize: 16,
      );
      final pts = Tensor.fromList([1, 2], [16.0, 16.0]);
      final labels = [SamPointType.positive];
      final sparse = enc.encodeSparse(pointsXY: pts, pointLabels: labels);
      expect(sparse.shape, equals([1, 32]));
    });

    test('two clicks (pos + neg) yields [2, embedDim]', () {
      final enc = SamPromptEncoder(
        embedDim: 32,
        imageEmbedH: 4,
        imageEmbedW: 4,
        imageSize: 32,
        maskInSize: 16,
      );
      final pts = Tensor.fromList([2, 2], [10.0, 20.0, 25.0, 5.0]);
      final labels = [SamPointType.positive, SamPointType.negative];
      expect(
        enc.encodeSparse(pointsXY: pts, pointLabels: labels).shape,
        equals([2, 32]),
      );
    });

    test('positive and negative clicks at the same coord differ only by '
        'the type embedding', () {
      final enc = SamPromptEncoder(
        embedDim: 32,
        imageEmbedH: 4,
        imageEmbedW: 4,
        imageSize: 32,
        maskInSize: 16,
      );
      final pt = Tensor.fromList([1, 2], [16.0, 16.0]);
      final posOut = enc
          .encodeSparse(pointsXY: pt, pointLabels: [SamPointType.positive])
          .toList();
      final negOut = enc
          .encodeSparse(pointsXY: pt, pointLabels: [SamPointType.negative])
          .toList();
      final ptEmb = enc.pointEmbeddings.toList();
      // posOut - negOut should equal pointEmbeddings[POS] - pointEmbeddings[NEG].
      final pos = SamPointType.positive.index;
      final neg = SamPointType.negative.index;
      for (int j = 0; j < 32; j++) {
        final expected = ptEmb[pos * 32 + j] - ptEmb[neg * 32 + j];
        final actual = posOut[j] - negOut[j];
        expect(
          (actual - expected).abs() < 1e-5,
          isTrue,
          reason: 'j=$j: expected $expected, got $actual',
        );
      }
    });

    test('length mismatch throws', () {
      final enc = SamPromptEncoder(
        embedDim: 32,
        imageEmbedH: 4,
        imageEmbedW: 4,
        imageSize: 32,
        maskInSize: 16,
      );
      final pts = Tensor.fromList([2, 2], [1.0, 2.0, 3.0, 4.0]);
      expect(
        () => enc.encodeSparse(
          pointsXY: pts,
          pointLabels: [SamPointType.positive],
        ),
        throwsArgumentError,
      );
    });
  });

  group('SamPromptEncoder — sparse (boxes)', () {
    test('one box yields [2, embedDim] (top-left + bottom-right)', () {
      final enc = SamPromptEncoder(
        embedDim: 32,
        imageEmbedH: 4,
        imageEmbedW: 4,
        imageSize: 32,
        maskInSize: 16,
      );
      final boxes = Tensor.fromList([1, 4], [4.0, 4.0, 28.0, 28.0]);
      expect(enc.encodeSparse(boxesXYXY: boxes).shape, equals([2, 32]));
    });

    test('N boxes + M points yields [M + 2N, embedDim]', () {
      final enc = SamPromptEncoder(
        embedDim: 32,
        imageEmbedH: 4,
        imageEmbedW: 4,
        imageSize: 32,
        maskInSize: 16,
      );
      final pts = Tensor.fromList([2, 2], [10.0, 20.0, 25.0, 5.0]);
      final labels = [SamPointType.positive, SamPointType.negative];
      final boxes = Tensor.fromList(
        [2, 4],
        [
          4, 4, 20, 20, //
          8, 8, 24, 24, //
        ],
      );
      final out = enc.encodeSparse(
        pointsXY: pts,
        pointLabels: labels,
        boxesXYXY: boxes,
      );
      expect(out.shape, equals([2 + 4, 32]));
    });

    test('no prompts yields [0, embedDim]', () {
      final enc = SamPromptEncoder(
        embedDim: 32,
        imageEmbedH: 4,
        imageEmbedW: 4,
        imageSize: 32,
        maskInSize: 16,
      );
      final out = enc.encodeSparse();
      expect(out.shape, equals([0, 32]));
    });
  });

  group('SamPromptEncoder — dense', () {
    test('encodeDense with mask=null broadcasts the no-mask embedding', () {
      final enc = SamPromptEncoder(
        embedDim: 8,
        imageEmbedH: 4,
        imageEmbedW: 4,
        imageSize: 32,
        maskInSize: 16,
      );
      final dense = enc.encodeDense();
      expect(dense.shape, equals([8, 4, 4]));
      // For each channel c, all H·W pixels equal noMaskEmbedding[c].
      final vals = dense.toList();
      final noMask = enc.noMaskEmbedding.toList();
      for (int c = 0; c < 8; c++) {
        final expected = noMask[c];
        for (int i = 0; i < 16; i++) {
          expect((vals[c * 16 + i] - expected).abs() < 1e-6, isTrue);
        }
      }
    });

    test('encodeDense with real mask returns [embedDim, H, W]', () {
      final enc = SamPromptEncoder(
        embedDim: 8,
        imageEmbedH: 4,
        imageEmbedW: 4,
        imageSize: 32,
        maskInSize: 16, // 16 / 2 / 2 = 4 = imageEmbedH ✓
      );
      // Mask input is [1, 1, 16, 16].
      final mask = _fake([1, 1, 16, 16], seed: 11);
      final dense = enc.encodeDense(mask: mask);
      expect(dense.shape, equals([8, 4, 4]));
    });

    test('rejects wrong-shape mask', () {
      final enc = SamPromptEncoder(
        embedDim: 8,
        imageEmbedH: 4,
        imageEmbedW: 4,
        imageSize: 32,
        maskInSize: 16,
      );
      expect(
        () => enc.encodeDense(mask: _fake([1, 1, 8, 8], seed: 12)),
        throwsArgumentError,
      );
    });
  });

  group('SamPromptEncoder — image PE', () {
    test('imagePositionEmbedding returns [embedDim, H, W]', () {
      final enc = SamPromptEncoder(
        embedDim: 8,
        imageEmbedH: 4,
        imageEmbedW: 4,
        imageSize: 32,
        maskInSize: 16,
      );
      expect(enc.imagePositionEmbedding().shape, equals([8, 4, 4]));
    });
  });
}
