import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

const int _imageSize = 16;
const int _patchSize = 4;
const int _numChannels = 3;
const int _numPatches = 16;
const int _patchPixels = _patchSize * _patchSize * _numChannels;

List<double> _randomPatches(math.Random rng) => List<double>.generate(
  _numPatches * _patchPixels,
  (_) => rng.nextDouble() * 2 - 1,
);

void main() {
  group('ViTObjectDetector', () {
    test('forward shapes (CPU)', () {
      final det = ViTObjectDetector(
        imageSize: _imageSize,
        patchSize: _patchSize,
        numChannels: _numChannels,
        embedDim: 16,
        numClasses: 5,
        numQueries: 3,
        numLayers: 1,
        numHeads: 2,
        seed: 0,
      );
      final rng = math.Random(0);
      final x = Tensor.fromList([
        _numPatches,
        _patchPixels,
      ], _randomPatches(rng));
      final out = det(x);
      expect(out['logits']!.shape, [3, 6]);
      expect(out['boxes']!.shape, [3, 4]);

      // Boxes must be in [0, 1] because of sigmoid.
      for (final v in out['boxes']!.toList()) {
        expect(v, greaterThanOrEqualTo(0.0));
        expect(v, lessThanOrEqualTo(1.0));
      }
    });

    test('one-step Adam decreases loss (fixed-order MSE)', () {
      final det = ViTObjectDetector(
        imageSize: _imageSize,
        patchSize: _patchSize,
        numChannels: _numChannels,
        embedDim: 16,
        numClasses: 5,
        numQueries: 3,
        numLayers: 1,
        numHeads: 2,
        seed: 1,
      );
      final params = det.parameters();
      final opt = Adam(params, lr: 3e-3);

      final rng = math.Random(2);
      final x = Tensor.fromList([
        _numPatches,
        _patchPixels,
      ], _randomPatches(rng));
      final gtClasses = Tensor.fromList([3], [1.0, 2.0, 5.0]);
      final gtBoxes = Tensor.fromList(
        [3, 4],
        [0.1, 0.1, 0.2, 0.2, 0.5, 0.5, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0],
      );

      double totalLoss() {
        final preds = det(x);
        final cls = preds['logits']!.crossEntropy(gtClasses).mean();
        final diff = preds['boxes']! - gtBoxes;
        final boxL = (diff * diff).mean();
        return (cls + boxL).toList()[0];
      }

      final start = totalLoss();
      for (int i = 0; i < 5; i++) {
        opt.zeroGrad();
        final preds = det(x);
        final cls = preds['logits']!.crossEntropy(gtClasses).mean();
        final diff = preds['boxes']! - gtBoxes;
        final boxL = (diff * diff).mean();
        final loss = cls + boxL;
        loss.backward();
        opt.step();
      }
      final end = totalLoss();
      expect(end, lessThan(start));
    });
  });

  group('HungarianAlgorithm', () {
    test('minimum-cost assignment on a small matrix', () {
      // Optimal is 0->1 (2) + 1->2 (7) + 2->0 (4) = 13.
      final costs = [
        [8, 2, 6],
        [5, 9, 7],
        [4, 6, 7],
      ];
      final assign = HungarianAlgorithm(costs).getAssignment();
      int total = 0;
      for (int i = 0; i < 3; i++) {
        total += costs[i][assign[i]];
      }
      expect(total, 13);
      expect(assign, [1, 2, 0]);
    });

    test('identity when diagonal is cheapest', () {
      final costs = [
        [1, 9, 9],
        [9, 1, 9],
        [9, 9, 1],
      ];
      final assign = HungarianAlgorithm(costs).getAssignment();
      expect(assign, [0, 1, 2]);
    });
  });
}
