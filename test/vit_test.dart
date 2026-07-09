// Smoke tests for the Vision Transformer backbone / heads.
//
// These aren't full parity tests — the point is that a small ViT:
//   * accepts a `[numPatches, patchPixels]` input,
//   * produces the expected output shape from the backbone / heads,
//   * has all parameters connected to the loss (nonzero grads),
//   * and can take one Adam step without exploding.
//
// Both CPU and GPU device placements are exercised.

import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

Tensor _randomImage(int numPatches, int patchPixels,
    {required Device device, int seed = 0}) {
  final rng = math.Random(seed);
  final vals = List<double>.generate(
    numPatches * patchPixels,
    (_) => rng.nextDouble() * 2 - 1,
  );
  return Tensor.fromList([numPatches, patchPixels], vals, device: device);
}

void main() {
  // Small config so tests stay fast on CPU too.
  const imageSize = 8;
  const patchSize = 4;
  const numChannels = 3;
  const embedDim = 16;
  const numLayers = 2;
  const numHeads = 2;
  const numPatches = (imageSize ~/ patchSize) * (imageSize ~/ patchSize); // 4
  const patchPixels = patchSize * patchSize * numChannels; // 48

  group('ViTBackbone', () {
    test('CPU forward has shape [numPatches + 1, embedDim]', () {
      final vit = ViTBackbone(
        imageSize: imageSize,
        patchSize: patchSize,
        numChannels: numChannels,
        embedDim: embedDim,
        numLayers: numLayers,
        numHeads: numHeads,
        device: Device.CPU,
        seed: 0,
      );
      final x = _randomImage(numPatches, patchPixels, device: Device.CPU);
      final y = vit(x);
      expect(y.shape, equals([numPatches + 1, embedDim]));
      for (final v in y.toList()) {
        expect(v.isFinite, isTrue);
      }
    });

    test('GPU forward matches CPU shape', () {
      final vit = ViTBackbone(
        imageSize: imageSize,
        patchSize: patchSize,
        numChannels: numChannels,
        embedDim: embedDim,
        numLayers: numLayers,
        numHeads: numHeads,
        device: Device.GPU,
        seed: 1,
      );
      final x = _randomImage(numPatches, patchPixels, device: Device.GPU,
          seed: 42);
      final y = vit(x);
      expect(y.shape, equals([numPatches + 1, embedDim]));
      for (final v in y.toList()) {
        expect(v.isFinite, isTrue);
      }
    });

    test('vitClsFeature returns [1, embedDim] equal to encoded row 0', () {
      final vit = ViTBackbone(
        imageSize: imageSize,
        patchSize: patchSize,
        numChannels: numChannels,
        embedDim: embedDim,
        numLayers: numLayers,
        numHeads: numHeads,
        device: Device.CPU,
        seed: 2,
      );
      final x = _randomImage(numPatches, patchPixels, device: Device.CPU);
      final y = vit(x);
      final cls = vitClsFeature(y);
      expect(cls.shape, equals([1, embedDim]));
      final yList = y.toList();
      final clsList = cls.toList();
      for (int j = 0; j < embedDim; j++) {
        expect((yList[j] - clsList[j]).abs() < 1e-6, isTrue,
            reason: 'col $j: encoded=${yList[j]} vs cls=${clsList[j]}');
      }
    });
  });

  group('ViTClassifier', () {
    test('CPU trains one step, loss decreases', () {
      const numClasses = 5;
      final clf = ViTClassifier(
        imageSize: imageSize,
        patchSize: patchSize,
        numChannels: numChannels,
        embedDim: embedDim,
        numClasses: numClasses,
        numLayers: numLayers,
        numHeads: numHeads,
        device: Device.CPU,
        seed: 3,
      );
      final x = _randomImage(numPatches, patchPixels, device: Device.CPU);
      final target = Tensor.fromList([1], [2], device: Device.CPU);
      final opt = Adam(clf.parameters(), lr: 1e-2);

      final logitsBefore = clf(x);
      final lossBefore = logitsBefore.crossEntropy(target).mean().toList()[0];

      for (int i = 0; i < 5; i++) {
        opt.zeroGrad();
        final logits = clf(x);
        final loss = logits.crossEntropy(target).mean();
        loss.backward();
        opt.step();
      }

      final logitsAfter = clf(x);
      final lossAfter = logitsAfter.crossEntropy(target).mean().toList()[0];
      expect(lossAfter < lossBefore, isTrue,
          reason: 'loss did not decrease: $lossBefore -> $lossAfter');
    });

    test('GPU forward produces [1, numClasses] logits', () {
      const numClasses = 5;
      final clf = ViTClassifier(
        imageSize: imageSize,
        patchSize: patchSize,
        numChannels: numChannels,
        embedDim: embedDim,
        numClasses: numClasses,
        numLayers: numLayers,
        numHeads: numHeads,
        device: Device.GPU,
        seed: 4,
      );
      final x = _randomImage(numPatches, patchPixels, device: Device.GPU);
      final logits = clf(x);
      expect(logits.shape, equals([1, numClasses]));
      for (final v in logits.toList()) {
        expect(v.isFinite, isTrue);
      }
    });
  });

  group('ViTFaceEmbedding', () {
    test('CPU output is [1, outputDim] and unit-norm', () {
      const outputDim = 8;
      final face = ViTFaceEmbedding(
        imageSize: imageSize,
        patchSize: patchSize,
        numChannels: numChannels,
        embedDim: embedDim,
        outputDim: outputDim,
        numLayers: numLayers,
        numHeads: numHeads,
        device: Device.CPU,
        seed: 5,
      );
      final x = _randomImage(numPatches, patchPixels, device: Device.CPU);
      final e = face(x);
      expect(e.shape, equals([1, outputDim]));
      final vals = e.toList();
      var sumSq = 0.0;
      for (final v in vals) {
        expect(v.isFinite, isTrue);
        sumSq += v * v;
      }
      expect((sumSq - 1.0).abs() < 1e-4, isTrue,
          reason: 'expected unit L2 norm, got sqrt(sumSq)=${math.sqrt(sumSq)}');
    });

    test('cosineSimilarity of a vector with itself is ~1', () {
      const outputDim = 8;
      final face = ViTFaceEmbedding(
        imageSize: imageSize,
        patchSize: patchSize,
        numChannels: numChannels,
        embedDim: embedDim,
        outputDim: outputDim,
        numLayers: numLayers,
        numHeads: numHeads,
        device: Device.CPU,
        seed: 6,
      );
      final x = _randomImage(numPatches, patchPixels, device: Device.CPU);
      final e1 = face(x);
      final e2 = face(x);
      final sim = ViTFaceEmbedding.cosineSimilarity(e1, e2).toList()[0];
      expect((sim - 1.0).abs() < 1e-4, isTrue,
          reason: 'self-similarity should be 1, got $sim');
    });

    test('GPU output has correct shape and finite values', () {
      const outputDim = embedDim; // triggers no-projection code path
      final face = ViTFaceEmbedding(
        imageSize: imageSize,
        patchSize: patchSize,
        numChannels: numChannels,
        embedDim: embedDim,
        outputDim: outputDim,
        numLayers: numLayers,
        numHeads: numHeads,
        device: Device.GPU,
        seed: 7,
      );
      final x = _randomImage(numPatches, patchPixels, device: Device.GPU);
      final e = face(x);
      expect(e.shape, equals([1, outputDim]));
      for (final v in e.toList()) {
        expect(v.isFinite, isTrue);
      }
    });
  });
}
