// Smoke tests for the multi-modal transformer modules.
//
// Covers:
//   * AudioTransformer / VideoTransformer / TextTransformer encoders
//     produce the right sequence shape and finite outputs.
//   * meanRows collapses [N, D] -> [1, D] with the expected values.
//   * MultiModalClassifier fuses audio+video (and optionally text)
//     into class logits and takes an Adam step that decreases loss.
//   * MultiModalEncoder concatenates unimodal sequences into a joint
//     sequence and passes it through a fusion encoder.

import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

Tensor _rand2D(int r, int c, {required Device device, int seed = 0}) {
  final rng = math.Random(seed);
  final vals = List<double>.generate(r * c, (_) => rng.nextDouble() * 2 - 1);
  return Tensor.fromList([r, c], vals, device: device);
}

Tensor _tokens(List<int> ids, {required Device device}) => Tensor.fromList(
  [ids.length],
  ids.map((i) => i.toDouble()).toList(),
  device: device,
);

void main() {
  group('meanRows', () {
    test('collapses [N, D] to [1, D] with correct values (CPU)', () {
      final x = Tensor.fromList([3, 2], [1, 2, 3, 4, 5, 6]);
      final m = meanRows(x);
      expect(m.shape, equals([1, 2]));
      expect(m.toList(), [3.0, 4.0]); // means of cols
    });

    test('GPU parity for a small tensor', () {
      final xCpu = Tensor.fromList([4, 3], List.generate(12, (i) => i * 0.5));
      final xGpu = Tensor.fromList(
        [4, 3],
        List.generate(12, (i) => i * 0.5),
        device: Device.GPU,
      );
      final mCpu = meanRows(xCpu).toList();
      final mGpu = meanRows(xGpu).toList();
      for (int i = 0; i < mCpu.length; i++) {
        expect((mCpu[i] - mGpu[i]).abs() < 1e-4, isTrue);
      }
    });
  });

  group('AudioTransformer', () {
    const featureDim = 40; // MFCC-ish
    const embedDim = 32;
    const maxSeqLen = 20;

    Tensor _mkAudio(int seqLen, {required Device device, int seed = 0}) =>
        _rand2D(seqLen, featureDim, device: device, seed: seed);

    test('CPU forward produces [seqLen, embedDim] with finite values', () {
      final m = AudioTransformer(
        featureDim: featureDim,
        maxSeqLen: maxSeqLen,
        embedDim: embedDim,
        numLayers: 2,
        numHeads: 4,
        device: Device.CPU,
        seed: 0,
      );
      final y = m(_mkAudio(15, device: Device.CPU));
      expect(y.shape, equals([15, embedDim]));
      for (final v in y.toList()) expect(v.isFinite, isTrue);
    });

    test('GPU forward has the same shape', () {
      final m = AudioTransformer(
        featureDim: featureDim,
        maxSeqLen: maxSeqLen,
        embedDim: embedDim,
        numLayers: 2,
        numHeads: 4,
        device: Device.GPU,
        seed: 1,
      );
      final y = m(_mkAudio(12, device: Device.GPU));
      expect(y.shape, equals([12, embedDim]));
      for (final v in y.toList()) expect(v.isFinite, isTrue);
    });

    test('poolMean returns [1, embedDim]', () {
      final m = AudioTransformer(
        featureDim: featureDim,
        maxSeqLen: maxSeqLen,
        embedDim: embedDim,
        numLayers: 1,
        numHeads: 4,
        device: Device.CPU,
        seed: 2,
      );
      final p = m.poolMean(_mkAudio(10, device: Device.CPU));
      expect(p.shape, equals([1, embedDim]));
    });
  });

  group('VideoTransformer', () {
    const frameDim = 24;
    const embedDim = 32;
    const maxFrames = 16;

    Tensor _mkVideo(int n, {required Device device, int seed = 0}) =>
        _rand2D(n, frameDim, device: device, seed: seed);

    test('CPU forward — with frame projection (frameDim != embedDim)', () {
      final m = VideoTransformer(
        frameFeatureDim: frameDim,
        maxFrames: maxFrames,
        embedDim: embedDim,
        numLayers: 2,
        numHeads: 4,
        device: Device.CPU,
        seed: 3,
      );
      expect(m.frameProjection, isNotNull);
      final y = m(_mkVideo(8, device: Device.CPU));
      expect(y.shape, equals([8, embedDim]));
    });

    test('no frame projection when frameFeatureDim == embedDim', () {
      final m = VideoTransformer(
        frameFeatureDim: embedDim,
        maxFrames: maxFrames,
        embedDim: embedDim,
        numLayers: 2,
        numHeads: 4,
        device: Device.CPU,
        seed: 4,
      );
      expect(m.frameProjection, isNull);
      final y = m(_rand2D(6, embedDim, device: Device.CPU));
      expect(y.shape, equals([6, embedDim]));
    });
  });

  group('TextTransformer', () {
    const vocabSize = 20;
    const embedDim = 32;
    const maxSeqLen = 16;

    test('CPU forward on token indices', () {
      final m = TextTransformer(
        vocabSize: vocabSize,
        maxSeqLen: maxSeqLen,
        embedDim: embedDim,
        numLayers: 2,
        numHeads: 4,
        device: Device.CPU,
        seed: 5,
      );
      final y = m(_tokens([1, 2, 3, 4, 5], device: Device.CPU));
      expect(y.shape, equals([5, embedDim]));
      for (final v in y.toList()) expect(v.isFinite, isTrue);
    });
  });

  group('MultiModalClassifier', () {
    const audioFeatDim = 16;
    const videoFeatDim = 20;
    const embedA = 24;
    const embedV = 24;
    const numClasses = 4;

    test('CPU audio+video forward produces [1, numClasses]', () {
      final clf = MultiModalClassifier(
        audio: AudioTransformer(
          featureDim: audioFeatDim,
          maxSeqLen: 12,
          embedDim: embedA,
          numLayers: 1,
          numHeads: 4,
          seed: 6,
        ),
        video: VideoTransformer(
          frameFeatureDim: videoFeatDim,
          maxFrames: 10,
          embedDim: embedV,
          numLayers: 1,
          numHeads: 4,
          seed: 7,
        ),
        numClasses: numClasses,
        seed: 8,
      );
      final logits = clf(
        _rand2D(8, audioFeatDim, device: Device.CPU, seed: 10),
        _rand2D(6, videoFeatDim, device: Device.CPU, seed: 11),
      );
      expect(logits.shape, equals([1, numClasses]));
    });

    test('CPU trains one class in 5 steps — loss decreases', () {
      final clf = MultiModalClassifier(
        audio: AudioTransformer(
          featureDim: audioFeatDim,
          maxSeqLen: 12,
          embedDim: embedA,
          numLayers: 1,
          numHeads: 4,
          seed: 9,
        ),
        video: VideoTransformer(
          frameFeatureDim: videoFeatDim,
          maxFrames: 10,
          embedDim: embedV,
          numLayers: 1,
          numHeads: 4,
          seed: 10,
        ),
        numClasses: numClasses,
        seed: 11,
      );
      final a = _rand2D(8, audioFeatDim, device: Device.CPU, seed: 12);
      final v = _rand2D(6, videoFeatDim, device: Device.CPU, seed: 13);
      final target = Tensor.fromList([1], [2]);
      final opt = Adam(clf.parameters(), lr: 1e-2);
      final before = clf(a, v).crossEntropy(target).mean().toList()[0];
      for (int i = 0; i < 5; i++) {
        opt.zeroGrad();
        final loss = clf(a, v).crossEntropy(target).mean();
        loss.backward();
        opt.step();
      }
      final after = clf(a, v).crossEntropy(target).mean().toList()[0];
      expect(
        after < before,
        isTrue,
        reason: 'multimodal loss did not decrease: $before -> $after',
      );
    });

    test('audio+video+text forward produces logits', () {
      final clf = MultiModalClassifier(
        audio: AudioTransformer(
          featureDim: audioFeatDim,
          maxSeqLen: 12,
          embedDim: embedA,
          numLayers: 1,
          numHeads: 4,
          seed: 12,
        ),
        video: VideoTransformer(
          frameFeatureDim: videoFeatDim,
          maxFrames: 10,
          embedDim: embedV,
          numLayers: 1,
          numHeads: 4,
          seed: 13,
        ),
        text: TextTransformer(
          vocabSize: 20,
          maxSeqLen: 12,
          embedDim: 16,
          numLayers: 1,
          numHeads: 4,
          seed: 14,
        ),
        numClasses: numClasses,
        seed: 15,
      );
      final logits = clf(
        _rand2D(8, audioFeatDim, device: Device.CPU, seed: 16),
        _rand2D(6, videoFeatDim, device: Device.CPU, seed: 17),
        textIn: _tokens([1, 2, 3, 4], device: Device.CPU),
      );
      expect(logits.shape, equals([1, numClasses]));
    });

    test('mismatched text encoder / textIn throws', () {
      final clf = MultiModalClassifier(
        audio: AudioTransformer(
          featureDim: audioFeatDim,
          maxSeqLen: 8,
          embedDim: embedA,
          numLayers: 1,
          numHeads: 4,
          seed: 18,
        ),
        video: VideoTransformer(
          frameFeatureDim: videoFeatDim,
          maxFrames: 8,
          embedDim: embedV,
          numLayers: 1,
          numHeads: 4,
          seed: 19,
        ),
        numClasses: numClasses,
        seed: 20,
      );
      expect(
        () => clf(
          _rand2D(4, audioFeatDim, device: Device.CPU),
          _rand2D(4, videoFeatDim, device: Device.CPU),
          textIn: _tokens([1, 2], device: Device.CPU),
        ),
        throwsArgumentError,
      );
    });
  });

  group('MultiModalEncoder', () {
    const jointDim = 24;

    test('joint sequence length = sum of unimodal sequence lengths', () {
      final enc = MultiModalEncoder(
        audio: AudioTransformer(
          featureDim: 16,
          maxSeqLen: 12,
          embedDim: jointDim,
          numLayers: 1,
          numHeads: 4,
          seed: 21,
        ),
        video: VideoTransformer(
          frameFeatureDim: 20,
          maxFrames: 10,
          embedDim: jointDim,
          numLayers: 1,
          numHeads: 4,
          seed: 22,
        ),
        text: TextTransformer(
          vocabSize: 20,
          maxSeqLen: 12,
          embedDim: jointDim,
          numLayers: 1,
          numHeads: 4,
          seed: 23,
        ),
        jointEmbedDim: jointDim,
        fusionLayers: 1,
        fusionHeads: 4,
        seed: 24,
      );
      final y = enc(
        _rand2D(8, 16, device: Device.CPU, seed: 25),
        _rand2D(6, 20, device: Device.CPU, seed: 26),
        textIn: _tokens([1, 2, 3, 4], device: Device.CPU),
      );
      expect(y.shape, equals([8 + 6 + 4, jointDim]));
      for (final v in y.toList()) expect(v.isFinite, isTrue);
    });
  });
}
