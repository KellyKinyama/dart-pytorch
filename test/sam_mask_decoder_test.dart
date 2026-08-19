@Timeout(Duration(minutes: 3))
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

/// Tiny mask-decoder config: embed 32, 4 heads, mlp 64, depth 2.
/// Numbers of mask tokens etc. mirror SAM's (1 IoU + 4 mask tokens).
const _tinyCfg = SamMaskDecoderConfig(
  embedDim: 32,
  numHeads: 4,
  mlpDim: 64,
  transformerDepth: 2,
  numMultimaskOutputs: 3,
  iouHeadDepth: 3,
  iouHeadHiddenDim: 32,
);

void main() {
  group('SamAttention', () {
    test('downsampleRate=1 preserves internal dim', () {
      final a = SamAttention(embedDim: 32, numHeads: 4);
      expect(a.internalDim, 32);
      expect(a.headDim, 8);
    });

    test('downsampleRate=2 halves internal dim', () {
      final a = SamAttention(
          embedDim: 32, numHeads: 4, downsampleRate: 2);
      expect(a.internalDim, 16);
      expect(a.headDim, 4);
    });

    test('forward returns [N_q, embedDim]', () {
      final a = SamAttention(
          embedDim: 32, numHeads: 4, downsampleRate: 2);
      final q = _fake([5, 32], seed: 1);
      final k = _fake([7, 32], seed: 2);
      final v = _fake([7, 32], seed: 3);
      expect(a(q, k, v).shape, equals([5, 32]));
    });

    test('rejects wrong-shape input', () {
      final a = SamAttention(embedDim: 32, numHeads: 4);
      final q = _fake([5, 32], seed: 4);
      expect(
        () => a(q, _fake([7, 33], seed: 5), _fake([7, 32], seed: 6)),
        throwsArgumentError,
      );
    });
  });

  group('SamMlpBlock', () {
    test('preserves last dim', () {
      final mlp = SamMlpBlock(embedDim: 32, mlpDim: 64);
      final x = _fake([5, 32], seed: 7);
      expect(mlp(x).shape, equals([5, 32]));
    });
  });

  group('SamMlp', () {
    test('numLayers=3 forward output dim matches outputDim', () {
      final mlp = SamMlp(
        inputDim: 32,
        hiddenDim: 64,
        outputDim: 4,
        numLayers: 3,
      );
      final x = _fake([1, 32], seed: 8);
      expect(mlp(x).shape, equals([1, 4]));
    });

    test('sigmoidOutput squashes to [0, 1]', () {
      final mlp = SamMlp(
        inputDim: 32,
        hiddenDim: 32,
        outputDim: 4,
        numLayers: 2,
        sigmoidOutput: true,
      );
      final x = _fake([3, 32], seed: 9);
      final out = mlp(x).toList();
      for (final v in out) {
        expect(v >= 0 && v <= 1, isTrue);
      }
    });
  });

  group('SamTwoWayAttentionBlock', () {
    test('preserves query and key shapes', () {
      final block = SamTwoWayAttentionBlock(
        embedDim: 32,
        numHeads: 4,
        mlpDim: 64,
      );
      final queries = _fake([5, 32], seed: 10);
      final keys = _fake([16, 32], seed: 11);
      final qPe = _fake([5, 32], seed: 12);
      final kPe = _fake([16, 32], seed: 13);
      final r = block(queries: queries, keys: keys, queryPe: qPe, keyPe: kPe);
      expect(r.queries.shape, equals([5, 32]));
      expect(r.keys.shape, equals([16, 32]));
    });

    test('skipFirstLayerPe=true still runs', () {
      final block = SamTwoWayAttentionBlock(
        embedDim: 32,
        numHeads: 4,
        mlpDim: 64,
        skipFirstLayerPe: true,
      );
      final r = block(
        queries: _fake([3, 32], seed: 14),
        keys: _fake([9, 32], seed: 15),
        queryPe: _fake([3, 32], seed: 16),
        keyPe: _fake([9, 32], seed: 17),
      );
      expect(r.queries.shape, equals([3, 32]));
    });
  });

  group('SamTwoWayTransformer', () {
    test('forward returns queries and keys of expected shapes', () {
      final tr = SamTwoWayTransformer(
        depth: 2,
        embedDim: 32,
        numHeads: 4,
        mlpDim: 64,
      );
      final imageEmb = _fake([1, 32, 4, 4], seed: 18); // H=W=4, so 16 pixels
      final imagePe = _fake([32, 4, 4], seed: 19);
      final pointEmb = _fake([6, 32], seed: 20); // 6 prompt tokens
      final r = tr(
        imageEmbedding: imageEmb,
        imagePe: imagePe,
        pointEmbedding: pointEmb,
      );
      expect(r.queries.shape, equals([6, 32]));
      expect(r.keys.shape, equals([16, 32]));
    });

    test('rejects wrong-shape image', () {
      final tr = SamTwoWayTransformer(
        depth: 1,
        embedDim: 32,
        numHeads: 4,
        mlpDim: 64,
      );
      // Wrong C.
      expect(
        () => tr(
          imageEmbedding: _fake([1, 33, 4, 4], seed: 21),
          imagePe: _fake([32, 4, 4], seed: 22),
          pointEmbedding: _fake([3, 32], seed: 23),
        ),
        throwsArgumentError,
      );
    });
  });

  group('SamMaskDecoder', () {
    test('numMaskTokens = 1 + numMultimaskOutputs', () {
      expect(_tinyCfg.numMaskTokens, 4);
    });

    test('forward produces expected mask and IoU shapes', () {
      final dec = SamMaskDecoder(_tinyCfg);
      // Small "image grid" of 4x4; upscaled by 2×2 = 4× -> masks
      // land on 16×16.
      final imageEmb = _fake([1, 32, 4, 4], seed: 100);
      final imagePe = _fake([32, 4, 4], seed: 101);
      final sparse = _fake([2, 32], seed: 102); // e.g. 2 prompts
      final dense = _fake([32, 4, 4], seed: 103);
      final out = dec(
        imageEmbedding: imageEmb,
        imagePe: imagePe,
        sparsePrompts: sparse,
        densePrompts: dense,
      );
      // 4 mask tokens = 1 (no-mask) + 3 (multimask). Upscale factor = 4.
      expect(out.masks.shape, equals([4, 16, 16]));
      expect(out.iouPredictions.shape, equals([4]));
    });

    test('select(multimask=false) returns [1, H, W] / [1]', () {
      final dec = SamMaskDecoder(_tinyCfg);
      final out = dec(
        imageEmbedding: _fake([1, 32, 4, 4], seed: 200),
        imagePe: _fake([32, 4, 4], seed: 201),
        sparsePrompts: _fake([2, 32], seed: 202),
        densePrompts: _fake([32, 4, 4], seed: 203),
      );
      final r = out.select(multimask: false);
      expect(r.masks.shape, equals([1, 16, 16]));
      expect(r.iouPredictions.shape, equals([1]));
    });

    test('select(multimask=true) returns 3 masks/IoU', () {
      final dec = SamMaskDecoder(_tinyCfg);
      final out = dec(
        imageEmbedding: _fake([1, 32, 4, 4], seed: 300),
        imagePe: _fake([32, 4, 4], seed: 301),
        sparsePrompts: _fake([2, 32], seed: 302),
        densePrompts: _fake([32, 4, 4], seed: 303),
      );
      final r = out.select(multimask: true);
      expect(r.masks.shape, equals([3, 16, 16]));
      expect(r.iouPredictions.shape, equals([3]));
    });

    test('empty sparse prompts still runs (mask decoder relies only '
        'on the prepended IoU + mask tokens)', () {
      final dec = SamMaskDecoder(_tinyCfg);
      final out = dec(
        imageEmbedding: _fake([1, 32, 4, 4], seed: 400),
        imagePe: _fake([32, 4, 4], seed: 401),
        sparsePrompts: Tensor.fromList([0, 32], const <double>[]),
        densePrompts: _fake([32, 4, 4], seed: 402),
      );
      expect(out.masks.shape, equals([4, 16, 16]));
      expect(out.iouPredictions.shape, equals([4]));
    });

    test('parameter list is non-empty and finite', () {
      final dec = SamMaskDecoder(_tinyCfg);
      final params = dec.parameters();
      expect(params, isNotEmpty);
      var checked = 0;
      for (final p in params) {
        for (final v in p.toList()) {
          expect(v.isFinite, isTrue);
        }
        checked++;
        if (checked > 50) break;
      }
    });
  });
}
