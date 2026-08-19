@Timeout(Duration(minutes: 3))
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

Tensor _fake(List<int> shape, {int seed = 0, Device device = Device.CPU}) {
  final rng = math.Random(seed);
  var n = 1;
  for (final d in shape) {
    n *= d;
  }
  final v = Float32List(n);
  for (int i = 0; i < n; i++) {
    v[i] = rng.nextDouble() - 0.5;
  }
  return Tensor.fromFloat32List(shape, v, device: device);
}

/// Tiny SAM config for CPU-viable tests: 32×32 image, patch=8 → 4×4
/// grid, embed=32, 2 layers (1 global at index 1, 1 windowed at 0),
/// window=2 (so we get 2×2=4 windows out of the 4×4 grid).
const _tinyCfg = SamImageEncoderConfig(
  imageSize: 32,
  patchSize: 8,
  embedDim: 32,
  numLayers: 2,
  numHeads: 4,
  mlpDim: 64,
  windowSize: 2,
  outChannels: 16,
  globalAttnIndices: [1],
);

void main() {
  group('SamImageEncoderConfig', () {
    test('vitB matches facebook/sam-vit-base', () {
      final cfg = SamImageEncoderConfig.vitB();
      expect(cfg.imageSize, 1024);
      expect(cfg.patchSize, 16);
      expect(cfg.embedDim, 768);
      expect(cfg.numLayers, 12);
      expect(cfg.numHeads, 12);
      expect(cfg.mlpDim, 3072);
      expect(cfg.windowSize, 14);
      expect(cfg.outChannels, 256);
      expect(cfg.gridSize, 64);
      expect(cfg.globalAttnIndices, equals([2, 5, 8, 11]));
    });

    test('vitL matches sam-vit-large', () {
      final cfg = SamImageEncoderConfig.vitL();
      expect(cfg.embedDim, 1024);
      expect(cfg.numLayers, 24);
      expect(cfg.numHeads, 16);
      expect(cfg.globalAttnIndices, equals([5, 11, 17, 23]));
    });

    test('vitH matches sam-vit-huge', () {
      final cfg = SamImageEncoderConfig.vitH();
      expect(cfg.embedDim, 1280);
      expect(cfg.numLayers, 32);
      expect(cfg.numHeads, 16);
      expect(cfg.globalAttnIndices, equals([7, 15, 23, 31]));
    });
  });

  group('LayerNorm2d', () {
    test('preserves NCHW shape', () {
      final ln = LayerNorm2d(4);
      final x = _fake([1, 4, 3, 3], seed: 1);
      expect(ln(x).shape, equals([1, 4, 3, 3]));
    });

    test(
      'output is per-pixel zero-mean unit-variance after gamma=1, beta=0',
      () {
        final ln = LayerNorm2d(8);
        final x = _fake([1, 8, 2, 2], seed: 2);
        final out = ln(x);
        final data = out.toList();
        // For each pixel (y, x), the channel axis should be normalised.
        const h = 2, w = 2, c = 8;
        for (int y = 0; y < h; y++) {
          for (int px = 0; px < w; px++) {
            double sum = 0;
            for (int ci = 0; ci < c; ci++) {
              sum += data[((0 * c + ci) * h + y) * w + px];
            }
            expect(
              sum.abs() < 1e-4,
              isTrue,
              reason: 'pixel ($y,$px) mean $sum',
            );
          }
        }
      },
    );
  });

  group('SamWindowedAttention', () {
    test('windowed forward preserves [H·W, D] shape', () {
      final attn = SamWindowedAttention(
        embedDim: 32,
        numHeads: 4,
        inputSize: 4,
        windowSize: 2,
      );
      final x = _fake([16, 32], seed: 3);
      expect(attn(x).shape, equals([16, 32]));
    });

    test('global attention (windowSize == inputSize) preserves shape', () {
      final attn = SamWindowedAttention(
        embedDim: 32,
        numHeads: 4,
        inputSize: 4,
        windowSize: 4,
      );
      final x = _fake([16, 32], seed: 4);
      expect(attn(x).shape, equals([16, 32]));
    });

    test('rejects wrong-shape input', () {
      final attn = SamWindowedAttention(
        embedDim: 32,
        numHeads: 4,
        inputSize: 4,
        windowSize: 2,
      );
      expect(() => attn(_fake([16, 33], seed: 5)), throwsArgumentError);
      expect(() => attn(_fake([15, 32], seed: 5)), throwsArgumentError);
    });

    test('rel-pos actually affects output — zeroing the tables changes it', () {
      final attn = SamWindowedAttention(
        embedDim: 32,
        numHeads: 4,
        inputSize: 4,
        windowSize: 2,
      );
      final x = _fake([16, 32], seed: 6);
      final withRelPos = attn(x).toList();

      // Zero out the rel-pos tables and re-run.
      final zeroH = List<double>.filled(attn.relPosH.length, 0);
      final zeroW = List<double>.filled(attn.relPosW.length, 0);
      attn.relPosH.assign(Tensor.fromList(attn.relPosH.shape, zeroH));
      attn.relPosW.assign(Tensor.fromList(attn.relPosW.shape, zeroW));
      final withoutRelPos = attn(x).toList();

      double maxDiff = 0;
      for (int i = 0; i < withRelPos.length; i++) {
        final d = (withRelPos[i] - withoutRelPos[i]).abs();
        if (d > maxDiff) maxDiff = d;
      }
      expect(
        maxDiff > 1e-3,
        isTrue,
        reason: 'rel-pos should meaningfully change attention output',
      );
    });
  });

  group('SamViTBlock', () {
    test('forward preserves shape (windowed block)', () {
      final block = SamViTBlock(
        embedDim: 32,
        numHeads: 4,
        inputSize: 4,
        windowSize: 2,
        mlpDim: 64,
      );
      final x = _fake([16, 32], seed: 7);
      expect(block(x).shape, equals([16, 32]));
    });

    test('forward preserves shape (global block)', () {
      final block = SamViTBlock(
        embedDim: 32,
        numHeads: 4,
        inputSize: 4,
        windowSize: 4, // global
        mlpDim: 64,
      );
      final x = _fake([16, 32], seed: 8);
      expect(block(x).shape, equals([16, 32]));
    });
  });

  group('SamImageEncoder', () {
    test('vitB config.gridSize = 64', () {
      expect(SamImageEncoderConfig.vitB().gridSize, 64);
    });

    test('forward produces [1, outChannels, gridSize, gridSize]', () {
      final enc = SamImageEncoder(_tinyCfg);
      final img = _fake([1, 3, 32, 32], seed: 9);
      final emb = enc(img);
      // Tiny: gridSize = 32/8 = 4, outChannels = 16.
      expect(emb.shape, equals([1, 16, 4, 4]));
    });

    test('rejects wrong-shape image', () {
      final enc = SamImageEncoder(_tinyCfg);
      expect(() => enc(_fake([1, 3, 33, 32], seed: 10)), throwsArgumentError);
      expect(() => enc(_fake([2, 3, 32, 32], seed: 11)), throwsArgumentError);
      expect(() => enc(_fake([1, 4, 32, 32], seed: 12)), throwsArgumentError);
    });

    test('parameter list is non-empty and finite', () {
      final enc = SamImageEncoder(_tinyCfg);
      final params = enc.parameters();
      expect(params, isNotEmpty);
      for (final p in params) {
        for (final v in p.toList()) {
          expect(v.isFinite, isTrue);
        }
      }
    });

    test('global-attention layers are dispatched at the right indices', () {
      final enc = SamImageEncoder(_tinyCfg);
      expect(enc.blocks.length, 2);
      // Windowed block: windowSize == 2 (from tinyCfg).
      expect(enc.blocks[0].attn.windowSize, 2);
      // Global block: windowSize == gridSize (4).
      expect(enc.blocks[1].attn.windowSize, 4);
    });
  });
}
