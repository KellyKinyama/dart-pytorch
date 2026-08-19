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
  group('SinusoidalTimestepEmbedding', () {
    test('output shape [embedDim]', () {
      final emb = SinusoidalTimestepEmbedding(freqDim: 32, embedDim: 64);
      final t = Tensor.fromList([1], [0.5]);
      final out = emb(t);
      expect(out.shape, equals([64]));
    });

    test('different timesteps produce different embeddings', () {
      final emb = SinusoidalTimestepEmbedding(freqDim: 32, embedDim: 64);
      final e0 = emb(Tensor.fromList([1], [0.0])).toList();
      final e1 = emb(Tensor.fromList([1], [1.0])).toList();
      double maxDiff = 0;
      for (int i = 0; i < e0.length; i++) {
        final d = (e0[i] - e1[i]).abs();
        if (d > maxDiff) maxDiff = d;
      }
      expect(maxDiff > 0.0, isTrue);
    });

    test('rejects wrong-shape input', () {
      final emb = SinusoidalTimestepEmbedding(freqDim: 32, embedDim: 64);
      expect(() => emb(Tensor.fromList([2], [0.5, 0.7])), throwsArgumentError);
    });

    test('rejects odd freqDim', () {
      expect(
        () => SinusoidalTimestepEmbedding(freqDim: 33, embedDim: 64),
        throwsArgumentError,
      );
    });
  });

  group('AdaLNZero', () {
    test('zero-init produces zero modulation vectors', () {
      final adaLn = AdaLNZero(embedDim: 16);
      final c = _fake([16], seed: 1).reshape([16]);
      final mod = adaLn(c).toList();
      for (int i = 0; i < mod.length; i++) {
        expect(
          mod[i].abs() < 1e-6,
          isTrue,
          reason: 'mod[$i] = ${mod[i]} — should be zero at init',
        );
      }
    });

    test('output shape [6, embedDim]', () {
      final adaLn = AdaLNZero(embedDim: 32);
      final c = _fake([32], seed: 2);
      expect(adaLn(c).shape, equals([6, 32]));
    });

    test('non-zero modulation after we perturb the projection', () {
      final adaLn = AdaLNZero(embedDim: 16);
      // Perturb the modulation weight so the block leaves the identity
      // regime. Any non-zero value across the 96-cell block is enough.
      final w = List<double>.filled(adaLn.modulation.weight.length, 0.0);
      w[0] = 0.5;
      adaLn.modulation.weight.assign(
        Tensor.fromList(
          adaLn.modulation.weight.shape,
          w,
          device: adaLn.modulation.weight.device,
        ),
      );
      final c = Tensor.fromList([16], List<double>.filled(16, 1.0));
      final mod = adaLn(c).toList();
      double maxAbs = 0;
      for (final v in mod) {
        if (v.abs() > maxAbs) maxAbs = v.abs();
      }
      expect(maxAbs > 0, isTrue);
    });
  });

  group('adaLNModulate + adaLNGate', () {
    test('modulate is xn * (1 + scale) + shift row-wise', () {
      final xn = Tensor.fromList([2, 3], [1, 2, 3, 4, 5, 6]);
      final scale = Tensor.fromList([3], [0.1, 0.0, -0.5]);
      final shift = Tensor.fromList([3], [10, 20, 30]);
      final out = adaLNModulate(xn, scale, shift).toList();
      // row 0: 1*(1.1)+10 = 11.1, 2*(1.0)+20 = 22.0, 3*(0.5)+30 = 31.5
      // row 1: 4*1.1+10 = 14.4, 5*1.0+20 = 25.0, 6*0.5+30 = 33.0
      expect((out[0] - 11.1).abs() < 1e-5, isTrue);
      expect((out[1] - 22.0).abs() < 1e-5, isTrue);
      expect((out[2] - 31.5).abs() < 1e-5, isTrue);
      expect((out[3] - 14.4).abs() < 1e-5, isTrue);
      expect((out[4] - 25.0).abs() < 1e-5, isTrue);
      expect((out[5] - 33.0).abs() < 1e-5, isTrue);
    });

    test('gate is x * gate row-wise', () {
      final x = Tensor.fromList([2, 3], [1, 2, 3, 4, 5, 6]);
      final gate = Tensor.fromList([3], [0.0, 2.0, -1.0]);
      final out = adaLNGate(x, gate).toList();
      expect(out, equals([0.0, 4.0, -3.0, 0.0, 10.0, -6.0]));
    });
  });

  group('F5DiTBlock', () {
    test('at init the block is an identity residual (zero modulation)', () {
      final block = F5DiTBlock(embedDim: 32, numHeads: 4, mlpDim: 64);
      final x = _fake([5, 32], seed: 7);
      final c = _fake([32], seed: 8);
      final y = block(x, c);
      // With AdaLNZero at init, gate_msa = gate_mlp = 0, so both
      // residual updates are zero. Output must equal input.
      final xVals = x.toList();
      final yVals = y.toList();
      double maxDiff = 0;
      for (int i = 0; i < xVals.length; i++) {
        final d = (xVals[i] - yVals[i]).abs();
        if (d > maxDiff) maxDiff = d;
      }
      expect(
        maxDiff < 1e-5,
        isTrue,
        reason: 'DiT block should be identity at init; max diff = $maxDiff',
      );
    });

    test('after perturbing modulation, block leaves identity regime', () {
      final block = F5DiTBlock(embedDim: 32, numHeads: 4, mlpDim: 64);
      // Perturb the AdaLN modulation weight matrix so gate_msa is
      // non-zero.
      final w = List<double>.filled(block.adaLn.modulation.weight.length, 0.1);
      block.adaLn.modulation.weight.assign(
        Tensor.fromList(
          block.adaLn.modulation.weight.shape,
          w,
          device: block.adaLn.modulation.weight.device,
        ),
      );
      final x = _fake([5, 32], seed: 11);
      final c = _fake([32], seed: 12);
      final y = block(x, c);
      final xVals = x.toList();
      final yVals = y.toList();
      double maxDiff = 0;
      for (int i = 0; i < xVals.length; i++) {
        final d = (xVals[i] - yVals[i]).abs();
        if (d > maxDiff) maxDiff = d;
      }
      expect(
        maxDiff > 1e-3,
        isTrue,
        reason: 'perturbed block should differ from identity',
      );
    });

    test('rejects wrong-shape input', () {
      final block = F5DiTBlock(embedDim: 32, numHeads: 4, mlpDim: 64);
      final c = _fake([32], seed: 13);
      expect(() => block(_fake([5, 33], seed: 14), c), throwsArgumentError);
    });
  });
}
