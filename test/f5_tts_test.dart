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

  group('gaussianNoise', () {
    test('produces the requested shape', () {
      final n = gaussianNoise([4, 5], seed: 1);
      expect(n.shape, equals([4, 5]));
    });

    test('empirical mean/std are close to N(0, 1) for large sample', () {
      final n = gaussianNoise([1000], seed: 2).toList();
      double sum = 0;
      for (final v in n) {
        sum += v;
      }
      final mean = sum / n.length;
      double sq = 0;
      for (final v in n) {
        sq += (v - mean) * (v - mean);
      }
      final std = math.sqrt(sq / n.length);
      expect(mean.abs() < 0.1, isTrue, reason: 'mean=$mean');
      expect((std - 1.0).abs() < 0.1, isTrue, reason: 'std=$std');
    });

    test('deterministic given the same seed', () {
      final a = gaussianNoise([16], seed: 42).toList();
      final b = gaussianNoise([16], seed: 42).toList();
      expect(a, equals(b));
    });
  });

  group('FlowMatchingSampler', () {
    test('linear velocity v(x, t) = 1 integrates from 0 to 1 exactly', () {
      // dx/dt = 1  =>  x(1) = x(0) + 1.
      const sampler = FlowMatchingSampler(numSteps: 8);
      final x0 = Tensor.fromList([3], [0.0, 5.0, -2.0]);
      final x1 = sampler.sample(
        initialNoise: x0,
        velocityField: (x, t) =>
            Tensor.fromList(x.shape, List<double>.filled(x.length, 1.0)),
      );
      final vals = x1.toList();
      expect((vals[0] - 1.0).abs() < 1e-5, isTrue);
      expect((vals[1] - 6.0).abs() < 1e-5, isTrue);
      expect((vals[2] - -1.0).abs() < 1e-5, isTrue);
    });

    test('midpoint solver is exact for constant velocity too', () {
      const sampler = FlowMatchingSampler(
        numSteps: 4,
        solver: FlowSolver.midpoint,
      );
      final x0 = Tensor.fromList([2], [0.0, 10.0]);
      final x1 = sampler.sample(
        initialNoise: x0,
        velocityField: (x, t) => Tensor.fromList(x.shape, [0.5, 2.0]),
      );
      final vals = x1.toList();
      // x0 + 1 * dv where dv = velocity * 1 for whole interval.
      expect((vals[0] - 0.5).abs() < 1e-5, isTrue);
      expect((vals[1] - 12.0).abs() < 1e-5, isTrue);
    });

    test('midpoint solver has smaller error than Euler on '
        'time-varying velocity', () {
      // dx/dt = 2·t → x(1) = x(0) + t² |₀¹ = x(0) + 1.
      Tensor v(Tensor x, Tensor t) {
        final tt = t.toList()[0];
        return Tensor.fromList(x.shape, List<double>.filled(x.length, 2 * tt));
      }

      const nSteps = 4;
      const euler = FlowMatchingSampler(numSteps: nSteps);
      const midpoint = FlowMatchingSampler(
        numSteps: nSteps,
        solver: FlowSolver.midpoint,
      );
      final x0 = Tensor.fromList([1], [0.0]);
      final xE = euler.sample(initialNoise: x0, velocityField: v);
      final xM = midpoint.sample(initialNoise: x0, velocityField: v);
      final eErr = (xE.toList()[0] - 1.0).abs();
      final mErr = (xM.toList()[0] - 1.0).abs();
      expect(
        mErr < eErr,
        isTrue,
        reason: 'midpoint err $mErr should beat Euler err $eErr',
      );
    });

    test('velocity shape mismatch throws', () {
      const sampler = FlowMatchingSampler(numSteps: 4);
      expect(
        () => sampler.sample(
          initialNoise: Tensor.fromList([3], [0, 1, 2]),
          velocityField: (x, t) => Tensor.fromList([2], [0, 0]),
        ),
        throwsArgumentError,
      );
    });

    test('numSteps=32 shape roundtrip on a 2-D noise shape', () {
      const sampler = FlowMatchingSampler(numSteps: 32);
      final z = gaussianNoise([4, 8], seed: 99);
      final out = sampler.sample(
        initialNoise: z,
        velocityField: (x, t) =>
            Tensor.fromList(x.shape, List<double>.filled(x.length, 0.0)),
      );
      // Zero velocity → sample equals input noise.
      expect(out.shape, equals([4, 8]));
      final zVals = z.toList();
      final outVals = out.toList();
      for (int i = 0; i < zVals.length; i++) {
        expect((zVals[i] - outVals[i]).abs() < 1e-5, isTrue);
      }
    });
  });
}
