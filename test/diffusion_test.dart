@Timeout(Duration(minutes: 3))
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

Tensor _fake([List<int> shape = const [1, 1, 8, 8], int seed = 0]) {
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

bool _gpuAvailable() {
  try {
    Tensor.fromList([1], [1.0], device: Device.GPU).toList();
    return true;
  } catch (_) {
    return false;
  }
}

void main() {
  group('NoiseSchedule.linear', () {
    test('T=1000 defaults: monotone α̅ from ≈1 down to ≈0', () {
      final sch = NoiseSchedule.linear();
      expect(sch.numTimesteps, 1000);
      expect(sch.betas.first, closeTo(1e-4, 1e-6));
      expect(sch.betas.last, closeTo(0.02, 1e-6));
      expect(sch.alphaBars.first, closeTo(1.0 - 1e-4, 1e-4));
      expect(sch.alphaBars.last, lessThan(1e-3));
      // Monotone decreasing.
      for (int t = 1; t < sch.numTimesteps; t++) {
        expect(
          sch.alphaBars[t] < sch.alphaBars[t - 1],
          isTrue,
          reason: 't=$t αb=${sch.alphaBars[t]} prev=${sch.alphaBars[t - 1]}',
        );
      }
    });

    test('sqrt(αb) and sqrt(1-αb) satisfy the identity', () {
      final sch = NoiseSchedule.linear(numTimesteps: 50);
      for (int t = 0; t < sch.numTimesteps; t++) {
        final s = sch.sqrtAlphaBars[t];
        final o = sch.sqrtOneMinusAlphaBars[t];
        final total = s * s + o * o;
        expect((total - 1.0).abs() < 1e-4, isTrue, reason: 't=$t total=$total');
      }
    });

    test('posteriorVariance[0] = 0; positive for t > 0', () {
      final sch = NoiseSchedule.linear(numTimesteps: 20);
      expect(sch.posteriorVariance[0], 0.0);
      for (int t = 1; t < sch.numTimesteps; t++) {
        expect(sch.posteriorVariance[t], greaterThan(0.0));
      }
    });

    test('rejects tiny T', () {
      expect(() => NoiseSchedule.linear(numTimesteps: 1), throwsArgumentError);
    });
  });

  group('forwardDiffuse', () {
    test('shape matches x0 and eps is same shape', () {
      final sch = NoiseSchedule.linear(numTimesteps: 100);
      final x0 = _fake();
      final r = sch.forwardDiffuse(x0, 42, seed: 7);
      expect(r.xT.shape, x0.shape);
      expect(r.eps.shape, x0.shape);
    });

    test('deterministic given supplied eps: x_t = √αb·x0 + √(1-αb)·eps', () {
      final sch = NoiseSchedule.linear(numTimesteps: 10);
      const t = 5;
      final x0 = Tensor.fromList([1, 1, 2, 2], [1.0, 0.5, 0.0, -0.5]);
      final eps = Tensor.fromList([1, 1, 2, 2], [0.1, 0.2, 0.3, 0.4]);
      final r = sch.forwardDiffuse(x0, t, eps: eps);
      final s = sch.sqrtAlphaBars[t];
      final o = sch.sqrtOneMinusAlphaBars[t];
      final expected = [
        s * 1.0 + o * 0.1,
        s * 0.5 + o * 0.2,
        s * 0.0 + o * 0.3,
        s * -0.5 + o * 0.4,
      ];
      final got = r.xT.toList();
      for (int i = 0; i < 4; i++) {
        expect(
          (got[i] - expected[i]).abs() < 1e-5,
          isTrue,
          reason: 'i=$i got=${got[i]} want=${expected[i]}',
        );
      }
    });

    test('t=T-1 pushes x_t close to pure noise stats (mean~0, var~1)', () {
      final sch = NoiseSchedule.linear();
      final x0 = Tensor.fromList([
        1,
        1,
        32,
        32,
      ], List<double>.filled(32 * 32, 1.0));
      final r = sch.forwardDiffuse(x0, sch.numTimesteps - 1, seed: 3);
      final vals = r.xT.toList();
      double sum = 0, sqSum = 0;
      for (final v in vals) {
        sum += v;
        sqSum += v * v;
      }
      final mean = sum / vals.length;
      final variance = sqSum / vals.length - mean * mean;
      expect(mean.abs() < 0.1, isTrue, reason: 'mean=$mean');
      // sqrt(αb_{T-1}) is very small; std ≈ sqrt(1-αb) ≈ 1.
      expect((variance - 1.0).abs() < 0.15, isTrue, reason: 'var=$variance');
    });

    test('rejects t out of range', () {
      final sch = NoiseSchedule.linear(numTimesteps: 5);
      final x0 = _fake();
      expect(() => sch.forwardDiffuse(x0, 5), throwsArgumentError);
      expect(() => sch.forwardDiffuse(x0, -1), throwsArgumentError);
    });
  });

  group('reverseStep', () {
    test('t=0 returns exact posterior mean (no random noise)', () {
      final sch = NoiseSchedule.linear(numTimesteps: 10);
      final xt = Tensor.fromList([1, 1, 2, 2], [0.2, 0.3, 0.4, 0.5]);
      final eps = Tensor.fromList([1, 1, 2, 2], [0.1, 0.1, 0.1, 0.1]);
      final r1 = sch.reverseStep(xt, 0, eps).toList();
      final r2 = sch.reverseStep(xt, 0, eps).toList();
      // Deterministic at t=0.
      expect(r1, equals(r2));
      // Manual mean check for pos 0.
      final inv = sch.sqrtRecipAlphas[0];
      final coefEps = sch.betas[0] / sch.sqrtOneMinusAlphaBars[0];
      final expected0 = (0.2 - coefEps * 0.1) * inv;
      expect((r1[0] - expected0).abs() < 1e-5, isTrue);
    });

    test('rejects epsHat shape mismatch', () {
      final sch = NoiseSchedule.linear(numTimesteps: 5);
      final xt = _fake();
      final wrong = _fake([1, 1, 4, 4]);
      expect(() => sch.reverseStep(xt, 2, wrong), throwsArgumentError);
    });
  });

  group('TinyUNet', () {
    test('forward preserves input shape [N,1,H,W]', () {
      final unet = TinyUNet(hidden: 8, totalTimesteps: 100);
      final x = _fake([2, 1, 16, 16]);
      final y = unet(x, 10);
      expect(y.shape, equals([2, 1, 16, 16]));
    });

    test('rejects non-single-channel input', () {
      final unet = TinyUNet();
      expect(() => unet(_fake([1, 3, 16, 16]), 0), throwsArgumentError);
    });

    test('rejects out-of-range t', () {
      final unet = TinyUNet(totalTimesteps: 100);
      expect(() => unet(_fake([1, 1, 16, 16]), 100), throwsArgumentError);
    });

    test('parameter list is non-empty and finite', () {
      final unet = TinyUNet();
      final params = unet.parameters();
      expect(params, isNotEmpty);
      for (final p in params) {
        for (final v in p.toList()) {
          expect(v.isFinite, isTrue);
        }
      }
    });
  });

  group('TinyUNet on GPU', () {
    final gpuOk = _gpuAvailable();
    if (!gpuOk) {
      test(
        'GPU unavailable → skipped',
        () {},
        skip: 'CUDA / native/lib/libmat_mul.so not usable in this env',
      );
      return;
    }
    test('CPU vs GPU forward parity', () {
      final cpu = TinyUNet(hidden: 8);
      final gpu = TinyUNet(hidden: 8, device: Device.GPU);
      // Sync every parameter.
      final cpuP = cpu.parameters();
      final gpuP = gpu.parameters();
      expect(cpuP.length, gpuP.length);
      for (int i = 0; i < cpuP.length; i++) {
        gpuP[i].assign(
          Tensor.fromList(cpuP[i].shape, cpuP[i].toList(), device: Device.GPU),
        );
      }
      final xVals = List<double>.generate(
        1 * 1 * 16 * 16,
        (i) => math.sin(i * 0.11),
      );
      final xCpu = Tensor.fromList([1, 1, 16, 16], xVals);
      final xGpu = Tensor.fromList([1, 1, 16, 16], xVals, device: Device.GPU);
      final cpuOut = cpu(xCpu, 42).toList();
      final gpuOut = gpu(xGpu, 42).toList();
      double maxDiff = 0;
      for (int i = 0; i < cpuOut.length; i++) {
        final d = (cpuOut[i] - gpuOut[i]).abs();
        if (d > maxDiff) maxDiff = d;
      }
      expect(maxDiff, lessThan(5e-3), reason: 'max diff = $maxDiff');
    });
  });
}
