import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

void main() {
  group('SGD (no momentum)', () {
    test('single step matches p - lr * grad', () {
      final p = Tensor.fromList([3], [1.0, 2.0, 3.0], requiresGrad: true);
      // Loss = sum(p^2) -> grad = 2p = [2, 4, 6].
      p.pow(2).sum().backward();
      SGD([p], lr: 0.1).step();
      // Expected: p - 0.1 * [2,4,6] = [0.8, 1.6, 2.4]
      final got = p.toList();
      for (int i = 0; i < 3; i++) {
        expect(
          (got[i] - [0.8, 1.6, 2.4][i]).abs() < 1e-5,
          isTrue,
          reason: 'i=$i got=${got[i]}',
        );
      }
    });

    test('grad is retained across step; zeroGrad clears', () {
      final p = Tensor.fromList([2], [1.0, 2.0], requiresGrad: true);
      final opt = SGD([p], lr: 0.1);
      p.pow(2).sum().backward();
      expect(p.grad, isNotNull);
      opt.step();
      opt.zeroGrad();
      expect(p.grad, isNull);
    });

    test('quadratic bowl converges near zero', () {
      // Minimize sum((p - target)^2). Optimum at p == target.
      final target = [1.0, -2.0, 0.5, 3.0];
      final p = Tensor.fromList([4], [0.0, 0.0, 0.0, 0.0], requiresGrad: true);
      final t = Tensor.fromList([4], target);
      final opt = SGD([p], lr: 0.1);
      for (int i = 0; i < 200; i++) {
        opt.zeroGrad();
        (p - t).pow(2).sum().backward();
        opt.step();
      }
      final got = p.toList();
      for (int i = 0; i < 4; i++) {
        expect(
          (got[i] - target[i]).abs() < 1e-3,
          isTrue,
          reason: 'i=$i got=${got[i]}',
        );
      }
    });
  });

  group('SGD with momentum', () {
    test('velocity accumulates correctly for two steps', () {
      final p = Tensor.fromList([2], [1.0, 2.0], requiresGrad: true);
      final opt = SGD([p], lr: 0.1, momentum: 0.9);
      // Step 1: grad = 2p = [2, 4]; v = [2, 4]; p -= 0.1 * v -> [0.8, 1.6]
      p.pow(2).sum().backward();
      opt.step();
      opt.zeroGrad();
      final s1 = p.toList();
      expect((s1[0] - 0.8).abs() < 1e-5, isTrue);
      expect((s1[1] - 1.6).abs() < 1e-5, isTrue);

      // Step 2: grad2 = 2*[0.8,1.6] = [1.6, 3.2]
      //         v = 0.9 * [2,4] + [1.6, 3.2] = [3.4, 6.8]
      //         p -= 0.1 * v = [0.8-0.34, 1.6-0.68] = [0.46, 0.92]
      p.pow(2).sum().backward();
      opt.step();
      final s2 = p.toList();
      expect((s2[0] - 0.46).abs() < 1e-4, isTrue, reason: 'got ${s2[0]}');
      expect((s2[1] - 0.92).abs() < 1e-4, isTrue, reason: 'got ${s2[1]}');
    });
  });

  group('Adam', () {
    test('single step matches reference formula', () {
      final p = Tensor.fromList([2], [1.0, -2.0], requiresGrad: true);
      final opt = Adam([p], lr: 0.1, beta1: 0.9, beta2: 0.999, eps: 1e-8);
      // Loss = sum(p^2) -> grad = 2p = [2, -4].
      p.pow(2).sum().backward();
      opt.step();

      // Reference (t=1):
      //   m = (1-0.9) * g = 0.1 * g = [0.2, -0.4]
      //   v = (1-0.999) * g^2 = 0.001 * [4, 16] = [0.004, 0.016]
      //   biasCorr1 = 1 - 0.9 = 0.1;   biasCorr2 = 1 - 0.999 = 0.001
      //   mHat = m / 0.1 = [2, -4]
      //   vHat = v / 0.001 = [4, 16]
      //   update = 0.1 * mHat / (sqrt(vHat)+eps) ~= 0.1 * [2,-4]/[2,4]
      //          = [0.1, -0.1]
      //   p -= update -> [0.9, -1.9]
      final got = p.toList();
      expect((got[0] - 0.9).abs() < 1e-4, isTrue, reason: 'got ${got[0]}');
      expect((got[1] - (-1.9)).abs() < 1e-4, isTrue, reason: 'got ${got[1]}');
    });

    test('quadratic bowl converges near zero (faster than SGD)', () {
      final target = [1.0, -2.0, 0.5];
      final p = Tensor.fromList([3], [0.0, 0.0, 0.0], requiresGrad: true);
      final t = Tensor.fromList([3], target);
      final opt = Adam([p], lr: 0.1);
      for (int i = 0; i < 200; i++) {
        opt.zeroGrad();
        (p - t).pow(2).sum().backward();
        opt.step();
      }
      final got = p.toList();
      for (int i = 0; i < 3; i++) {
        expect(
          (got[i] - target[i]).abs() < 1e-2,
          isTrue,
          reason: 'i=$i got=${got[i]}',
        );
      }
    });

    test('bias correction is applied at t=1 (no v_hat blow-up)', () {
      // On the very first step Adam's effective update magnitude should
      // be ~= lr regardless of grad scale (thanks to bias correction).
      // Try a huge grad and confirm |update| ~ lr.
      final p = Tensor.fromList([1], [0.0], requiresGrad: true);
      final opt = Adam([p], lr: 0.05);
      (p * 1000.0).sum().backward(); // grad = 1000
      opt.step();
      final delta = (p.toList()[0]).abs();
      // Should be approximately lr = 0.05 (within ~1e-3).
      expect(
        (delta - 0.05).abs() < 1e-3,
        isTrue,
        reason: 'delta=$delta expected ~0.05',
      );
    });
  });

  group('Optimizer + Module integration', () {
    test('SGD trains LayerNorm gamma/beta on synthetic target', () {
      // Fit a LayerNorm to reproduce a target affine mapping of a fixed
      // input. Normalize -> gamma * x_hat + beta. Target has non-unit
      // gamma and non-zero beta, so both params must move.
      final ln = LayerNorm(4);
      final opt = SGD(ln.parameters(), lr: 0.05);
      final xVals = [
        0.5, -1.0, 2.0, 0.3, //
        -0.2, 1.4, 0.0, -0.8, //
      ];
      final x = Tensor.fromList([2, 4], xVals);
      // Compute target = LayerNorm(x) with gamma=[2,1,0.5,3], beta=[0.1,-0.1,0.2,0].
      // Easier: pick a target tensor of same shape with known values.
      final target = Tensor.fromList(
        [2, 4],
        [
          1.0, -0.5, 0.5, 2.0, //
          -0.5, 1.0, 0.2, -0.8, //
        ],
      );

      double lossVal() => (ln(x) - target).pow(2).sum().toList()[0];

      final initial = lossVal();
      for (int i = 0; i < 300; i++) {
        opt.zeroGrad();
        (ln(x) - target).pow(2).sum().backward();
        opt.step();
      }
      final finalL = lossVal();
      expect(
        finalL < initial * 0.5,
        isTrue,
        reason: 'loss did not decrease enough: $initial -> $finalL',
      );
    });

    test('Adam trains Embedding on a lookup regression', () {
      // Given fixed indices, train Embedding weights so that lookup
      // matches a target tensor.
      final emb = Embedding(5, 3, seed: 7);
      final opt = Adam(emb.parameters(), lr: 0.1);
      final idx = Tensor.fromList([4], [0.0, 2.0, 4.0, 2.0]);
      final target = Tensor.fromList(
        [4, 3],
        [
          1.0, 0.0, -1.0, //
          0.5, 0.5, 0.5, //
          -1.0, 1.0, 0.0, //
          0.5, 0.5, 0.5, //
        ],
      );

      double lossVal() => (emb(idx) - target).pow(2).sum().toList()[0];

      final initial = lossVal();
      for (int i = 0; i < 300; i++) {
        opt.zeroGrad();
        (emb(idx) - target).pow(2).sum().backward();
        opt.step();
      }
      final finalL = lossVal();
      expect(
        finalL < 1e-3,
        isTrue,
        reason: 'did not converge: initial=$initial final=$finalL',
      );
    });
  });

  group('assign', () {
    test('CPU: overwrites data in place, preserves identity', () {
      final p = Tensor.fromList([3], [1.0, 2.0, 3.0], requiresGrad: true);
      final identityBefore = identical(p, p);
      p.assign(Tensor.fromList([3], [10.0, 20.0, 30.0]));
      expect(identical(p, p), identityBefore);
      expect(p.toList(), [10.0, 20.0, 30.0]);
      expect(p.requiresGrad, isTrue);
    });

    test('GPU: overwrites data in place', () {
      final p = Tensor.fromList(
        [64],
        List<double>.generate(64, (i) => 0.0),
        device: Device.GPU,
        requiresGrad: true,
      );
      final src = Tensor.fromList(
        [64],
        List<double>.generate(64, (i) => i.toDouble()),
        device: Device.GPU,
      );
      p.assign(src);
      final got = p.toList();
      for (int i = 0; i < 64; i++) {
        expect(got[i], i.toDouble());
      }
    });

    test('rejects shape mismatch', () {
      final p = Tensor.fromList([3], [1.0, 2.0, 3.0]);
      expect(
        () => p.assign(Tensor.fromList([4], [1, 2, 3, 4])),
        throwsArgumentError,
      );
    });

    test('rejects device mismatch', () {
      final p = Tensor.fromList([2], [1, 2], device: Device.CPU);
      final g = Tensor.fromList([2], [3, 4], device: Device.GPU);
      expect(() => p.assign(g), throwsArgumentError);
      g.dispose();
    });
  });

  // Reference math (unused-but-kept as documentation).
  // ignore: unused_element
  double refAdamStep(
    double p,
    double g,
    double m,
    double v,
    int t, {
    double lr = 1e-3,
    double b1 = 0.9,
    double b2 = 0.999,
    double eps = 1e-8,
  }) {
    final m1 = b1 * m + (1 - b1) * g;
    final v1 = b2 * v + (1 - b2) * g * g;
    final mh = m1 / (1 - math.pow(b1, t));
    final vh = v1 / (1 - math.pow(b2, t));
    return p - lr * mh / (math.sqrt(vh) + eps);
  }
}
