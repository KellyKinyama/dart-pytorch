/// Higher-order gradient tests (A3 of `doc/parity-plan.md`).
///
/// Exercises the functional `Tensor.grad` entry point: first-order parity
/// with `backward`, non-mutation of public `.grad`, and second-order
/// derivatives via `createGraph`.
library;

import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

void expectClose(List<double> got, List<double> want, {double tol = 1e-4}) {
  expect(got.length, want.length);
  for (int i = 0; i < got.length; i++) {
    expect(
      (got[i] - want[i]).abs() < tol,
      isTrue,
      reason: 'index $i: got ${got[i]} want ${want[i]}',
    );
  }
}

void main() {
  group('Tensor.grad — first order (CPU)', () {
    test('matches backward() for sum(x^2)', () {
      final x = Tensor.fromList([3], [1.0, 2.0, 3.0], requiresGrad: true);
      final y = (x * x).sum();
      final g = Tensor.grads([y], [x]);
      expectClose(g[0]!.toList(), [2, 4, 6]);
    });

    test('does not mutate public .grad', () {
      final x = Tensor.fromList([3], [1.0, 2.0, 3.0], requiresGrad: true);
      final y = (x * x).sum();
      Tensor.grads([y], [x]);
      expect(x.grad, isNull, reason: 'grad() must not touch .grad');

      // A subsequent backward still works and is unaffected.
      (x * x).sum().backward();
      expectClose(x.grad!.toList(), [2, 4, 6]);
    });

    test('multiple inputs: d/da(a*b)=b, d/db(a*b)=a', () {
      final a = Tensor.fromList([3], [2.0, 3.0, 4.0], requiresGrad: true);
      final b = Tensor.fromList([3], [5.0, 6.0, 7.0], requiresGrad: true);
      final y = (a * b).sum();
      final g = Tensor.grads([y], [a, b]);
      expectClose(g[0]!.toList(), [5, 6, 7]);
      expectClose(g[1]!.toList(), [2, 3, 4]);
    });

    test('gradOutputs seed scales the result', () {
      final x = Tensor.fromList([3], [1.0, 2.0, 3.0], requiresGrad: true);
      final y = x * x; // non-scalar output
      final seed = Tensor.fromList([3], [1.0, 1.0, 1.0]);
      final g = Tensor.grads([y], [x], gradOutputs: [seed]);
      expectClose(g[0]!.toList(), [2, 4, 6]);
    });
  });

  group('Tensor.grad — second order (CPU)', () {
    test('d²/dx² sum(x²) = 2', () {
      final x = Tensor.fromList([3], [1.0, 2.0, 3.0], requiresGrad: true);
      final y = (x * x).sum();
      final g1 = Tensor.grads([y], [x], createGraph: true)[0]!;
      expectClose(g1.toList(), [2, 4, 6]);
      final g2 = Tensor.grads([g1], [x])[0]!;
      expectClose(g2.toList(), [2, 2, 2]);
    });

    test('d²/dx² sum(x³) = 6x', () {
      final x = Tensor.fromList([3], [1.0, 2.0, 3.0], requiresGrad: true);
      final y = x.pow(3.0).sum();
      final g1 = Tensor.grads([y], [x], createGraph: true)[0]!;
      expectClose(g1.toList(), [3, 12, 27]); // 3x²
      final g2 = Tensor.grads([g1], [x])[0]!;
      expectClose(g2.toList(), [6, 12, 18]); // 6x
    });

    test('second derivative of tanh: d²/dx² = -2 tanh (1 - tanh²)', () {
      final xs = [0.5, -1.0, 1.5];
      final x = Tensor.fromList([3], xs, requiresGrad: true);
      final y = x.tanh().sum();
      final g1 = Tensor.grads([y], [x], createGraph: true)[0]!;
      final g2 = Tensor.grads([g1], [x])[0]!;
      final want = [
        for (final v in xs)
          () {
            final t = _tanh(v);
            return -2 * t * (1 - t * t);
          }(),
      ];
      expectClose(g2.toList(), want, tol: 1e-3);
    });
  });
}

double _tanh(double x) {
  final e2 = 1.0 - 2.0 / (1.0 + math.exp(2 * x.abs()));
  return x >= 0 ? e2 : -e2;
}
