/// Engine-robustness tests (A6 of `doc/parity-plan.md`): iterative
/// (stack-safe) traversal on deep graphs, gradient hooks, and anomaly
/// detection.
library;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

void expectClose(List<double> got, List<double> want, {double tol = 1e-4}) {
  expect(got.length, want.length);
  for (int i = 0; i < got.length; i++) {
    expect((got[i] - want[i]).abs() < tol, isTrue,
        reason: 'index $i: got ${got[i]} want ${want[i]}');
  }
}

void main() {
  group('iterative traversal (CPU)', () {
    test('deep chain backprops without stack overflow', () {
      // A recursive visit would blow the stack here; the worklist must not.
      final x = Tensor.fromList([1], [1.0], requiresGrad: true);
      var y = x;
      const depth = 20000;
      for (int i = 0; i < depth; i++) {
        y = y + 1.0; // d/dx of a long add chain is still 1
      }
      y.backward();
      expectClose(x.grad!.toList(), [1.0]);
    });

    test('deep multiply-by-one chain keeps gradient exact', () {
      final x = Tensor.fromList([2], [3.0, -2.0], requiresGrad: true);
      var y = x;
      for (int i = 0; i < 5000; i++) {
        y = y * 1.0;
      }
      y.sum().backward();
      expectClose(x.grad!.toList(), [1.0, 1.0]);
    });
  });

  group('gradient hooks (CPU)', () {
    test('hook observes the gradient', () {
      final x = Tensor.fromList([3], [1.0, 2.0, 3.0], requiresGrad: true);
      List<double>? seen;
      x.registerHook((g) {
        seen = g.toList();
        return null;
      });
      (x * x).sum().backward();
      expect(seen, isNotNull);
      expectClose(seen!, [2, 4, 6]);
    });

    test('hook can replace the gradient (scale by 10)', () {
      final x = Tensor.fromList([3], [1.0, 2.0, 3.0], requiresGrad: true);
      x.registerHook((g) => g * 10.0);
      (x * x).sum().backward();
      expectClose(x.grad!.toList(), [20, 40, 60]);
    });

    test('removeHook stops it firing', () {
      final x = Tensor.fromList([3], [1.0, 2.0, 3.0], requiresGrad: true);
      var fired = false;
      final id = x.registerHook((g) {
        fired = true;
        return null;
      });
      x.removeHook(id);
      (x * x).sum().backward();
      expect(fired, isFalse);
    });

    test('hook fires in Tensor.grads too', () {
      final x = Tensor.fromList([2], [2.0, 3.0], requiresGrad: true);
      var fired = false;
      x.registerHook((g) {
        fired = true;
        return null;
      });
      Tensor.grads([(x * x).sum()], [x]);
      expect(fired, isTrue);
    });
  });

  group('anomaly detection (CPU)', () {
    tearDown(() => Tensor.detectAnomaly = false);

    test('off by default: non-finite grad does not throw', () {
      final x = Tensor.fromList([2], [0.0, 1.0], requiresGrad: true);
      // d/dx log(x) = 1/x -> Inf at x=0, but detection is off.
      expect(() => x.log().sum().backward(), returnsNormally);
    });

    test('on: throws a clear anomaly error on non-finite grad', () {
      Tensor.detectAnomaly = true;
      final x = Tensor.fromList([2], [0.0, 1.0], requiresGrad: true);
      expect(
        () => x.log().sum().backward(),
        throwsA(
          isA<StateError>().having(
            (e) => e.message,
            'message',
            contains('anomaly'),
          ),
        ),
      );
    });
  });
}
