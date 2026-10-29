/// In-place op tracking tests (A5 of `doc/parity-plan.md`).
///
/// Covers the value correctness of `zero_` / `add_` / `mul_` and the
/// version-counter guard that catches a tensor being mutated in-place
/// after an op saved it for backward.
library;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

void expectClose(List<double> got, List<double> want, {double tol = 1e-6}) {
  expect(got.length, want.length);
  for (int i = 0; i < got.length; i++) {
    expect((got[i] - want[i]).abs() < tol, isTrue,
        reason: 'index $i: got ${got[i]} want ${want[i]}');
  }
}

void main() {
  group('in-place op values (CPU)', () {
    test('add_ scalar and tensor', () {
      final x = Tensor.fromList([3], [1.0, 2.0, 3.0]);
      expect(identical(x.add_(1.0), x), isTrue);
      expectClose(x.toList(), [2, 3, 4]);
      x.add_(Tensor.fromList([3], [10.0, 20.0, 30.0]));
      expectClose(x.toList(), [12, 23, 34]);
    });

    test('mul_ scalar', () {
      final x = Tensor.fromList([3], [1.0, 2.0, 3.0]);
      x.mul_(2.0);
      expectClose(x.toList(), [2, 4, 6]);
    });

    test('zero_', () {
      final x = Tensor.fromList([4], [1.0, -2.0, 3.0, 4.0]);
      x.zero_();
      expectClose(x.toList(), [0, 0, 0, 0]);
    });
  });

  group('version guard catches unsafe in-place (CPU)', () {
    test('mutating a saved mul operand after forward throws', () {
      final a = Tensor.fromList([3], [1.0, 2.0, 3.0], requiresGrad: true);
      final b = Tensor.fromList([3], [4.0, 5.0, 6.0], requiresGrad: true);
      final y = (a * b).sum();
      a.mul_(2.0); // a was saved by `*` for its backward
      expect(y.backward, throwsStateError);
    });

    test('mutating a saved matmul operand throws', () {
      final A = Tensor.fromList([2, 2], [1.0, 2.0, 3.0, 4.0],
          requiresGrad: true);
      final B = Tensor.fromList([2, 2], [5.0, 6.0, 7.0, 8.0],
          requiresGrad: true);
      final y = A.matmul(B).sum();
      B.add_(1.0);
      expect(y.backward, throwsStateError);
    });

    test('guard also fires through Tensor.grads', () {
      final x = Tensor.fromList([3], [1.0, 2.0, 3.0], requiresGrad: true);
      final y = x.pow(2.0).sum();
      x.add_(0.5); // x saved by pow backward
      expect(() => Tensor.grads([y], [x]), throwsStateError);
    });
  });

  group('safe in-place does not false-positive (CPU)', () {
    test('mutating an add operand is allowed (add backward ignores values)', () {
      final a = Tensor.fromList([3], [1.0, 2.0, 3.0], requiresGrad: true);
      final b = Tensor.fromList([3], [4.0, 5.0, 6.0], requiresGrad: true);
      final y = (a + b).sum();
      a.add_(100.0); // add backward is identity — safe
      y.backward();
      expectClose(a.grad!.toList(), [1, 1, 1]);
      expectClose(b.grad!.toList(), [1, 1, 1]);
    });

    test('in-place before the op uses post-mutation values', () {
      final a = Tensor.fromList([3], [1.0, 2.0, 3.0], requiresGrad: true);
      final b = Tensor.fromList([3], [4.0, 5.0, 6.0], requiresGrad: true);
      a.mul_(2.0); // mutate BEFORE building the graph — fine
      final y = (a * b).sum();
      y.backward();
      // d/db = a (post-mutation) = [2,4,6]
      expectClose(b.grad!.toList(), [2, 4, 6]);
      // d/da = b = [4,5,6]
      expectClose(a.grad!.toList(), [4, 5, 6]);
    });
  });
}
