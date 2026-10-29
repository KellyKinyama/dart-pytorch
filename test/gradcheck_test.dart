/// Gradient-check coverage for every differentiable op in the tensor
/// core (A1 of `doc/parity-plan.md`). Each op's analytic backward is
/// validated against central finite differences via `expectGradCheck`.
library;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

import 'gradcheck.dart';

void main() {
  group('gradcheck — elementwise binary (CPU)', () {
    test('add', () {
      expectGradCheck(
        (t) => t[0] + t[1],
        [
          [1.0, -2.0, 3.0, 0.5],
          [0.3, 1.5, -0.7, 2.0],
        ],
        [
          [2, 2],
          [2, 2],
        ],
      );
    });

    test('sub', () {
      expectGradCheck(
        (t) => t[0] - t[1],
        [
          [1.0, -2.0, 3.0, 0.5],
          [0.3, 1.5, -0.7, 2.0],
        ],
        [
          [2, 2],
          [2, 2],
        ],
      );
    });

    test('mul', () {
      expectGradCheck(
        (t) => t[0] * t[1],
        [
          [1.0, -2.0, 3.0, 0.5],
          [0.3, 1.5, -0.7, 2.0],
        ],
        [
          [2, 2],
          [2, 2],
        ],
      );
    });

    test('div', () {
      expectGradCheck(
        (t) => t[0] / t[1],
        [
          [1.0, -2.0, 3.0, 0.5],
          [0.8, 1.5, -0.7, 2.0],
        ],
        [
          [2, 2],
          [2, 2],
        ],
      );
    });

    test('scalar mul', () {
      expectGradCheck(
        (t) => t[0] * 2.5,
        [
          [1.0, -2.0, 3.0, 0.5],
        ],
        [
          [2, 2],
        ],
      );
    });
  });

  group('gradcheck — broadcasting (CPU)', () {
    test('add [2,3] + [3] (row broadcast)', () {
      expectGradCheck(
        (t) => t[0] + t[1],
        [
          [1.0, -2.0, 3.0, 0.5, -1.0, 2.0], // [2,3]
          [0.3, 1.5, -0.7], // [3]
        ],
        [
          [2, 3],
          [3],
        ],
      );
    });

    test('mul [2,3] * [2,1] (column broadcast)', () {
      expectGradCheck(
        (t) => t[0] * t[1],
        [
          [1.0, -2.0, 3.0, 0.5, -1.0, 2.0], // [2,3]
          [1.5, -0.5], // [2,1]
        ],
        [
          [2, 3],
          [2, 1],
        ],
      );
    });

    test('sub [2,3] - [1,3] (leading-dim broadcast)', () {
      expectGradCheck(
        (t) => t[0] - t[1],
        [
          [1.0, -2.0, 3.0, 0.5, -1.0, 2.0], // [2,3]
          [0.3, 1.5, -0.7], // [1,3]
        ],
        [
          [2, 3],
          [1, 3],
        ],
      );
    });

    test('div [2,3] / [2,1] (both grads, column broadcast)', () {
      expectGradCheck(
        (t) => t[0] / t[1],
        [
          [1.0, -2.0, 3.0, 0.5, -1.0, 2.0], // [2,3]
          [1.5, 2.0], // [2,1]
        ],
        [
          [2, 3],
          [2, 1],
        ],
      );
    });

    test('mul [2,1,3] * [4,3] (3D x 2D broadcast)', () {
      expectGradCheck(
        (t) => t[0] * t[1],
        [
          [1.0, -2.0, 3.0, 0.5, -1.0, 2.0], // [2,1,3]
          [0.3, 1.5, -0.7, 2.0, 1.0, -0.5, 0.8, -1.2, 0.4, 1.1, -0.6, 0.9], // [4,3]
        ],
        [
          [2, 1, 3],
          [4, 3],
        ],
      );
    });
  });

  group('gradcheck — unary activations / math (CPU)', () {
    test('relu', () {
      // Seeds kept away from 0 so the kink doesn't break central diff.
      expectGradCheck(
        (t) => t[0].relu(),
        [
          [0.7, -0.5, 1.2, -0.9],
        ],
        [
          [2, 2],
        ],
      );
    });

    test('sigmoid', () {
      expectGradCheck(
        (t) => t[0].sigmoid(),
        [
          [0.5, -1.0, 2.0, -0.3],
        ],
        [
          [2, 2],
        ],
      );
    });

    test('tanh', () {
      expectGradCheck(
        (t) => t[0].tanh(),
        [
          [0.5, -1.0, 1.5, -0.3],
        ],
        [
          [2, 2],
        ],
      );
    });

    test('abs', () {
      expectGradCheck(
        (t) => t[0].abs(),
        [
          [0.7, -0.5, 1.2, -0.9],
        ],
        [
          [2, 2],
        ],
      );
    });

    test('log (positive domain)', () {
      expectGradCheck(
        (t) => t[0].log(),
        [
          [0.5, 1.0, 2.0, 3.0],
        ],
        [
          [2, 2],
        ],
      );
    });

    test('pow (exp=2)', () {
      expectGradCheck(
        (t) => t[0].pow(2.0),
        [
          [0.7, -0.5, 1.2, -0.9],
        ],
        [
          [2, 2],
        ],
      );
    });

    test('pow (exp=0.5, positive domain)', () {
      expectGradCheck(
        (t) => t[0].pow(0.5),
        [
          [0.5, 1.0, 2.0, 3.0],
        ],
        [
          [2, 2],
        ],
      );
    });
  });

  group('gradcheck — shape / reduction (CPU)', () {
    test('reshape', () {
      expectGradCheck(
        (t) => t[0].reshape([3, 2]),
        [
          [1.0, -2.0, 3.0, 0.5, -1.0, 2.0],
        ],
        [
          [2, 3],
        ],
      );
    });

    test('transpose', () {
      expectGradCheck(
        (t) => t[0].transpose(),
        [
          [1.0, -2.0, 3.0, 0.5, -1.0, 2.0],
        ],
        [
          [2, 3],
        ],
      );
    });

    test('sum', () {
      expectGradCheck(
        (t) => t[0].sum(),
        [
          [1.0, -2.0, 3.0, 0.5],
        ],
        [
          [2, 2],
        ],
      );
    });

    test('mean', () {
      expectGradCheck(
        (t) => t[0].mean(),
        [
          [1.0, -2.0, 3.0, 0.5],
        ],
        [
          [2, 2],
        ],
      );
    });
  });

  group('gradcheck — linear algebra (CPU)', () {
    test('matmul', () {
      expectGradCheck(
        (t) => t[0].matmul(t[1]),
        [
          [1.0, -2.0, 3.0, 0.5, -1.0, 2.0], // [2,3]
          [0.3, 1.5, -0.7, 2.0, 1.0, -0.5], // [3,2]
        ],
        [
          [2, 3],
          [3, 2],
        ],
      );
    });
  });

  group('gradcheck — nn ops (CPU)', () {
    test('softmax (row-wise)', () {
      expectGradCheck(
        (t) => t[0].softmax(),
        [
          [1.0, -2.0, 3.0, 0.5, -1.0, 2.0],
        ],
        [
          [2, 3],
        ],
      );
    });

    test('layerNorm', () {
      expectGradCheck(
        (t) => t[0].layerNorm(t[1], t[2]),
        [
          [1.0, -2.0, 3.0, 0.5, -1.0, 2.0], // x [2,3]
          [1.1, 0.9, 1.0], // gamma [3]
          [0.0, 0.1, -0.1], // beta [3]
        ],
        [
          [2, 3],
          [3],
          [3],
        ],
      );
    });

    test('rmsNorm', () {
      expectGradCheck(
        (t) => t[0].rmsNorm(t[1]),
        [
          [1.0, -2.0, 3.0, 0.5, -1.0, 2.0], // x [2,3]
          [1.1, 0.9, 1.0], // gamma [3]
        ],
        [
          [2, 3],
          [3],
        ],
      );
    });

    test('crossEntropy (logits checked, targets constant)', () {
      expectGradCheck(
        (t) => t[0].crossEntropy(t[1]),
        [
          [1.0, -2.0, 3.0, 0.5, -1.0, 2.0], // logits [2,3]
          [0.0, 2.0], // targets [2]
        ],
        [
          [2, 3],
          [2],
        ],
        check: [true, false],
      );
    });

    test('embedding (table checked, indices constant)', () {
      expectGradCheck(
        (t) => t[0].embedding(t[1]),
        [
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], // table [3,2]
          [0.0, 2.0, 1.0], // indices [3]
        ],
        [
          [3, 2],
          [3],
        ],
        check: [true, false],
      );
    });

    test('concat (axis=1)', () {
      expectGradCheck(
        (t) => TensorConcat.concat([t[0], t[1]], axis: 1),
        [
          [1.0, -2.0, 3.0, 0.5], // [2,2]
          [0.3, 1.5, -0.7, 2.0], // [2,2]
        ],
        [
          [2, 2],
          [2, 2],
        ],
      );
    });
  });
}
