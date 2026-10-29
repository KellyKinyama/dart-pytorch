/// Custom autograd `Function` API tests (A4 of `doc/parity-plan.md`).
///
/// Verifies that a user-defined [AutogradFunction] splices into the tape,
/// passes gradcheck, matches a built-in op it reimplements, and supports
/// higher-order differentiation via `createGraph`.
library;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

import 'gradcheck.dart';

/// out = a * b + c — exercises multiple inputs and broadcasting reduction.
class _MulAdd extends AutogradFunction {
  @override
  List<Tensor> forward(AutogradContext ctx, List<Tensor> inputs) {
    ctx.saveForBackward([inputs[0], inputs[1]]);
    return [inputs[0] * inputs[1] + inputs[2]];
  }

  @override
  List<Tensor?> backward(AutogradContext ctx, List<Tensor> gradOutputs) {
    final g = gradOutputs[0];
    final a = ctx.savedTensors[0];
    final b = ctx.savedTensors[1];
    // d/da = g*b, d/db = g*a, d/dc = g.
    return [g * b, g * a, g];
  }
}

/// Reimplements ReLU as a custom function to dogfood the API.
class _ReluFn extends AutogradFunction {
  @override
  List<Tensor> forward(AutogradContext ctx, List<Tensor> inputs) {
    final x = inputs[0];
    ctx.saveForBackward([x]);
    final d = x.toList();
    final out = [for (final v in d) v < 0 ? 0.0 : v];
    return [Tensor.fromList(x.shape, out, device: x.device)];
  }

  @override
  List<Tensor?> backward(AutogradContext ctx, List<Tensor> gradOutputs) {
    final x = ctx.savedTensors[0];
    final g = gradOutputs[0].toList();
    final xd = x.toList();
    final gx = [for (int i = 0; i < xd.length; i++) xd[i] > 0 ? g[i] : 0.0];
    return [Tensor.fromList(x.shape, gx, device: x.device)];
  }
}

/// out = x^3 via a custom function whose backward is itself differentiable
/// (3 x^2), so second-order flows through.
class _Cube extends AutogradFunction {
  @override
  List<Tensor> forward(AutogradContext ctx, List<Tensor> inputs) {
    ctx.saveForBackward([inputs[0]]);
    return [inputs[0].pow(3.0)];
  }

  @override
  List<Tensor?> backward(AutogradContext ctx, List<Tensor> gradOutputs) {
    final x = ctx.savedTensors[0];
    return [gradOutputs[0] * (x.pow(2.0) * 3.0)];
  }
}

void main() {
  group('AutogradFunction (CPU)', () {
    test('MulAdd passes gradcheck', () {
      expectGradCheck(
        (t) => _MulAdd().apply([t[0], t[1], t[2]])[0],
        [
          [1.0, -2.0, 3.0, 0.5], // a
          [0.3, 1.5, -0.7, 2.0], // b
          [0.1, 0.2, -0.3, 0.4], // c
        ],
        [
          [2, 2],
          [2, 2],
          [2, 2],
        ],
      );
    });

    test('MulAdd with broadcasting input (c is [1,2])', () {
      expectGradCheck(
        (t) => _MulAdd().apply([t[0], t[1], t[2]])[0],
        [
          [1.0, -2.0, 3.0, 0.5], // a [2,2]
          [0.3, 1.5, -0.7, 2.0], // b [2,2]
          [0.1, -0.2], // c [1,2] broadcast
        ],
        [
          [2, 2],
          [2, 2],
          [1, 2],
        ],
      );
    });

    test('custom ReLU matches the built-in and gradchecks', () {
      final x = Tensor.fromList(
        [4],
        [0.7, -0.5, 1.2, -0.9],
        requiresGrad: true,
      );
      final builtin = x.relu().sum();
      final custom = _ReluFn().apply([x])[0].sum();
      expect(
        (builtin.toList()[0] - custom.toList()[0]).abs() < 1e-6,
        isTrue,
      );
      expectGradCheck(
        (t) => _ReluFn().apply([t[0]])[0],
        [
          [0.7, -0.5, 1.2, -0.9],
        ],
        [
          [2, 2],
        ],
      );
    });

    test('participates in Tensor.grads and higher-order (Cube)', () {
      final x = Tensor.fromList([3], [1.0, 2.0, 3.0], requiresGrad: true);
      final y = _Cube().apply([x])[0].sum();
      final g1 = Tensor.grads([y], [x], createGraph: true)[0]!;
      expect(_close(g1.toList(), [3, 12, 27]), isTrue); // 3x^2
      final g2 = Tensor.grads([g1], [x])[0]!;
      expect(_close(g2.toList(), [6, 12, 18]), isTrue); // 6x
    });

    test('no graph wired when no input requires grad', () {
      final x = Tensor.fromList([3], [1.0, -2.0, 3.0]); // requiresGrad false
      final out = _ReluFn().apply([x])[0];
      expect(out.requiresGrad, isFalse);
    });
  });
}

bool _close(List<double> a, List<double> b, {double tol = 1e-4}) {
  if (a.length != b.length) return false;
  for (int i = 0; i < a.length; i++) {
    if ((a[i] - b[i]).abs() >= tol) return false;
  }
  return true;
}
