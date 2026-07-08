import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

void expectClose(
  List<double> got,
  List<double> want, {
  double tol = 1e-4,
  String? reason,
}) {
  expect(got.length, want.length, reason: reason);
  for (int i = 0; i < got.length; i++) {
    expect(
      (got[i] - want[i]).abs() < tol,
      isTrue,
      reason:
          '${reason ?? ''} index $i: got ${got[i]} want ${want[i]} '
          '(diff ${(got[i] - want[i]).abs()})',
    );
  }
}

void main() {
  group('autograd basics (CPU)', () {
    test('y = x^2 gives dy/dx = 2x', () {
      final x = Tensor.fromList([3], [1.0, 2.0, 3.0], requiresGrad: true);
      final y = (x * x).sum();
      y.backward();
      expectClose(x.grad!.toList(), [2, 4, 6]);
    });

    test('y = a * x + b — leaves get correct grads', () {
      final a = Tensor.fromList([3], [2.0, 3.0, 4.0], requiresGrad: true);
      final x = Tensor.fromList([3], [5.0, 6.0, 7.0], requiresGrad: true);
      final b = Tensor.fromList([3], [1.0, 1.0, 1.0], requiresGrad: true);
      final loss = (a * x + b).sum();
      loss.backward();
      // d/da = x, d/dx = a, d/db = 1
      expectClose(a.grad!.toList(), [5, 6, 7]);
      expectClose(x.grad!.toList(), [2, 3, 4]);
      expectClose(b.grad!.toList(), [1, 1, 1]);
    });

    test('multi-use leaf accumulates gradients', () {
      // y = x*x + x  =>  dy/dx = 2x + 1
      final x = Tensor.fromList([3], [1.0, 2.0, 3.0], requiresGrad: true);
      final y = (x * x + x).sum();
      y.backward();
      expectClose(x.grad!.toList(), [3, 5, 7]);
    });

    test('subtraction: y = (a - b).sum()', () {
      final a = Tensor.fromList([2], [5.0, 6.0], requiresGrad: true);
      final b = Tensor.fromList([2], [2.0, 3.0], requiresGrad: true);
      (a - b).sum().backward();
      expectClose(a.grad!.toList(), [1, 1]);
      expectClose(b.grad!.toList(), [-1, -1]);
    });

    test('division: y = (a / b).sum() — d/da = 1/b, d/db = -a/b^2', () {
      final a = Tensor.fromList([2], [4.0, 9.0], requiresGrad: true);
      final b = Tensor.fromList([2], [2.0, 3.0], requiresGrad: true);
      (a / b).sum().backward();
      expectClose(a.grad!.toList(), [1 / 2, 1 / 3]);
      expectClose(b.grad!.toList(), [-4 / 4, -9 / 9]);
    });

    test('scalar num operand: y = (x * 3).sum() -> dy/dx = 3', () {
      final x = Tensor.fromList([4], [1.0, 2.0, 3.0, 4.0], requiresGrad: true);
      (x * 3.0).sum().backward();
      expectClose(x.grad!.toList(), [3, 3, 3, 3]);
    });

    test('scalar tensor broadcast: y = (x * s).sum() -> ds = sum(x)', () {
      final x = Tensor.fromList([4], [1.0, 2.0, 3.0, 4.0], requiresGrad: true);
      final s = Tensor.fromList([1], [3.0], requiresGrad: true);
      (x * s).sum().backward();
      expectClose(x.grad!.toList(), [3, 3, 3, 3]);
      expectClose(s.grad!.toList(), [10.0]); // 1+2+3+4
    });

    test('matmul: y = (A @ B).sum() -> dA = ones @ B^T, dB = A^T @ ones', () {
      final aData = <double>[1, 2, 3, 4, 5, 6];
      final bData = <double>[7, 8, 9, 10, 11, 12];
      final a = Tensor.fromList([2, 3], aData, requiresGrad: true);
      final b = Tensor.fromList([3, 2], bData, requiresGrad: true);
      (a.matmul(b)).sum().backward();

      // dA[i,k] = sum_j 1 * B[k,j] = row-sums of B: [15, 19, 23]
      expectClose(a.grad!.toList(), [15, 19, 23, 15, 19, 23]);
      // dB[k,j] = sum_i A[i,k] * 1 = col-sums of A: A col 0=[1,4]=5, col1=[2,5]=7, col2=[3,6]=9
      expectClose(b.grad!.toList(), [5, 5, 7, 7, 9, 9]);
    });

    test('transpose is its own reverse: y = T(x).sum() -> dx = ones', () {
      final x = Tensor.fromList(
        [2, 3],
        [1.0, 2, 3, 4, 5, 6],
        requiresGrad: true,
      );
      x.transpose().sum().backward();
      expectClose(x.grad!.toList(), [1, 1, 1, 1, 1, 1]);
    });

    test('sigmoid: numerical gradient check', () {
      const h = 1e-3;
      final xs = [-1.0, 0.0, 0.5, 2.0];
      for (final xv in xs) {
        final x = Tensor.fromList([1], [xv], requiresGrad: true);
        x.sigmoid().sum().backward();
        final analytical = x.grad!.toList()[0];
        final f1 = Tensor.fromList([1], [xv + h]).sigmoid().toList()[0];
        final f0 = Tensor.fromList([1], [xv - h]).sigmoid().toList()[0];
        final numerical = (f1 - f0) / (2 * h);
        expect(
          (analytical - numerical).abs() < 1e-3,
          isTrue,
          reason:
              'sigmoid grad at $xv: analytical=$analytical numerical=$numerical',
        );
      }
    });

    test('tanh: numerical gradient check', () {
      const h = 1e-3;
      for (final xv in [-1.0, 0.0, 0.5, 2.0]) {
        final x = Tensor.fromList([1], [xv], requiresGrad: true);
        x.tanh().sum().backward();
        final analytical = x.grad!.toList()[0];
        final f1 = Tensor.fromList([1], [xv + h]).tanh().toList()[0];
        final f0 = Tensor.fromList([1], [xv - h]).tanh().toList()[0];
        final numerical = (f1 - f0) / (2 * h);
        expect(
          (analytical - numerical).abs() < 1e-3,
          isTrue,
          reason:
              'tanh grad at $xv: analytical=$analytical numerical=$numerical',
        );
      }
    });

    test('log: dy/dx = 1/x', () {
      final x = Tensor.fromList([3], [1.0, 2.0, 4.0], requiresGrad: true);
      x.log().sum().backward();
      expectClose(x.grad!.toList(), [1.0, 0.5, 0.25]);
    });

    test('pow: y = (x^3).sum() -> dy/dx = 3 x^2', () {
      final x = Tensor.fromList([3], [1.0, 2.0, 3.0], requiresGrad: true);
      x.pow(3).sum().backward();
      expectClose(x.grad!.toList(), [3, 12, 27]);
    });

    test('relu backward on CPU', () {
      final x = Tensor.fromList(
        [4],
        [-2.0, -0.5, 1.0, 3.0],
        requiresGrad: true,
      );
      x.relu().sum().backward();
      expectClose(x.grad!.toList(), [0, 0, 1, 1]);
    });

    test('mean backward: dX = 1/n uniformly', () {
      final x = Tensor.fromList([4], [1.0, 2, 3, 4], requiresGrad: true);
      x.mean().backward();
      expectClose(x.grad!.toList(), [0.25, 0.25, 0.25, 0.25]);
    });

    test('zeroGrad clears grad', () {
      final x = Tensor.fromList([2], [1.0, 2.0], requiresGrad: true);
      x.sum().backward();
      expect(x.grad, isNotNull);
      x.zeroGrad();
      expect(x.grad, isNull);
    });

    test('backward on non-requiresGrad throws', () {
      final x = Tensor.fromList([2], [1.0, 2.0]);
      expect(() => x.backward(), throwsStateError);
    });
  });

  group('autograd on GPU (parity)', () {
    test('y = x^2 sum -> dy/dx = 2x on GPU', () {
      final x = Tensor.fromList(
        [3],
        [1.0, 2.0, 3.0],
        device: Device.GPU,
        requiresGrad: true,
      );
      (x * x).sum().backward();
      expectClose(x.grad!.toList(), [2, 4, 6]);
    });

    test('matmul on GPU: gradients match CPU', () {
      final aData = <double>[1, 2, 3, 4, 5, 6];
      final bData = <double>[7, 8, 9, 10, 11, 12];
      final a = Tensor.fromList(
        [2, 3],
        aData,
        device: Device.GPU,
        requiresGrad: true,
      );
      final b = Tensor.fromList(
        [3, 2],
        bData,
        device: Device.GPU,
        requiresGrad: true,
      );
      (a.matmul(b)).sum().backward();
      expectClose(a.grad!.toList(), [15, 19, 23, 15, 19, 23]);
      expectClose(b.grad!.toList(), [5, 5, 7, 7, 9, 9]);
    });

    test('sigmoid chain on GPU matches CPU', () {
      final data = List<double>.generate(8, (i) => (i - 4) * 0.5);
      final xCpu = Tensor.fromList([8], data, requiresGrad: true);
      final xGpu = Tensor.fromList(
        [8],
        data,
        device: Device.GPU,
        requiresGrad: true,
      );
      // loss = sum(sigmoid(x) * x)
      (xCpu.sigmoid() * xCpu).sum().backward();
      (xGpu.sigmoid() * xGpu).sum().backward();
      expectClose(xGpu.grad!.toList(), xCpu.grad!.toList(), tol: 1e-4);
    });

    test('relu backward throws on GPU with helpful message', () {
      final x = Tensor.fromList(
        [4],
        [-1.0, 0.0, 1.0, 2.0],
        device: Device.GPU,
        requiresGrad: true,
      );
      final loss = x.relu().sum();
      expect(() => loss.backward(), throwsA(isA<UnimplementedError>()));
    });
  });

  group('one-step training loop (CPU)', () {
    test('linear regression converges toward target after N SGD steps', () {
      // Target: y = 3*x0 + 4*x1 + 5. Model: y_hat = W @ x + b.
      var w = Tensor.fromList([1, 2], [0.0, 0.0], requiresGrad: true);
      var b = Tensor.fromList([1, 1], [0.0], requiresGrad: true);

      const lr = 0.05;
      final xs = [
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 3.0],
        [0.5, 4.0],
      ];
      final ys = xs.map((x) => 3 * x[0] + 4 * x[1] + 5).toList();

      double lastLoss = double.infinity;
      for (int epoch = 0; epoch < 200; epoch++) {
        double epochLoss = 0;
        for (int i = 0; i < xs.length; i++) {
          final x = Tensor.fromList([2, 1], xs[i]);
          final yTarget = ys[i];
          final yPred = (w.matmul(x) + b); // shape [1,1]
          final diff = yPred - yTarget;
          final loss = (diff * diff).sum();
          epochLoss += loss.toList()[0];
          loss.backward();

          // Manual SGD step: rebuild leaves from data - lr * grad.
          final wVals = w.toList();
          final wG = w.grad!.toList();
          w = Tensor.fromList(
            [1, 2],
            [wVals[0] - lr * wG[0], wVals[1] - lr * wG[1]],
            requiresGrad: true,
          );
          final bVals = b.toList();
          final bG = b.grad!.toList();
          b = Tensor.fromList(
            [1, 1],
            [bVals[0] - lr * bG[0]],
            requiresGrad: true,
          );
        }
        lastLoss = epochLoss;
      }
      expect(
        lastLoss < 0.5,
        isTrue,
        reason: 'expected loss to drop below 0.5, got $lastLoss',
      );
      expectClose(w.toList(), [3.0, 4.0], tol: 0.3);
      expectClose(b.toList(), [5.0], tol: 0.3);
    });
  });

  group('broadcast + / - backward preserves shared gradient handle', () {
    // Regression: `+` and `-` used to pass out._grad to both operand
    // reduction paths directly. When one path went through
    // _reduceForBroadcast (which disposes its input), the other
    // operand's aliased grad handle was invalidated — showing up as
    // a null-check in the next matmul backward on GPU. The fix
    // clones for the branch that reads g directly when both operands
    // require grad.
    for (final device in [Device.CPU, Device.GPU]) {
      test('+ row-broadcast with both operands requires_grad ($device)', () {
        final a = Tensor.fromList(
          [4, 8],
          List<double>.generate(32, (i) => (i % 5) * 0.1),
          device: device,
          requiresGrad: true,
        );
        final b = Tensor.fromList(
          [1, 8],
          List<double>.generate(8, (i) => (i % 3) * 0.2),
          device: device,
          requiresGrad: true,
        );
        final y = a + b;
        // Force another op that reads the aliased grad after the
        // reduction path runs — matmul backward exercises the same
        // failure mode observed in Linear-with-bias.
        final w = Tensor.fromList(
          [8, 4],
          List<double>.generate(32, (i) => (i % 7) * 0.05),
          device: device,
          requiresGrad: true,
        );
        final z = y.matmul(w);
        z.sum().backward();
        expect(a.grad, isNotNull);
        expect(b.grad, isNotNull);
        expect(w.grad, isNotNull);
        expect(a.grad!.shape, [4, 8]);
        expect(b.grad!.shape, [1, 8]);
      });
    }
  });
}
