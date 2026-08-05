import 'dart:math' as math;

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

/// Reference: y_j = x_j / sqrt(mean(x^2) + eps) * gamma_j.
List<double> _refRmsNormRow(
  List<double> x,
  List<double> g, {
  double eps = 1e-6,
}) {
  double sq = 0;
  for (final v in x) {
    sq += v * v;
  }
  final invRms = 1.0 / math.sqrt(sq / x.length + eps);
  return [for (int j = 0; j < x.length; j++) x[j] * invRms * g[j]];
}

void main() {
  group('RMSNorm forward (CPU)', () {
    test('matches manual computation on a 3x4 tensor', () {
      final xv = <double>[
        1, 2, 3, 4, //
        -1, 0.5, -0.5, 2, //
        0.1, -0.1, 0.2, -0.2, //
      ];
      final gv = <double>[0.9, 1.1, 1.0, 1.2];

      final x = Tensor.fromList([3, 4], xv);
      final g = Tensor.fromList([4], gv);

      final y = x.rmsNorm(g);
      final want = <double>[
        ..._refRmsNormRow(xv.sublist(0, 4), gv),
        ..._refRmsNormRow(xv.sublist(4, 8), gv),
        ..._refRmsNormRow(xv.sublist(8, 12), gv),
      ];
      expectClose(y.toList(), want, tol: 1e-5);
    });

    test('gamma=1 gives unit-RMS rows', () {
      final xv = <double>[3, 4, 0, 0, 1, 2, 2, 1];
      final gv = <double>[1, 1, 1, 1];
      final x = Tensor.fromList([2, 4], xv);
      final g = Tensor.fromList([4], gv);

      final y = x.rmsNorm(g).toList();
      for (int i = 0; i < 2; i++) {
        double sq = 0;
        for (int j = 0; j < 4; j++) {
          sq += y[i * 4 + j] * y[i * 4 + j];
        }
        final rms = math.sqrt(sq / 4);
        expect((rms - 1.0).abs() < 1e-4, isTrue, reason: 'row $i rms=$rms');
      }
    });

    test('rank-3 input reshapes correctly', () {
      final xv = <double>[
        for (int i = 0; i < 2 * 3 * 4; i++) (i + 1).toDouble(),
      ];
      final x = Tensor.fromList([2, 3, 4], xv);
      final g = Tensor.fromList([4], <double>[1, 1, 1, 1]);
      final y = x.rmsNorm(g);
      expect(y.shape, [2, 3, 4]);
      final first = y.toList().sublist(0, 4);
      final want = _refRmsNormRow(<double>[1, 2, 3, 4], <double>[1, 1, 1, 1]);
      expectClose(first, want, tol: 1e-5);
    });

    test('RMSNorm module forward matches direct op call', () {
      final xv = <double>[0.5, -1.0, 2.0, 0.25, 1.5, -0.5];
      final x = Tensor.fromList([2, 3], xv);
      final rn = RMSNorm(3);
      final yMod = rn(x).toList();
      final yOp = x.rmsNorm(rn.gamma).toList();
      expectClose(yMod, yOp, tol: 1e-6);
    });
  });

  group('RMSNorm backward (CPU)', () {
    test('dGamma / dX finite-difference check', () {
      const eps = 1e-6;
      final xData = <double>[0.5, -1.0, 2.0, 0.25, 1.5, -0.5];
      final gData = <double>[1.1, 0.9, 1.0];

      final x = Tensor.fromList([2, 3], xData, requiresGrad: true);
      final gamma = Tensor.fromList([3], gData, requiresGrad: true);
      final y = x.rmsNorm(gamma, eps: eps);
      final loss = y.sum();
      loss.backward();

      final gXAnal = x.grad!.toList();
      final gGAnal = gamma.grad!.toList();

      const h = 1e-3;
      double lossAt(List<double> xd, List<double> gd) {
        final x2 = Tensor.fromList([2, 3], xd);
        final g2 = Tensor.fromList([3], gd);
        double s = 0;
        for (final v in x2.rmsNorm(g2, eps: eps).toList()) {
          s += v;
        }
        return s;
      }

      for (int i = 0; i < 6; i++) {
        final plus = List<double>.of(xData);
        final minus = List<double>.of(xData);
        plus[i] += h;
        minus[i] -= h;
        final fd = (lossAt(plus, gData) - lossAt(minus, gData)) / (2 * h);
        expect(
          (gXAnal[i] - fd).abs() < 5e-3,
          isTrue,
          reason: 'dX[$i] anal=${gXAnal[i]} fd=$fd',
        );
      }
      for (int j = 0; j < 3; j++) {
        final plus = List<double>.of(gData);
        final minus = List<double>.of(gData);
        plus[j] += h;
        minus[j] -= h;
        final fd = (lossAt(xData, plus) - lossAt(xData, minus)) / (2 * h);
        expect(
          (gGAnal[j] - fd).abs() < 5e-3,
          isTrue,
          reason: 'dGamma[$j] anal=${gGAnal[j]} fd=$fd',
        );
      }
    });
  });
}
