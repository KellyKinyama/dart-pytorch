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

// Manual reference implementation for a single row.
List<double> _refLayerNormRow(
  List<double> x,
  List<double> g,
  List<double> b, {
  double eps = 1e-5,
}) {
  final n = x.length;
  final mean = x.reduce((a, v) => a + v) / n;
  double sq = 0;
  for (final v in x) {
    sq += (v - mean) * (v - mean);
  }
  final invStd = 1.0 / math.sqrt(sq / n + eps);
  return List<double>.generate(n, (i) => (x[i] - mean) * invStd * g[i] + b[i]);
}

/// Numerical gradient via central differences on the sum(layerNorm(...))
/// loss for a single input tensor. Returns a flat list matching `param`.
List<double> _numGrad({
  required List<double> xVals,
  required List<double> gVals,
  required List<double> bVals,
  required int r,
  required int c,
  required String wrt, // 'x' | 'g' | 'b'
  double h = 1e-3,
  double eps = 1e-5,
}) {
  double loss(List<double> xs, List<double> gs, List<double> bs) {
    double total = 0;
    for (int i = 0; i < r; i++) {
      final row = _refLayerNormRow(
        xs.sublist(i * c, (i + 1) * c),
        gs,
        bs,
        eps: eps,
      );
      for (final v in row) {
        total += v;
      }
    }
    return total;
  }

  final target = wrt == 'x'
      ? List<double>.of(xVals)
      : wrt == 'g'
      ? List<double>.of(gVals)
      : List<double>.of(bVals);

  final out = List<double>.filled(target.length, 0);
  for (int i = 0; i < target.length; i++) {
    final orig = target[i];
    target[i] = orig + h;
    final lp = wrt == 'x'
        ? loss(target, gVals, bVals)
        : wrt == 'g'
        ? loss(xVals, target, bVals)
        : loss(xVals, gVals, target);
    target[i] = orig - h;
    final lm = wrt == 'x'
        ? loss(target, gVals, bVals)
        : wrt == 'g'
        ? loss(xVals, target, bVals)
        : loss(xVals, gVals, target);
    target[i] = orig;
    out[i] = (lp - lm) / (2 * h);
  }
  return out;
}

void main() {
  group('LayerNorm forward (CPU)', () {
    test('matches manual computation on a 3x4 tensor', () {
      final xv = <double>[
        1, 2, 3, 4, //
        5, 6, 7, 8, //
        -1, 0, 1, 2, //
      ];
      final gv = <double>[1.0, 1.0, 1.0, 1.0];
      final bv = <double>[0.0, 0.0, 0.0, 0.0];

      final x = Tensor.fromList([3, 4], xv);
      final g = Tensor.fromList([4], gv);
      final b = Tensor.fromList([4], bv);

      final y = x.layerNorm(g, b);
      final want = <double>[
        ..._refLayerNormRow(xv.sublist(0, 4), gv, bv),
        ..._refLayerNormRow(xv.sublist(4, 8), gv, bv),
        ..._refLayerNormRow(xv.sublist(8, 12), gv, bv),
      ];
      expectClose(y.toList(), want);
    });

    test('gamma/beta are applied per-channel', () {
      final xv = <double>[0.0, 1.0, 2.0, 3.0];
      final gv = <double>[2.0, 0.5, 3.0, 1.0];
      final bv = <double>[0.1, -0.2, 0.3, 0.0];
      final x = Tensor.fromList([1, 4], xv);
      final g = Tensor.fromList([4], gv);
      final b = Tensor.fromList([4], bv);
      final y = x.layerNorm(g, b);
      final want = _refLayerNormRow(xv, gv, bv);
      expectClose(y.toList(), want);
    });
  });

  group('LayerNorm backward (CPU) — numerical gradient check', () {
    test('grad w.r.t. x matches finite differences', () {
      final xv = <double>[0.4, -1.1, 2.3, 0.7, 1.5, -0.6];
      final gv = <double>[1.0, 1.2, 0.9];
      final bv = <double>[0.0, 0.0, 0.0];
      const r = 2, c = 3;

      final x = Tensor.fromList([r, c], xv, requiresGrad: true);
      final g = Tensor.fromList([c], gv);
      final b = Tensor.fromList([c], bv);
      x.layerNorm(g, b).sum().backward();

      final want = _numGrad(
        xVals: xv,
        gVals: gv,
        bVals: bv,
        r: r,
        c: c,
        wrt: 'x',
      );
      expectClose(x.grad!.toList(), want, tol: 1e-3);
    });

    test('grad w.r.t. gamma matches finite differences', () {
      final xv = <double>[0.4, -1.1, 2.3, 0.7, 1.5, -0.6];
      final gv = <double>[1.0, 1.2, 0.9];
      final bv = <double>[0.0, 0.0, 0.0];
      const r = 2, c = 3;

      final x = Tensor.fromList([r, c], xv);
      final g = Tensor.fromList([c], gv, requiresGrad: true);
      final b = Tensor.fromList([c], bv);
      x.layerNorm(g, b).sum().backward();

      final want = _numGrad(
        xVals: xv,
        gVals: gv,
        bVals: bv,
        r: r,
        c: c,
        wrt: 'g',
      );
      expectClose(g.grad!.toList(), want, tol: 1e-3);
    });

    test('grad w.r.t. beta matches finite differences', () {
      final xv = <double>[0.4, -1.1, 2.3, 0.7, 1.5, -0.6];
      final gv = <double>[1.0, 1.2, 0.9];
      final bv = <double>[0.1, -0.2, 0.3];
      const r = 2, c = 3;

      final x = Tensor.fromList([r, c], xv);
      final g = Tensor.fromList([c], gv);
      final b = Tensor.fromList([c], bv, requiresGrad: true);
      x.layerNorm(g, b).sum().backward();

      final want = _numGrad(
        xVals: xv,
        gVals: gv,
        bVals: bv,
        r: r,
        c: c,
        wrt: 'b',
      );
      expectClose(b.grad!.toList(), want, tol: 1e-3);
    });
  });

  group('LayerNorm CPU/GPU parity', () {
    test('forward matches between CPU and GPU', () {
      final xv = List<double>.generate(64, (i) => math.sin(i * 0.3));
      final gv = List<double>.generate(8, (i) => 1.0 + i * 0.05);
      final bv = List<double>.generate(8, (i) => i * 0.01);

      final xc = Tensor.fromList([8, 8], xv, device: Device.CPU);
      final gc = Tensor.fromList([8], gv, device: Device.CPU);
      final bc = Tensor.fromList([8], bv, device: Device.CPU);
      final yc = xc.layerNorm(gc, bc);

      final xg = Tensor.fromList([8, 8], xv, device: Device.GPU);
      final gg = Tensor.fromList([8], gv, device: Device.GPU);
      final bg = Tensor.fromList([8], bv, device: Device.GPU);
      final yg = xg.layerNorm(gg, bg);

      expectClose(yg.toList(), yc.toList(), tol: 1e-4);
    });

    test('backward grads match between CPU and GPU', () {
      final xv = List<double>.generate(24, (i) => (i - 12) * 0.1);
      final gv = <double>[1.0, 0.7, 1.3, 0.5, 1.1, 0.9];
      final bv = <double>[0.0, 0.1, -0.1, 0.2, -0.2, 0.05];

      Tensor xc = Tensor.fromList(
        [4, 6],
        xv,
        device: Device.CPU,
        requiresGrad: true,
      );
      Tensor gc = Tensor.fromList(
        [6],
        gv,
        device: Device.CPU,
        requiresGrad: true,
      );
      Tensor bc = Tensor.fromList(
        [6],
        bv,
        device: Device.CPU,
        requiresGrad: true,
      );
      xc.layerNorm(gc, bc).sum().backward();

      Tensor xg = Tensor.fromList(
        [4, 6],
        xv,
        device: Device.GPU,
        requiresGrad: true,
      );
      Tensor gg = Tensor.fromList(
        [6],
        gv,
        device: Device.GPU,
        requiresGrad: true,
      );
      Tensor bg = Tensor.fromList(
        [6],
        bv,
        device: Device.GPU,
        requiresGrad: true,
      );
      xg.layerNorm(gg, bg).sum().backward();

      expectClose(xg.grad!.toList(), xc.grad!.toList(), tol: 1e-3);
      expectClose(gg.grad!.toList(), gc.grad!.toList(), tol: 1e-3);
      expectClose(bg.grad!.toList(), bc.grad!.toList(), tol: 1e-3);
    });
  });

  group('LayerNorm module', () {
    test(
      'initializes gamma=1, beta=0 and yields zero-mean/unit-var output',
      () {
        final ln = LayerNorm(4);
        expect(ln.parameters().length, 2);
        expect(ln.gamma.toList(), [1.0, 1.0, 1.0, 1.0]);
        expect(ln.beta.toList(), [0.0, 0.0, 0.0, 0.0]);

        final x = Tensor.fromList([2, 4], [1.0, 2, 3, 4, -2, -1, 0, 1]);
        final y = ln(x).toList();
        // Row means should be ~0, row variances ~1.
        for (int i = 0; i < 2; i++) {
          final row = y.sublist(i * 4, (i + 1) * 4);
          final m = row.reduce((a, v) => a + v) / 4;
          double sq = 0;
          for (final v in row) {
            sq += (v - m) * (v - m);
          }
          expect(m.abs() < 1e-4, isTrue, reason: 'row $i mean=$m');
          expect(
            ((sq / 4) - 1).abs() < 1e-2,
            isTrue,
            reason: 'row $i var=${sq / 4}',
          );
        }
      },
    );

    test('parameters accumulate gradients and can be zeroed', () {
      final ln = LayerNorm(3);
      final x = Tensor.fromList(
        [2, 3],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        requiresGrad: true,
      );
      ln(x).sum().backward();
      expect(ln.gamma.grad, isNotNull);
      expect(ln.beta.grad, isNotNull);

      ln.zeroGrad();
      expect(ln.gamma.grad, isNull);
      expect(ln.beta.grad, isNull);
    });
  });
}
