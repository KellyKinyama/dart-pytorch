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

List<double> _refSoftmaxRow(List<double> x) {
  final m = x.reduce(math.max);
  final es = x.map((v) => math.exp(v - m)).toList();
  final s = es.reduce((a, b) => a + b);
  return es.map((e) => e / s).toList();
}

void main() {
  group('softmax forward (CPU)', () {
    test('row sums to 1 and matches reference', () {
      final xv = <double>[1, 2, 3, -1, 0, 5, 100, 101, 102];
      final x = Tensor.fromList([3, 3], xv);
      final y = x.softmax().toList();
      final want = <double>[
        ..._refSoftmaxRow(xv.sublist(0, 3)),
        ..._refSoftmaxRow(xv.sublist(3, 6)),
        ..._refSoftmaxRow(xv.sublist(6, 9)),
      ];
      expectClose(y, want);
      // Each row sums to ~1.
      for (int i = 0; i < 3; i++) {
        final s = y.sublist(i * 3, (i + 1) * 3).reduce((a, b) => a + b);
        expect((s - 1.0).abs() < 1e-5, isTrue);
      }
    });
  });

  group('softmax backward (CPU) — numerical grad check', () {
    test('matches finite differences on scalar loss = sum(softmax(x))', () {
      // Loss = sum(softmax(x)) is constant (each row sums to 1), so grad
      // must be exactly zero. This makes for a super-clean sanity check.
      final xv = <double>[0.4, -1.1, 2.3, 0.7, 1.5, -0.6];
      final x = Tensor.fromList([2, 3], xv, requiresGrad: true);
      x.softmax().sum().backward();
      for (final v in x.grad!.toList()) {
        expect(v.abs() < 1e-5, isTrue, reason: 'grad=$v');
      }
    });

    test('matches finite differences on weighted loss', () {
      // Use loss = sum(w * softmax(x)) with fixed w — non-trivial grad.
      final xv = <double>[0.4, -1.1, 2.3, 0.7, 1.5, -0.6];
      final wv = <double>[1.0, 0.5, -0.3, 0.8, -1.2, 0.4];
      const r = 2, c = 3;

      final x = Tensor.fromList([r, c], xv, requiresGrad: true);
      final w = Tensor.fromList([r, c], wv);
      (x.softmax() * w).sum().backward();

      // Numerical grad.
      double loss(List<double> xs) {
        double total = 0;
        for (int i = 0; i < r; i++) {
          final y = _refSoftmaxRow(xs.sublist(i * c, (i + 1) * c));
          for (int j = 0; j < c; j++) {
            total += y[j] * wv[i * c + j];
          }
        }
        return total;
      }

      final h = 1e-3;
      final want = List<double>.filled(xv.length, 0);
      final probe = List<double>.of(xv);
      for (int i = 0; i < xv.length; i++) {
        final orig = probe[i];
        probe[i] = orig + h;
        final lp = loss(probe);
        probe[i] = orig - h;
        final lm = loss(probe);
        probe[i] = orig;
        want[i] = (lp - lm) / (2 * h);
      }
      expectClose(x.grad!.toList(), want, tol: 1e-3);
    });
  });

  group('softmax CPU/GPU parity', () {
    test('forward matches', () {
      final xv = List<double>.generate(64, (i) => math.sin(i * 0.3));
      final xc = Tensor.fromList([8, 8], xv, device: Device.CPU);
      final xg = Tensor.fromList([8, 8], xv, device: Device.GPU);
      expectClose(xg.softmax().toList(), xc.softmax().toList());
    });

    test('backward matches', () {
      final xv = List<double>.generate(24, (i) => (i - 12) * 0.1);
      final xc = Tensor.fromList(
        [4, 6],
        xv,
        device: Device.CPU,
        requiresGrad: true,
      );
      final xg = Tensor.fromList(
        [4, 6],
        xv,
        device: Device.GPU,
        requiresGrad: true,
      );
      // (softmax * softmax) — nontrivial nonlinear scalar loss.
      (xc.softmax() * xc.softmax()).sum().backward();
      (xg.softmax() * xg.softmax()).sum().backward();
      expectClose(xg.grad!.toList(), xc.grad!.toList(), tol: 1e-3);
    });
  });

  group('cross-entropy forward (CPU)', () {
    test('matches -log(softmax(x)[target]) row-by-row', () {
      final xv = <double>[1.0, 2.0, 3.0, 0.5, -1.0, 4.0];
      final tv = <double>[2.0, 0.0]; // target class per row
      final x = Tensor.fromList([2, 3], xv);
      final t = Tensor.fromList([2], tv);
      final loss = x.crossEntropy(t).toList();

      final want = <double>[];
      for (int i = 0; i < 2; i++) {
        final row = xv.sublist(i * 3, (i + 1) * 3);
        final y = _refSoftmaxRow(row);
        want.add(-math.log(y[tv[i].toInt()]));
      }
      expectClose(loss, want);
    });
  });

  group('cross-entropy backward (CPU) — numerical grad check', () {
    test('grad = (softmax(x) - one_hot) scaled by 1/R for mean loss', () {
      final xv = <double>[0.4, -1.1, 2.3, 0.7, 1.5, -0.6];
      final tv = <double>[1.0, 2.0];
      const r = 2, c = 3;

      final x = Tensor.fromList([r, c], xv, requiresGrad: true);
      final t = Tensor.fromList([r], tv);
      x.crossEntropy(t).mean().backward();

      double loss(List<double> xs) {
        double total = 0;
        for (int i = 0; i < r; i++) {
          final row = xs.sublist(i * c, (i + 1) * c);
          final m = row.reduce(math.max);
          double s = 0;
          for (final v in row) {
            s += math.exp(v - m);
          }
          final lse = m + math.log(s);
          total += lse - row[tv[i].toInt()];
        }
        return total / r;
      }

      final h = 1e-3;
      final want = List<double>.filled(xv.length, 0);
      final probe = List<double>.of(xv);
      for (int i = 0; i < xv.length; i++) {
        final orig = probe[i];
        probe[i] = orig + h;
        final lp = loss(probe);
        probe[i] = orig - h;
        final lm = loss(probe);
        probe[i] = orig;
        want[i] = (lp - lm) / (2 * h);
      }
      expectClose(x.grad!.toList(), want, tol: 1e-3);
    });
  });

  group('cross-entropy CPU/GPU parity', () {
    test('forward matches', () {
      final xv = List<double>.generate(20, (i) => math.sin(i * 0.2));
      final tv = <double>[0.0, 3.0, 1.0, 4.0];
      final xc = Tensor.fromList([4, 5], xv, device: Device.CPU);
      final tc = Tensor.fromList([4], tv, device: Device.CPU);
      final xg = Tensor.fromList([4, 5], xv, device: Device.GPU);
      final tg = Tensor.fromList([4], tv, device: Device.GPU);
      expectClose(xg.crossEntropy(tg).toList(), xc.crossEntropy(tc).toList());
    });

    test('backward matches', () {
      final xv = List<double>.generate(20, (i) => (i - 10) * 0.15);
      final tv = <double>[0.0, 3.0, 1.0, 4.0];
      final xc = Tensor.fromList(
        [4, 5],
        xv,
        device: Device.CPU,
        requiresGrad: true,
      );
      final tc = Tensor.fromList([4], tv, device: Device.CPU);
      final xg = Tensor.fromList(
        [4, 5],
        xv,
        device: Device.GPU,
        requiresGrad: true,
      );
      final tg = Tensor.fromList([4], tv, device: Device.GPU);
      xc.crossEntropy(tc).mean().backward();
      xg.crossEntropy(tg).mean().backward();
      expectClose(xg.grad!.toList(), xc.grad!.toList(), tol: 1e-3);
    });
  });

  group('embedding forward (CPU)', () {
    test('picks rows out of the table', () {
      final table = Tensor.fromList(
        [4, 3],
        [
          0.0, 0.1, 0.2, //
          1.0, 1.1, 1.2, //
          2.0, 2.1, 2.2, //
          3.0, 3.1, 3.2, //
        ],
      );
      final idx = Tensor.fromList([3], [2.0, 0.0, 3.0]);
      final out = table.embedding(idx).toList();
      expectClose(out, [
        2.0, 2.1, 2.2, //
        0.0, 0.1, 0.2, //
        3.0, 3.1, 3.2, //
      ]);
    });

    test('out-of-range indices yield zero rows', () {
      final table = Tensor.fromList([2, 2], [1.0, 2.0, 3.0, 4.0]);
      final idx = Tensor.fromList([2], [5.0, 0.0]);
      final out = table.embedding(idx).toList();
      expectClose(out, [0.0, 0.0, 1.0, 2.0]);
    });
  });

  group('embedding backward (CPU)', () {
    test('grads scatter-add into duplicated rows', () {
      // Indices [0, 1, 0] should accumulate row 0's grad twice.
      final table = Tensor.fromList(
        [2, 2],
        [1.0, 2.0, 3.0, 4.0],
        requiresGrad: true,
      );
      final idx = Tensor.fromList([3], [0.0, 1.0, 0.0]);
      // Loss = sum(embedding(idx)) -> upstream grad is all ones,
      // shape [3, 2].
      table.embedding(idx).sum().backward();
      // Row 0 got 2 hits, row 1 got 1 hit; each dim contributes 1.
      expectClose(table.grad!.toList(), [2.0, 2.0, 1.0, 1.0]);
    });
  });

  group('embedding CPU/GPU parity', () {
    test('forward matches', () {
      final tv = List<double>.generate(
        24,
        (i) => (i * 0.7 - 5).clamp(-3.0, 5.0),
      );
      final iv = <double>[3.0, 0.0, 5.0, 2.0, 7.0, 1.0];
      final tc = Tensor.fromList([8, 3], tv, device: Device.CPU);
      final ic = Tensor.fromList([6], iv, device: Device.CPU);
      final tg = Tensor.fromList([8, 3], tv, device: Device.GPU);
      final ig = Tensor.fromList([6], iv, device: Device.GPU);
      expectClose(tg.embedding(ig).toList(), tc.embedding(ic).toList());
    });

    test('backward matches (scatter-add semantics)', () {
      final tv = List<double>.generate(24, (i) => (i - 12) * 0.1);
      final iv = <double>[3.0, 0.0, 3.0, 2.0, 0.0, 1.0];
      final tc = Tensor.fromList(
        [8, 3],
        tv,
        device: Device.CPU,
        requiresGrad: true,
      );
      final ic = Tensor.fromList([6], iv, device: Device.CPU);
      final tg = Tensor.fromList(
        [8, 3],
        tv,
        device: Device.GPU,
        requiresGrad: true,
      );
      final ig = Tensor.fromList([6], iv, device: Device.GPU);
      tc.embedding(ic).sum().backward();
      tg.embedding(ig).sum().backward();
      expectClose(tg.grad!.toList(), tc.grad!.toList(), tol: 1e-4);
    });
  });

  group('Embedding module', () {
    test('constructs weight and forwards', () {
      final emb = Embedding(10, 4, seed: 42);
      expect(emb.parameters().length, 1);
      expect(emb.weight.shape, [10, 4]);
      final idx = Tensor.fromList([3], [1.0, 4.0, 9.0]);
      final out = emb(idx);
      expect(out.shape, [3, 4]);
    });

    test('backward accumulates into weight.grad', () {
      final emb = Embedding(5, 3, seed: 1);
      final idx = Tensor.fromList([4], [0.0, 2.0, 2.0, 4.0]);
      emb(idx).sum().backward();
      expect(emb.weight.grad, isNotNull);
      final g = emb.weight.grad!.toList();
      // Row 2 saw 2 hits → each of its dims should be 2.0.
      expect((g[6] - 2.0).abs() < 1e-5, isTrue, reason: 'row 2 dim 0');
      expect((g[7] - 2.0).abs() < 1e-5, isTrue, reason: 'row 2 dim 1');
      expect((g[8] - 2.0).abs() < 1e-5, isTrue, reason: 'row 2 dim 2');
      // Row 1 saw 0 hits → zero.
      expect(g[3], 0.0);
    });
  });
}
