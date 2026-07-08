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

// Reference SDPA: q [N,Dk], k [M,Dk], v [M,Dv] -> [N,Dv].
List<double> _refSDPA(
  List<double> q,
  List<double> k,
  List<double> v,
  int n,
  int m,
  int dk,
  int dv,
) {
  final scale = 1.0 / math.sqrt(dk);
  // scores [N,M]
  final scores = List<double>.filled(n * m, 0);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      double s = 0;
      for (int d = 0; d < dk; d++) {
        s += q[i * dk + d] * k[j * dk + d];
      }
      scores[i * m + j] = s * scale;
    }
  }
  // softmax rows
  final attn = List<double>.filled(n * m, 0);
  for (int i = 0; i < n; i++) {
    final row = scores.sublist(i * m, (i + 1) * m);
    final s = _refSoftmaxRow(row);
    for (int j = 0; j < m; j++) {
      attn[i * m + j] = s[j];
    }
  }
  // out = attn @ v  ->  [N, Dv]
  final out = List<double>.filled(n * dv, 0);
  for (int i = 0; i < n; i++) {
    for (int d = 0; d < dv; d++) {
      double s = 0;
      for (int j = 0; j < m; j++) {
        s += attn[i * m + j] * v[j * dv + d];
      }
      out[i * dv + d] = s;
    }
  }
  return out;
}

void main() {
  group('scaled dot-product attention (forward)', () {
    test('matches reference on 2x2 hand-computed case', () {
      // N=M=2, Dk=Dv=2.
      final qv = [1.0, 0.0, 0.0, 1.0];
      final kv = [1.0, 0.0, 0.0, 1.0];
      final vv = [10.0, 20.0, 30.0, 40.0];
      final q = Tensor.fromList([2, 2], qv);
      final k = Tensor.fromList([2, 2], kv);
      final v = Tensor.fromList([2, 2], vv);
      final got = q.scaledDotProductAttention(k, v).toList();
      final want = _refSDPA(qv, kv, vv, 2, 2, 2, 2);
      expectClose(got, want);
    });

    test('rectangular N=3, M=4, Dk=5, Dv=2 matches reference', () {
      final rng = math.Random(11);
      final qv = List<double>.generate(3 * 5, (_) => rng.nextDouble() - 0.5);
      final kv = List<double>.generate(4 * 5, (_) => rng.nextDouble() - 0.5);
      final vv = List<double>.generate(4 * 2, (_) => rng.nextDouble() - 0.5);
      final q = Tensor.fromList([3, 5], qv);
      final k = Tensor.fromList([4, 5], kv);
      final v = Tensor.fromList([4, 2], vv);
      final got = q.scaledDotProductAttention(k, v).toList();
      final want = _refSDPA(qv, kv, vv, 3, 4, 5, 2);
      expectClose(got, want);
    });

    test('causal mask: position i cannot attend beyond i', () {
      // With a strong negative mask above the diagonal, row i's output
      // should equal the same SDPA computed only over positions <= i.
      const n = 3;
      final rng = math.Random(3);
      final qv = List<double>.generate(n * 2, (_) => rng.nextDouble());
      final kv = List<double>.generate(n * 2, (_) => rng.nextDouble());
      final vv = List<double>.generate(n * 2, (_) => rng.nextDouble());

      final maskVals = <double>[];
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          maskVals.add(j > i ? -1e9 : 0.0);
        }
      }
      final q = Tensor.fromList([n, 2], qv);
      final k = Tensor.fromList([n, 2], kv);
      final v = Tensor.fromList([n, 2], vv);
      final mask = Tensor.fromList([n, n], maskVals);
      final got = q.scaledDotProductAttention(k, v, mask: mask).toList();

      // Reference: for row 0, output = v[0]. (only allowed key)
      expectClose(
        got.sublist(0, 2),
        vv.sublist(0, 2),
        tol: 1e-4,
        reason: 'row 0 should equal v[0]',
      );
    });
  });

  group('scaled dot-product attention (backward)', () {
    test('grad wrt Q matches numerical differentiation', () {
      final qv = [0.3, -0.2, 0.1, 0.4];
      final kv = [1.0, 0.5, -0.5, 0.2];
      final vv = [1.0, 2.0, 3.0, 4.0];
      final q = Tensor.fromList([2, 2], qv, requiresGrad: true);
      final k = Tensor.fromList([2, 2], kv);
      final v = Tensor.fromList([2, 2], vv);
      q.scaledDotProductAttention(k, v).sum().backward();

      double loss(List<double> qs) {
        final out = _refSDPA(qs, kv, vv, 2, 2, 2, 2);
        return out.reduce((a, b) => a + b);
      }

      final h = 1e-3;
      final want = List<double>.filled(qv.length, 0);
      final probe = List<double>.of(qv);
      for (int i = 0; i < qv.length; i++) {
        final orig = probe[i];
        probe[i] = orig + h;
        final lp = loss(probe);
        probe[i] = orig - h;
        final lm = loss(probe);
        probe[i] = orig;
        want[i] = (lp - lm) / (2 * h);
      }
      expectClose(q.grad!.toList(), want, tol: 1e-3);
    });

    test('grad wrt V matches numerical differentiation', () {
      final qv = [0.3, -0.2, 0.1, 0.4];
      final kv = [1.0, 0.5, -0.5, 0.2];
      final vv = [1.0, 2.0, 3.0, 4.0];
      final q = Tensor.fromList([2, 2], qv);
      final k = Tensor.fromList([2, 2], kv);
      final v = Tensor.fromList([2, 2], vv, requiresGrad: true);
      q.scaledDotProductAttention(k, v).sum().backward();

      double loss(List<double> vs) {
        final out = _refSDPA(qv, kv, vs, 2, 2, 2, 2);
        return out.reduce((a, b) => a + b);
      }

      final h = 1e-3;
      final want = List<double>.filled(vv.length, 0);
      final probe = List<double>.of(vv);
      for (int i = 0; i < vv.length; i++) {
        final orig = probe[i];
        probe[i] = orig + h;
        final lp = loss(probe);
        probe[i] = orig - h;
        final lm = loss(probe);
        probe[i] = orig;
        want[i] = (lp - lm) / (2 * h);
      }
      expectClose(v.grad!.toList(), want, tol: 1e-3);
    });
  });

  group('Linear module', () {
    test('shapes and parameters', () {
      final lin = Linear(4, 3);
      expect(lin.weight.shape, [3, 4]);
      expect(lin.bias!.shape, [1, 3]);
      expect(lin.parameters().length, 2);

      final linNoBias = Linear(4, 3, bias: false);
      expect(linNoBias.parameters().length, 1);
      expect(linNoBias.bias, isNull);
    });

    test('forward matches x @ W.T + b', () {
      final lin = Linear(3, 2, seed: 42);
      final x = Tensor.fromList([2, 3], [1, 2, 3, 4, 5, 6]);
      final got = lin(x).toList();

      final w = lin.weight.toList(); // [2, 3]
      final b = lin.bias!.toList(); // [1, 2]
      final want = <double>[];
      for (int i = 0; i < 2; i++) {
        for (int o = 0; o < 2; o++) {
          double s = b[o];
          for (int d = 0; d < 3; d++) {
            s += x.toList()[i * 3 + d] * w[o * 3 + d];
          }
          want.add(s);
        }
      }
      expectClose(got, want, tol: 1e-4);
    });

    test('trains via SGD on synthetic regression', () {
      // Target: y = x * [2, -1, 0.5] + 0.3 (single output). Learn it.
      final lin = Linear(3, 1, seed: 3);
      final opt = SGD(lin.parameters(), lr: 0.05);
      final rng = math.Random(9);
      final xVals = List<double>.generate(
        20 * 3,
        (_) => rng.nextDouble() * 2 - 1,
      );
      final x = Tensor.fromList([20, 3], xVals);
      final yVals = <double>[];
      for (int i = 0; i < 20; i++) {
        final xi = xVals.sublist(i * 3, (i + 1) * 3);
        yVals.add(xi[0] * 2.0 + xi[1] * -1.0 + xi[2] * 0.5 + 0.3);
      }
      final y = Tensor.fromList([20, 1], yVals);

      double lossVal() => (lin(x) - y).pow(2).mean().toList()[0];

      final initial = lossVal();
      for (int i = 0; i < 400; i++) {
        opt.zeroGrad();
        (lin(x) - y).pow(2).mean().backward();
        opt.step();
      }
      final finalL = lossVal();
      expect(
        finalL < initial * 0.01,
        isTrue,
        reason: 'loss did not converge: initial=$initial final=$finalL',
      );
    });
  });

  group('Attention + Linear integration', () {
    test('single self-attention block trains toward target', () {
      // A tiny self-attention block over one sequence:
      //   Q = Wq(x), K = Wk(x), V = Wv(x)
      //   out = softmax(Q @ K.T / sqrt(d)) @ V
      // Fit `out` to a fixed target via SGD on the projection weights.
      final wq = Linear(3, 3, seed: 1, bias: false);
      final wk = Linear(3, 3, seed: 2, bias: false);
      final wv = Linear(3, 3, seed: 3, bias: false);
      final params = [
        ...wq.parameters(),
        ...wk.parameters(),
        ...wv.parameters(),
      ];
      final opt = Adam(params, lr: 0.05);

      final x = Tensor.fromList([2, 3], [1.0, 0.0, -1.0, 0.5, -0.5, 0.5]);
      final target = Tensor.fromList([2, 3], [0.1, 0.2, 0.3, -0.1, 0.4, 0.5]);

      Tensor forward() {
        final q = wq(x);
        final k = wk(x);
        final v = wv(x);
        return q.scaledDotProductAttention(k, v);
      }

      double lossVal() => (forward() - target).pow(2).sum().toList()[0];

      final initial = lossVal();
      for (int i = 0; i < 400; i++) {
        opt.zeroGrad();
        (forward() - target).pow(2).sum().backward();
        opt.step();
      }
      final finalL = lossVal();
      expect(
        finalL < initial * 0.1,
        isTrue,
        reason: 'loss did not decrease enough: $initial -> $finalL',
      );
    });
  });
}
