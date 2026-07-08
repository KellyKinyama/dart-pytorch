/// Tests for the Attention-Free Transformer (AFT) primitives and
/// modules.
library;

import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

const double _tol = 1e-4;
const double _numGradTol = 3e-2;

void expectClose(List<double> got, List<double> want, {double tol = 1e-4}) {
  expect(got.length, want.length,
      reason: 'length got=${got.length} want=${want.length}');
  for (int i = 0; i < got.length; i++) {
    expect((got[i] - want[i]).abs() < tol, isTrue,
        reason: 'idx $i got=${got[i]} want=${want[i]} (tol=$tol)');
  }
}

List<double> _rand(int n, {int seed = 0, double scale = 0.5}) {
  final r = math.Random(seed);
  return List<double>.generate(n, (_) => (r.nextDouble() * 2 - 1) * scale);
}

double _sigmoid(double z) => 1.0 / (1.0 + math.exp(-z));

/// Reference AFT-Full forward in plain Dart doubles (no autograd).
/// Mirrors the docstring/kernel exactly so `aftFull` can be validated.
List<double> _referenceAft(
  List<double> q,
  List<double> k,
  List<double> v,
  List<double> w,
  int t,
  int d, {
  bool masked = false,
}) {
  final out = List<double>.filled(t * d, 0.0);
  for (int i = 0; i < t; i++) {
    final limit = masked ? i + 1 : t;
    for (int dd = 0; dd < d; dd++) {
      double maxVal = -1e30;
      for (int tp = 0; tp < limit; tp++) {
        final s = k[tp * d + dd] + w[i * t + tp];
        if (s > maxVal) maxVal = s;
      }
      double num = 0.0;
      double den = 0.0;
      for (int tp = 0; tp < limit; tp++) {
        final e = math.exp(k[tp * d + dd] + w[i * t + tp] - maxVal);
        num += e * v[tp * d + dd];
        den += e;
      }
      out[i * d + dd] = _sigmoid(q[i * d + dd]) * num / (den + 1e-6);
    }
  }
  return out;
}

void _closeList(
  List<double> a,
  List<double> b, {
  double tol = _tol,
  String? name,
}) {
  expect(a.length, b.length);
  for (int i = 0; i < a.length; i++) {
    expect(
      a[i],
      closeTo(b[i], tol),
      reason: '${name ?? 'idx'} $i: got ${a[i]} vs ref ${b[i]}',
    );
  }
}

/// Finite-difference gradient of `f(x)` wrt `x`.
List<double> _numGrad(
  List<double> x,
  double Function(List<double>) f, {
  double eps = 5e-3,
}) {
  final g = List<double>.filled(x.length, 0.0);
  for (int i = 0; i < x.length; i++) {
    final orig = x[i];
    x[i] = orig + eps;
    final fp = f(x);
    x[i] = orig - eps;
    final fm = f(x);
    x[i] = orig;
    g[i] = (fp - fm) / (2 * eps);
  }
  return g;
}

void main() {
  group('Tensor.aftFull forward', () {
    test('matches Dart-double reference (full, no mask)', () {
      const t = 4;
      const d = 3;
      final q = _rand(t * d, seed: 1);
      final k = _rand(t * d, seed: 2);
      final v = _rand(t * d, seed: 3);
      final w = _rand(t * t, seed: 4);
      final out = TensorAft.aftFull(
        Tensor.fromList([t, d], q),
        Tensor.fromList([t, d], k),
        Tensor.fromList([t, d], v),
        Tensor.fromList([t, t], w),
      );
      expect(out.shape, [t, d]);
      _closeList(out.toList(), _referenceAft(q, k, v, w, t, d));
    });

    test('matches reference (masked / causal)', () {
      const t = 5;
      const d = 4;
      final q = _rand(t * d, seed: 10);
      final k = _rand(t * d, seed: 11);
      final v = _rand(t * d, seed: 12);
      final w = _rand(t * t, seed: 13);
      final out = TensorAft.aftFull(
        Tensor.fromList([t, d], q),
        Tensor.fromList([t, d], k),
        Tensor.fromList([t, d], v),
        Tensor.fromList([t, t], w),
        masked: true,
      );
      _closeList(out.toList(), _referenceAft(q, k, v, w, t, d, masked: true));
    });

    test('rejects shape mismatches', () {
      final q = Tensor.fromList([3, 2], _rand(6));
      final k = Tensor.fromList([3, 2], _rand(6));
      final v = Tensor.fromList([3, 2], _rand(6));
      final wBad = Tensor.fromList([4, 4], _rand(16));
      expect(() => TensorAft.aftFull(q, k, v, wBad), throwsArgumentError);
      final qBad = Tensor.fromList([4, 2], _rand(8));
      final wOk = Tensor.fromList([3, 3], _rand(9));
      expect(() => TensorAft.aftFull(qBad, k, v, wOk), throwsArgumentError);
    });
  });

  group('Tensor.aftFull backward (finite-difference)', () {
    test('grads for q/k/v/w match numerical (full, small)', () {
      const t = 3;
      const d = 2;
      final q0 = _rand(t * d, seed: 20);
      final k0 = _rand(t * d, seed: 21);
      final v0 = _rand(t * d, seed: 22);
      final w0 = _rand(t * t, seed: 23);

      double loss(
        List<double> q,
        List<double> k,
        List<double> v,
        List<double> w,
      ) {
        final out = _referenceAft(q, k, v, w, t, d);
        double s = 0.0;
        for (final o in out) {
          s += o * o;
        }
        return 0.5 * s;
      }

      final gQNum = _numGrad(
        List<double>.from(q0),
        (qq) => loss(qq, k0, v0, w0),
      );
      final gKNum = _numGrad(
        List<double>.from(k0),
        (kk) => loss(q0, kk, v0, w0),
      );
      final gVNum = _numGrad(
        List<double>.from(v0),
        (vv) => loss(q0, k0, vv, w0),
      );
      final gWNum = _numGrad(
        List<double>.from(w0),
        (ww) => loss(q0, k0, v0, ww),
      );

      final qT = Tensor.fromList([t, d], q0, requiresGrad: true);
      final kT = Tensor.fromList([t, d], k0, requiresGrad: true);
      final vT = Tensor.fromList([t, d], v0, requiresGrad: true);
      final wT = Tensor.fromList([t, t], w0, requiresGrad: true);
      final out = TensorAft.aftFull(qT, kT, vT, wT);
      // Loss = 0.5 * sum(out^2)  =>  d/d out = out
      final scalar = (out * out).sum() * 0.5;
      scalar.backward();

      _closeList(qT.grad!.toList(), gQNum, tol: _numGradTol, name: 'gQ');
      _closeList(kT.grad!.toList(), gKNum, tol: _numGradTol, name: 'gK');
      _closeList(vT.grad!.toList(), gVNum, tol: _numGradTol, name: 'gV');
      _closeList(wT.grad!.toList(), gWNum, tol: _numGradTol, name: 'gW');
    });

    test('masked backward: gradient rows above the diagonal in W are 0', () {
      const t = 4;
      const d = 2;
      final qT = Tensor.fromList(
        [t, d],
        _rand(t * d, seed: 30),
        requiresGrad: true,
      );
      final kT = Tensor.fromList(
        [t, d],
        _rand(t * d, seed: 31),
        requiresGrad: true,
      );
      final vT = Tensor.fromList(
        [t, d],
        _rand(t * d, seed: 32),
        requiresGrad: true,
      );
      final wT = Tensor.fromList(
        [t, t],
        _rand(t * t, seed: 33),
        requiresGrad: true,
      );
      final out = TensorAft.aftFull(qT, kT, vT, wT, masked: true);
      out.sum().backward();
      final gW = wT.grad!.toList();
      for (int i = 0; i < t; i++) {
        for (int j = i + 1; j < t; j++) {
          expect(
            gW[i * t + j],
            0.0,
            reason: 'W[$i,$j] should have 0 grad under causal mask',
          );
        }
      }
    });
  });

  group('Tensor.sliceTopLeft', () {
    test('forward and backward round-trip', () {
      final t = Tensor.fromList(
        [4, 4],
        List<double>.generate(16, (i) => i.toDouble()),
        requiresGrad: true,
      );
      final s = TensorAft.sliceTopLeft(t, 2, 3);
      expect(s.shape, [2, 3]);
      expect(s.toList(), [0, 1, 2, 4, 5, 6]);

      s.sum().backward();
      final expected = <double>[];
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          expected.add((i < 2 && j < 3) ? 1.0 : 0.0);
        }
      }
      _closeList(t.grad!.toList(), expected);
    });
  });

  group('AFTAttention module', () {
    test('preserves shape [T, embedDim]', () {
      const t = 5;
      const d = 8;
      final m = AFTAttention(d, maxSeqLen: 16, seed: 40);
      final x = Tensor.fromList([t, d], _rand(t * d, seed: 41));
      final out = m(x);
      expect(out.shape, [t, d]);
    });

    test('causal AFT: reducing loss with SGD', () {
      const t = 4;
      const d = 6;
      final m = AFTAttention(d, maxSeqLen: t, masked: true, seed: 42);
      final x = Tensor.fromList([t, d], _rand(t * d, seed: 43));
      final y = Tensor.fromList([t, d], _rand(t * d, seed: 44));
      final opt = SGD(m.parameters(), lr: 0.05);

      double loss() {
        final o = m(x);
        return ((o - y) * (o - y)).mean().toList()[0];
      }

      final l0 = loss();
      for (int step = 0; step < 30; step++) {
        opt.zeroGrad();
        final o = m(x);
        final l = ((o - y) * (o - y)).mean();
        l.backward();
        opt.step();
      }
      final l1 = loss();
      expect(l1, lessThan(l0), reason: 'loss should decrease (l0=$l0 l1=$l1)');
    });
  });

  group('AFTLanguageModel', () {
    test('forward produces [seqLen, vocab] logits', () {
      final lm = AFTLanguageModel(
        vocabSize: 16,
        embedDim: 16,
        numLayers: 2,
        maxLen: 8,
        seed: 50,
      );
      final tokens = Tensor.fromList([6], [1, 3, 5, 7, 9, 11]);
      final logits = lm(tokens);
      expect(logits.shape, [6, 16]);
    });

    test('rejects non-1D tokens and seqLen > maxLen', () {
      final lm = AFTLanguageModel(
        vocabSize: 8,
        embedDim: 8,
        numLayers: 1,
        maxLen: 4,
        seed: 51,
      );
      expect(
        () => lm(Tensor.fromList([2, 2], [0, 1, 2, 3])),
        throwsArgumentError,
      );
      expect(
        () => lm(Tensor.fromList([5], [0, 1, 2, 3, 4])),
        throwsArgumentError,
      );
    });

    test('overfits a tiny next-token task (loss -> near zero)', () {
      final lm = AFTLanguageModel(
        vocabSize: 6,
        embedDim: 16,
        numLayers: 1,
        maxLen: 5,
        seed: 52,
      );
      final x = Tensor.fromList([5], [0, 1, 2, 3, 4]);
      final y = Tensor.fromList([5], [1, 2, 3, 4, 5]);
      final opt = Adam(lm.parameters(), lr: 5e-3);
      final l0 = lm(x).crossEntropy(y).mean().toList()[0];
      for (int step = 0; step < 200; step++) {
        opt.zeroGrad();
        lm(x).crossEntropy(y).mean().backward();
        opt.step();
      }
      final l1 = lm(x).crossEntropy(y).mean().toList()[0];
      expect(
        l1,
        lessThan(0.1),
        reason: 'expected loss to drop near zero (l0=$l0 l1=$l1)',
      );
    });
  });

  group('AFT on GPU (parity)', () {
    test('aftFull GPU vs CPU parity (masked + unmasked)', () {
      const t = 6;
      const d = 4;
      for (final masked in [false, true]) {
        final qData = _rand(t * d, seed: 1);
        final kData = _rand(t * d, seed: 2);
        final vData = _rand(t * d, seed: 3);
        final wData = _rand(t * t, seed: 4);

        final qCpu = Tensor.fromList([t, d], qData, requiresGrad: true);
        final kCpu = Tensor.fromList([t, d], kData, requiresGrad: true);
        final vCpu = Tensor.fromList([t, d], vData, requiresGrad: true);
        final wCpu = Tensor.fromList([t, t], wData, requiresGrad: true);

        final qGpu = Tensor.fromList(
          [t, d],
          qData,
          device: Device.GPU,
          requiresGrad: true,
        );
        final kGpu = Tensor.fromList(
          [t, d],
          kData,
          device: Device.GPU,
          requiresGrad: true,
        );
        final vGpu = Tensor.fromList(
          [t, d],
          vData,
          device: Device.GPU,
          requiresGrad: true,
        );
        final wGpu = Tensor.fromList(
          [t, t],
          wData,
          device: Device.GPU,
          requiresGrad: true,
        );

        final outCpu = TensorAft.aftFull(qCpu, kCpu, vCpu, wCpu, masked: masked);
        final outGpu = TensorAft.aftFull(qGpu, kGpu, vGpu, wGpu, masked: masked);
        expectClose(outGpu.toList(), outCpu.toList(), tol: 1e-5);

        outCpu.sum().backward();
        outGpu.sum().backward();
        expectClose(qGpu.grad!.toList(), qCpu.grad!.toList(), tol: 1e-5);
        expectClose(kGpu.grad!.toList(), kCpu.grad!.toList(), tol: 1e-5);
        expectClose(vGpu.grad!.toList(), vCpu.grad!.toList(), tol: 1e-5);
        expectClose(wGpu.grad!.toList(), wCpu.grad!.toList(), tol: 1e-5);
      }
    });

    test('sliceTopLeft GPU vs CPU parity (forward + backward)', () {
      const bigR = 8;
      const bigC = 8;
      const r = 5;
      const c = 3;
      final data = _rand(bigR * bigC, seed: 7);
      final xCpu = Tensor.fromList([bigR, bigC], data, requiresGrad: true);
      final xGpu = Tensor.fromList(
        [bigR, bigC],
        data,
        device: Device.GPU,
        requiresGrad: true,
      );
      final yCpu = TensorAft.sliceTopLeft(xCpu, r, c);
      final yGpu = TensorAft.sliceTopLeft(xGpu, r, c);
      expectClose(yGpu.toList(), yCpu.toList(), tol: 1e-6);
      yCpu.sum().backward();
      yGpu.sum().backward();
      expectClose(xGpu.grad!.toList(), xCpu.grad!.toList(), tol: 1e-6);
    });

    test('AFTAttention on GPU: forward + backward populates all param grads', () {
      final layer = AFTAttention(
        4,
        maxSeqLen: 8,
        masked: true,
        device: Device.GPU,
        seed: 42,
      );
      final x = Tensor.fromList(
        [5, 4],
        _rand(5 * 4, seed: 11),
        device: Device.GPU,
        requiresGrad: true,
      );
      final y = layer(x);
      expect(y.shape, [5, 4]);
      y.sum().backward();
      for (final p in layer.parameters()) {
        expect(p.grad, isNotNull, reason: 'param without grad after backward');
      }
    });

    test('AFTLanguageModel on GPU trains a tiny sequence (loss decreases)', () {
      final vocab = 5;
      final seq = 6;
      final lm = AFTLanguageModel(
        vocabSize: vocab,
        embedDim: 8,
        numLayers: 1,
        maxLen: 8,
        ffnDim: 16,
        dropoutP: 0.0,
        device: Device.GPU,
        seed: 3,
      );
      final tokens = Tensor.fromList(
        [seq],
        List<double>.generate(seq, (i) => (i % vocab).toDouble()),
        device: Device.GPU,
      );
      final targets = Tensor.fromList(
        [seq],
        List<double>.generate(seq, (i) => ((i + 1) % vocab).toDouble()),
        device: Device.GPU,
      );
      final opt = SGD(lm.parameters(), lr: 0.5);
      final l0 = lm(tokens).crossEntropy(targets).mean().toList()[0];
      for (int step = 0; step < 40; step++) {
        opt.zeroGrad();
        lm(tokens).crossEntropy(targets).mean().backward();
        opt.step();
      }
      final l1 = lm(tokens).crossEntropy(targets).mean().toList()[0];
      expect(
        l1,
        lessThan(l0 * 0.5),
        reason: 'expected GPU AFT loss to drop (l0=$l0 l1=$l1)',
      );
    });
  });
}
