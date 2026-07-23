import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

// Naive CPU matmul for reference values.
List<double> cpuMatmul(List<double> a, List<double> b, int m, int k, int n) {
  final out = List<double>.filled(m * n, 0.0);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      double s = 0;
      for (int p = 0; p < k; p++) {
        s += a[i * k + p] * b[p * n + j];
      }
      out[i * n + j] = s;
    }
  }
  return out;
}

void expectClose(List<double> got, List<double> want, {double tol = 1e-4}) {
  expect(got.length, want.length);
  for (int i = 0; i < got.length; i++) {
    expect(
      (got[i] - want[i]).abs() < tol,
      isTrue,
      reason: 'index $i: got ${got[i]} want ${want[i]}',
    );
  }
}

void main() {
  test('calculate', () {
    expect(calculate(), 42);
  });

  // ---------------------------------------------------------------------
  // matmul: CPU path (default for small tensors), GPU path (explicit),
  // and mixed-device rejection.
  // ---------------------------------------------------------------------
  group('Tensor.matmul', () {
    test('2x3 @ 3x2 identity-like (CPU default)', () {
      final a = Tensor.fromList([2, 3], [1, 2, 3, 4, 5, 6]);
      final b = Tensor.fromList([3, 2], [7, 8, 9, 10, 11, 12]);
      expect(a.device, Device.CPU);
      final c = a.matmul(b);
      expect(c.device, Device.CPU);
      // Row 0: [1*7+2*9+3*11, 1*8+2*10+3*12]  = [58, 64]
      // Row 1: [4*7+5*9+6*11, 4*8+5*10+6*12]  = [139, 154]
      expect(c.shape, [2, 2]);
      expectClose(c.toList(), [58, 64, 139, 154]);
    });

    test('square 4x4 matches CPU reference (CPU path)', () {
      final aData = List<double>.generate(16, (i) => (i + 1).toDouble());
      final bData = List<double>.generate(16, (i) => (i * 0.5 - 3).toDouble());
      final a = Tensor.fromList([4, 4], aData);
      final b = Tensor.fromList([4, 4], bData);
      final c = a.matmul(b);
      expect(c.shape, [4, 4]);
      expectClose(c.toList(), cpuMatmul(aData, bData, 4, 4, 4));
    });

    test('non-tile-aligned 33x17 @ 17x9 on GPU', () {
      const m = 33, k = 17, n = 9;
      final aData = List<double>.generate(m * k, (i) => ((i % 7) - 3) * 0.25);
      final bData = List<double>.generate(k * n, (i) => ((i % 11) - 5) * 0.5);
      final a = Tensor.fromList([m, k], aData, device: Device.GPU);
      final b = Tensor.fromList([k, n], bData, device: Device.GPU);
      expect(a.device, Device.GPU);
      final c = a.matmul(b);
      expect(c.device, Device.GPU);
      expect(c.shape, [m, n]);
      expectClose(c.toList(), cpuMatmul(aData, bData, m, k, n), tol: 1e-3);
      a.dispose();
      b.dispose();
      c.dispose();
    });

    test('mixed devices throw', () {
      final a = Tensor.fromList([2, 2], [1, 2, 3, 4]);
      final b = Tensor.fromList([2, 2], [5, 6, 7, 8], device: Device.GPU);
      expect(() => a.matmul(b), throwsArgumentError);
      b.dispose();
    });
  });

  // ---------------------------------------------------------------------
  // Device transfer.
  // ---------------------------------------------------------------------
  group('Tensor.to', () {
    test('CPU -> GPU -> CPU round-trip preserves data', () {
      final data = List<double>.generate(12, (i) => i - 5.5);
      final a = Tensor.fromList([3, 4], data);
      expect(a.device, Device.CPU);
      final g = a.to(Device.GPU);
      expect(g.device, Device.GPU);
      final back = g.to(Device.CPU);
      expect(back.device, Device.CPU);
      expectClose(back.toList(), data);
      g.dispose();
    });

    test('to(sameDevice) returns identical tensor', () {
      final a = Tensor.fromList([2, 2], [1, 2, 3, 4]);
      expect(identical(a.to(Device.CPU), a), isTrue);
    });
  });

  // ---------------------------------------------------------------------
  // CPU-only elementwise + activations + rearrangement + reductions.
  // ---------------------------------------------------------------------
  group('CPU ops', () {
    test('add / sub / mul / div elementwise', () {
      final a = Tensor.fromList([2, 2], [1, 2, 3, 4]);
      final b = Tensor.fromList([2, 2], [5, 6, 7, 8]);
      expectClose((a + b).toList(), [6, 8, 10, 12]);
      expectClose((a - b).toList(), [-4, -4, -4, -4]);
      expectClose((a * b).toList(), [5, 12, 21, 32]);
      expectClose((a / b).toList(), [1 / 5, 2 / 6, 3 / 7, 4 / 8]);
    });

    test('scalar broadcast', () {
      final a = Tensor.fromList([2, 2], [1, 2, 3, 4]);
      expectClose((a + 10).toList(), [11, 12, 13, 14]);
      expectClose((a * 0.5).toList(), [0.5, 1.0, 1.5, 2.0]);
    });

    test('row broadcast add [1,N] into [M,N]', () {
      final a = Tensor.fromList([2, 3], [1, 2, 3, 4, 5, 6]);
      final bias = Tensor.fromList([1, 3], [10, 20, 30]);
      expectClose((a + bias).toList(), [11, 22, 33, 14, 25, 36]);
    });

    test('relu / sigmoid / tanh / abs', () {
      final t = Tensor.fromList([4], [-1.0, 0.0, 1.0, 2.0]);
      expectClose(t.relu().toList(), [0, 0, 1, 2]);
      expectClose(t.abs().toList(), [1, 0, 1, 2]);
      // Sigmoid & tanh: check a couple of anchor values.
      final s = t.sigmoid().toList();
      expect((s[1] - 0.5).abs() < 1e-6, isTrue);
      expect((s[0] - 1 / (1 + 2.718281828)).abs() < 1e-4, isTrue);
      final th = t.tanh().toList();
      expect(th[1].abs() < 1e-6, isTrue);
    });

    test('tanh is numerically stable for large |x| (no NaN/Inf)', () {
      // Regression: the old `(exp(2x) - 1) / (exp(2x) + 1)` form
      // overflowed to Inf/Inf = NaN for |x| >~ 44. GPT-2's GELU
      // feeds tanh values in the hundreds/thousands.
      final t = Tensor.fromList(
        [6],
        [-10000.0, -100.0, -20.0, 20.0, 100.0, 10000.0],
      );
      final out = t.tanh().toList();
      for (final v in out) {
        expect(v.isFinite, isTrue, reason: 'tanh produced $v');
      }
      // Values well outside [-1, 1] would indicate math error.
      for (final v in out) {
        expect(v.abs() <= 1.0 + 1e-6, isTrue);
      }
      // Sign is preserved.
      expect(out[0], lessThan(0));
      expect(out[5], greaterThan(0));
    });

    test('log / pow', () {
      final t = Tensor.fromList([3], [1.0, 2.0, 4.0]);
      expectClose(t.log().toList(), [
        0.0,
        0.6931471805599453,
        1.3862943611198906,
      ]);
      expectClose(t.pow(2).toList(), [1, 4, 16]);
    });

    test('transpose 2D', () {
      final t = Tensor.fromList([2, 3], [1, 2, 3, 4, 5, 6]);
      final tt = t.transpose();
      expect(tt.shape, [3, 2]);
      expectClose(tt.toList(), [1, 4, 2, 5, 3, 6]);
    });

    test('sum / mean', () {
      final t = Tensor.fromList([4], [1, 2, 3, 4]);
      expectClose(t.sum().toList(), [10]);
      expectClose(t.mean().toList(), [2.5]);
    });
  });

  // ---------------------------------------------------------------------
  // GPU parity: same ops as above, forced onto GPU, must match CPU
  // outputs within a slightly looser tolerance for reductions.
  // ---------------------------------------------------------------------
  group('GPU ops', () {
    test('add / sub / mul / div elementwise', () {
      final a = Tensor.fromList([2, 2], [1, 2, 3, 4], device: Device.GPU);
      final b = Tensor.fromList([2, 2], [5, 6, 7, 8], device: Device.GPU);
      final s = a + b;
      final d = a - b;
      final m = a * b;
      final q = a / b;
      expect(s.device, Device.GPU);
      expectClose(s.toList(), [6, 8, 10, 12]);
      expectClose(d.toList(), [-4, -4, -4, -4]);
      expectClose(m.toList(), [5, 12, 21, 32]);
      expectClose(q.toList(), [1 / 5, 2 / 6, 3 / 7, 4 / 8]);
      a.dispose();
      b.dispose();
      s.dispose();
      d.dispose();
      m.dispose();
      q.dispose();
    });

    test('scalar broadcast', () {
      final a = Tensor.fromList([2, 2], [1, 2, 3, 4], device: Device.GPU);
      final s = a + 10;
      final m = a * 0.5;
      expectClose(s.toList(), [11, 12, 13, 14]);
      expectClose(m.toList(), [0.5, 1.0, 1.5, 2.0]);
      a.dispose();
      s.dispose();
      m.dispose();
    });

    test('row broadcast add [1,N] into [M,N]', () {
      final a = Tensor.fromList([2, 3], [1, 2, 3, 4, 5, 6], device: Device.GPU);
      final bias = Tensor.fromList([1, 3], [10, 20, 30], device: Device.GPU);
      final r = a + bias;
      expectClose(r.toList(), [11, 22, 33, 14, 25, 36]);
      a.dispose();
      bias.dispose();
      r.dispose();
    });

    test('relu / sigmoid / tanh / abs', () {
      final t = Tensor.fromList([4], [-1.0, 0.0, 1.0, 2.0], device: Device.GPU);
      final r = t.relu();
      final s = t.sigmoid();
      final th = t.tanh();
      final ab = t.abs();
      expectClose(r.toList(), [0, 0, 1, 2]);
      expectClose(ab.toList(), [1, 0, 1, 2]);
      final sv = s.toList();
      expect((sv[1] - 0.5).abs() < 1e-6, isTrue);
      expect((sv[0] - 1 / (1 + 2.718281828)).abs() < 1e-4, isTrue);
      expect(th.toList()[1].abs() < 1e-6, isTrue);
      t.dispose();
      r.dispose();
      s.dispose();
      th.dispose();
      ab.dispose();
    });

    test('log / pow', () {
      final t = Tensor.fromList([3], [1.0, 2.0, 4.0], device: Device.GPU);
      final l = t.log();
      final p = t.pow(2);
      expectClose(l.toList(), [0.0, 0.6931471805599453, 1.3862943611198906]);
      expectClose(p.toList(), [1, 4, 16]);
      t.dispose();
      l.dispose();
      p.dispose();
    });

    test('transpose 2D (non-32-aligned)', () {
      final t = Tensor.fromList([2, 3], [1, 2, 3, 4, 5, 6], device: Device.GPU);
      final tt = t.transpose();
      expect(tt.shape, [3, 2]);
      expectClose(tt.toList(), [1, 4, 2, 5, 3, 6]);
      t.dispose();
      tt.dispose();
    });

    test('sum / mean', () {
      final t = Tensor.fromList([4], [1, 2, 3, 4], device: Device.GPU);
      final s = t.sum();
      final m = t.mean();
      expectClose(s.toList(), [10]);
      expectClose(m.toList(), [2.5]);
      t.dispose();
      s.dispose();
      m.dispose();
    });

    test('CPU/GPU parity on a chained expression', () {
      final data = List<double>.generate(64, (i) => (i - 32) * 0.1);
      final cpu = Tensor.fromList([8, 8], data);
      final gpu = Tensor.fromList([8, 8], data, device: Device.GPU);
      final cpuOut = ((cpu * 2.0) + 1.5).relu().transpose();
      final gpuOut = ((gpu * 2.0) + 1.5).relu().transpose();
      expectClose(gpuOut.toList(), cpuOut.toList(), tol: 1e-5);
      gpu.dispose();
      gpuOut.dispose();
    });
  });
}
