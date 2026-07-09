// GPU/CPU parity tests for `Tensor` and its ops.
//
// Modelled on `dart_cuda/test/core/tensor/gpu_tensor_test.dart` but
// adapted to `dart-pytorch`'s API (no `.dispose()`, no `Tensor.zeros`,
// no `.getRow`/`.slice`; instead uses `Tensor.fill`, `.toList()`,
// `Device.CPU`/`Device.GPU`, and `requiresGrad`).
//
// The tests come in two tiers:
//   * "small" — hand-picked values with expected outputs (correctness).
//   * "large / parity" — CPU vs. GPU parity at LayerNorm-, attention-
//     and FFN-scale shapes (D=384, T=192, hidden=1536). These stress
//     the same kernels that fire in the bigger shakespeare demos and
//     are where the loss=0 divergence at `embed >= 192` bug hides.
//
// Requires `native/lib/libmat_mul.so` and a CUDA-capable GPU.
//
// Run: `dart test test/gpu_tensor_test.dart`.

import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

const double _tol = 1e-3;
const double _tolBig = 1e-2; // wider tolerance for large-D parity

Matcher closeToList(List<double> expected, [double tol = _tol]) =>
    pairwiseCompare<num, double>(
      expected,
      (e, a) => (a.toDouble() - e).abs() < tol,
      'each element within $tol',
    );

/// Assert two length-matched lists are elementwise close.
void expectClose(List<double> a, List<double> b, {double tol = _tol}) {
  expect(a.length, b.length);
  for (int i = 0; i < a.length; i++) {
    expect(
      (a[i] - b[i]).abs() < tol,
      isTrue,
      reason:
          'index $i differs by ${(a[i] - b[i]).abs()} (a=${a[i]} b=${b[i]})',
    );
  }
}

/// Deterministic pseudo-random values in `(-0.5, 0.5)` for parity data.
List<double> _fake(int n, int seed) {
  final r = math.Random(seed);
  return List<double>.generate(n, (_) => r.nextDouble() - 0.5);
}

void main() {
  // ---------------------------------------------------------------
  // FFI bridge — CPU and GPU
  // ---------------------------------------------------------------
  group('Tensor — FFI bridge', () {
    test('fromList round-trips through toList (CPU)', () {
      final values = [1.0, 2.0, 3.0, 4.0];
      final x = Tensor.fromList([2, 2], values);
      expect(x.toList(), closeToList(values));
    });

    test('fromList round-trips through toList (GPU)', () {
      final values = [1.0, 2.0, 3.0, 4.0];
      final x = Tensor.fromList([2, 2], values, device: Device.GPU);
      expect(x.toList(), closeToList(values));
    });

    test('shape and length are populated', () {
      final x = Tensor.fromList([2, 3], List.filled(6, 0.0));
      expect(x.shape, equals([2, 3]));
      expect(x.length, equals(6));
    });

    test('to(GPU) then to(CPU) preserves values', () {
      final values = List<double>.generate(24, (i) => i.toDouble());
      final cpu = Tensor.fromList([2, 3, 4], values);
      final gpu = cpu.to(Device.GPU);
      final back = gpu.to(Device.CPU);
      expect(back.toList(), closeToList(values));
    });

    test('clone preserves values and is independent', () {
      final x = Tensor.fromList([1, 3], [1.0, 2.0, 3.0], device: Device.GPU);
      final y = x.clone();
      expect(y.toList(), closeToList([1.0, 2.0, 3.0]));
      // Cloned tensors have independent grad state.
      expect(y.grad, isNull);
    });
  });

  // ---------------------------------------------------------------
  // Factories
  // ---------------------------------------------------------------
  group('Tensor — factories', () {
    test('Tensor.fill yields the given constant (CPU)', () {
      final x = Tensor.fill([2, 2], 7.5);
      expect(x.toList(), everyElement(closeTo(7.5, _tol)));
    });

    test('Tensor.fill yields the given constant (GPU)', () {
      final x = Tensor.fill([4, 4], -1.25, device: Device.GPU);
      expect(x.toList(), everyElement(closeTo(-1.25, _tol)));
    });
  });

  // ---------------------------------------------------------------
  // Element-wise ops — small correctness
  // ---------------------------------------------------------------
  group('Tensor — element-wise ops (small)', () {
    for (final device in [Device.CPU, Device.GPU]) {
      final tag = device == Device.CPU ? 'CPU' : 'GPU';

      test('addition ($tag)', () {
        final a = Tensor.fromList([1, 3], [1, 2, 3], device: device);
        final b = Tensor.fromList([1, 3], [10, 20, 30], device: device);
        expect((a + b).toList(), closeToList([11, 22, 33]));
      });

      test('subtraction ($tag)', () {
        final a = Tensor.fromList([1, 3], [5, 5, 5], device: device);
        final b = Tensor.fromList([1, 3], [1, 2, 3], device: device);
        expect((a - b).toList(), closeToList([4, 3, 2]));
      });

      test('hadamard product ($tag)', () {
        final a = Tensor.fromList([1, 3], [2, 3, 4], device: device);
        final b = Tensor.fromList([1, 3], [5, 6, 7], device: device);
        expect((a * b).toList(), closeToList([10, 18, 28]));
      });

      test('elementwise division ($tag)', () {
        final a = Tensor.fromList([1, 3], [10, 20, 30], device: device);
        final b = Tensor.fromList([1, 3], [2, 4, 5], device: device);
        expect((a / b).toList(), closeToList([5, 5, 6]));
      });

      test('scalar add / mul / div ($tag)', () {
        final a = Tensor.fromList([1, 3], [1, 2, 3], device: device);
        expect((a + 10.0).toList(), closeToList([11, 12, 13]));
        expect((a * 4.0).toList(), closeToList([4, 8, 12]));
        expect((a / 2.0).toList(), closeToList([0.5, 1.0, 1.5]));
      });
    }
  });

  // ---------------------------------------------------------------
  // Element-wise ops — CPU vs GPU parity at big shapes
  // ---------------------------------------------------------------
  group('Tensor — element-wise parity (large)', () {
    test('add [192, 384] parity', () {
      final aData = _fake(192 * 384, 1);
      final bData = _fake(192 * 384, 2);
      final cpu = Tensor.fromList([192, 384], aData) +
          Tensor.fromList([192, 384], bData);
      final gpu = Tensor.fromList([192, 384], aData, device: Device.GPU) +
          Tensor.fromList([192, 384], bData, device: Device.GPU);
      expectClose(gpu.toList(), cpu.toList());
    });

    test('mul [192, 384] parity', () {
      final aData = _fake(192 * 384, 3);
      final bData = _fake(192 * 384, 4);
      final cpu = Tensor.fromList([192, 384], aData) *
          Tensor.fromList([192, 384], bData);
      final gpu = Tensor.fromList([192, 384], aData, device: Device.GPU) *
          Tensor.fromList([192, 384], bData, device: Device.GPU);
      expectClose(gpu.toList(), cpu.toList());
    });
  });

  // ---------------------------------------------------------------
  // Reductions and unary ops
  // ---------------------------------------------------------------
  group('Tensor — reductions and unary ops', () {
    test('sum reduces to a scalar (CPU)', () {
      final x = Tensor.fromList([2, 2], [1, 2, 3, 4]);
      final s = x.sum();
      expect(s.toList().first, closeTo(10.0, _tol));
    });

    test('sum reduces to a scalar (GPU)', () {
      final x = Tensor.fromList([2, 2], [1, 2, 3, 4], device: Device.GPU);
      final s = x.sum();
      expect(s.toList().first, closeTo(10.0, _tol));
    });

    test('mean matches sum / N (GPU, large)', () {
      final data = _fake(192 * 384, 5);
      final expected = data.reduce((a, b) => a + b) / data.length;
      final x = Tensor.fromList([192, 384], data, device: Device.GPU);
      expect(x.mean().toList().first, closeTo(expected, _tolBig));
    });

    test('relu zeros negatives (CPU vs GPU parity)', () {
      final data = List<double>.generate(200, (i) => (i - 100) * 0.1);
      final cpu = Tensor.fromList([200], data).relu();
      final gpu =
          Tensor.fromList([200], data, device: Device.GPU).relu();
      expectClose(gpu.toList(), cpu.toList());
    });

    test('sigmoid matches numerical reference (GPU)', () {
      final values = [-2.0, -0.5, 0.0, 0.5, 2.0];
      final x = Tensor.fromList([1, values.length], values,
          device: Device.GPU);
      final expected =
          values.map((v) => 1.0 / (1.0 + math.exp(-v))).toList();
      expectClose(x.sigmoid().toList(), expected);
    });

    test('tanh matches numerical reference (GPU)', () {
      final values = [-1.0, 0.0, 1.0];
      final x = Tensor.fromList([1, values.length], values,
          device: Device.GPU);
      final expected = values.map((v) {
        final ep = math.exp(v);
        final en = math.exp(-v);
        return (ep - en) / (ep + en);
      }).toList();
      expectClose(x.tanh().toList(), expected);
    });

    test('abs on GPU matches |x|', () {
      final x = Tensor.fromList([1, 4], [-1.5, 0.0, 2.5, -4.0],
          device: Device.GPU);
      expect(x.abs().toList(), closeToList([1.5, 0.0, 2.5, 4.0]));
    });

    test('pow(2) squares each element (CPU vs GPU parity)', () {
      final data = _fake(500, 6);
      final cpu = Tensor.fromList([500], data).pow(2.0);
      final gpu =
          Tensor.fromList([500], data, device: Device.GPU).pow(2.0);
      expectClose(gpu.toList(), cpu.toList());
    });

    test('log on positive values (GPU)', () {
      final values = [1.0, math.e, math.e * math.e];
      final x = Tensor.fromList([1, values.length], values,
          device: Device.GPU);
      expectClose(x.log().toList(), [0.0, 1.0, 2.0]);
    });
  });

  // ---------------------------------------------------------------
  // Matrix multiplication
  // ---------------------------------------------------------------
  group('Tensor — matrix multiplication', () {
    test('2x3 @ 3x2 = 2x2 (CPU)', () {
      final a = Tensor.fromList([2, 3], [1, 2, 3, 4, 5, 6]);
      final b = Tensor.fromList([3, 2], [7, 8, 9, 10, 11, 12]);
      final c = a.matmul(b);
      expect(c.shape, equals([2, 2]));
      expect(c.toList(), closeToList([58, 64, 139, 154]));
    });

    test('2x3 @ 3x2 = 2x2 (GPU)', () {
      final a = Tensor.fromList([2, 3], [1, 2, 3, 4, 5, 6],
          device: Device.GPU);
      final b = Tensor.fromList([3, 2], [7, 8, 9, 10, 11, 12],
          device: Device.GPU);
      final c = a.matmul(b);
      expect(c.shape, equals([2, 2]));
      expect(c.toList(), closeToList([58, 64, 139, 154]));
    });

    test('identity matmul preserves input (GPU)', () {
      final a = Tensor.fromList([2, 2], [1, 2, 3, 4], device: Device.GPU);
      final i = Tensor.fromList([2, 2], [1, 0, 0, 1], device: Device.GPU);
      expect(a.matmul(i).toList(), closeToList([1, 2, 3, 4]));
    });

    // Attention-scale QK^T: [T, D] @ [D, T] with T=192, D=64.
    test('attention-scale matmul parity: [192,64] @ [64,192]', () {
      final aData = _fake(192 * 64, 10);
      final bData = _fake(64 * 192, 11);
      final cpu = Tensor.fromList([192, 64], aData)
          .matmul(Tensor.fromList([64, 192], bData));
      final gpu = Tensor.fromList([192, 64], aData, device: Device.GPU)
          .matmul(Tensor.fromList([64, 192], bData, device: Device.GPU));
      expectClose(gpu.toList(), cpu.toList(), tol: _tolBig);
    });

    // FFN-scale: [T, D] @ [D, 4D] with T=192, D=384, hidden=1536.
    // This is the matmul that appears in the FFN hidden layer of the
    // "big" GPT demo and is the biggest single matmul the model runs.
    test('FFN-scale matmul parity: [192,384] @ [384,1536]', () {
      final aData = _fake(192 * 384, 20);
      final bData = _fake(384 * 1536, 21);
      final cpu = Tensor.fromList([192, 384], aData)
          .matmul(Tensor.fromList([384, 1536], bData));
      final gpu = Tensor.fromList([192, 384], aData, device: Device.GPU)
          .matmul(Tensor.fromList([384, 1536], bData, device: Device.GPU));
      expectClose(gpu.toList(), cpu.toList(), tol: _tolBig);
    });
  });

  // ---------------------------------------------------------------
  // Slicing / views
  // ---------------------------------------------------------------
  group('Tensor — reshape and transpose', () {
    test('reshape preserves data (GPU)', () {
      final x = Tensor.fromList([2, 3], [1, 2, 3, 4, 5, 6],
          device: Device.GPU);
      final v = x.reshape([3, 2]);
      expect(v.shape, equals([3, 2]));
      expect(v.toList(), closeToList([1, 2, 3, 4, 5, 6]));
    });

    test('transpose swaps rows and columns (CPU vs GPU parity)', () {
      final data = _fake(64 * 192, 30);
      final cpu = Tensor.fromList([64, 192], data).transpose();
      final gpu =
          Tensor.fromList([64, 192], data, device: Device.GPU).transpose();
      expect(gpu.shape, equals([192, 64]));
      expectClose(gpu.toList(), cpu.toList());
    });
  });

  // ---------------------------------------------------------------
  // Broadcast (scalar and row)
  // ---------------------------------------------------------------
  group('Tensor — broadcast', () {
    test('add: [N, M] + [1, 1] scalar broadcast forward+backward (GPU)', () {
      final a = Tensor.fromList([2, 2], [1, 2, 3, 4],
          device: Device.GPU, requiresGrad: true);
      final b =
          Tensor.fromList([1, 1], [10], device: Device.GPU, requiresGrad: true);
      final c = a + b;
      c.sum().backward();
      expect(c.toList(), closeToList([11, 12, 13, 14]));
      expect(a.grad!.toList(), closeToList([1, 1, 1, 1]));
      expect(b.grad!.toList(), closeToList([4]));
    });

    test('mul: [N, M] * [1, 1] scalar broadcast (GPU)', () {
      final a = Tensor.fromList([2, 2], [1, 2, 3, 4],
          device: Device.GPU, requiresGrad: true);
      final b =
          Tensor.fromList([1, 1], [3], device: Device.GPU, requiresGrad: true);
      final c = a * b;
      c.sum().backward();
      expect(c.toList(), closeToList([3, 6, 9, 12]));
      expect(a.grad!.toList(), closeToList([3, 3, 3, 3]));
      expect(b.grad!.toList(), closeToList([10]));
    });

    test('add: [B, N] + [1, N] row bias forward+backward (GPU)', () {
      final a = Tensor.fromList(
        [3, 4],
        [
          1, 2, 3, 4, //
          5, 6, 7, 8, //
          9, 10, 11, 12,
        ],
        device: Device.GPU,
        requiresGrad: true,
      );
      final b = Tensor.fromList([1, 4], [10, 20, 30, 40],
          device: Device.GPU, requiresGrad: true);
      final c = a + b;
      c.sum().backward();
      expect(
        c.toList(),
        closeToList([
          11, 22, 33, 44, //
          15, 26, 37, 48, //
          19, 30, 41, 52,
        ]),
      );
      expect(a.grad!.toList(), closeToList(List<double>.filled(12, 1)));
      expect(b.grad!.toList(), closeToList([3, 3, 3, 3]));
    });
  });

  // ---------------------------------------------------------------
  // Autograd basics
  // ---------------------------------------------------------------
  group('Tensor — autograd basics', () {
    test('sum.backward yields ones gradient (GPU)', () {
      final x = Tensor.fromList([2, 2], [1, 2, 3, 4],
          device: Device.GPU, requiresGrad: true);
      x.sum().backward();
      expect(x.grad!.toList(), closeToList([1.0, 1.0, 1.0, 1.0]));
    });

    test('mean.backward yields 1/N gradient (GPU)', () {
      final x = Tensor.fromList([1, 4], [1, 2, 3, 4],
          device: Device.GPU, requiresGrad: true);
      x.mean().backward();
      expect(x.grad!.toList(), closeToList([0.25, 0.25, 0.25, 0.25]));
    });

    test('zeroGrad clears gradients (GPU)', () {
      final x = Tensor.fromList([1, 2], [3.0, 4.0],
          device: Device.GPU, requiresGrad: true);
      x.sum().backward();
      expect(x.grad!.toList(), closeToList([1.0, 1.0]));
      // dart-pytorch's `zeroGrad` sets `_grad = null` (in contrast to
      // dart_cuda which overwrites with zeros in place).
      x.zeroGrad();
      expect(x.grad, isNull);
    });
  });

  // ---------------------------------------------------------------
  // Autograd parity at LARGE dims — the loss=0 bug hides here.
  //
  // These tests build the same op chains used inside the big
  // shakespeare demos (LayerNorm-scale reductions, FFN-scale
  // matmuls, attention-scale softmax) and compare CPU vs GPU
  // gradients directly. A regression here == a regression in the
  // GPT/AFT/MoE big demos.
  // ---------------------------------------------------------------
  group('Tensor — autograd parity (large)', () {
    test('matmul backward parity [192,384] @ [384,1536]', () {
      final aData = _fake(192 * 384, 40);
      final bData = _fake(384 * 1536, 41);

      final aCpu = Tensor.fromList([192, 384], aData, requiresGrad: true);
      final bCpu = Tensor.fromList([384, 1536], bData, requiresGrad: true);
      aCpu.matmul(bCpu).sum().backward();

      final aGpu = Tensor.fromList([192, 384], aData,
          device: Device.GPU, requiresGrad: true);
      final bGpu = Tensor.fromList([384, 1536], bData,
          device: Device.GPU, requiresGrad: true);
      aGpu.matmul(bGpu).sum().backward();

      expectClose(aGpu.grad!.toList(), aCpu.grad!.toList(), tol: _tolBig);
      expectClose(bGpu.grad!.toList(), bCpu.grad!.toList(), tol: _tolBig);
    });

    test('elementwise chain backward parity: (x*x + x).sum() at [192,384]',
        () {
      final data = _fake(192 * 384, 50);
      final xCpu = Tensor.fromList([192, 384], data, requiresGrad: true);
      (xCpu * xCpu + xCpu).sum().backward();

      final xGpu = Tensor.fromList([192, 384], data,
          device: Device.GPU, requiresGrad: true);
      (xGpu * xGpu + xGpu).sum().backward();

      expectClose(xGpu.grad!.toList(), xCpu.grad!.toList(), tol: _tolBig);
    });

    test('sigmoid chain backward parity at [192,384]', () {
      final data = _fake(192 * 384, 60);
      final xCpu = Tensor.fromList([192, 384], data, requiresGrad: true);
      (xCpu.sigmoid() * xCpu).sum().backward();

      final xGpu = Tensor.fromList([192, 384], data,
          device: Device.GPU, requiresGrad: true);
      (xGpu.sigmoid() * xGpu).sum().backward();

      expectClose(xGpu.grad!.toList(), xCpu.grad!.toList(), tol: _tolBig);
    });

    test('relu backward parity at [192,384]', () {
      final data = _fake(192 * 384, 70);
      final xCpu = Tensor.fromList([192, 384], data, requiresGrad: true);
      xCpu.relu().sum().backward();

      final xGpu = Tensor.fromList([192, 384], data,
          device: Device.GPU, requiresGrad: true);
      xGpu.relu().sum().backward();

      expectClose(xGpu.grad!.toList(), xCpu.grad!.toList(), tol: _tolBig);
    });
  });

  // ---------------------------------------------------------------
  // Repeated forward+backward — the loss=0 bug fires between
  // steps 3 and 10, not on the first backward pass. These tests
  // catch cross-step state corruption (grad accumulator leaks,
  // handle reuse, kernel scratch overwrites).
  // ---------------------------------------------------------------
  group('Tensor — repeated forward/backward parity', () {
    test('10 matmul + backward iterations at FFN scale keep parity', () {
      final rng = math.Random(80);
      // Reuse the same input across iterations (mimics an
      // optimizer loop where params persist).
      final aData = _fake(64 * 384, 81);
      final bData = _fake(384 * 1536, 82);

      final aCpu = Tensor.fromList([64, 384], aData, requiresGrad: true);
      final bCpu = Tensor.fromList([384, 1536], bData, requiresGrad: true);
      final aGpu = Tensor.fromList([64, 384], aData,
          device: Device.GPU, requiresGrad: true);
      final bGpu = Tensor.fromList([384, 1536], bData,
          device: Device.GPU, requiresGrad: true);

      for (int step = 0; step < 10; step++) {
        aCpu.zeroGrad();
        bCpu.zeroGrad();
        aGpu.zeroGrad();
        bGpu.zeroGrad();

        // Multiply by an iteration-dependent scalar so the graph
        // isn't identical across steps (mimics fresh loss values).
        final scale = 0.5 + rng.nextDouble();
        (aCpu.matmul(bCpu) * scale).sum().backward();
        (aGpu.matmul(bGpu) * scale).sum().backward();

        expectClose(
          aGpu.grad!.toList(),
          aCpu.grad!.toList(),
          tol: _tolBig,
        );
        expectClose(
          bGpu.grad!.toList(),
          bCpu.grad!.toList(),
          tol: _tolBig,
        );
      }
    });
  });
}
