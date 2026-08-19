@Timeout(Duration(minutes: 2))
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

Tensor _rand(List<int> shape, {int seed = 0, Device device = Device.CPU}) {
  final rng = math.Random(seed);
  var n = 1;
  for (final d in shape) {
    n *= d;
  }
  final v = Float32List(n);
  for (int i = 0; i < n; i++) {
    v[i] = rng.nextDouble() - 0.5;
  }
  return Tensor.fromFloat32List(shape, v, device: device);
}

bool _gpuAvailable() {
  try {
    Tensor.fromList([1], [1.0], device: Device.GPU).toList();
    return true;
  } catch (_) {
    return false;
  }
}

void main() {
  group('ConvTranspose1d output-length formula', () {
    final cases = <(int, int, int, int, int, int)>[
      // (L, kernel, stride, padding, outputPadding, expected)
      (4, 2, 1, 0, 0, 5), // (4-1)*1 + 2 = 5
      (4, 4, 2, 1, 0, 8), // (4-1)*2 - 2 + 4 = 8
      (10, 16, 8, 4, 0, 80), // HiFi-GAN stage 1: (10-1)*8 - 8 + 16 = 80
      (2, 2, 2, 0, 1, 5), // outputPadding widens
    ];
    for (final (l, k, s, p, op, expected) in cases) {
      test('L=$l k=$k s=$s p=$p op=$op -> $expected', () {
        final ct = ConvTranspose1d(
          1,
          1,
          kernelSize: k,
          stride: s,
          padding: p,
          outputPadding: op,
        );
        expect(ct.outputLength(l), expected);
      });
    }
  });

  group('ConvTranspose1d hand-computed', () {
    test('1x1x2, k=2, s=1, p=0, no bias', () {
      // Input X = [a, b], kernel W = [c, d].
      // Output at ol:
      //   ol=0: X[0]*W[0] = a*c
      //   ol=1: X[0]*W[1] + X[1]*W[0] = a*d + b*c
      //   ol=2: X[1]*W[1] = b*d
      final ct = ConvTranspose1d(1, 1, kernelSize: 2, bias: false);
      ct.weight.assign(Tensor.fromList([1, 1, 2], [2.0, 3.0]));
      final x = Tensor.fromList([1, 1, 2], [4.0, 5.0]);
      final y = ct(x);
      expect(y.shape, equals([1, 1, 3]));
      final got = y.toList();
      expect(got, equals([8.0, 12.0 + 10.0, 15.0]));
    });

    test('1x1x2, k=2, s=2, no bias — no overlap', () {
      final ct = ConvTranspose1d(1, 1, kernelSize: 2, stride: 2, bias: false);
      ct.weight.assign(Tensor.fromList([1, 1, 2], [2.0, 3.0]));
      final x = Tensor.fromList([1, 1, 2], [4.0, 5.0]);
      final y = ct(x);
      expect(y.shape, equals([1, 1, 4]));
      // Each input pixel maps to its own [k*W[0], k*W[1]] block.
      expect(y.toList(), equals([8.0, 12.0, 10.0, 15.0]));
    });

    test('bias per channel', () {
      final ct = ConvTranspose1d(1, 2, kernelSize: 1);
      ct.weight.assign(Tensor.fromList([1, 2, 1], [1.0, 1.0]));
      ct.bias!.assign(Tensor.fromList([2], [10.0, 20.0]));
      final x = Tensor.fromList([1, 1, 2], [1.0, 2.0]);
      final y = ct(x).toList();
      expect(y, equals([11.0, 12.0, 21.0, 22.0]));
    });

    test('rejects outputPadding >= stride', () {
      expect(
        () => ConvTranspose1d(1, 1, kernelSize: 2, stride: 2, outputPadding: 2),
        throwsArgumentError,
      );
    });

    test('rejects non-3D input', () {
      final ct = ConvTranspose1d(1, 1, kernelSize: 2);
      expect(() => ct(Tensor.fromList([2], [1.0, 2.0])), throwsArgumentError);
    });
  });

  group('ConvTranspose1d on GPU', () {
    final gpuOk = _gpuAvailable();
    if (!gpuOk) {
      test(
        'GPU unavailable → skipped',
        () {},
        skip: 'CUDA / native/lib/libmat_mul.so not usable in this env',
      );
      return;
    }

    test('CPU vs GPU parity, stride=1', () {
      final cpu = ConvTranspose1d(4, 6, kernelSize: 3);
      final gpu = ConvTranspose1d(4, 6, kernelSize: 3, device: Device.GPU);
      gpu.weight.assign(
        Tensor.fromList(
          cpu.weight.shape,
          cpu.weight.toList(),
          device: Device.GPU,
        ),
      );
      gpu.bias!.assign(
        Tensor.fromList(
          cpu.bias!.shape,
          cpu.bias!.toList(),
          device: Device.GPU,
        ),
      );
      final xCpu = _rand([2, 4, 8], seed: 42);
      final xGpu = _rand([2, 4, 8], seed: 42, device: Device.GPU);
      final yCpu = cpu(xCpu).toList();
      final yGpu = gpu(xGpu).toList();
      for (int i = 0; i < yCpu.length; i++) {
        expect(
          (yCpu[i] - yGpu[i]).abs() < 1e-4,
          isTrue,
          reason: 'i=$i cpu=${yCpu[i]} gpu=${yGpu[i]}',
        );
      }
    });

    test('CPU vs GPU parity, stride=8 (HiFi-GAN-like)', () {
      final cpu = ConvTranspose1d(16, 8, kernelSize: 16, stride: 8, padding: 4);
      final gpu = ConvTranspose1d(
        16,
        8,
        kernelSize: 16,
        stride: 8,
        padding: 4,
        device: Device.GPU,
      );
      gpu.weight.assign(
        Tensor.fromList(
          cpu.weight.shape,
          cpu.weight.toList(),
          device: Device.GPU,
        ),
      );
      gpu.bias!.assign(
        Tensor.fromList(
          cpu.bias!.shape,
          cpu.bias!.toList(),
          device: Device.GPU,
        ),
      );
      final xCpu = _rand([1, 16, 20], seed: 7);
      final xGpu = _rand([1, 16, 20], seed: 7, device: Device.GPU);
      final yCpu = cpu(xCpu).toList();
      final yGpu = gpu(xGpu).toList();
      for (int i = 0; i < yCpu.length; i++) {
        expect(
          (yCpu[i] - yGpu[i]).abs() < 1e-4,
          isTrue,
          reason: 'i=$i cpu=${yCpu[i]} gpu=${yGpu[i]}',
        );
      }
    });
  });
}
