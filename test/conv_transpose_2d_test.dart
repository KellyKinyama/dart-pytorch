@Timeout(Duration(minutes: 3))
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

/// Assign into a `ConvTranspose2d`'s weight/bias for deterministic
/// hand-computed reference tests.
void _setWeight(ConvTranspose2d ct, List<double> vals) {
  ct.weight.assign(Tensor.fromList(ct.weight.shape, vals));
}

void _setBias(ConvTranspose2d ct, List<double> vals) {
  ct.bias!.assign(Tensor.fromList([ct.bias!.shape[0]], vals));
}

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
    final probe = Tensor.fromList([2], [1.0, 2.0], device: Device.GPU);
    probe.toList();
    return true;
  } catch (_) {
    return false;
  }
}

void main() {
  group('ConvTranspose2d output shape formula', () {
    // Hout = (H - 1) * s - 2 * p + k + op
    final cases = <(int, int, int, int, int, int)>[
      // (H, kernel, stride, padding, outputPadding, expected)
      (2, 2, 1, 0, 0, 3), // classic 2->3
      (2, 2, 2, 0, 0, 4), // stride-2 upsample
      (4, 3, 1, 1, 0, 4), // same-size (padding=1, k=3)
      (2, 4, 2, 1, 0, 4), // 2->4, k=4, s=2, p=1
      (2, 2, 2, 0, 1, 5), // outputPadding widens by 1
      (3, 3, 2, 1, 1, 6), // (3-1)*2 - 2 + 3 + 1 = 6
    ];
    for (final (h, k, s, p, op, expected) in cases) {
      test('H=$h k=$k s=$s p=$p op=$op -> $expected', () {
        final ct = ConvTranspose2d(
          1,
          1,
          kernel: k,
          stride: s,
          padding: p,
          outputPadding: op,
        );
        expect(ct.outputHeight(h), expected);
        expect(ct.outputWidth(h), expected);
      });
    }
  });

  group('ConvTranspose2d hand-computed cases', () {
    test('1x1x2x2, kernel=2, stride=1, padding=0, no bias', () {
      final ct = ConvTranspose2d(1, 1, kernel: 2, bias: false);
      _setWeight(ct, [1.0, 2.0, 3.0, 4.0]);
      final x = Tensor.fromList([1, 1, 2, 2], [1.0, 2.0, 3.0, 4.0]);
      final y = ct(x);
      expect(y.shape, equals([1, 1, 3, 3]));
      final expected = <double>[
        1, 4, 4, //
        6, 20, 16, //
        9, 24, 16, //
      ];
      final got = y.toList();
      for (int i = 0; i < 9; i++) {
        expect(
          (got[i] - expected[i]).abs() < 1e-5,
          isTrue,
          reason: 'pos $i: got=${got[i]}, want=${expected[i]}',
        );
      }
    });

    test('1x1x2x2, kernel=2, stride=2, padding=0, no bias', () {
      final ct = ConvTranspose2d(1, 1, kernel: 2, stride: 2, bias: false);
      _setWeight(ct, [1.0, 2.0, 3.0, 4.0]);
      final x = Tensor.fromList([1, 1, 2, 2], [1.0, 2.0, 3.0, 4.0]);
      final y = ct(x);
      expect(y.shape, equals([1, 1, 4, 4]));
      final expected = <double>[
        1, 2, 2, 4, //
        3, 4, 6, 8, //
        3, 6, 4, 8, //
        9, 12, 12, 16, //
      ];
      final got = y.toList();
      for (int i = 0; i < 16; i++) {
        expect(
          (got[i] - expected[i]).abs() < 1e-5,
          isTrue,
          reason: 'pos $i: got=${got[i]}, want=${expected[i]}',
        );
      }
    });

    test('bias adds per channel', () {
      final ct = ConvTranspose2d(1, 2, kernel: 1);
      // Weight [Cin=1, Cout=2, 1, 1] = identity-like.
      _setWeight(ct, [1.0, 1.0]);
      _setBias(ct, [10.0, 20.0]);
      final x = Tensor.fromList([1, 1, 2, 2], [1.0, 2.0, 3.0, 4.0]);
      final y = ct(x);
      expect(y.shape, equals([1, 2, 2, 2]));
      final got = y.toList();
      // Channel 0: [1+10, 2+10, 3+10, 4+10] = [11, 12, 13, 14]
      // Channel 1: [1+20, 2+20, 3+20, 4+20] = [21, 22, 23, 24]
      expect(got.sublist(0, 4), equals([11.0, 12.0, 13.0, 14.0]));
      expect(got.sublist(4, 8), equals([21.0, 22.0, 23.0, 24.0]));
    });

    test('padding=1 with kernel=3, stride=1 preserves spatial size', () {
      final ct = ConvTranspose2d(
        1,
        1,
        kernel: 3,
        stride: 1,
        padding: 1,
        bias: false,
      );
      final x = _rand([1, 1, 5, 5], seed: 7);
      final y = ct(x);
      expect(y.shape, equals([1, 1, 5, 5]));
    });

    test('outputPadding widens the destination by op rows/cols', () {
      final ct = ConvTranspose2d(
        1,
        1,
        kernel: 2,
        stride: 2,
        outputPadding: 1,
        bias: false,
      );
      final x = _rand([1, 1, 2, 2], seed: 3);
      final y = ct(x);
      // (2-1)*2 - 0 + 2 + 1 = 5
      expect(y.shape, equals([1, 1, 5, 5]));
    });

    test('rejects outputPadding >= stride (PyTorch parity)', () {
      expect(
        () => ConvTranspose2d(1, 1, kernel: 3, stride: 2, outputPadding: 2),
        throwsArgumentError,
      );
    });

    test('rejects non-NCHW input', () {
      final ct = ConvTranspose2d(1, 1, kernel: 2);
      expect(
        () => ct(Tensor.fromList([4], [1.0, 2.0, 3.0, 4.0])),
        throwsArgumentError,
      );
    });

    test('multi-channel: [1,2,2,2] -> [1,3,3,3]', () {
      final ct = ConvTranspose2d(2, 3, kernel: 2, bias: false);
      _setWeight(ct, List<double>.filled(2 * 3 * 2 * 2, 1.0));
      // Uniform weights of 1 → every output pixel is the sum over
      // in-channels of contributing inputs. With X = ones([1,2,2,2]),
      // every out pixel receives contributions from both channels.
      final x = Tensor.fromList([1, 2, 2, 2], List<double>.filled(8, 1.0));
      final y = ct(x);
      expect(y.shape, equals([1, 3, 3, 3]));
      // Each output channel sees the same overlap pattern; with W=1 and
      // X=1 the classical 2x2 -> 3x3 pattern of "corners=1, edges=2,
      // centre=4" applies per input-channel and both channels sum.
      // So per out-channel: [[2,4,2],[4,8,4],[2,4,2]].
      final expected = <double>[
        2, 4, 2, 4, 8, 4, 2, 4, 2, //
        2, 4, 2, 4, 8, 4, 2, 4, 2, //
        2, 4, 2, 4, 8, 4, 2, 4, 2, //
      ];
      expect(y.toList(), equals(expected));
    });
  });

  group('ConvTranspose2d on GPU', () {
    final gpuOk = _gpuAvailable();
    if (!gpuOk) {
      test(
        'GPU unavailable → skipped',
        () {},
        skip: 'CUDA / native/lib/libmat_mul.so not usable in this env',
      );
      return;
    }

    test('CPU vs GPU forward parity, stride=1', () {
      final cpu = ConvTranspose2d(4, 8, kernel: 3);
      final gpu = ConvTranspose2d(4, 8, kernel: 3, device: Device.GPU);
      // Sync weights + biases.
      final wVals = cpu.weight.toList();
      final bVals = cpu.bias!.toList();
      gpu.weight.assign(
        Tensor.fromList(cpu.weight.shape, wVals, device: Device.GPU),
      );
      gpu.bias!.assign(
        Tensor.fromList(cpu.bias!.shape, bVals, device: Device.GPU),
      );

      final xCpu = _rand([2, 4, 5, 5], seed: 42);
      final xGpu = _rand([2, 4, 5, 5], seed: 42, device: Device.GPU);
      final yCpu = cpu(xCpu).toList();
      final yGpu = gpu(xGpu).toList();
      expect(yCpu.length, yGpu.length);
      for (int i = 0; i < yCpu.length; i++) {
        expect(
          (yCpu[i] - yGpu[i]).abs() < 1e-4,
          isTrue,
          reason: 'pos $i cpu=${yCpu[i]} gpu=${yGpu[i]}',
        );
      }
    });

    test('CPU vs GPU forward parity, stride=2', () {
      final cpu = ConvTranspose2d(4, 8, kernel: 3, stride: 2, padding: 1);
      final gpu = ConvTranspose2d(
        4,
        8,
        kernel: 3,
        stride: 2,
        padding: 1,
        device: Device.GPU,
      );
      final wVals = cpu.weight.toList();
      final bVals = cpu.bias!.toList();
      gpu.weight.assign(
        Tensor.fromList(cpu.weight.shape, wVals, device: Device.GPU),
      );
      gpu.bias!.assign(
        Tensor.fromList(cpu.bias!.shape, bVals, device: Device.GPU),
      );

      final xCpu = _rand([1, 4, 8, 8], seed: 99);
      final xGpu = _rand([1, 4, 8, 8], seed: 99, device: Device.GPU);
      final yCpu = cpu(xCpu).toList();
      final yGpu = gpu(xGpu).toList();
      for (int i = 0; i < yCpu.length; i++) {
        expect(
          (yCpu[i] - yGpu[i]).abs() < 1e-4,
          isTrue,
          reason: 'pos $i cpu=${yCpu[i]} gpu=${yGpu[i]}',
        );
      }
    });
  });
}
