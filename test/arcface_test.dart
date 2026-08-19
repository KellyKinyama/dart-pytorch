@Timeout(Duration(minutes: 2))
library;

import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

Tensor _rowsFromDouble(int rows, int cols, double v) =>
    Tensor.fromList([rows, cols], List<double>.filled(rows * cols, v));

bool _gpuAvailable() {
  try {
    Tensor.fromList([1], [1.0], device: Device.GPU).toList();
    return true;
  } catch (_) {
    return false;
  }
}

void main() {
  group('ArcFace', () {
    test('logitsForInference returns s·cosθ with no margin twist', () {
      final af = ArcFace(4, 3, scale: 10.0, margin: 0.5);
      // Force weight matrix so we can hand-compute cosines.
      af.weight.weight.assign(
        Tensor.fromList(
          [3, 4],
          [
            1, 0, 0, 0, //
            0, 1, 0, 0, //
            0, 0, 1, 0, //
          ],
        ),
      );
      // Two batch rows, both aligned exactly with class 0 (cosθ = 1
      // against class 0, 0 against class 1 & 2).
      final e = Tensor.fromList(
        [2, 4],
        [
          5, 0, 0, 0, //
          3, 0, 0, 0, //
        ],
      );
      final out = af.logitsForInference(e).toList();
      // s * cosθ = 10 * [1, 0, 0] for every row.
      expect(out.sublist(0, 3), equals([10.0, 0.0, 0.0]));
      expect(out.sublist(3, 6), equals([10.0, 0.0, 0.0]));
    });

    test('target column uses cos(θ+m); others use cos(θ)', () {
      final af = ArcFace(4, 3, scale: 1.0, margin: 0.5);
      af.weight.weight.assign(
        Tensor.fromList(
          [3, 4],
          [
            1, 0, 0, 0, //
            0, 1, 0, 0, //
            0, 0, 1, 0, //
          ],
        ),
      );
      final e = Tensor.fromList([1, 4], [1, 0, 0, 0]);
      final labels = Tensor.fromList([1], [0.0]); // target = class 0
      final out = af(e, labels).toList();
      // cosθ against class 0 is 1 → cos(1 + 0.5·rad_wrap) but with
      // clamp to 1 we get cos(θ+m) with θ=0 → cos(0.5) ≈ 0.8776
      expect(
        (out[0] - math.cos(0.5)).abs() < 1e-5,
        isTrue,
        reason: 'target col: got ${out[0]}, want ${math.cos(0.5)}',
      );
      // Non-target columns are s·cosθ = 1·0 = 0.
      expect(out[1], closeTo(0.0, 1e-6));
      expect(out[2], closeTo(0.0, 1e-6));
    });

    test('margin=0 collapses to plain scaled cosine', () {
      final af = ArcFace(4, 3, scale: 2.0, margin: 0.0);
      af.weight.weight.assign(
        Tensor.fromList(
          [3, 4],
          [
            1, 0, 0, 0, //
            0, 1, 0, 0, //
            0, 0, 1, 0, //
          ],
        ),
      );
      final e = Tensor.fromList(
        [2, 4],
        [
          1, 0, 0, 0, //
          0, 1, 0, 0, //
        ],
      );
      final labels = Tensor.fromList([2], [0.0, 1.0]);
      final trainOut = af(e, labels).toList();
      final inferOut = af.logitsForInference(e).toList();
      for (int i = 0; i < trainOut.length; i++) {
        expect(
          (trainOut[i] - inferOut[i]).abs() < 1e-5,
          isTrue,
          reason: 'i=$i: train=${trainOut[i]} infer=${inferOut[i]}',
        );
      }
    });

    test('normalises embeddings — magnitude does not affect cosine', () {
      final af = ArcFace(4, 2, scale: 1.0, margin: 0.0);
      af.weight.weight.assign(
        Tensor.fromList(
          [2, 4],
          [
            1, 0, 0, 0, //
            0, 1, 0, 0, //
          ],
        ),
      );
      final small = _rowsFromDouble(1, 4, 0.0);
      Tensor.fromList(
        [1, 4],
        [0.1, 0, 0, 0],
      ).toList().asMap().forEach((i, v) => small.toList()[i] = v);
      final sSmall = af.logitsForInference(
        Tensor.fromList([1, 4], [0.1, 0, 0, 0]),
      );
      final sBig = af.logitsForInference(
        Tensor.fromList([1, 4], [1000.0, 0, 0, 0]),
      );
      final aSmall = sSmall.toList();
      final aBig = sBig.toList();
      for (int i = 0; i < aSmall.length; i++) {
        expect(
          (aSmall[i] - aBig[i]).abs() < 1e-4,
          isTrue,
          reason: 'i=$i: small=${aSmall[i]} big=${aBig[i]}',
        );
      }
    });

    test('rejects wrong-shape embeddings', () {
      final af = ArcFace(4, 2);
      expect(
        () => af.logitsForInference(Tensor.fromList([3], [1.0, 2.0, 3.0])),
        throwsArgumentError,
      );
      expect(
        () => af.logitsForInference(Tensor.fromList([1, 5], [1, 2, 3, 4, 5])),
        throwsArgumentError,
      );
    });

    test('rejects out-of-range label', () {
      final af = ArcFace(4, 2);
      final e = Tensor.fromList([1, 4], [1, 0, 0, 0]);
      expect(() => af(e, Tensor.fromList([1], [5.0])), throwsArgumentError);
    });

    test('parameter list surfaces the weight matrix', () {
      final af = ArcFace(8, 10);
      final params = af.parameters();
      expect(params.length, 1);
      expect(params.first.shape, equals([10, 8]));
    });
  });

  group('ArcFace on GPU', () {
    final gpuOk = _gpuAvailable();
    if (!gpuOk) {
      test(
        'GPU unavailable → skipped',
        () {},
        skip: 'CUDA / native/lib/libmat_mul.so not usable in this env',
      );
      return;
    }
    test('CPU vs GPU inference logits agree', () {
      final cpu = ArcFace(16, 4, scale: 30.0, margin: 0.35);
      final gpu = ArcFace(16, 4, scale: 30.0, margin: 0.35, device: Device.GPU);
      final wVals = cpu.weight.weight.toList();
      gpu.weight.weight.assign(
        Tensor.fromList([4, 16], wVals, device: Device.GPU),
      );
      final xVals = <double>[
        for (int i = 0; i < 3 * 16; i++) math.sin(i * 0.37),
      ];
      final xCpu = Tensor.fromList([3, 16], xVals);
      final xGpu = Tensor.fromList([3, 16], xVals, device: Device.GPU);
      final cpuLogits = cpu.logitsForInference(xCpu).toList();
      final gpuLogits = gpu.logitsForInference(xGpu).toList();
      for (int i = 0; i < cpuLogits.length; i++) {
        expect(
          (cpuLogits[i] - gpuLogits[i]).abs() < 5e-4,
          isTrue,
          reason: 'i=$i cpu=${cpuLogits[i]} gpu=${gpuLogits[i]}',
        );
      }
    });
  });
}
