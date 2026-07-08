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

void main() {
  group('Tensor.concat (axis=1)', () {
    test('two 2x2 tensors', () {
      final a = Tensor.fromList([2, 2], [1, 2, 3, 4]);
      final b = Tensor.fromList([2, 3], [5, 6, 7, 8, 9, 10]);
      final out = TensorConcat.concat([a, b]);
      expect(out.shape, [2, 5]);
      expect(out.toList(), [1, 2, 5, 6, 7, 3, 4, 8, 9, 10]);
    });

    test('single input returns equivalent data', () {
      final a = Tensor.fromList([2, 3], [1, 2, 3, 4, 5, 6]);
      final out = TensorConcat.concat([a]);
      expect(out.shape, [2, 3]);
      expect(out.toList(), a.toList());
    });

    test('rejects device mismatch', () {
      final a = Tensor.fromList([2, 2], [1, 2, 3, 4], device: Device.CPU);
      final b = Tensor.fromList([2, 2], [5, 6, 7, 8], device: Device.GPU);
      expect(() => TensorConcat.concat([a, b]), throwsArgumentError);
      b.dispose();
    });

    test('rejects row mismatch', () {
      final a = Tensor.fromList([2, 2], [1, 2, 3, 4]);
      final b = Tensor.fromList([3, 2], [1, 2, 3, 4, 5, 6]);
      expect(() => TensorConcat.concat([a, b]), throwsArgumentError);
    });

    test('rejects axis != 1', () {
      final a = Tensor.fromList([2, 2], [1, 2, 3, 4]);
      expect(() => TensorConcat.concat([a], axis: 0), throwsArgumentError);
    });

    test('gradient slices upstream grad back into each input', () {
      final a = Tensor.fromList([2, 2], [1, 2, 3, 4], requiresGrad: true);
      final b = Tensor.fromList(
        [2, 3],
        [5, 6, 7, 8, 9, 10],
        requiresGrad: true,
      );
      // Loss = sum(out) — upstream grad is all ones.
      TensorConcat.concat([a, b]).sum().backward();
      expect(a.grad!.toList(), [1, 1, 1, 1]);
      expect(b.grad!.toList(), [1, 1, 1, 1, 1, 1]);
    });

    test('gradient matches numerical differentiation on weighted sum', () {
      final av = [0.3, -0.2, 0.1, 0.4];
      final bv = [1.0, -0.5, 0.2, 0.7, 0.1, -0.3];
      final wv = [0.5, -1.0, 0.3, 0.2, 0.9, -0.4, 0.1, -0.6, 0.7, 0.8];
      final a = Tensor.fromList([2, 2], av, requiresGrad: true);
      final b = Tensor.fromList([2, 3], bv, requiresGrad: true);
      final w = Tensor.fromList([2, 5], wv);
      (TensorConcat.concat([a, b]) * w).sum().backward();

      double loss(List<double> avs, List<double> bvs) {
        double total = 0;
        for (int r = 0; r < 2; r++) {
          for (int c = 0; c < 2; c++) {
            total += avs[r * 2 + c] * wv[r * 5 + c];
          }
          for (int c = 0; c < 3; c++) {
            total += bvs[r * 3 + c] * wv[r * 5 + 2 + c];
          }
        }
        return total;
      }

      final h = 1e-3;
      final wantA = List<double>.filled(av.length, 0);
      final probeA = List<double>.of(av);
      for (int i = 0; i < av.length; i++) {
        final o = probeA[i];
        probeA[i] = o + h;
        final lp = loss(probeA, bv);
        probeA[i] = o - h;
        final lm = loss(probeA, bv);
        probeA[i] = o;
        wantA[i] = (lp - lm) / (2 * h);
      }
      expectClose(a.grad!.toList(), wantA, tol: 1e-3);

      final wantB = List<double>.filled(bv.length, 0);
      final probeB = List<double>.of(bv);
      for (int i = 0; i < bv.length; i++) {
        final o = probeB[i];
        probeB[i] = o + h;
        final lp = loss(av, probeB);
        probeB[i] = o - h;
        final lm = loss(av, probeB);
        probeB[i] = o;
        wantB[i] = (lp - lm) / (2 * h);
      }
      expectClose(b.grad!.toList(), wantB, tol: 1e-3);
    });

    test('GPU inputs round-trip correctly', () {
      final a = Tensor.fromList(
        [4, 2],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        device: Device.GPU,
      );
      final b = Tensor.fromList(
        [4, 2],
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        device: Device.GPU,
      );
      final out = TensorConcat.concat([a, b]);
      expect(out.device, Device.GPU);
      expect(out.shape, [4, 4]);
      expect(out.toList(), [
        1, 2, 10, 20, //
        3, 4, 30, 40, //
        5, 6, 50, 60, //
        7, 8, 70, 80, //
      ]);
    });
  });

  group('MultiHeadAttention', () {
    test('constructs, forwards, shapes', () {
      final mha = MultiHeadAttention(8, 2, seed: 1);
      expect(mha.embedDim, 8);
      expect(mha.numHeads, 2);
      expect(mha.headDim, 4);
      // 3*2 head projections + 1 output = 7 Linears without bias -> 7 weights.
      expect(mha.parameters().length, 7);
      final x = Tensor.fromList([
        3,
        8,
      ], List<double>.generate(24, (i) => (i - 12) * 0.1));
      final y = mha(x);
      expect(y.shape, [3, 8]);
    });

    test('rejects embedDim not divisible by numHeads', () {
      expect(() => MultiHeadAttention(7, 2), throwsArgumentError);
    });

    test('is differentiable end-to-end', () {
      final mha = MultiHeadAttention(4, 2, seed: 2);
      final x = Tensor.fromList(
        [2, 4],
        [1.0, 0.0, -1.0, 0.5, 0.5, -0.5, 0.5, 0.0],
      );
      mha(x).sum().backward();
      // Every parameter should have a gradient.
      for (final p in mha.parameters()) {
        expect(p.grad, isNotNull, reason: 'param ${p.shape} missing grad');
        // Sanity: some gradient should be non-zero.
        final anyNonZero = p.grad!.toList().any((v) => v.abs() > 1e-8);
        expect(
          anyNonZero,
          isTrue,
          reason: 'grad for param ${p.shape} is entirely zero',
        );
      }
    });

    test('trains toward a synthetic target with Adam', () {
      final mha = MultiHeadAttention(4, 2, seed: 3);
      final opt = Adam(mha.parameters(), lr: 0.05);
      final x = Tensor.fromList(
        [3, 4],
        [1.0, 0.0, -1.0, 0.5, 0.5, -0.5, 0.5, 0.0, -0.5, 1.0, 0.0, 0.5],
      );
      final target = Tensor.fromList(
        [3, 4],
        [
          0.1, 0.2, 0.3, -0.1, //
          -0.2, 0.4, 0.0, 0.5, //
          0.3, -0.3, 0.2, 0.1, //
        ],
      );

      double lossVal() => (mha(x) - target).pow(2).mean().toList()[0];

      final initial = lossVal();
      for (int i = 0; i < 400; i++) {
        opt.zeroGrad();
        (mha(x) - target).pow(2).mean().backward();
        opt.step();
      }
      final finalL = lossVal();
      expect(
        finalL < initial * 0.2,
        isTrue,
        reason: 'loss did not converge: initial=$initial final=$finalL',
      );
    });

    test('train()/eval() propagates to attention dropout', () {
      final mha = MultiHeadAttention(4, 2, dropoutP: 0.5, seed: 4);
      expect(mha.training, isTrue);
      expect(mha.attnDropout!.training, isTrue);
      mha.eval();
      expect(mha.attnDropout!.training, isFalse);
      mha.train();
      expect(mha.attnDropout!.training, isTrue);
    });
  });

  group('TransformerBlock', () {
    test('forward preserves shape and gradient flows to every parameter', () {
      final block = TransformerBlock(8, 2, seed: 7);
      final x = Tensor.fromList([
        4,
        8,
      ], List<double>.generate(32, (i) => math.sin(i * 0.3)));
      final y = block(x);
      expect(y.shape, [4, 8]);

      y.sum().backward();
      for (final p in block.parameters()) {
        expect(p.grad, isNotNull, reason: 'param ${p.shape} missing grad');
      }
    });

    test('eval() disables dropout — output becomes deterministic', () {
      final block = TransformerBlock(4, 2, dropoutP: 0.5, seed: 11);
      block.eval();
      final x = Tensor.fromList(
        [2, 4],
        [1.0, 0.0, -1.0, 0.5, -0.5, 0.5, 0.0, 1.0],
      );
      final y1 = block(x).toList();
      final y2 = block(x).toList();
      expectClose(
        y1,
        y2,
        tol: 1e-6,
        reason: 'eval mode should be deterministic',
      );
    });

    test('trains to reduce loss on synthetic target', () {
      final block = TransformerBlock(4, 2, seed: 5);
      final opt = Adam(block.parameters(), lr: 0.05);
      final x = Tensor.fromList(
        [3, 4],
        [
          1.0, 0.0, -1.0, 0.5, //
          -0.5, 0.5, 0.0, 1.0, //
          0.2, -0.2, 0.4, -0.4, //
        ],
      );
      final target = Tensor.fromList(
        [3, 4],
        [
          0.1, -0.1, 0.2, -0.2, //
          -0.3, 0.3, 0.1, 0.4, //
          0.5, 0.5, -0.5, -0.5, //
        ],
      );

      double lossVal() => (block(x) - target).pow(2).mean().toList()[0];

      final initial = lossVal();
      for (int i = 0; i < 300; i++) {
        opt.zeroGrad();
        (block(x) - target).pow(2).mean().backward();
        opt.step();
      }
      final finalL = lossVal();
      expect(
        finalL < initial * 0.5,
        isTrue,
        reason: 'loss did not decrease enough: $initial -> $finalL',
      );
    });

    test('parameter count matches expected composition', () {
      // TransformerBlock = LN1(2) + LN2(2) + MHA(3H*W + Wo=7 for H=2) +
      //                    FFN1(2) + FFN2(2) = 15 tensors.
      final block = TransformerBlock(8, 2);
      // LN* have 2 params each (gamma, beta); Linear* have 2 (W, b) each;
      // MHA has 7 weights (bias:false).
      // Total: 2 + 2 + 7 + 2 + 2 = 15.
      expect(block.parameters().length, 15);
    });
  });
}
