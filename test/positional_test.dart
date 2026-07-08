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

// Reference sinusoidal PE.
List<double> _refSinusoidalPE(int n, int d) {
  final out = List<double>.filled(n * d, 0);
  for (int pos = 0; pos < n; pos++) {
    for (int i = 0; i < d; i++) {
      final pair = i ~/ 2;
      final freq = math.pow(10000.0, -2.0 * pair / d);
      final angle = pos * freq;
      out[pos * d + i] = i.isEven ? math.sin(angle) : math.cos(angle);
    }
  }
  return out;
}

void main() {
  group('SinusoidalPositionalEncoding', () {
    test('has no trainable parameters', () {
      final pe = SinusoidalPositionalEncoding(8);
      expect(pe.parameters(), isEmpty);
    });

    test('output = x + PE matches reference formula', () {
      const n = 5, d = 4;
      final pe = SinusoidalPositionalEncoding(d);
      final xVals = List<double>.filled(n * d, 0.0); // zeros so out == PE
      final x = Tensor.fromList([n, d], xVals);
      final out = pe(x).toList();
      final want = _refSinusoidalPE(n, d);
      expectClose(out, want, tol: 1e-5);
    });

    test('adds to input rather than replacing it', () {
      const n = 3, d = 6;
      final pe = SinusoidalPositionalEncoding(d);
      final xVals = List<double>.generate(n * d, (i) => (i + 1).toDouble());
      final x = Tensor.fromList([n, d], xVals);
      final out = pe(x).toList();
      final wantPE = _refSinusoidalPE(n, d);
      final want = List<double>.generate(n * d, (i) => xVals[i] + wantPE[i]);
      expectClose(out, want, tol: 1e-5);
    });

    test('gradient flows through unchanged to x', () {
      const n = 3, d = 4;
      final pe = SinusoidalPositionalEncoding(d);
      final x = Tensor.fromList(
        [n, d],
        List<double>.generate(n * d, (i) => (i - 6) * 0.1),
        requiresGrad: true,
      );
      pe(x).sum().backward();
      // PE has no params. Grad wrt x is all ones (since it's a pure add).
      final g = x.grad!.toList();
      for (final v in g) {
        expect(
          (v - 1.0).abs() < 1e-5,
          isTrue,
          reason: 'expected grad=1, got $v',
        );
      }
    });

    test('rejects mismatched embedding dim', () {
      final pe = SinusoidalPositionalEncoding(4);
      final x = Tensor.fromList([2, 6], List<double>.filled(12, 0));
      expect(() => pe(x), throwsArgumentError);
    });

    test('works with different seq lengths without preallocating', () {
      final pe = SinusoidalPositionalEncoding(4);
      final short = Tensor.fromList([2, 4], List<double>.filled(8, 0));
      final long = Tensor.fromList([10, 4], List<double>.filled(40, 0));
      expect(pe(short).shape, [2, 4]);
      expect(pe(long).shape, [10, 4]);
    });

    test('GPU path matches CPU path', () {
      const n = 4, d = 8;
      final pe = SinusoidalPositionalEncoding(d);
      final xVals = List<double>.generate(n * d, (i) => math.sin(i * 0.7));
      final xC = Tensor.fromList([n, d], xVals, device: Device.CPU);
      final xG = Tensor.fromList([n, d], xVals, device: Device.GPU);
      expectClose(pe(xG).toList(), pe(xC).toList(), tol: 1e-4);
    });
  });

  group('LearnedPositionalEmbedding', () {
    test('exposes one trainable table parameter', () {
      final pe = LearnedPositionalEmbedding(16, 8);
      expect(pe.parameters().length, 1);
      expect(pe.parameters()[0].shape, [16, 8]);
    });

    test('rejects seqLen > maxLen', () {
      final pe = LearnedPositionalEmbedding(4, 8);
      final x = Tensor.fromList([5, 8], List<double>.filled(40, 0));
      expect(() => pe(x), throwsArgumentError);
    });

    test('rejects mismatched embedding dim', () {
      final pe = LearnedPositionalEmbedding(8, 4);
      final x = Tensor.fromList([2, 6], List<double>.filled(12, 0));
      expect(() => pe(x), throwsArgumentError);
    });

    test('output = x + table[0..n-1]', () {
      final pe = LearnedPositionalEmbedding(4, 3, seed: 5);
      final xVals = [
        1.0, 2.0, 3.0, //
        4.0, 5.0, 6.0, //
      ];
      final x = Tensor.fromList([2, 3], xVals);
      final out = pe(x).toList();

      // Compute reference by manually looking up positions 0 and 1.
      final table = pe.parameters()[0].toList();
      final want = <double>[];
      for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 3; c++) {
          want.add(xVals[r * 3 + c] + table[r * 3 + c]);
        }
      }
      expectClose(out, want, tol: 1e-5);
    });

    test('gradient reaches the position table', () {
      final pe = LearnedPositionalEmbedding(4, 3, seed: 5);
      final x = Tensor.fromList(
        [2, 3],
        List<double>.filled(6, 0),
        requiresGrad: true,
      );
      // Loss = sum(pe(x)); upstream grad is all ones. Only rows 0 and 1
      // of the table should receive grads (positions 2 and 3 unused).
      pe(x).sum().backward();
      final g = pe.parameters()[0].grad!.toList();
      // Rows 0, 1: each dim should have grad 1.
      for (int i = 0; i < 6; i++) {
        expect(
          (g[i] - 1.0).abs() < 1e-5,
          isTrue,
          reason: 'row 0/1 dim $i g=${g[i]}',
        );
      }
      // Rows 2, 3: unused, grad should be 0.
      for (int i = 6; i < 12; i++) {
        expect(g[i], 0.0, reason: 'row 2/3 dim $i g=${g[i]}');
      }
    });

    test('train()/eval() propagates to embedding table submodule', () {
      final pe = LearnedPositionalEmbedding(8, 4);
      expect(pe.training, isTrue);
      pe.eval();
      expect(pe.training, isFalse);
      pe.train();
      expect(pe.training, isTrue);
    });
  });

  group('Positional encoding + TransformerBlock integration', () {
    test('sinusoidal PE fed into a transformer block produces gradients', () {
      final pe = SinusoidalPositionalEncoding(4);
      final block = TransformerBlock(4, 2, seed: 5);
      final params = block.parameters();

      final x = Tensor.fromList(
        [3, 4],
        [
          0.1, 0.2, 0.3, 0.4, //
          -0.1, 0.0, 0.1, 0.2, //
          0.3, -0.2, 0.4, -0.4, //
        ],
      );
      final y = block(pe(x));
      y.sum().backward();
      for (final p in params) {
        expect(p.grad, isNotNull);
      }
    });

    test('learned PE trains alongside TransformerBlock via Adam', () {
      final pe = LearnedPositionalEmbedding(8, 4, seed: 1);
      final block = TransformerBlock(4, 2, seed: 2);
      final params = [...pe.parameters(), ...block.parameters()];
      final opt = Adam(params, lr: 0.05);

      final x = Tensor.fromList(
        [3, 4],
        [
          0.1, -0.1, 0.2, -0.2, //
          0.3, 0.3, -0.3, -0.3, //
          -0.4, 0.4, 0.1, 0.0, //
        ],
      );
      final target = Tensor.fromList(
        [3, 4],
        [
          0.5, -0.5, 0.2, -0.2, //
          -0.1, 0.1, 0.4, 0.4, //
          0.3, 0.3, -0.3, -0.3, //
        ],
      );

      double lossVal() => (block(pe(x)) - target).pow(2).mean().toList()[0];

      final initial = lossVal();
      for (int i = 0; i < 300; i++) {
        opt.zeroGrad();
        (block(pe(x)) - target).pow(2).mean().backward();
        opt.step();
      }
      final finalL = lossVal();
      expect(
        finalL < initial * 0.5,
        isTrue,
        reason: 'loss did not decrease enough: $initial -> $finalL',
      );
    });
  });
}
