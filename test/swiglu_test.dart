import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

void main() {
  group('SwiGluFfn', () {
    test('output has expected shape [N, dim]', () {
      final ffn = SwiGluFfn(8, 22);
      final x = Tensor.fromList(
        [3, 8],
        [for (int i = 0; i < 24; i++) 0.1 * i - 1.0],
      );
      final y = ffn(x);
      expect(y.shape, [3, 8]);
    });

    test('rank-3 input passes through via Linear reshape', () {
      final ffn = SwiGluFfn(4, 11);
      final x = Tensor.fromList(
        [2, 5, 4],
        [for (int i = 0; i < 40; i++) 0.05 * i],
      );
      final y = ffn(x);
      expect(y.shape, [2, 5, 4]);
    });

    test('parameters() returns exactly the three Linear weights', () {
      final ffn = SwiGluFfn(6, 16);
      final ps = ffn.parameters();
      // Three bias-free Linears -> three weight tensors.
      expect(ps.length, 3);
      expect(ps[0].shape, [16, 6]); // gate: [hidden, dim]
      expect(ps[1].shape, [16, 6]); // up
      expect(ps[2].shape, [6, 16]); // down
    });

    test('matches manual reference on a fixed 2x4 input', () {
      final ffn = SwiGluFfn(4, 6, seed: 42);

      // Grab the weights out of the FFN so we can compute reference
      // in pure Dart.
      final wGate = ffn.gateProj.weight.toList(); // [6, 4]
      final wUp = ffn.upProj.weight.toList(); // [6, 4]
      final wDown = ffn.downProj.weight.toList(); // [4, 6]

      final xv = <double>[0.5, -1.0, 2.0, 0.25, 1.0, -0.5, 0.75, 1.5];
      final x = Tensor.fromList([2, 4], xv);

      final y = ffn(x).toList();

      // Reference: for each row, compute:
      //   gate = W_gate @ x    (shape 6)
      //   up   = W_up   @ x    (shape 6)
      //   silu = gate * sigmoid(gate)
      //   h    = silu * up     (shape 6)
      //   out  = W_down @ h    (shape 4)
      final want = <double>[];
      for (int r = 0; r < 2; r++) {
        final x1 = xv.sublist(r * 4, (r + 1) * 4);
        final gate = List<double>.filled(6, 0);
        final up = List<double>.filled(6, 0);
        for (int i = 0; i < 6; i++) {
          for (int j = 0; j < 4; j++) {
            gate[i] += wGate[i * 4 + j] * x1[j];
            up[i] += wUp[i * 4 + j] * x1[j];
          }
        }
        final silu = List<double>.generate(
          6,
          (i) => gate[i] * (1.0 / (1.0 + math.exp(-gate[i]))),
        );
        final h = List<double>.generate(6, (i) => silu[i] * up[i]);
        final out = List<double>.filled(4, 0);
        for (int i = 0; i < 4; i++) {
          for (int j = 0; j < 6; j++) {
            out[i] += wDown[i * 6 + j] * h[j];
          }
        }
        want.addAll(out);
      }

      for (int i = 0; i < y.length; i++) {
        expect(
          (y[i] - want[i]).abs() < 1e-5,
          isTrue,
          reason: 'index $i: got ${y[i]} want ${want[i]}',
        );
      }
    });
  });
}
