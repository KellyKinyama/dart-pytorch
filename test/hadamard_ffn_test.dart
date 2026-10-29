import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

void main() {
  group('HadamardFFN', () {
    test('has 3*dim parameters', () {
      final f = HadamardFFN(16);
      expect(f.parameters().length, 3);
      expect(f.parameters().fold<int>(0, (a, p) => a + p.length), 3 * 16);
    });

    test('preserves [N, dim] shape with finite output', () {
      final f = HadamardFFN(8);
      final rng = math.Random(1);
      final x = Tensor.fromList(
        [4, 8],
        [for (var i = 0; i < 32; i++) rng.nextDouble() * 2 - 1],
      );
      final y = f(x);
      expect(y.shape, [4, 8]);
      expect(y.toList().every((v) => v.isFinite), isTrue);
    });

    test('rejects a non-power-of-two dim', () {
      expect(() => HadamardFFN(6), throwsArgumentError);
    });

    test('trains: a few steps reduce a toy regression loss', () {
      final f = HadamardFFN(8, seed: 2);
      final x = Tensor.fromList(
        [4, 8],
        [for (var i = 0; i < 32; i++) (i % 5) * 0.1 - 0.2],
      );
      final target = Tensor.fromList(
        [4, 8],
        [for (var i = 0; i < 32; i++) 0.3],
      );
      final opt = Adam(f.parameters(), lr: 1e-2);
      double loss() => ((f(x) - target).pow(2.0)).mean().toList().first;
      final before = loss();
      for (var s = 0; s < 30; s++) {
        opt.zeroGrad();
        final l = ((f(x) - target).pow(2.0)).mean();
        l.backward();
        opt.step();
      }
      expect(loss(), lessThan(before));
    });
  });
}
