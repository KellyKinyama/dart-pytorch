import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

void main() {
  group('EngramMemory', () {
    test('parameters = one [slots, dim] table per order', () {
      final e = EngramMemory(dim: 16, slots: 64, orders: [2, 3]);
      expect(e.parameters().length, 2);
      expect(e.parameters().fold<int>(0, (a, p) => a + p.length), 2 * 64 * 16);
    });

    test('returns a [N, dim] contribution', () {
      final e = EngramMemory(dim: 8, slots: 32);
      final ids = Tensor.fromList([5], [1, 2, 3, 2, 1]);
      final out = e(ids);
      expect(out.shape, [5, 8]);
      expect(out.toList().every((v) => v.isFinite), isTrue);
    });

    test('is deterministic for the same tokens', () {
      final e = EngramMemory(dim: 8, slots: 32);
      final ids = Tensor.fromList([4], [7, 8, 9, 7]);
      expect(e(ids).toList(), e(ids).toList());
    });

    test('trains: steps reduce a toy loss', () {
      final e = EngramMemory(dim: 8, slots: 32, seed: 3);
      final ids = Tensor.fromList([4], [1, 2, 3, 4]);
      final target = Tensor.fromList([4, 8], [for (var i = 0; i < 32; i++) 0.5]);
      final opt = Adam(e.parameters(), lr: 1e-2);
      double loss() => ((e(ids) - target).pow(2.0)).mean().toList().first;
      final before = loss();
      for (var s = 0; s < 30; s++) {
        opt.zeroGrad();
        final l = ((e(ids) - target).pow(2.0)).mean();
        l.backward();
        opt.step();
      }
      expect(loss(), lessThan(before));
    });
  });
}
