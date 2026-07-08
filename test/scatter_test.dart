/// Tests for scatterRowsAdd + gather roundtrip with autograd.
library;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

void main() {
  group('scatterRowsAdd', () {
    test('CPU: basic gather/scatter roundtrip', () {
      // full[0..4], take rows [3, 1, 3] -> subset [3, D], scatter back.
      final full = Tensor.fromList([
        4,
        2,
      ], [10.0, 11, 20, 21, 30, 31, 40, 41], requiresGrad: true);
      final indices = Tensor.fromList([3], [3.0, 1.0, 3.0]);
      final subset = full.embedding(indices); // [3, 2]
      expect(subset.toList(), [40, 41, 20, 21, 40, 41]);
      // Scatter back into a [4, 2] zero-init buffer.
      final scattered = subset.scatterRowsAdd(indices, 4);
      expect(scattered.shape, [4, 2]);
      // Row 3 hit twice (indices [3, 3]), row 1 once, rows 0/2 zero.
      expect(scattered.toList(), [0, 0, 20, 21, 0, 0, 80, 82]);
    });

    test('CPU: scatterRowsAdd backward gathers grad at indices', () {
      final subset = Tensor.fromList([
        2,
        3,
      ], [1.0, 2, 3, 4, 5, 6], requiresGrad: true);
      final indices = Tensor.fromList([2], [2.0, 0.0]);
      final scattered = subset.scatterRowsAdd(indices, 3); // [3, 3]
      // Sum-loss so upstream grad = ones([3, 3]).
      scattered.sum().backward();
      // subset.grad[k, :] = 1 for each element (all rows visible).
      expect(subset.grad!.toList(), [1, 1, 1, 1, 1, 1]);
    });

    test('GPU: gather/scatter roundtrip parity vs CPU', () {
      final fullCpu = Tensor.fromList([
        5,
        4,
      ], List<double>.generate(20, (i) => i.toDouble()), requiresGrad: true);
      final fullGpu = Tensor.fromList([
        5,
        4,
      ], List<double>.generate(20, (i) => i.toDouble()),
          requiresGrad: true, device: Device.GPU);
      final idxCpu = Tensor.fromList([4], [1.0, 3.0, 0.0, 3.0]);
      final idxGpu = Tensor.fromList([4], [1.0, 3.0, 0.0, 3.0],
          device: Device.GPU);
      final subsetCpu = fullCpu.embedding(idxCpu);
      final subsetGpu = fullGpu.embedding(idxGpu);
      final sc = subsetCpu.scatterRowsAdd(idxCpu, 5);
      final sg = subsetGpu.scatterRowsAdd(idxGpu, 5);
      final cList = sc.toList();
      final gList = sg.toList();
      expect(cList.length, gList.length);
      for (int i = 0; i < cList.length; i++) {
        expect((cList[i] - gList[i]).abs(), lessThan(1e-5));
      }
      // Backward parity: loss = sum. full.grad should match too.
      sc.sum().backward();
      sg.sum().backward();
      final gc = fullCpu.grad!.toList();
      final gg = fullGpu.grad!.toList();
      for (int i = 0; i < gc.length; i++) {
        expect((gc[i] - gg[i]).abs(), lessThan(1e-5));
      }
    });
  });
}
