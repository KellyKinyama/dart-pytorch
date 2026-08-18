import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

void main() {
  group('Conv1d', () {
    test('output shape follows (L + 2p - k) / s + 1', () {
      final layer = Conv1d(
        inChannels: 3,
        outChannels: 5,
        kernelSize: 3,
        padding: 1,
      );
      final x = Tensor.fill([2, 3, 10], 0.0, device: Device.CPU);
      final y = layer(x);
      expect(y.shape, equals([2, 5, 10]));
    });

    test('stride 2 halves the temporal dim', () {
      final layer = Conv1d(
        inChannels: 4,
        outChannels: 8,
        kernelSize: 3,
        stride: 2,
        padding: 1,
      );
      final x = Tensor.fill([1, 4, 16], 1.0, device: Device.CPU);
      final y = layer(x);
      expect(y.shape, equals([1, 8, 8]));
    });

    test('identity conv with kernel=1 and known weight matches matmul', () {
      final layer = Conv1d(inChannels: 2, outChannels: 3, kernelSize: 1);
      // weight [Cout=3, Cin=2, K=1] = [[[1], [2]], [[3], [4]], [[5], [6]]]
      layer.loadFromPytorch(
        Float32List.fromList([1, 2, 3, 4, 5, 6]),
        Float32List.fromList([0, 0, 0]),
      );
      final input = Tensor.fromFloat32List(
        [1, 2, 3],
        Float32List.fromList([10, 20, 30, 40, 50, 60]),
        device: Device.CPU,
      );
      final y = layer(input);
      expect(y.shape, equals([1, 3, 3]));
      final got = y.toList();
      // Expected output for each time step t:
      //   y[o, t] = sum_c W[o, c] * x[c, t]
      // t=0 (x = [10, 40]): 1*10+2*40=90, 3*10+4*40=190, 5*10+6*40=290
      // t=1 (x = [20, 50]): 1*20+2*50=120, 3*20+4*50=260, 5*20+6*50=400
      // t=2 (x = [30, 60]): 1*30+2*60=150, 3*30+4*60=330, 5*30+6*60=510
      final want = [90, 120, 150, 190, 260, 330, 290, 400, 510];
      for (int i = 0; i < 9; i++) {
        expect(got[i], closeTo(want[i], 1e-4));
      }
    });

    test('bias is added per output channel', () {
      final layer = Conv1d(inChannels: 1, outChannels: 2, kernelSize: 1);
      layer.loadFromPytorch(
        Float32List.fromList([0, 0]),
        Float32List.fromList([5, -7]),
      );
      final input = Tensor.fromFloat32List(
        [1, 1, 3],
        Float32List.fromList([0, 0, 0]),
        device: Device.CPU,
      );
      final got = layer(input).toList();
      expect(got, equals([5.0, 5.0, 5.0, -7.0, -7.0, -7.0]));
    });

    test('zero padding: kernel spans past the boundary', () {
      final layer = Conv1d(
        inChannels: 1,
        outChannels: 1,
        kernelSize: 3,
        padding: 1,
      );
      layer.loadFromPytorch(
        Float32List.fromList([1, 1, 1]), // sum of 3 taps
        Float32List.fromList([0]),
      );
      final input = Tensor.fromFloat32List(
        [1, 1, 5],
        Float32List.fromList([2, 3, 4, 5, 6]),
        device: Device.CPU,
      );
      final got = layer(input).toList();
      // With zero-pad at both ends:
      //   y[0] = 0 + 2 + 3 = 5
      //   y[1] = 2 + 3 + 4 = 9
      //   y[2] = 3 + 4 + 5 = 12
      //   y[3] = 4 + 5 + 6 = 15
      //   y[4] = 5 + 6 + 0 = 11
      expect(got, equals([5.0, 9.0, 12.0, 15.0, 11.0]));
    });
  });
}
