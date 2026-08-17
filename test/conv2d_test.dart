import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

Tensor _fromNCHW(int n, int c, int h, int w, List<double> data) =>
    Tensor.fromList([n, c, h, w], data);

void main() {
  group('Conv2d', () {
    test('1x1 identity kernel copies the input', () {
      final conv = Conv2d(3, 3, kernel: 1, bias: false);
      // Zero the weight, then set diagonal.
      final wVals = List<double>.filled(3 * 3, 0.0);
      wVals[0] = 1; // out=0, in=0
      wVals[4] = 1; // out=1, in=1
      wVals[8] = 1; // out=2, in=2
      conv.weight.assign(Tensor.fromList([3, 3, 1, 1], wVals));

      final x = _fromNCHW(1, 3, 2, 2, [
        1, 2, 3, 4, // c=0
        5, 6, 7, 8, // c=1
        9, 10, 11, 12, // c=2
      ]);
      final y = conv(x);
      expect(y.shape, [1, 3, 2, 2]);
      expect(y.toList(), x.toList());
    });

    test('output shape matches (H + 2p - K)/s + 1', () {
      final conv = Conv2d(2, 4, kernel: 3, stride: 1, padding: 1);
      final x = _fromNCHW(1, 2, 8, 8, List<double>.filled(1 * 2 * 8 * 8, 0.0));
      final y = conv(x);
      expect(y.shape, [1, 4, 8, 8]);
    });

    test('padding=1 preserves spatial size for 3x3', () {
      final conv = Conv2d(1, 1, kernel: 3, stride: 1, padding: 1, bias: false);
      final wVals = List<double>.filled(9, 0.0);
      wVals[4] = 1; // center = 1, everything else 0 -> identity
      conv.weight.assign(Tensor.fromList([1, 1, 3, 3], wVals));

      final x = _fromNCHW(1, 1, 3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
      final y = conv(x);
      expect(y.shape, [1, 1, 3, 3]);
      expect(y.toList(), x.toList());
    });

    test('all-ones 3x3 kernel sums a 3x3 neighborhood', () {
      final conv = Conv2d(1, 1, kernel: 3, stride: 1, padding: 1, bias: false);
      conv.weight.assign(Tensor.fromList([1, 1, 3, 3], List.filled(9, 1.0)));

      final x = _fromNCHW(1, 1, 3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
      final y = conv(x).toList();
      // Sum of the 3x3 neighborhood for each cell (zero padding).
      // Center cell (1,1) touches all 9 values: 45.
      expect(y[1 * 3 + 1], closeTo(45.0, 1e-6));
      // Top-left (0,0) touches [1,2,4,5] -> 12.
      expect(y[0], closeTo(12.0, 1e-6));
      // Bottom-right (2,2) touches [5,6,8,9] -> 28.
      expect(y[8], closeTo(28.0, 1e-6));
    });

    test('bias is added per output channel', () {
      final conv = Conv2d(1, 2, kernel: 1);
      conv.weight.assign(Tensor.fromList([2, 1, 1, 1], [1, 1]));
      conv.bias!.assign(Tensor.fromList([2], [10.0, -5.0]));

      final x = _fromNCHW(1, 1, 2, 2, [1, 2, 3, 4]);
      final y = conv(x).toList();
      // First channel: x + 10 = [11, 12, 13, 14].
      expect(y.sublist(0, 4), [11, 12, 13, 14]);
      // Second channel: x - 5 = [-4, -3, -2, -1].
      expect(y.sublist(4, 8), [-4, -3, -2, -1]);
    });

    test('stride=2 halves spatial dims', () {
      final conv = Conv2d(1, 1, kernel: 3, stride: 2, padding: 1, bias: false);
      conv.weight.assign(Tensor.fromList([1, 1, 3, 3], List.filled(9, 1.0)));
      final x = _fromNCHW(1, 1, 4, 4, [
        1, 2, 3, 4, //
        5, 6, 7, 8, //
        9, 10, 11, 12, //
        13, 14, 15, 16, //
      ]);
      final y = conv(x);
      expect(y.shape, [1, 1, 2, 2]);
    });

    test('rejects wrong-rank input', () {
      final conv = Conv2d(3, 3, kernel: 3);
      final x = Tensor.fromList([3, 3, 3], List.filled(27, 0.0));
      expect(() => conv(x), throwsA(isA<ArgumentError>()));
    });

    test('rejects channel-count mismatch', () {
      final conv = Conv2d(3, 3, kernel: 3);
      final x = _fromNCHW(1, 4, 3, 3, List.filled(36, 0.0));
      expect(() => conv(x), throwsA(isA<ArgumentError>()));
    });
  });
}
