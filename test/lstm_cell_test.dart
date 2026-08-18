import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

void main() {
  group('LSTMCell', () {
    test('zero weights → h_next = 0, c_next = 0', () {
      final cell = LSTMCell(4, 3);
      cell.loadFromPytorch(
        weightIh: Float32List(12 * 4),
        weightHh: Float32List(12 * 3),
        biasIh: Float32List(12),
        biasHh: Float32List(12),
      );
      final x = Tensor.fill([1, 4], 1.0, device: Device.CPU);
      final h = Tensor.fill([1, 3], 0.0, device: Device.CPU);
      final c = Tensor.fill([1, 3], 0.0, device: Device.CPU);
      final r = cell(x, h, c);
      // gates = 0 → i=0.5, f=0.5, g=tanh(0)=0, o=0.5
      // c_new = 0.5 * 0 + 0.5 * 0 = 0
      // h_new = 0.5 * tanh(0) = 0
      final gotH = r.h.toList();
      final gotC = r.c.toList();
      for (int i = 0; i < 3; i++) {
        expect(gotC[i], closeTo(0.0, 1e-6));
        expect(gotH[i], closeTo(0.0, 1e-6));
      }
    });

    test('shape check for [B=2, H=8]', () {
      final cell = LSTMCell(5, 8);
      cell.loadFromPytorch(
        weightIh: Float32List(32 * 5),
        weightHh: Float32List(32 * 8),
        biasIh: Float32List(32),
        biasHh: Float32List(32),
      );
      final x = Tensor.fill([2, 5], 0.0, device: Device.CPU);
      final h = Tensor.fill([2, 8], 0.0, device: Device.CPU);
      final c = Tensor.fill([2, 8], 0.0, device: Device.CPU);
      final r = cell(x, h, c);
      expect(r.h.shape, equals([2, 8]));
      expect(r.c.shape, equals([2, 8]));
    });

    test('gate order (i, f, g, o): forget-only gates hold state', () {
      // Set weights + biases so f = 1 (large positive bias), i = 0, g = 0, o = 0
      // Then c_new should equal c_prev, and h_new should be 0.
      final cell = LSTMCell(1, 2);
      // Layout: [ih_weight for i, f, g, o] each row length inputSize.
      // We set the FORGET row bias very high; other biases very negative.
      final wIh = Float32List(8); // all zeros
      final wHh = Float32List(16);
      final bIh = Float32List(8);
      final bHh = Float32List(8);
      // With PyTorch order (i, f, g, o) at offsets 0, 2, 4, 6 for hidden=2:
      bIh[0] = -20; bIh[1] = -20; // i gate biases
      bIh[2] = 20;  bIh[3] = 20;  // f gate biases (sigmoid ≈ 1)
      bIh[4] = 0;   bIh[5] = 0;   // g gate biases (tanh(0) = 0)
      bIh[6] = -20; bIh[7] = -20; // o gate biases (sigmoid ≈ 0)
      cell.loadFromPytorch(
        weightIh: wIh, weightHh: wHh,
        biasIh: bIh, biasHh: bHh,
      );

      final x = Tensor.fill([1, 1], 0.0, device: Device.CPU);
      final h = Tensor.fill([1, 2], 0.0, device: Device.CPU);
      final c = Tensor.fromFloat32List(
        [1, 2],
        Float32List.fromList([0.7, -0.4]),
        device: Device.CPU,
      );
      final r = cell(x, h, c);
      final gotC = r.c.toList();
      final gotH = r.h.toList();
      expect(gotC[0], closeTo(0.7, 1e-4));
      expect(gotC[1], closeTo(-0.4, 1e-4));
      // o ≈ 0 → h ≈ 0
      expect(gotH[0], closeTo(0.0, 1e-4));
      expect(gotH[1], closeTo(0.0, 1e-4));
    });

    test('numeric match vs hand-computed reference', () {
      final cell = LSTMCell(2, 2);
      // Small hand-picked weights.
      final wIh = Float32List.fromList([
        0.1, 0.2, // i row 0
        0.3, -0.1, // i row 1
        -0.2, 0.4, // f row 0
        0.5, 0.0, // f row 1
        0.1, -0.3, // g row 0
        0.2, 0.2, // g row 1
        0.0, 0.5, // o row 0
        -0.4, 0.1, // o row 1
      ]);
      final wHh = Float32List(16); // zeros
      final bIh = Float32List(8);
      final bHh = Float32List(8);
      cell.loadFromPytorch(
        weightIh: wIh, weightHh: wHh, biasIh: bIh, biasHh: bHh,
      );

      final x = Tensor.fromFloat32List(
        [1, 2],
        Float32List.fromList([0.5, -0.3]),
        device: Device.CPU,
      );
      final h = Tensor.fill([1, 2], 0.0, device: Device.CPU);
      final c = Tensor.fill([1, 2], 0.0, device: Device.CPU);
      final r = cell(x, h, c);
      final gotH = r.h.toList();
      final gotC = r.c.toList();

      // Hand-compute
      double sigmoid(double v) => 1 / (1 + math.exp(-v));
      double tanh(double v) {
        final e = math.exp(2 * v);
        return (e - 1) / (e + 1);
      }
      final iGate = [
        sigmoid(0.1 * 0.5 + 0.2 * (-0.3)),
        sigmoid(0.3 * 0.5 + (-0.1) * (-0.3)),
      ];
      final fGate = [
        sigmoid(-0.2 * 0.5 + 0.4 * (-0.3)),
        sigmoid(0.5 * 0.5 + 0.0 * (-0.3)),
      ];
      final gGate = [
        tanh(0.1 * 0.5 + (-0.3) * (-0.3)),
        tanh(0.2 * 0.5 + 0.2 * (-0.3)),
      ];
      final oGate = [
        sigmoid(0.0 * 0.5 + 0.5 * (-0.3)),
        sigmoid(-0.4 * 0.5 + 0.1 * (-0.3)),
      ];
      final cWant = [
        fGate[0] * 0 + iGate[0] * gGate[0],
        fGate[1] * 0 + iGate[1] * gGate[1],
      ];
      final hWant = [oGate[0] * tanh(cWant[0]), oGate[1] * tanh(cWant[1])];
      for (int i = 0; i < 2; i++) {
        expect(gotC[i], closeTo(cWant[i], 1e-5));
        expect(gotH[i], closeTo(hWant[i], 1e-5));
      }
    });
  });
}
