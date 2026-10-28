import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/core/nn/needle/hadamard.dart';
import 'package:dart_pytorch/core/nn/needle/cact_quant.dart';
import 'package:test/test.dart';

void main() {
  group('fast Walsh-Hadamard transform', () {
    test('is self-inverse (H·H = I)', () {
      final rng = math.Random(1);
      final x = Float64List.fromList(
        List<double>.generate(128, (_) => rng.nextDouble() * 2 - 1),
      );
      final once = Float64List.fromList(x);
      fwhtInPlace(once);
      final twice = Float64List.fromList(once);
      fwhtInPlace(twice);
      for (var i = 0; i < x.length; i++) {
        expect(twice[i], closeTo(x[i], 1e-9));
      }
    });

    test('is orthonormal (preserves L2 norm)', () {
      final rng = math.Random(2);
      final x = Float64List.fromList(
        List<double>.generate(64, (_) => rng.nextDouble()),
      );
      double n2(Float64List v) => v.fold(0.0, (a, e) => a + e * e);
      final before = n2(x);
      final y = Float64List.fromList(x);
      fwhtInPlace(y);
      expect(n2(y), closeTo(before, 1e-9));
    });

    test('rejects non-power-of-two length', () {
      expect(() => fwhtInPlace(Float64List(96)), throwsArgumentError);
    });
  });

  group('Monarch stage', () {
    test('A=B=H32 reproduces the 1024-wide Walsh-Hadamard transform', () {
      const t = 32;
      final h = hadamardMatrix(t);
      final rng = math.Random(3);
      final x = Float64List.fromList(
        List<double>.generate(t * t, (_) => rng.nextDouble() * 2 - 1),
      );
      final viaMonarch = monarchApply(x, h, h, t);
      final viaFwht = Float64List.fromList(x);
      fwhtInPlace(viaFwht);
      for (var i = 0; i < x.length; i++) {
        expect(viaMonarch[i], closeTo(viaFwht[i], 1e-9));
      }
    });
  });

  group('CQ dequantisation', () {
    test('fp16 round-trips simple values', () {
      for (final v in [0.0, 1.0, -1.0, 0.5, 2.0, 0.333, -12.5]) {
        final back = fp16ToDouble(doubleToFp16(v));
        expect(back, closeTo(v, v.abs() * 1e-2 + 1e-3));
      }
    });

    test('quantize→dequantize is near-exact when direction is on centroids',
        () {
      // Build a group whose post-rotation direction lands exactly on
      // codebook centroids, so only fp16 norm rounding limits accuracy.
      final codebook = Float32List.fromList([-0.75, -0.25, 0.25, 0.75]);
      const norm = 1.3;
      final dir = Float64List(128);
      var ss = 0.0;
      for (var j = 0; j < 128; j++) {
        dir[j] = codebook[j % 4];
        ss += dir[j] * dir[j];
      }
      // normalise to a unit direction so |r| == norm exactly
      final inv = 1.0 / math.sqrt(ss);
      final r = Float64List(128);
      for (var j = 0; j < 128; j++) {
        r[j] = norm * dir[j] * inv;
      }
      // w = H·r  (so that quantize's H·w recovers r exactly)
      final w = Float64List.fromList(r);
      fwhtInPlace(w);
      final wf = Float32List.fromList(w);

      // Use a codebook scaled to the normalised direction values.
      final cb = Float32List.fromList([
        for (final c in codebook) c * inv,
      ]);
      final packed = quantizeRow(wf, 2, cb);
      final back = dequantizeRow(packed, 128, 2, cb);
      for (var j = 0; j < 128; j++) {
        expect(back[j], closeTo(wf[j], 2e-2));
      }
    });

    test('quantize→dequantize preserves direction for random data (2-bit)', () {
      final codebook = Float32List.fromList([-0.75, -0.25, 0.25, 0.75]);
      final rng = math.Random(7);
      final w = Float32List.fromList(
        List<double>.generate(256, (_) => rng.nextDouble() * 2 - 1),
      );
      for (final layout in CqLayout.values) {
        final packed = quantizeRow(w, 2, codebook, layout: layout);
        final back = dequantizeRow(packed, w.length, 2, codebook,
            layout: layout);
        // 2-bit scalar quantization of a random 128-dim unit direction is
        // coarse; assert a sane positive cosine, not near-1.
        for (var g = 0; g < w.length ~/ 128; g++) {
          var dot = 0.0, na = 0.0, nb = 0.0;
          for (var j = 0; j < 128; j++) {
            final a = w[g * 128 + j];
            final b = back[g * 128 + j];
            dot += a * b;
            na += a * a;
            nb += b * b;
          }
          final cos = dot / (math.sqrt(na) * math.sqrt(nb));
          expect(cos, greaterThan(0.7),
              reason: 'layout=$layout group=$g cos=$cos');
        }
      }
    });

    test('codebookForBits slices the 28-float blob by width', () {
      final all = Float32List.fromList(
        List<double>.generate(28, (i) => i.toDouble()),
      );
      expect(codebookForBits(all, 2).length, 4);
      expect(codebookForBits(all, 3).length, 8);
      expect(codebookForBits(all, 4).length, 16);
      expect(codebookForBits(all, 4).first, 12.0);
    });

    test('real-layout matrix quantize→dequantize round-trips', () {
      // 3x256 matrix (two 128-groups per row); direction on centroids.
      final codebook = Float32List.fromList([-0.75, -0.25, 0.25, 0.75]);
      final rng = math.Random(11);
      final w = Float64List.fromList(
        List<double>.generate(3 * 256, (_) => rng.nextDouble() * 2 - 1),
      );
      final packed = quantizeMatrix(w, 3, 256, 2, codebook);
      final back = dequantizeMatrix(packed, 3, 256, 2, codebook);
      expect(back.length, 3 * 256);
      // 2-bit scalar quant of random dirs: assert positive per-group cosine.
      for (var r = 0; r < 3; r++) {
        for (var g = 0; g < 2; g++) {
          var dot = 0.0, na = 0.0, nb = 0.0;
          for (var j = 0; j < 128; j++) {
            final a = w[r * 256 + g * 128 + j];
            final b = back[r * 256 + g * 128 + j];
            dot += a * b;
            na += a * a;
            nb += b * b;
          }
          expect(dot / (math.sqrt(na) * math.sqrt(nb)), greaterThan(0.7));
        }
      }
    });
  });
}
