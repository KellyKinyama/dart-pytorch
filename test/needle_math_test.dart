import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/core/nn/needle/needle_math.dart';
import 'package:dart_pytorch/core/nn/needle/needle_engram.dart';
import 'package:test/test.dart';

double rowRms(Float64List x, int r, int dim) {
  var ss = 0.0;
  for (var i = 0; i < dim; i++) {
    final v = x[r * dim + i];
    ss += v * v;
  }
  return math.sqrt(ss / dim);
}

void main() {
  group('ZCRMSNorm / rms_unit', () {
    test('scale 0 yields unit-RMS rows', () {
      final x = Float64List.fromList([1, 2, 3, 4, -5, 6, 7, -8]);
      zcRmsNorm(x, 2, 4, Float64List(4));
      expect(rowRms(x, 0, 4), closeTo(1.0, 1e-3));
      expect(rowRms(x, 1, 4), closeTo(1.0, 1e-3));
    });

    test('scale c scales the row by (1 + c)', () {
      final x = Float64List.fromList([1, 2, 3, 4]);
      final scale = Float64List.fromList([0.5, 0.5, 0.5, 0.5]);
      zcRmsNorm(x, 1, 4, scale);
      expect(rowRms(x, 0, 4), closeTo(1.5, 1e-3));
    });

    test('rms_unit normalises without a scale', () {
      final x = Float64List.fromList([2, 0, 0, 0, 0, 3, 4, 0]);
      final u = rmsUnit(x, 2, 4);
      expect(rowRms(u, 0, 4), closeTo(1.0, 1e-5));
      expect(rowRms(u, 1, 4), closeTo(1.0, 1e-5));
    });
  });

  group('activations', () {
    test('silu(0)=0, sigmoid(0)=0.5', () {
      expect(silu(0), closeTo(0, 1e-12));
      expect(sigmoid(0), closeTo(0.5, 1e-12));
    });
    test('softmax rows are a distribution', () {
      final x = Float64List.fromList([1, 2, 3, 0, 0, 0]);
      softmaxRows(x, 2, 3);
      expect(x[0] + x[1] + x[2], closeTo(1.0, 1e-12));
      expect(x[3], closeTo(1 / 3, 1e-12));
      expect(x.every((v) => v > 0), isTrue);
    });
  });

  group('RoPE (half-split)', () {
    test('position 0 is identity', () {
      final (cos, sin) = ropeTables(4, 8, 100000.0);
      final x = Float64List.fromList(
        List<double>.generate(8, (i) => (i + 1).toDouble()),
      );
      final x0 = Float64List.fromList(x);
      // only rotate the first position (T=1 view)
      applyRopeHead(x0, 1, 8, cos, sin);
      for (var i = 0; i < 8; i++) {
        expect(x0[i], closeTo(x[i], 1e-9));
      }
    });

    test('rotation preserves each (x1,x2) pair norm', () {
      final (cos, sin) = ropeTables(6, 8, 100000.0);
      final rng = math.Random(4);
      final x = Float64List.fromList(
        List<double>.generate(6 * 8, (_) => rng.nextDouble() * 2 - 1),
      );
      final before = Float64List.fromList(x);
      applyRopeHead(x, 6, 8, cos, sin);
      const half = 4;
      for (var t = 0; t < 6; t++) {
        for (var i = 0; i < half; i++) {
          final b = before[t * 8 + i] * before[t * 8 + i] +
              before[t * 8 + half + i] * before[t * 8 + half + i];
          final a = x[t * 8 + i] * x[t * 8 + i] +
              x[t * 8 + half + i] * x[t * 8 + half + i];
          expect(a, closeTo(b, 1e-9));
        }
      }
    });
  });

  group('engram hashing', () {
    test('shape and range', () {
      final idx = engramIndices([1, 2, 3, 4, 5]);
      expect(idx.length, 5);
      expect(idx.first.length, 6); // 2 orders * 3 heads
      for (final row in idx) {
        for (final v in row) {
          expect(v, inInclusiveRange(0, 18431));
        }
      }
    });

    test('is deterministic', () {
      final a = engramIndices([7, 8, 9]);
      final b = engramIndices([7, 8, 9]);
      for (var t = 0; t < a.length; t++) {
        expect(a[t], b[t]);
      }
    });

    test('position 0 depends only on the first token (shift boundary)', () {
      // engram_indices uses shift_right, so position 0 sees only token[0];
      // extending the sequence must not change row 0.
      final short = engramIndices([42, 99]);
      final long = engramIndices([42, 99, 7, 3, 1]);
      expect(short[0], long[0]);
      expect(short[1], long[1]);
    });

    test('order-2 and order-3 tables differ on a varied sequence', () {
      final idx = engramIndices([1, 2, 3, 4, 5, 6]);
      // columns 0..2 are order-2 heads, 3..5 are order-3 heads.
      var anyDiff = false;
      for (var t = 2; t < idx.length; t++) {
        if (idx[t][0] != idx[t][3]) anyDiff = true;
      }
      expect(anyDiff, isTrue);
    });
  });
}

Matcher inInclusiveRange(int lo, int hi) =>
    allOf(greaterThanOrEqualTo(lo), lessThanOrEqualTo(hi));
