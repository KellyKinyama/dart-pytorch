import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

void main() {
  group('RopeCache', () {
    test('position 0 is identity', () {
      final rope = RopeCache(maxCtx: 4, headDim: 4);
      final q = Tensor.fromList([1, 4], [1.0, 2.0, 3.0, 4.0]);
      final out = Tensor.noGrad(() => rope.apply(q, startPos: 0));
      final data = out.toList();
      expect(data[0], closeTo(1.0, 1e-5));
      expect(data[1], closeTo(2.0, 1e-5));
      expect(data[2], closeTo(3.0, 1e-5));
      expect(data[3], closeTo(4.0, 1e-5));
    });

    test('matches closed-form rotate_half at position m', () {
      // Reference implementation, exactly the GPT-NeoX math:
      //   q_out = q * cos + rotate_half(q) * sin
      //   where rotate_half([a|b]) = [-b|a]
      const headDim = 8;
      const base = 10000.0;
      final halfD = headDim ~/ 2;
      final invFreq = List<double>.generate(
        halfD,
        (i) => 1.0 / math.pow(base, 2.0 * i / headDim),
      );
      List<double> refApply(List<double> q, int m) {
        final cos = List<double>.filled(headDim, 0);
        final sin = List<double>.filled(headDim, 0);
        for (int j = 0; j < halfD; j++) {
          final a = m * invFreq[j];
          cos[j] = cos[j + halfD] = math.cos(a);
          sin[j] = sin[j + halfD] = math.sin(a);
        }
        final rot = List<double>.filled(headDim, 0);
        for (int j = 0; j < halfD; j++) {
          rot[j] = -q[j + halfD];
          rot[j + halfD] = q[j];
        }
        return List<double>.generate(
          headDim,
          (j) => q[j] * cos[j] + rot[j] * sin[j],
        );
      }

      final rope = RopeCache(maxCtx: 32, headDim: headDim);
      final q = List<double>.generate(headDim, (i) => (i + 1).toDouble() * 0.1);
      for (final m in [0, 1, 3, 7, 15]) {
        final ourOut = Tensor.noGrad(
          () => rope.apply(Tensor.fromList([1, headDim], q), startPos: m),
        );
        final ref = refApply(q, m);
        final got = ourOut.toList();
        for (int j = 0; j < headDim; j++) {
          expect(got[j], closeTo(ref[j], 1e-5), reason: 'm=$m j=$j');
        }
      }
    });

    test('multi-row window respects contiguous positions', () {
      const headDim = 4;
      final rope = RopeCache(maxCtx: 8, headDim: headDim);
      final q = Tensor.fromList(
        [3, headDim],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
      );
      // Applying with startPos=2 should equal applying each row
      // individually at positions 2, 3, 4.
      final block = Tensor.noGrad(() => rope.apply(q, startPos: 2)).toList();
      for (int i = 0; i < 3; i++) {
        final row = q.toList().sublist(i * headDim, (i + 1) * headDim);
        final singleRow = Tensor.noGrad(
          () => rope.apply(Tensor.fromList([1, headDim], row), startPos: 2 + i),
        ).toList();
        for (int j = 0; j < headDim; j++) {
          expect(
            block[i * headDim + j],
            closeTo(singleRow[j], 1e-5),
            reason: 'row $i col $j',
          );
        }
      }
    });

    test('rejects mismatched headDim', () {
      final rope = RopeCache(maxCtx: 4, headDim: 4);
      final q = Tensor.fromList([1, 6], [1.0, 2, 3, 4, 5, 6]);
      expect(() => rope.apply(q), throwsArgumentError);
    });

    test('rejects out-of-range window', () {
      final rope = RopeCache(maxCtx: 4, headDim: 4);
      final q = Tensor.fromList([3, 4], [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
      expect(() => rope.apply(q, startPos: 2), throwsArgumentError);
    });
  });
}
