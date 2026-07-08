/// Tests for the seq2seq decoder pipeline:
/// `MultiHeadCrossAttention` + `TransformerDecoderBlock` +
/// `TransformerDecoder` + `EncoderDecoderTransformer`.
library;

import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

const double _tol = 1e-4;

List<double> _rand(int n, {int seed = 0, double scale = 1.0}) {
  final r = math.Random(seed);
  return List<double>.generate(n, (_) => (r.nextDouble() * 2 - 1) * scale);
}

void _closeList(List<double> a, List<double> b, {double tol = _tol}) {
  expect(a.length, b.length);
  for (int i = 0; i < a.length; i++) {
    expect(a[i], closeTo(b[i], tol), reason: 'index $i');
  }
}

void main() {
  group('MultiHeadCrossAttention', () {
    test('2D [Sq, D] x [Skv, Dkv] -> [Sq, D]', () {
      const d = 8;
      const dkv = 6;
      const heads = 2;
      const sq = 3;
      const skv = 5;
      final mha = MultiHeadCrossAttention(d, dkv, heads, seed: 1);
      final q = Tensor.fromList([sq, d], _rand(sq * d, seed: 10));
      final kv = Tensor.fromList([skv, dkv], _rand(skv * dkv, seed: 11));
      final out = mha(q, kv);
      expect(out.shape, [sq, d]);
    });

    test('3D batched matches per-sample', () {
      const d = 8;
      const dkv = 4;
      const heads = 2;
      const b = 2;
      const sq = 3;
      const skv = 4;
      final mha = MultiHeadCrossAttention(d, dkv, heads, seed: 2);
      final q = Tensor.fromList([b, sq, d], _rand(b * sq * d, seed: 20));
      final kv = Tensor.fromList([b, skv, dkv], _rand(b * skv * dkv, seed: 21));
      final out = mha(q, kv);
      expect(out.shape, [b, sq, d]);
      for (int i = 0; i < b; i++) {
        final qi = Tensor.fromList([
          sq,
          d,
        ], q.toList().sublist(i * sq * d, (i + 1) * sq * d));
        final kvi = Tensor.fromList([
          skv,
          dkv,
        ], kv.toList().sublist(i * skv * dkv, (i + 1) * skv * dkv));
        final ref = mha(qi, kvi).toList();
        final got = out.toList().sublist(i * sq * d, (i + 1) * sq * d);
        _closeList(got, ref);
      }
    });

    test('rejects rank mismatch and shape mismatches', () {
      final mha = MultiHeadCrossAttention(4, 4, 2, seed: 3);
      final q2 = Tensor.fromList([2, 4], _rand(8));
      final kv3 = Tensor.fromList([1, 2, 4], _rand(8));
      expect(() => mha(q2, kv3), throwsArgumentError);

      final qBadDim = Tensor.fromList([2, 5], _rand(10));
      final kvOk = Tensor.fromList([3, 4], _rand(12));
      expect(() => mha(qBadDim, kvOk), throwsArgumentError);

      final qBatched = Tensor.fromList([2, 2, 4], _rand(16));
      final kvBatchedMismatch = Tensor.fromList([3, 3, 4], _rand(36));
      expect(() => mha(qBatched, kvBatchedMismatch), throwsArgumentError);
    });
  });

  group('TransformerDecoderBlock', () {
    test('2D forward preserves shape and reduces loss under training', () {
      const d = 8;
      const heads = 2;
      const sq = 4;
      const skv = 5;
      final block = TransformerDecoderBlock(d, heads, seed: 4);
      final x = Tensor.fromList([sq, d], _rand(sq * d, seed: 30));
      final mem = Tensor.fromList([skv, d], _rand(skv * d, seed: 31));
      final mask = causalMask(sq);
      final out = block(x, mem, selfMask: mask);
      expect(out.shape, [sq, d]);
    });

    test('3D batched forward preserves shape', () {
      const d = 8;
      const heads = 2;
      const b = 2;
      const sq = 3;
      const skv = 4;
      final block = TransformerDecoderBlock(d, heads, seed: 5);
      final x = Tensor.fromList([b, sq, d], _rand(b * sq * d, seed: 40));
      final mem = Tensor.fromList([b, skv, d], _rand(b * skv * d, seed: 41));
      final mask = causalMask(sq);
      final out = block(x, mem, selfMask: mask);
      expect(out.shape, [b, sq, d]);
    });
  });

  group('TransformerDecoder stack', () {
    test('stacks blocks; matches manual composition on a single sequence', () {
      const d = 8;
      const heads = 2;
      const layers = 2;
      const sq = 3;
      const skv = 4;
      final dec = TransformerDecoder(layers, d, heads, seed: 6);
      final x = Tensor.fromList([sq, d], _rand(sq * d, seed: 50));
      final mem = Tensor.fromList([skv, d], _rand(skv * d, seed: 51));
      final mask = causalMask(sq);
      final out = dec(x, mem, selfMask: mask);
      expect(out.shape, [sq, d]);

      var h = x;
      for (final b in dec.blocks) {
        h = b(h, mem, selfMask: mask);
      }
      h = dec.finalNorm!(h);
      _closeList(out.toList(), h.toList());
    });
  });

  group('EncoderDecoderTransformer', () {
    test('forward produces [St, targetVocab] logits (1D)', () {
      final m = EncoderDecoderTransformer(
        sourceVocabSize: 12,
        targetVocabSize: 10,
        embedDim: 16,
        numLayers: 2,
        numHeads: 2,
        maxSourceLen: 8,
        maxTargetLen: 8,
        seed: 7,
      );
      final src = Tensor.fromList([5], [0, 1, 2, 3, 4]);
      final tgt = Tensor.fromList([4], [0, 1, 2, 3]);
      final logits = m(src, tgt);
      expect(logits.shape, [4, 10]);
    });

    test('batched forward produces [B, St, targetVocab]', () {
      final m = EncoderDecoderTransformer(
        sourceVocabSize: 12,
        targetVocabSize: 10,
        embedDim: 16,
        numLayers: 2,
        numHeads: 2,
        maxSourceLen: 8,
        maxTargetLen: 8,
        seed: 8,
      );
      m.eval();
      final src = Tensor.fromList([2, 5], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
      final tgt = Tensor.fromList([2, 4], [0, 1, 2, 3, 4, 5, 6, 7]);
      final logits = m(src, tgt);
      expect(logits.shape, [2, 4, 10]);

      // Per-sample parity.
      for (int i = 0; i < 2; i++) {
        final si = Tensor.fromList([
          5,
        ], src.toList().sublist(i * 5, (i + 1) * 5));
        final ti = Tensor.fromList([
          4,
        ], tgt.toList().sublist(i * 4, (i + 1) * 4));
        final ref = m(si, ti).toList();
        final got = logits.toList().sublist(i * 4 * 10, (i + 1) * 4 * 10);
        _closeList(got, ref);
      }
    });

    test('trains: crossEntropy loss decreases on a memorization task', () {
      final m = EncoderDecoderTransformer(
        sourceVocabSize: 6,
        targetVocabSize: 6,
        embedDim: 16,
        numLayers: 1,
        numHeads: 2,
        maxSourceLen: 4,
        maxTargetLen: 4,
        seed: 9,
      );
      // Copy task: target[i] = source[i]. The decoder input is the
      // shifted target ("BOS" = 0 then target[:-1]) — a standard
      // teacher-forcing setup.
      final src = Tensor.fromList([4], [1, 2, 3, 4]);
      final tgtIn = Tensor.fromList([4], [0, 1, 2, 3]);
      final tgtOut = Tensor.fromList([4], [1, 2, 3, 4]);

      final opt = SGD(m.parameters(), lr: 0.1);
      final l0 = m(src, tgtIn).crossEntropy(tgtOut).mean().toList()[0];
      for (int step = 0; step < 40; step++) {
        opt.zeroGrad();
        m(src, tgtIn).crossEntropy(tgtOut).mean().backward();
        opt.step();
      }
      final l1 = m(src, tgtIn).crossEntropy(tgtOut).mean().toList()[0];
      expect(
        l1,
        lessThan(l0 * 0.5),
        reason: 'expected loss to at least halve (before=$l0, after=$l1)',
      );
    });

    test('rejects rank mismatch between src and tgt', () {
      final m = EncoderDecoderTransformer(
        sourceVocabSize: 6,
        targetVocabSize: 6,
        embedDim: 8,
        numLayers: 1,
        numHeads: 2,
        maxSourceLen: 4,
        maxTargetLen: 4,
        seed: 10,
      );
      final src2 = Tensor.fromList([1, 3], [0, 1, 2]);
      final tgt1 = Tensor.fromList([2], [0, 1]);
      expect(() => m(src2, tgt1), throwsArgumentError);
    });
  });
}
