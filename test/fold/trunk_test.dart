library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/core/nn/fold/trunk.dart';
import 'package:test/test.dart';

Float32List _rand(int len, int seed) {
  final rng = math.Random(seed);
  final t = Float32List(len);
  for (var i = 0; i < len; i++) {
    t[i] = rng.nextDouble() * 2 - 1;
  }
  return t;
}

/// Permute the rows of `s` `[n, d]` by `perm` (out[perm[i]] = in[i]).
Float32List _permSeq(Float32List s, int n, int d, List<int> perm) {
  final out = Float32List(n * d);
  for (var i = 0; i < n; i++) {
    for (var c = 0; c < d; c++) {
      out[perm[i] * d + c] = s[i * d + c];
    }
  }
  return out;
}

/// Permute both axes of pair `z` `[n, n, d]` by `perm`.
Float32List _permPair(Float32List z, int n, int d, List<int> perm) {
  final out = Float32List(n * n * d);
  for (var i = 0; i < n; i++) {
    for (var j = 0; j < n; j++) {
      final src = (i * n + j) * d;
      final dst = (perm[i] * n + perm[j]) * d;
      for (var c = 0; c < d; c++) {
        out[dst + c] = z[src + c];
      }
    }
  }
  return out;
}

void main() {
  // Small, head-divisible dims.
  const cfg = FoldingTrunkConfig(
    cS: 8,
    cZ: 4,
    seqHeadWidth: 4, // 2 sequence heads
    pairHeadWidth: 2, // 2 pair heads
  );
  const n = 6;

  test('config head counts', () {
    expect(cfg.seqHeads, 2);
    expect(cfg.pairHeads, 2);
  });

  test('block preserves shapes and stays finite', () {
    final blk = TriangularSelfAttentionBlock.random(cfg, seed: 1);
    final s = _rand(n * cfg.cS, 10);
    final z = _rand(n * n * cfg.cZ, 11);
    final out = blk.call(s, z, n);
    expect(out.s.length, n * cfg.cS);
    expect(out.z.length, n * n * cfg.cZ);
    expect(out.s.every((v) => v.isFinite), isTrue);
    expect(out.z.every((v) => v.isFinite), isTrue);
  });

  test('block is equivariant to a joint residue permutation', () {
    final blk = TriangularSelfAttentionBlock.random(cfg, seed: 2);
    final s = _rand(n * cfg.cS, 20);
    final z = _rand(n * n * cfg.cZ, 21);

    // Reference, then permute the outputs.
    final ref = blk.call(s, z, n);
    final perm = [3, 0, 5, 1, 4, 2];
    final refSperm = _permSeq(ref.s, n, cfg.cS, perm);
    final refZperm = _permPair(ref.z, n, cfg.cZ, perm);

    // Permute the inputs, then run the block.
    final sP = _permSeq(s, n, cfg.cS, perm);
    final zP = _permPair(z, n, cfg.cZ, perm);
    final outP = blk.call(sP, zP, n);

    for (var i = 0; i < refSperm.length; i++) {
      expect(outP.s[i], closeTo(refSperm[i], 1e-4), reason: 's[$i]');
    }
    for (var i = 0; i < refZperm.length; i++) {
      expect(outP.z[i], closeTo(refZperm[i], 1e-4), reason: 'z[$i]');
    }
  });

  test('stacking multiple blocks remains finite', () {
    var s = _rand(n * cfg.cS, 30);
    var z = _rand(n * n * cfg.cZ, 31);
    for (var b = 0; b < 4; b++) {
      final blk = TriangularSelfAttentionBlock.random(cfg, seed: 100 + b);
      final out = blk.call(s, z, n);
      s = out.s;
      z = out.z;
    }
    expect(s.every((v) => v.isFinite), isTrue);
    expect(z.every((v) => v.isFinite), isTrue);
  });
}
