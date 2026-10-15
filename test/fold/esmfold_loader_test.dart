library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/core/nn/fold/esmfold_loader.dart';
import 'package:dart_pytorch/core/nn/fold/structure_module.dart';
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

void main() {
  const cfg = FoldingTrunkConfig(
    cS: 8,
    cZ: 4,
    seqHeadWidth: 4,
    pairHeadWidth: 2,
  );
  const n = 5;

  test('trunk block round-trips through the HF key map', () {
    final src = TriangularSelfAttentionBlock.random(cfg, seed: 1);
    final dst = TriangularSelfAttentionBlock.random(cfg, seed: 999);

    final map = dumpTrunkBlock(src, 'trunk.blocks.0.');
    final consumed = <String>{};
    loadTrunkBlock(dst, map, 'trunk.blocks.0.', consumed);

    // Every dumped key was consumed (schema fully covers the module).
    expect(consumed, containsAll(map.keys));
    expect(consumed.length, map.length);

    // The loaded module must now be numerically identical.
    final s = _rand(n * cfg.cS, 10);
    final z = _rand(n * n * cfg.cZ, 11);
    final a = src.call(s, z, n);
    final b = dst.call(s, z, n);
    for (var i = 0; i < a.s.length; i++) {
      expect(b.s[i], equals(a.s[i]), reason: 's[$i]');
    }
    for (var i = 0; i < a.z.length; i++) {
      expect(b.z[i], equals(a.z[i]), reason: 'z[$i]');
    }
  });

  test('trunk block map carries the expected HF keys', () {
    final blk = TriangularSelfAttentionBlock.random(cfg, seed: 2);
    final map = dumpTrunkBlock(blk, 'trunk.blocks.7.');
    for (final k in [
      'trunk.blocks.7.layernorm_1.weight',
      'trunk.blocks.7.sequence_to_pair.o_proj.bias',
      'trunk.blocks.7.pair_to_sequence.linear.weight',
      'trunk.blocks.7.seq_attention.proj.weight',
      'trunk.blocks.7.tri_mul_out.linear_z.weight',
      'trunk.blocks.7.tri_mul_in.layer_norm_in.bias',
      'trunk.blocks.7.tri_att_start.mha.linear_q.weight',
      'trunk.blocks.7.tri_att_end.mha.linear_g.bias',
      'trunk.blocks.7.mlp_seq.mlp.1.weight',
      'trunk.blocks.7.mlp_pair.mlp.3.bias',
    ]) {
      expect(map.containsKey(k), isTrue, reason: 'missing $k');
    }
    // seq_attention.proj has no bias in HF.
    expect(map.containsKey('trunk.blocks.7.seq_attention.proj.bias'), isFalse);
  });

  group('structure module', () {
    const ipaCfg = IpaConfig(
      cS: 16,
      cZ: 8,
      cHidden: 4,
      noHeads: 2,
      noQkPoints: 3,
      noVPoints: 4,
    );
    const nn = 5;

    test('round-trips through the HF key map', () {
      final src = StructureModule.random(ipaCfg, noBlocks: 3, seed: 3);
      final dst = StructureModule.random(ipaCfg, noBlocks: 3, seed: 888);

      final map = dumpStructureModule(src, 'trunk.structure_module.');
      final consumed = <String>{};
      loadStructureModule(dst, map, 'trunk.structure_module.', consumed);
      expect(consumed.length, map.length);

      final s = _rand(nn * ipaCfg.cS, 20);
      final z = _rand(nn * nn * ipaCfg.cZ, 21);
      final a = src.call(s, z);
      final b = dst.call(s, z);
      for (var i = 0; i < nn; i++) {
        expect(b.atoms[i].cAlpha.x, equals(a.atoms[i].cAlpha.x));
        expect(b.atoms[i].n.y, equals(a.atoms[i].n.y));
        expect(b.atoms[i].c.z, equals(a.atoms[i].c.z));
      }
    });

    test('carries the expected structure-module keys', () {
      final sm = StructureModule.random(ipaCfg, noBlocks: 2, seed: 4);
      final map = dumpStructureModule(sm, 'trunk.structure_module.');
      for (final k in [
        'trunk.structure_module.layer_norm_s.weight',
        'trunk.structure_module.linear_in.bias',
        'trunk.structure_module.ipa.linear_q.weight',
        'trunk.structure_module.ipa.linear_kv_points.bias',
        'trunk.structure_module.ipa.linear_out.weight',
        'trunk.structure_module.ipa.head_weights',
        'trunk.structure_module.transition.layers.0.linear_2.weight',
        'trunk.structure_module.transition.layer_norm.bias',
        'trunk.structure_module.bb_update.linear.weight',
      ]) {
        expect(map.containsKey(k), isTrue, reason: 'missing $k');
      }
      expect(map['trunk.structure_module.bb_update.linear.weight']!.shape,
          [6, ipaCfg.cS]);
      expect(
          map['trunk.structure_module.ipa.head_weights']!.shape, [ipaCfg.noHeads]);
    });
  });
}
