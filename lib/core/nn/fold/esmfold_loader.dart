/// Pretrained-weight loader for HuggingFace `facebook/esmfold_v1`.
///
/// Maps the HF `EsmForProteinFolding` safetensors state dict onto the plain-Dart
/// folding-trunk and structure-module modules in this package. Key strings are
/// taken verbatim from `transformers/models/esm/modeling_esmfold.py`, so a real
/// checkpoint loads without renaming. The ESM-2 3B stem is loaded separately via
/// [ESM2HFLoader] with the `esm.` prefix.
///
/// Each function also has a `dump*` counterpart that serializes a module back to
/// an HF-keyed map; together they enable a weightless round-trip test of the
/// mapping (every tensor placed, shape-checked, and recovered bit-for-bit).
library;

import 'dart:typed_data';

import '../../tensor/tensor.dart';
import 'structure_module.dart';
import 'trunk.dart';

/// Thrown when a checkpoint tensor is missing or mis-shaped.
class EsmfoldLoadError implements Exception {
  final String message;
  EsmfoldLoadError(this.message);
  @override
  String toString() => 'EsmfoldLoadError: $message';
}

// --------------------------------------------------------------------------
// Low-level copy helpers
// --------------------------------------------------------------------------

void _copyInto(Float32List dst, Map<String, Tensor> m, String key,
    Set<String> consumed) {
  final t = m[key];
  if (t == null) throw EsmfoldLoadError('missing tensor "$key"');
  final s = t.toFloat32List();
  if (s.length != dst.length) {
    throw EsmfoldLoadError(
        'shape mismatch for "$key": got ${s.length}, want ${dst.length}');
  }
  dst.setAll(0, s);
  consumed.add(key);
}

void _loadDense(Dense d, Map<String, Tensor> m, String p, Set<String> c) {
  _copyInto(d.weight, m, '$p.weight', c);
  if (d.bias != null) _copyInto(d.bias!, m, '$p.bias', c);
}

void _loadLn(Float32List g, Float32List b, Map<String, Tensor> m, String p,
    Set<String> c) {
  _copyInto(g, m, '$p.weight', c);
  _copyInto(b, m, '$p.bias', c);
}

void _dumpDense(Dense d, Map<String, Tensor> m, String p) {
  m['$p.weight'] = Tensor.fromList([d.outDim, d.inDim], d.weight.toList());
  if (d.bias != null) {
    m['$p.bias'] = Tensor.fromList([d.outDim], d.bias!.toList());
  }
}

void _dumpLn(Float32List g, Float32List b, Map<String, Tensor> m, String p) {
  m['$p.weight'] = Tensor.fromList([g.length], g.toList());
  m['$p.bias'] = Tensor.fromList([b.length], b.toList());
}

// --------------------------------------------------------------------------
// Trunk block
// --------------------------------------------------------------------------

void loadTrunkBlock(TriangularSelfAttentionBlock blk, Map<String, Tensor> m,
    String p, Set<String> consumed) {
  _loadLn(blk.ln1G, blk.ln1B, m, '${p}layernorm_1', consumed);

  final s2p = blk.sequenceToPair;
  _loadLn(s2p.lnGamma, s2p.lnBeta, m, '${p}sequence_to_pair.layernorm', consumed);
  _loadDense(s2p.proj, m, '${p}sequence_to_pair.proj', consumed);
  _loadDense(s2p.oProj, m, '${p}sequence_to_pair.o_proj', consumed);

  final p2s = blk.pairToSequence;
  _loadLn(p2s.lnGamma, p2s.lnBeta, m, '${p}pair_to_sequence.layernorm', consumed);
  _loadDense(p2s.linear, m, '${p}pair_to_sequence.linear', consumed);

  final sa = blk.seqAttention;
  _loadDense(sa.proj, m, '${p}seq_attention.proj', consumed);
  _loadDense(sa.oProj, m, '${p}seq_attention.o_proj', consumed);
  _loadDense(sa.gProj, m, '${p}seq_attention.g_proj', consumed);

  _loadTriMul(blk.triMulOut, m, '${p}tri_mul_out', consumed);
  _loadTriMul(blk.triMulIn, m, '${p}tri_mul_in', consumed);
  _loadTriAtt(blk.triAttStart, m, '${p}tri_att_start', consumed);
  _loadTriAtt(blk.triAttEnd, m, '${p}tri_att_end', consumed);

  _loadMlp(blk.mlpSeq, m, '${p}mlp_seq', consumed);
  _loadMlp(blk.mlpPair, m, '${p}mlp_pair', consumed);
}

void _loadTriMul(TriangleMultiplication t, Map<String, Tensor> m, String p,
    Set<String> c) {
  _loadLn(t.lnInG, t.lnInB, m, '$p.layer_norm_in', c);
  _loadLn(t.lnOutG, t.lnOutB, m, '$p.layer_norm_out', c);
  _loadDense(t.aP, m, '$p.linear_a_p', c);
  _loadDense(t.aG, m, '$p.linear_a_g', c);
  _loadDense(t.bP, m, '$p.linear_b_p', c);
  _loadDense(t.bG, m, '$p.linear_b_g', c);
  _loadDense(t.g, m, '$p.linear_g', c);
  _loadDense(t.z, m, '$p.linear_z', c);
}

void _loadTriAtt(TriangleAttention t, Map<String, Tensor> m, String p,
    Set<String> c) {
  _loadLn(t.lnG, t.lnB, m, '$p.layer_norm', c);
  _loadDense(t.linearBias, m, '$p.linear', c);
  _loadDense(t.q, m, '$p.mha.linear_q', c);
  _loadDense(t.k, m, '$p.mha.linear_k', c);
  _loadDense(t.v, m, '$p.mha.linear_v', c);
  _loadDense(t.o, m, '$p.mha.linear_o', c);
  _loadDense(t.gate, m, '$p.mha.linear_g', c);
}

void _loadMlp(ResidueMLP r, Map<String, Tensor> m, String p, Set<String> c) {
  _loadLn(r.lnGamma, r.lnBeta, m, '$p.mlp.0', c);
  _loadDense(r.l1, m, '$p.mlp.1', c);
  _loadDense(r.l2, m, '$p.mlp.3', c);
}

Map<String, Tensor> dumpTrunkBlock(
    TriangularSelfAttentionBlock blk, String p) {
  final m = <String, Tensor>{};
  _dumpLn(blk.ln1G, blk.ln1B, m, '${p}layernorm_1');
  final s2p = blk.sequenceToPair;
  _dumpLn(s2p.lnGamma, s2p.lnBeta, m, '${p}sequence_to_pair.layernorm');
  _dumpDense(s2p.proj, m, '${p}sequence_to_pair.proj');
  _dumpDense(s2p.oProj, m, '${p}sequence_to_pair.o_proj');
  final p2s = blk.pairToSequence;
  _dumpLn(p2s.lnGamma, p2s.lnBeta, m, '${p}pair_to_sequence.layernorm');
  _dumpDense(p2s.linear, m, '${p}pair_to_sequence.linear');
  final sa = blk.seqAttention;
  _dumpDense(sa.proj, m, '${p}seq_attention.proj');
  _dumpDense(sa.oProj, m, '${p}seq_attention.o_proj');
  _dumpDense(sa.gProj, m, '${p}seq_attention.g_proj');
  _dumpTriMul(blk.triMulOut, m, '${p}tri_mul_out');
  _dumpTriMul(blk.triMulIn, m, '${p}tri_mul_in');
  _dumpTriAtt(blk.triAttStart, m, '${p}tri_att_start');
  _dumpTriAtt(blk.triAttEnd, m, '${p}tri_att_end');
  _dumpMlp(blk.mlpSeq, m, '${p}mlp_seq');
  _dumpMlp(blk.mlpPair, m, '${p}mlp_pair');
  return m;
}

void _dumpTriMul(TriangleMultiplication t, Map<String, Tensor> m, String p) {
  _dumpLn(t.lnInG, t.lnInB, m, '$p.layer_norm_in');
  _dumpLn(t.lnOutG, t.lnOutB, m, '$p.layer_norm_out');
  _dumpDense(t.aP, m, '$p.linear_a_p');
  _dumpDense(t.aG, m, '$p.linear_a_g');
  _dumpDense(t.bP, m, '$p.linear_b_p');
  _dumpDense(t.bG, m, '$p.linear_b_g');
  _dumpDense(t.g, m, '$p.linear_g');
  _dumpDense(t.z, m, '$p.linear_z');
}

void _dumpTriAtt(TriangleAttention t, Map<String, Tensor> m, String p) {
  _dumpLn(t.lnG, t.lnB, m, '$p.layer_norm');
  _dumpDense(t.linearBias, m, '$p.linear');
  _dumpDense(t.q, m, '$p.mha.linear_q');
  _dumpDense(t.k, m, '$p.mha.linear_k');
  _dumpDense(t.v, m, '$p.mha.linear_v');
  _dumpDense(t.o, m, '$p.mha.linear_o');
  _dumpDense(t.gate, m, '$p.mha.linear_g');
}

void _dumpMlp(ResidueMLP r, Map<String, Tensor> m, String p) {
  _dumpLn(r.lnGamma, r.lnBeta, m, '$p.mlp.0');
  _dumpDense(r.l1, m, '$p.mlp.1');
  _dumpDense(r.l2, m, '$p.mlp.3');
}

// --------------------------------------------------------------------------
// Structure module
// --------------------------------------------------------------------------

void loadStructureModule(
    StructureModule sm, Map<String, Tensor> m, String p, Set<String> consumed) {
  _loadLn(sm.lnSGamma, sm.lnSBeta, m, '${p}layer_norm_s', consumed);
  _loadLn(sm.lnZGamma, sm.lnZBeta, m, '${p}layer_norm_z', consumed);
  _loadDense(sm.linearIn, m, '${p}linear_in', consumed);

  final ipa = sm.ipa;
  _loadDense(ipa.linearQ, m, '${p}ipa.linear_q', consumed);
  _loadDense(ipa.linearKv, m, '${p}ipa.linear_kv', consumed);
  _loadDense(ipa.linearQPoints, m, '${p}ipa.linear_q_points', consumed);
  _loadDense(ipa.linearKvPoints, m, '${p}ipa.linear_kv_points', consumed);
  _loadDense(ipa.linearB, m, '${p}ipa.linear_b', consumed);
  _loadDense(ipa.linearOut, m, '${p}ipa.linear_out', consumed);
  _copyInto(ipa.headWeights, m, '${p}ipa.head_weights', consumed);

  _loadLn(sm.lnIpaGamma, sm.lnIpaBeta, m, '${p}layer_norm_ipa', consumed);

  final tr = sm.transition;
  _loadDense(tr.l1, m, '${p}transition.layers.0.linear_1', consumed);
  _loadDense(tr.l2, m, '${p}transition.layers.0.linear_2', consumed);
  _loadDense(tr.l3, m, '${p}transition.layers.0.linear_3', consumed);
  _loadLn(tr.lnGamma, tr.lnBeta, m, '${p}transition.layer_norm', consumed);

  _loadDense(sm.bbUpdate.linear, m, '${p}bb_update.linear', consumed);
}

Map<String, Tensor> dumpStructureModule(StructureModule sm, String p) {
  final m = <String, Tensor>{};
  _dumpLn(sm.lnSGamma, sm.lnSBeta, m, '${p}layer_norm_s');
  _dumpLn(sm.lnZGamma, sm.lnZBeta, m, '${p}layer_norm_z');
  _dumpDense(sm.linearIn, m, '${p}linear_in');
  final ipa = sm.ipa;
  _dumpDense(ipa.linearQ, m, '${p}ipa.linear_q');
  _dumpDense(ipa.linearKv, m, '${p}ipa.linear_kv');
  _dumpDense(ipa.linearQPoints, m, '${p}ipa.linear_q_points');
  _dumpDense(ipa.linearKvPoints, m, '${p}ipa.linear_kv_points');
  _dumpDense(ipa.linearB, m, '${p}ipa.linear_b');
  _dumpDense(ipa.linearOut, m, '${p}ipa.linear_out');
  m['${p}ipa.head_weights'] =
      Tensor.fromList([ipa.headWeights.length], ipa.headWeights.toList());
  _dumpLn(sm.lnIpaGamma, sm.lnIpaBeta, m, '${p}layer_norm_ipa');
  final tr = sm.transition;
  _dumpDense(tr.l1, m, '${p}transition.layers.0.linear_1');
  _dumpDense(tr.l2, m, '${p}transition.layers.0.linear_2');
  _dumpDense(tr.l3, m, '${p}transition.layers.0.linear_3');
  _dumpLn(tr.lnGamma, tr.lnBeta, m, '${p}transition.layer_norm');
  _dumpDense(sm.bbUpdate.linear, m, '${p}bb_update.linear');
  return m;
}
