/// Load a `.cact` archive into a float [NeedleModelF] by dequantising every
/// tensor named by [NeedleWeights.map]. CQ matrices go through
/// [dequantizeMatrix]; fp16/fp32 arrays through the direct readers.
///
/// The `mhc_phi` tensors are quantised along the wide `n*d` axis (so the CQ
/// groups stay 128-aligned) and are transposed here into the `[L, n*d,
/// lanes]` order the forward pass indexes.
library;

import 'dart:typed_data';

import 'cact_format.dart';
import 'cact_quant.dart';
import 'needle_forward.dart';
import 'needle_weights.dart';

NeedleModelF loadNeedleModel(CactFile file) {
  final w = NeedleWeights.map(file);
  final cfg = w.config;
  final n = cfg.numLanes;
  final nd = n * cfg.dModel;

  Float64List cq(CactTensor t) {
    final inDim = t.shape.last;
    final outDim =
        t.shape.sublist(0, t.shape.length - 1).fold<int>(1, (a, b) => a * b);
    return Float64List.fromList(dequantizeMatrix(
      file.blob(t),
      outDim,
      inDim,
      t.bits,
      codebookForBits(file.codebooks, t.bits),
    ));
  }

  Float64List f16(CactTensor t) => readFp16Array(file.blob(t), t.nbytes ~/ 2);
  Float64List f32(CactTensor t) => readFp32Array(file.blob(t), t.nbytes ~/ 4);
  Int32List perm(CactTensor t) =>
      Int32List.fromList([for (final v in f32(t)) v.round()]);

  // phi: stored [L*laneCount, nd] quantised along nd; forward wants
  // [L, nd, laneCount].
  Float64List phi(CactTensor t, int laneCount) {
    final cols = t.shape.last; // nd
    final dq = cq(t);
    final l = cfg.numLayers;
    final out = Float64List(l * nd * laneCount);
    for (var li = 0; li < l; li++) {
      for (var j = 0; j < laneCount; j++) {
        for (var k = 0; k < nd; k++) {
          out[li * nd * laneCount + k * laneCount + j] =
              dq[(li * laneCount + j) * cols + k];
        }
      }
    }
    return out;
  }

  final p1 = perm(w.hadaP1);
  final p2 = perm(w.hadaP2);

  final blocks = [
    for (final b in w.blocks)
      BlockW(
        normIn: f16(b.normIn),
        postNorm: f16(b.postNorm),
        preHada: f16(b.preHada),
        attnGate: f16(b.attnGate)[0],
        attn: AttnW(
          qProj: cq(b.qProj),
          kProj: cq(b.kProj),
          vProj: cq(b.vProj),
          qTaps: f16(b.qTaps),
          kTaps: f16(b.kTaps),
          vTaps: f16(b.vTaps),
          qNorm: f16(b.qNorm),
          kNorm: f16(b.kNorm),
          gateProj: cq(b.gateProj),
          outProj: cq(b.outProj),
        ),
        mlp: MlpW(
          condV: f16(b.condV),
          condU: f16(b.condU),
          d1: f16(b.d1),
          d2: f16(b.d2),
          b2: f16(b.b2),
          d3: f16(b.d3),
          d4: f16(b.d4),
          w1a: f16(b.w1a),
          w1b: f16(b.w1b),
          w2a: f16(b.w2a),
          w2b: f16(b.w2b),
          w3a: f16(b.w3a),
          w3b: f16(b.w3b),
          p1: p1,
          p2: p2,
        ),
      ),
  ];

  final mhc = MhcW(
    aPre: f16(w.mhc.aPre),
    aPost: f16(w.mhc.aPost),
    aRes: f16(w.mhc.aRes),
    bPre: f16(w.mhc.bPre),
    bPost: f16(w.mhc.bPost),
    bRes: f16(w.mhc.bRes),
    phiPre: phi(w.mhc.phiPre, n),
    phiPost: phi(w.mhc.phiPost, n),
    phiRes: phi(w.mhc.phiRes, n * n),
  );

  final engram = [
    for (final e in w.engram)
      EngramW(
        site: e.site,
        tables: cq(e.tables),
        keyProj: cq(e.keyProj),
        valueProj: cq(e.valueProj),
        taps: f16(e.taps),
      ),
  ];

  return NeedleModelF(
    config: cfg,
    embedding: cq(w.embedding),
    blocks: blocks,
    mhc: mhc,
    engram: engram,
    finalNorm: f16(w.finalNorm),
  );
}
