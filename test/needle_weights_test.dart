import 'dart:typed_data';

import 'package:dart_pytorch/core/nn/needle/cact_format.dart';
import 'package:dart_pytorch/core/nn/needle/needle_config.dart';
import 'package:dart_pytorch/core/nn/needle/needle_weights.dart';
import 'package:test/test.dart';

/// Build a synthetic `.cact` directory that matches the Needle 3 contract,
/// so the walker can be validated without a 29 MB weights file.
class _DirBuilder {
  final List<CactTensor> recs = [];
  CactTensor add(CactDType dt, List<int> shape, {int bits = 0}) {
    final t = CactTensor(
      index: recs.length,
      dtype: dt,
      ndim: shape.length,
      shape: shape,
      offset: 0,
      nbytes: shape.fold(1, (a, b) => a * b),
      group: 128,
      bits: bits,
    );
    recs.add(t);
    return t;
  }
}

CactFile _synthFile({int layers = 20}) {
  final b = _DirBuilder();
  b.add(CactDType.cq, [8192, 768], bits: 4); // embedding

  for (var l = 0; l < layers; l++) {
    b.add(CactDType.fp16, [768]); // norm_in
    b.add(CactDType.cq, [576, 768], bits: 2); // q_proj
    b.add(CactDType.cq, [96, 768], bits: 2); // k_proj
    b.add(CactDType.cq, [128, 768], bits: 2); // v_proj
    b.add(CactDType.fp16, [3, 576]); // q_taps
    b.add(CactDType.fp16, [3, 96]); // k_taps
    b.add(CactDType.fp16, [3, 128]); // v_taps
    b.add(CactDType.fp16, [48]); // q_norm
    b.add(CactDType.fp16, [48]); // k_norm
    b.add(CactDType.cq, [768, 768], bits: 2); // gate_proj
    b.add(CactDType.cq, [768, 768], bits: 2); // out_proj
    b.add(CactDType.fp16, [768]); // post_norm
    b.add(CactDType.fp16, [1]); // attn_gate
    b.add(CactDType.fp16, [768]); // pre_hada
    for (var k = 0; k < 5; k++) {
      b.add(CactDType.fp16, [1024]); // d1,d2,b2,d3,d4
    }
    for (var k = 0; k < 6; k++) {
      b.add(CactDType.fp16, [32, 32]); // w1a..w3b
    }
    b.add(CactDType.fp16, [768, 8]); // cond_v
    b.add(CactDType.fp16, [8, 1024]); // cond_u
  }

  // mHC: 6 fp16 lane maps, 3 cq4 phi.
  for (var k = 0; k < 6; k++) {
    b.add(CactDType.fp16, [4]);
  }
  for (var k = 0; k < 3; k++) {
    b.add(CactDType.cq, [layers * 4, 768], bits: 4);
  }

  b.add(CactDType.fp32, [1024]); // hada_p1
  b.add(CactDType.fp32, [1024]); // hada_p2

  final sites = NeedleConfig(
    vocab: 8192, dModel: 768, numLayers: layers, kvBits: 8, ropeTheta: 1e5,
  ).activeEngramSites();
  for (var s = 0; s < sites.length; s++) {
    b.add(CactDType.cq, [2, 18432], bits: 2); // tables
    b.add(CactDType.cq, [128, 768], bits: 2); // key_proj
    b.add(CactDType.cq, [768, 128], bits: 2); // value_proj
    b.add(CactDType.fp16, [3, 128]); // taps
  }

  b.add(CactDType.fp16, [768]); // final_norm
  b.add(CactDType.fp16, [1]); // heads.manifest (1 head)
  // one confidence head, 6 tensors
  b.add(CactDType.cq, [4, 768], bits: 4); // probes
  b.add(CactDType.fp16, [4]); // gain
  b.add(CactDType.cq, [4, 768], bits: 4); // query
  b.add(CactDType.fp16, [4]); // row_bias
  b.add(CactDType.cq, [768, 768], bits: 4); // proj
  b.add(CactDType.fp16, [768]); // bias
  b.add(CactDType.raw, [111000]); // tokenizer

  final raw = List<int>.filled(49, 0);
  raw[0] = kCactTagNeedle3;
  raw[1] = b.recs.length; // nTensors
  raw[2] = 28; // codebookLen
  raw[4] = 8; // kv_bits
  raw[5] = 8192; // vocab
  raw[7] = 768; // d_model
  raw[10] = layers; // layers
  final header = CactHeader(rawFields: raw, ropeTheta: 100000.0);
  return CactFile(
    header: header,
    codebooks: Float32List(28),
    directory: b.recs,
  );
}

void main() {
  group('NeedleConfig ladder rule', () {
    const cfg = NeedleConfig(
      vocab: 8192, dModel: 768, numLayers: 20, kvBits: 8, ropeTheta: 1e5,
    );
    test('ladderBlocks matches the published selections', () {
      expect(NeedleConfig.ladderBlocks(2), [0, 19]);
      expect(NeedleConfig.ladderBlocks(4), [0, 9, 14, 19]);
      expect(NeedleConfig.ladderBlocks(8), [0, 4, 6, 9, 11, 14, 16, 19]);
    });
    test('full model keeps all engram sites and global layers', () {
      expect(cfg.activeEngramSites(), [3, 7, 11, 15, 19]);
      expect(cfg.activeGlobalLayers(), [4, 9, 14, 19]);
    });
    test('projection dims follow head geometry', () {
      expect(cfg.qProjDim, 576);
      expect(cfg.kProjDim, 96);
      expect(cfg.vProjDim, 128);
    });
  });

  group('NeedleWeights.map on a synthetic contract directory', () {
    test('shipped 20-layer file has 581 records and slices cleanly', () {
      final file = _synthFile(layers: 20);
      expect(file.directory.length, 581);
      final w = NeedleWeights.map(file);
      expect(w.blocks.length, 20);
      expect(w.engram.length, 5);
      expect(w.heads.length, 1);
      expect(w.embedding.dtype, CactDType.cq);
      expect(w.tokenizer.dtype, CactDType.raw);
      expect(w.hadaP1.dtype, CactDType.fp32);
      expect(w.blocks.first.qProj.shape, [576, 768]);
      expect(w.engram.map((e) => e.site), [3, 7, 11, 15, 19]);
    });

    test('a reordered directory is refused', () {
      final file = _synthFile(layers: 20);
      // Swap embedding (cq) with the first block norm (fp16).
      final d = List<CactTensor>.of(file.directory);
      final tmp = d[0];
      d[0] = d[1];
      d[1] = tmp;
      final bad = CactFile(
        header: file.header,
        codebooks: file.codebooks,
        directory: d,
      );
      expect(() => NeedleWeights.map(bad), throwsFormatException);
    });
  });
}
