@Timeout(Duration(minutes: 3))
library;

import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

/// Zero out the biases our `Expert` allocates by default so the roundtrip
/// compare doesn't drift on random init that HF checkpoints don't ship.
void _zeroExpertBiases(MoEFeedForward moe) {
  for (final e in [...moe.routedExperts, ...moe.sharedExperts]) {
    if (e.w1.bias != null) {
      final n = e.w1.bias!.length;
      e.w1.bias!.assign(Tensor.fromList(
          e.w1.bias!.shape, List<double>.filled(n, 0.0),
          device: e.w1.bias!.device));
    }
    if (e.w2.bias != null) {
      final n = e.w2.bias!.length;
      e.w2.bias!.assign(Tensor.fromList(
          e.w2.bias!.shape, List<double>.filled(n, 0.0),
          device: e.w2.bias!.device));
    }
    if (e.w3 != null && e.w3!.bias != null) {
      final n = e.w3!.bias!.length;
      e.w3!.bias!.assign(Tensor.fromList(
          e.w3!.bias!.shape, List<double>.filled(n, 0.0),
          device: e.w3!.bias!.device));
    }
  }
}

/// Same tiny config as `test/deepseek_v2_test.dart`.
const _tinyCfg = DeepSeekV2Config(
  vocabSize: 128,
  maxCtx: 32,
  embedDim: 32,
  numLayers: 2,
  firstKDenseReplace: 1,
  denseFfnDim: 64,
  moeExpertHiddenDim: 32,
  numRoutedExperts: 4,
  numSharedExperts: 1,
  numExpertsPerTok: 2,
  numExpertGroups: 1,
  topKGroups: 1,
  mlaConfig: MLAConfig(
    embedDim: 32,
    numHeads: 4,
    qLoraRank: null,
    kvLoraRank: 12,
    qkNopeHeadDim: 6,
    qkRopeHeadDim: 4,
    vHeadDim: 6,
  ),
);

/// Same tiny config but with Q compression, exercising the
/// `q_a_proj + q_a_layernorm + q_b_proj` path.
const _tinyCompressedCfg = DeepSeekV2Config(
  vocabSize: 64,
  maxCtx: 16,
  embedDim: 32,
  numLayers: 2,
  firstKDenseReplace: 1,
  denseFfnDim: 64,
  moeExpertHiddenDim: 32,
  numRoutedExperts: 4,
  numSharedExperts: 1,
  numExpertsPerTok: 2,
  numExpertGroups: 1,
  topKGroups: 1,
  mlaConfig: MLAConfig(
    embedDim: 32,
    numHeads: 4,
    qLoraRank: 16, // compressed
    kvLoraRank: 12,
    qkNopeHeadDim: 6,
    qkRopeHeadDim: 4,
    vHeadDim: 6,
  ),
);

/// Build a HF-style DeepSeek-V2 state_dict from an in-memory model so
/// the loader can be roundtripped without downloading real weights.
Map<String, Tensor> _dumpToHF(DeepSeekV2Model m) {
  // HF DeepSeek-V2 checkpoints ship no biases on the MoE experts —
  // zero them here so the source model matches what the loader
  // reconstructs.
  for (final block in m.blocks) {
    if (block.moeFfn != null) {
      _zeroExpertBiases(block.moeFfn!);
    }
  }
  final state = <String, Tensor>{};
  final cfg = m.config;
  final mla = cfg.mlaConfig;
  final h = mla.numHeads;
  final nopeD = mla.qkNopeHeadDim;
  final ropeD = mla.qkRopeHeadDim;
  final vD = mla.vHeadDim;
  final headDim = mla.qkHeadDim;
  final kvL = mla.kvLoraRank;
  final d = cfg.embedDim;

  state['model.embed_tokens.weight'] = m.embedIn.weight;

  Tensor concatRows(List<Tensor> parts, int totalRows, int cols) {
    final buf = Float32List(totalRows * cols);
    var row = 0;
    for (final p in parts) {
      final vals = p.toList();
      final rowsHere = p.shape[0];
      for (int i = 0; i < rowsHere * cols; i++) {
        buf[row * cols + i] = vals[i];
      }
      row += rowsHere;
    }
    return Tensor.fromFloat32List([totalRows, cols], buf);
  }

  Tensor concatCols(List<Tensor> parts, int rows, int totalCols) {
    final buf = Float32List(rows * totalCols);
    var colBase = 0;
    for (final p in parts) {
      final cs = p.shape[1];
      final vals = p.toList();
      for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cs; c++) {
          buf[r * totalCols + colBase + c] = vals[r * cs + c];
        }
      }
      colBase += cs;
    }
    return Tensor.fromFloat32List([rows, totalCols], buf);
  }

  Tensor transpose2D(Tensor t) {
    final r = t.shape[0];
    final c = t.shape[1];
    final data = t.toList();
    final buf = Float32List(r * c);
    for (int i = 0; i < r; i++) {
      for (int j = 0; j < c; j++) {
        buf[j * r + i] = data[i * c + j];
      }
    }
    return Tensor.fromFloat32List([c, r], buf);
  }

  for (int i = 0; i < cfg.numLayers; i++) {
    final block = m.blocks[i];
    final p = 'model.layers.$i';
    state['$p.input_layernorm.weight'] = block.attnLn.gamma;
    state['$p.post_attention_layernorm.weight'] = block.ffnLn.gamma;

    // ---- MLA ----
    final attn = block.attn;
    // Per-head fused Q: rows [head_h] = [nope | rope].
    final qRowsPerHead = <Tensor>[];
    for (int hh = 0; hh < h; hh++) {
      qRowsPerHead.add(concatRows(
        [attn.qUpNope[hh].weight, attn.qUpRope[hh].weight],
        headDim,
        mla.qInDim,
      ));
    }
    final qFused = concatRows(qRowsPerHead, h * headDim, mla.qInDim);
    if (mla.qLoraRank != null) {
      state['$p.self_attn.q_a_proj.weight'] = attn.qDown!.weight;
      state['$p.self_attn.q_a_layernorm.weight'] = attn.qLn!.gamma;
      state['$p.self_attn.q_b_proj.weight'] = qFused;
    } else {
      state['$p.self_attn.q_proj.weight'] = qFused;
    }
    // KV low-rank fused: rows [0, kvL) is kvDown, [kvL, kvL+ropeD) is kRope.
    state['$p.self_attn.kv_a_proj_with_mqa.weight'] = concatRows(
      [attn.kvDown.weight, attn.kRope.weight],
      kvL + ropeD,
      d,
    );
    state['$p.self_attn.kv_a_layernorm.weight'] = attn.kvLn.gamma;
    // kv_b_proj: per-head [nope | v].
    final kvRowsPerHead = <Tensor>[];
    for (int hh = 0; hh < h; hh++) {
      kvRowsPerHead.add(concatRows(
        [attn.kUpNope[hh].weight, attn.vUp[hh].weight],
        nopeD + vD,
        kvL,
      ));
    }
    state['$p.self_attn.kv_b_proj.weight'] =
        concatRows(kvRowsPerHead, h * (nopeD + vD), kvL);
    state['$p.self_attn.o_proj.weight'] = attn.oProj.weight;

    // ---- FFN ----
    if (!block.isMoE) {
      final ffn = block.denseFfn!;
      state['$p.mlp.gate_proj.weight'] = ffn.gateProj.weight;
      state['$p.mlp.up_proj.weight'] = ffn.upProj.weight;
      state['$p.mlp.down_proj.weight'] = ffn.downProj.weight;
    } else {
      final moe = block.moeFfn!;
      // Router: our gateW is [D, E]; HF wants [E, D].
      state['$p.mlp.gate.weight'] = transpose2D(moe.gateW);
      for (int j = 0; j < cfg.numRoutedExperts; j++) {
        final e = moe.routedExperts[j];
        final ep = '$p.mlp.experts.$j';
        state['$ep.gate_proj.weight'] = e.w1.weight;
        state['$ep.up_proj.weight'] = e.w3!.weight;
        state['$ep.down_proj.weight'] = e.w2.weight;
      }
      // Shared experts: our N separate bodies fused into one big body.
      final sharedGateParts = <Tensor>[
        for (final e in moe.sharedExperts) e.w1.weight
      ];
      final sharedUpParts = <Tensor>[
        for (final e in moe.sharedExperts) e.w3!.weight
      ];
      final sharedDownParts = <Tensor>[
        for (final e in moe.sharedExperts) e.w2.weight
      ];
      final sharedFfn = cfg.numSharedExperts * cfg.moeExpertHiddenDim;
      state['$p.mlp.shared_experts.gate_proj.weight'] =
          concatRows(sharedGateParts, sharedFfn, d);
      state['$p.mlp.shared_experts.up_proj.weight'] =
          concatRows(sharedUpParts, sharedFfn, d);
      state['$p.mlp.shared_experts.down_proj.weight'] =
          concatCols(sharedDownParts, d, sharedFfn);
    }
  }

  state['model.norm.weight'] = m.finalNorm.gamma;
  if (!cfg.tieWordEmbeddings) {
    state['lm_head.weight'] = m.untiedHead!.weight;
  }
  return state;
}

void main() {
  group('DeepSeekV2HFLoader roundtrip', () {
    test('Lite-style (no Q compression) — outputs match after load', () {
      final src = DeepSeekV2Model(_tinyCfg);
      final state = _dumpToHF(src);

      final dst = DeepSeekV2Model(DeepSeekV2Config(
        vocabSize: _tinyCfg.vocabSize,
        maxCtx: _tinyCfg.maxCtx,
        embedDim: _tinyCfg.embedDim,
        numLayers: _tinyCfg.numLayers,
        firstKDenseReplace: _tinyCfg.firstKDenseReplace,
        denseFfnDim: _tinyCfg.denseFfnDim,
        moeExpertHiddenDim: _tinyCfg.moeExpertHiddenDim,
        numRoutedExperts: _tinyCfg.numRoutedExperts,
        numSharedExperts: _tinyCfg.numSharedExperts,
        numExpertsPerTok: _tinyCfg.numExpertsPerTok,
        mlaConfig: _tinyCfg.mlaConfig,
        seed: 999, // different init
      ));

      final report = DeepSeekV2HFLoader.loadMap(dst, state);
      expect(report.unusedKeys, isEmpty);

      final tokens = Tensor.fromList([4], [1.0, 5.0, 9.0, 13.0]);
      final srcOut = src(tokens).toList();
      final dstOut = dst(tokens).toList();
      expect(srcOut.length, dstOut.length);
      for (int i = 0; i < srcOut.length; i++) {
        expect(
          (srcOut[i] - dstOut[i]).abs() < 1e-4,
          isTrue,
          reason: 'i=$i src=${srcOut[i]} dst=${dstOut[i]}',
        );
      }
    });

    test('Compressed-Q path — outputs match after load', () {
      final src = DeepSeekV2Model(_tinyCompressedCfg);
      final state = _dumpToHF(src);
      // Should include the q_a_* + q_b_* keys.
      expect(
          state.containsKey('model.layers.0.self_attn.q_a_proj.weight'), isTrue);
      expect(state.containsKey('model.layers.0.self_attn.q_proj.weight'),
          isFalse);

      final dst = DeepSeekV2Model(DeepSeekV2Config(
        vocabSize: _tinyCompressedCfg.vocabSize,
        maxCtx: _tinyCompressedCfg.maxCtx,
        embedDim: _tinyCompressedCfg.embedDim,
        numLayers: _tinyCompressedCfg.numLayers,
        firstKDenseReplace: _tinyCompressedCfg.firstKDenseReplace,
        denseFfnDim: _tinyCompressedCfg.denseFfnDim,
        moeExpertHiddenDim: _tinyCompressedCfg.moeExpertHiddenDim,
        numRoutedExperts: _tinyCompressedCfg.numRoutedExperts,
        numSharedExperts: _tinyCompressedCfg.numSharedExperts,
        numExpertsPerTok: _tinyCompressedCfg.numExpertsPerTok,
        mlaConfig: _tinyCompressedCfg.mlaConfig,
        seed: 42,
      ));
      final report = DeepSeekV2HFLoader.loadMap(dst, state);
      expect(report.unusedKeys, isEmpty);

      final tokens = Tensor.fromList([3], [1.0, 5.0, 9.0]);
      final srcOut = src(tokens).toList();
      final dstOut = dst(tokens).toList();
      for (int i = 0; i < srcOut.length; i++) {
        expect(
          (srcOut[i] - dstOut[i]).abs() < 1e-4,
          isTrue,
          reason: 'i=$i src=${srcOut[i]} dst=${dstOut[i]}',
        );
      }
    });

    test('rejects a missing tensor', () {
      final src = DeepSeekV2Model(_tinyCfg);
      final state = _dumpToHF(src);
      state.remove('model.norm.weight');
      expect(
        () => DeepSeekV2HFLoader.loadMap(src, state),
        throwsArgumentError,
      );
    });

    test('rejects a shape mismatch', () {
      final src = DeepSeekV2Model(_tinyCfg);
      final state = _dumpToHF(src);
      state['model.norm.weight'] =
          Tensor.fromList([16], List<double>.filled(16, 1.0));
      expect(
        () => DeepSeekV2HFLoader.loadMap(src, state),
        throwsArgumentError,
      );
    });

    test('silently absorbs the recomputed rotary buffer', () {
      final src = DeepSeekV2Model(_tinyCfg);
      final state = _dumpToHF(src);
      state['model.layers.0.self_attn.rotary_emb.inv_freq'] =
          Tensor.fromList([2], [1.0, 0.5]);
      final report = DeepSeekV2HFLoader.loadMap(src, state);
      expect(report.unusedKeys, isEmpty);
    });
  });
}
