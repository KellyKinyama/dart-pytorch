import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:dart_pytorch/core/nn/deepseek_v2_hf_loader.dart';

const _cfg = DeepSeekV2Config(
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

void main() {
  final src = DeepSeekV2Model(_cfg);
  final dst = DeepSeekV2Model(
    DeepSeekV2Config(
      vocabSize: _cfg.vocabSize,
      maxCtx: _cfg.maxCtx,
      embedDim: _cfg.embedDim,
      numLayers: _cfg.numLayers,
      firstKDenseReplace: _cfg.firstKDenseReplace,
      denseFfnDim: _cfg.denseFfnDim,
      moeExpertHiddenDim: _cfg.moeExpertHiddenDim,
      numRoutedExperts: _cfg.numRoutedExperts,
      numSharedExperts: _cfg.numSharedExperts,
      numExpertsPerTok: _cfg.numExpertsPerTok,
      mlaConfig: _cfg.mlaConfig,
      seed: 999,
    ),
  );

  // Dump source's HF-style state — reuse the same code that lives in
  // the test file, inlined so I can print progress.
  final state = <String, Tensor>{};
  state['model.embed_tokens.weight'] = src.embedIn.weight;
  final cfg = _cfg;
  final mla = cfg.mlaConfig;
  final h = mla.numHeads;
  final nopeD = mla.qkNopeHeadDim;
  final ropeD = mla.qkRopeHeadDim;
  final vD = mla.vHeadDim;
  final headDim = mla.qkHeadDim;
  final kvL = mla.kvLoraRank;
  final d = cfg.embedDim;

  Tensor concatRows(List<Tensor> parts, int totalRows, int cols) {
    final buf = <double>[];
    for (final p in parts) {
      buf.addAll(p.toList());
    }
    return Tensor.fromList([totalRows, cols], buf, device: Device.CPU);
  }

  Tensor concatCols(List<Tensor> parts, int rows, int totalCols) {
    final buf = List<double>.filled(rows * totalCols, 0);
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
    return Tensor.fromList([rows, totalCols], buf, device: Device.CPU);
  }

  Tensor transpose2D(Tensor t) {
    final r = t.shape[0];
    final c = t.shape[1];
    final data = t.toList();
    final buf = List<double>.filled(r * c, 0);
    for (int i = 0; i < r; i++) {
      for (int j = 0; j < c; j++) {
        buf[j * r + i] = data[i * c + j];
      }
    }
    return Tensor.fromList([c, r], buf, device: Device.CPU);
  }

  for (int i = 0; i < cfg.numLayers; i++) {
    final block = src.blocks[i];
    final p = 'model.layers.$i';
    state['$p.input_layernorm.weight'] = block.attnLn.gamma;
    state['$p.post_attention_layernorm.weight'] = block.ffnLn.gamma;

    final attn = block.attn;
    final qRowsPerHead = <Tensor>[];
    for (int hh = 0; hh < h; hh++) {
      qRowsPerHead.add(
        concatRows(
          [attn.qUpNope[hh].weight, attn.qUpRope[hh].weight],
          headDim,
          mla.qInDim,
        ),
      );
    }
    final qFused = concatRows(qRowsPerHead, h * headDim, mla.qInDim);
    if (mla.qLoraRank != null) {
      state['$p.self_attn.q_a_proj.weight'] = attn.qDown!.weight;
      state['$p.self_attn.q_a_layernorm.weight'] = attn.qLn!.gamma;
      state['$p.self_attn.q_b_proj.weight'] = qFused;
    } else {
      state['$p.self_attn.q_proj.weight'] = qFused;
    }
    state['$p.self_attn.kv_a_proj_with_mqa.weight'] = concatRows(
      [attn.kvDown.weight, attn.kRope.weight],
      kvL + ropeD,
      d,
    );
    state['$p.self_attn.kv_a_layernorm.weight'] = attn.kvLn.gamma;
    final kvRowsPerHead = <Tensor>[];
    for (int hh = 0; hh < h; hh++) {
      kvRowsPerHead.add(
        concatRows(
          [attn.kUpNope[hh].weight, attn.vUp[hh].weight],
          nopeD + vD,
          kvL,
        ),
      );
    }
    state['$p.self_attn.kv_b_proj.weight'] = concatRows(
      kvRowsPerHead,
      h * (nopeD + vD),
      kvL,
    );
    state['$p.self_attn.o_proj.weight'] = attn.oProj.weight;

    if (!block.isMoE) {
      final ffn = block.denseFfn!;
      state['$p.mlp.gate_proj.weight'] = ffn.gateProj.weight;
      state['$p.mlp.up_proj.weight'] = ffn.upProj.weight;
      state['$p.mlp.down_proj.weight'] = ffn.downProj.weight;
    } else {
      final moe = block.moeFfn!;
      state['$p.mlp.gate.weight'] = transpose2D(moe.gateW);
      for (int j = 0; j < cfg.numRoutedExperts; j++) {
        final e = moe.routedExperts[j];
        final ep = '$p.mlp.experts.$j';
        state['$ep.gate_proj.weight'] = e.w1.weight;
        state['$ep.up_proj.weight'] = e.w3!.weight;
        state['$ep.down_proj.weight'] = e.w2.weight;
      }
      final sharedGateParts = <Tensor>[
        for (final e in moe.sharedExperts) e.w1.weight,
      ];
      final sharedUpParts = <Tensor>[
        for (final e in moe.sharedExperts) e.w3!.weight,
      ];
      final sharedDownParts = <Tensor>[
        for (final e in moe.sharedExperts) e.w2.weight,
      ];
      final sharedFfn = cfg.numSharedExperts * cfg.moeExpertHiddenDim;
      state['$p.mlp.shared_experts.gate_proj.weight'] = concatRows(
        sharedGateParts,
        sharedFfn,
        d,
      );
      state['$p.mlp.shared_experts.up_proj.weight'] = concatRows(
        sharedUpParts,
        sharedFfn,
        d,
      );
      state['$p.mlp.shared_experts.down_proj.weight'] = concatCols(
        sharedDownParts,
        d,
        sharedFfn,
      );
    }
  }
  state['model.norm.weight'] = src.finalNorm.gamma;
  if (!cfg.tieWordEmbeddings) {
    state['lm_head.weight'] = src.untiedHead!.weight;
  }

  final report = DeepSeekV2HFLoader.loadMap(dst, state);
  print('load report: $report');

  // Iterate ALL params and see which ones differ.
  final srcP = src.parameters();
  final dstP = dst.parameters();
  print('num params: src=${srcP.length}, dst=${dstP.length}');
  int diffCount = 0;
  for (int i = 0; i < srcP.length; i++) {
    final a = srcP[i].toList();
    final b = dstP[i].toList();
    if (a.length != b.length) {
      print('param $i shape mismatch');
      continue;
    }
    double maxD = 0;
    for (int j = 0; j < a.length; j++) {
      final d = (a[j] - b[j]).abs();
      if (d > maxD) maxD = d;
    }
    if (maxD > 1e-6) {
      diffCount++;
      if (diffCount < 15) {
        print('param $i shape=${srcP[i].shape} max diff=$maxD');
      }
    }
  }
  print('total diverged params: $diffCount / ${srcP.length}');

  final t = Tensor.fromList([4], [1.0, 5.0, 9.0, 13.0]);
  print('src output[0] = ${src(t).toList()[0]}');
  print('dst output[0] = ${dst(t).toList()[0]}');
}
