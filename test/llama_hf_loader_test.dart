import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

/// Build a state-dict for the given [Llama] using the exact HF-Llama
/// naming scheme that [LlamaHFLoader.loadMap] expects. Re-concatenates
/// per-head q/k/v Linears into the single fused HF projection matrix.
Map<String, Tensor> _dumpToHFMap(Llama m) {
  final state = <String, Tensor>{};
  final cfg = m.config;
  final d = cfg.embedDim;
  final h = cfg.numHeads;
  final kvH = cfg.numKvHeads;
  final headDim = d ~/ h;

  state['model.embed_tokens.weight'] = m.embedIn.weight;

  Tensor concatRows(List<Tensor> parts, int totalRows, int cols) {
    final buf = List<double>.filled(totalRows * cols, 0.0);
    var row = 0;
    for (final p in parts) {
      final vals = p.toList();
      final rowsHere = p.shape[0];
      for (int i = 0; i < rowsHere * cols; i++) {
        buf[row * cols + i] = vals[i];
      }
      row += rowsHere;
    }
    return Tensor.fromList([totalRows, cols], buf);
  }

  for (int i = 0; i < cfg.numLayers; i++) {
    final blk = m.blocks[i];
    final p = 'model.layers.$i';
    state['$p.input_layernorm.weight'] = blk.attnNorm.gamma;
    state['$p.post_attention_layernorm.weight'] = blk.ffnNorm.gamma;

    state['$p.self_attn.q_proj.weight'] = concatRows(
      [for (final l in blk.attn.wq) l.weight],
      h * headDim,
      d,
    );
    state['$p.self_attn.k_proj.weight'] = concatRows(
      [for (final l in blk.attn.wk) l.weight],
      kvH * headDim,
      d,
    );
    state['$p.self_attn.v_proj.weight'] = concatRows(
      [for (final l in blk.attn.wv) l.weight],
      kvH * headDim,
      d,
    );
    state['$p.self_attn.o_proj.weight'] = blk.attn.wo.weight;

    state['$p.mlp.gate_proj.weight'] = blk.ffn.gateProj.weight;
    state['$p.mlp.up_proj.weight'] = blk.ffn.upProj.weight;
    state['$p.mlp.down_proj.weight'] = blk.ffn.downProj.weight;
  }

  state['model.norm.weight'] = m.finalNorm.gamma;
  if (!cfg.tieWeights) {
    state['lm_head.weight'] = m.untiedHead!.weight;
  }
  return state;
}

void main() {
  group('LlamaHFLoader', () {
    test('roundtrip: dump tied model -> load into fresh -> logits match', () {
      final cfg = LlamaConfig(
        vocabSize: 32,
        maxCtx: 8,
        embedDim: 8,
        numLayers: 2,
        numHeads: 4,
        numKvHeads: 2,
        ffnDim: 16,
        seed: 123,
      );
      final src = Llama(cfg);
      final state = _dumpToHFMap(src);

      final dst = Llama(LlamaConfig(
        vocabSize: cfg.vocabSize,
        maxCtx: cfg.maxCtx,
        embedDim: cfg.embedDim,
        numLayers: cfg.numLayers,
        numHeads: cfg.numHeads,
        numKvHeads: cfg.numKvHeads,
        ffnDim: cfg.ffnDim,
        seed: 999, // deliberately different init seed
      ));

      final report = LlamaHFLoader.loadMap(dst, state);
      expect(report.unusedKeys, isEmpty);
      // 1 embed + numLayers*(2 rmsnorm + 4 attn + 3 mlp) + 1 finalNorm
      expect(report.consumedCount, 1 + cfg.numLayers * 9 + 1);

      final tokens = Tensor.fromList([4], [0.0, 3.0, 1.0, 2.0]);
      final srcLogits = src(tokens).toList();
      final dstLogits = dst(tokens).toList();
      expect(srcLogits.length, dstLogits.length);
      for (int i = 0; i < srcLogits.length; i++) {
        expect(
          (srcLogits[i] - dstLogits[i]).abs() < 1e-5,
          isTrue,
          reason: 'logit $i: src=${srcLogits[i]} dst=${dstLogits[i]}',
        );
      }
    });

    test('roundtrip: dump untied model -> load into fresh -> match', () {
      final cfg = LlamaConfig(
        vocabSize: 16,
        maxCtx: 8,
        embedDim: 8,
        numLayers: 1,
        numHeads: 2,
        numKvHeads: 1,
        ffnDim: 16,
        tieWeights: false,
        seed: 7,
      );
      final src = Llama(cfg);
      final state = _dumpToHFMap(src);
      expect(state.containsKey('lm_head.weight'), isTrue);

      final dst = Llama(LlamaConfig(
        vocabSize: cfg.vocabSize,
        maxCtx: cfg.maxCtx,
        embedDim: cfg.embedDim,
        numLayers: cfg.numLayers,
        numHeads: cfg.numHeads,
        numKvHeads: cfg.numKvHeads,
        ffnDim: cfg.ffnDim,
        tieWeights: false,
        seed: 42,
      ));
      final report = LlamaHFLoader.loadMap(dst, state);
      expect(report.unusedKeys, isEmpty);

      final tokens = Tensor.fromList([3], [0.0, 1.0, 2.0]);
      final srcLogits = src(tokens).toList();
      final dstLogits = dst(tokens).toList();
      for (int i = 0; i < srcLogits.length; i++) {
        expect(
          (srcLogits[i] - dstLogits[i]).abs() < 1e-5,
          isTrue,
          reason: 'logit $i: src=${srcLogits[i]} dst=${dstLogits[i]}',
        );
      }
    });

    test('rejects missing tensor', () {
      final cfg = LlamaConfig(
        vocabSize: 16,
        maxCtx: 4,
        embedDim: 4,
        numLayers: 1,
        numHeads: 2,
        numKvHeads: 1,
        ffnDim: 8,
      );
      final m = Llama(cfg);
      final state = _dumpToHFMap(m);
      state.remove('model.norm.weight');
      expect(
        () => LlamaHFLoader.loadMap(m, state),
        throwsArgumentError,
      );
    });

    test('rejects shape mismatch', () {
      final cfg = LlamaConfig(
        vocabSize: 16,
        maxCtx: 4,
        embedDim: 4,
        numLayers: 1,
        numHeads: 2,
        numKvHeads: 1,
        ffnDim: 8,
      );
      final m = Llama(cfg);
      final state = _dumpToHFMap(m);
      // Replace the norm weight with a wrong-shape tensor.
      state['model.norm.weight'] = Tensor.fromList([8], List.filled(8, 1.0));
      expect(
        () => LlamaHFLoader.loadMap(m, state),
        throwsArgumentError,
      );
    });

    test('tied model with redundant lm_head.weight is accepted', () {
      final cfg = LlamaConfig(
        vocabSize: 16,
        maxCtx: 4,
        embedDim: 4,
        numLayers: 1,
        numHeads: 2,
        numKvHeads: 1,
        ffnDim: 8,
      );
      final m = Llama(cfg);
      final state = _dumpToHFMap(m);
      // Some tied HF checkpoints ship lm_head anyway — verify it's
      // consumed silently.
      state['lm_head.weight'] = m.embedIn.weight;
      final report = LlamaHFLoader.loadMap(m, state);
      expect(report.unusedKeys, isEmpty);
    });

    test('llama32_1BConfig has expected shape parameters', () {
      final cfg = LlamaHFLoader.llama32_1BConfig();
      expect(cfg.vocabSize, 128256);
      expect(cfg.embedDim, 2048);
      expect(cfg.numLayers, 16);
      expect(cfg.numHeads, 32);
      expect(cfg.numKvHeads, 8);
      expect(cfg.ffnDim, 8192);
      expect(cfg.ropeBase, 500000.0);
      expect(cfg.tieWeights, isTrue);
    });

    test('llama31_8BConfig has tieWeights=false', () {
      final cfg = LlamaHFLoader.llama31_8BConfig();
      expect(cfg.tieWeights, isFalse);
      expect(cfg.numLayers, 32);
      expect(cfg.ffnDim, 14336);
    });
  });
}
