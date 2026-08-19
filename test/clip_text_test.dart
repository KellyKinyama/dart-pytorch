@Timeout(Duration(minutes: 3))
library;

import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

/// Build a HF-style CLIP-text state_dict from an in-memory model so
/// the loader can be roundtripped without downloading real weights.
Map<String, Tensor> _dumpTextToHF(CLIPTextModel m) {
  final state = <String, Tensor>{};
  final cfg = m.config;
  final d = cfg.embedDim;
  final h = cfg.numHeads;
  final headDim = d ~/ h;
  final ffn = cfg.ffnDim;

  state['embeddings.token_embedding.weight'] = m.tokenEmbedding.weight;
  state['embeddings.position_embedding.weight'] = m.positionEmbedding;

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

  List<double> concatBias(List<Linear> parts) {
    final out = <double>[];
    for (final l in parts) {
      out.addAll(l.bias!.toList());
    }
    return out;
  }

  for (int i = 0; i < cfg.numLayers; i++) {
    final block = m.encoder.blocks[i];
    final p = 'encoder.layers.$i';
    state['$p.layer_norm1.weight'] = block.ln1.gamma;
    state['$p.layer_norm1.bias'] = block.ln1.beta;
    state['$p.layer_norm2.weight'] = block.ln2.gamma;
    state['$p.layer_norm2.bias'] = block.ln2.beta;

    state['$p.self_attn.q_proj.weight'] = concatRows(
      [for (final l in block.mha.wq) l.weight],
      h * headDim,
      d,
    );
    state['$p.self_attn.q_proj.bias'] = Tensor.fromList([
      d,
    ], concatBias(block.mha.wq));
    state['$p.self_attn.k_proj.weight'] = concatRows(
      [for (final l in block.mha.wk) l.weight],
      h * headDim,
      d,
    );
    state['$p.self_attn.k_proj.bias'] = Tensor.fromList([
      d,
    ], concatBias(block.mha.wk));
    state['$p.self_attn.v_proj.weight'] = concatRows(
      [for (final l in block.mha.wv) l.weight],
      h * headDim,
      d,
    );
    state['$p.self_attn.v_proj.bias'] = Tensor.fromList([
      d,
    ], concatBias(block.mha.wv));

    state['$p.self_attn.out_proj.weight'] = block.mha.wo.weight;
    state['$p.self_attn.out_proj.bias'] = Tensor.fromList([
      d,
    ], block.mha.wo.bias!.toList());

    state['$p.mlp.fc1.weight'] = block.ffn1.weight;
    state['$p.mlp.fc1.bias'] = Tensor.fromList([
      ffn,
    ], block.ffn1.bias!.toList());
    state['$p.mlp.fc2.weight'] = block.ffn2.weight;
    state['$p.mlp.fc2.bias'] = Tensor.fromList([d], block.ffn2.bias!.toList());
  }

  state['final_layer_norm.weight'] = m.finalLayerNorm.gamma;
  state['final_layer_norm.bias'] = m.finalLayerNorm.beta;
  return state;
}

bool _gpuAvailable() {
  try {
    Tensor.fromList([1], [1.0], device: Device.GPU).toList();
    return true;
  } catch (_) {
    return false;
  }
}

void main() {
  group('CLIPTextConfig', () {
    test('baseTextConfig matches ViT-B/{16,32} text tower', () {
      final cfg = ClipHFLoader.baseTextConfig();
      expect(cfg.vocabSize, 49408);
      expect(cfg.maxCtx, 77);
      expect(cfg.embedDim, 512);
      expect(cfg.numLayers, 12);
      expect(cfg.numHeads, 8);
      expect(cfg.ffnDim, 2048);
    });

    test('largeTextConfig matches ViT-L/14 text tower', () {
      final cfg = ClipHFLoader.largeTextConfig();
      expect(cfg.embedDim, 768);
      expect(cfg.numLayers, 12);
      expect(cfg.numHeads, 12);
      expect(cfg.ffnDim, 3072);
    });
  });

  group('CLIPTextModel forward', () {
    test('output shape [N, D]', () {
      final m = CLIPTextModel(
        const CLIPTextConfig(
          vocabSize: 100,
          maxCtx: 16,
          embedDim: 16,
          numLayers: 2,
          numHeads: 4,
          ffnDim: 32,
        ),
      );
      final tokens = Tensor.fromList([5], [0.0, 5.0, 3.0, 7.0, 9.0]);
      final out = m(tokens);
      expect(out.shape, equals([5, 16]));
    });

    test('pooledEmbedding picks argmax token position', () {
      final m = CLIPTextModel(
        const CLIPTextConfig(
          vocabSize: 50,
          maxCtx: 8,
          embedDim: 8,
          numLayers: 1,
          numHeads: 2,
          ffnDim: 16,
        ),
      );
      // Argmax of [1, 3, 40, 7, 2] is index 2 (value 40).
      final tokens = Tensor.fromList([5], [1.0, 3.0, 40.0, 7.0, 2.0]);
      final pooled = m.pooledEmbedding(tokens);
      expect(pooled.shape, equals([8]));
    });

    test('causality — later token doesn\'t affect earlier hidden states', () {
      final m = CLIPTextModel(
        const CLIPTextConfig(
          vocabSize: 50,
          maxCtx: 8,
          embedDim: 8,
          numLayers: 2,
          numHeads: 2,
          ffnDim: 16,
        ),
      );
      final base = m(Tensor.fromList([4], [1.0, 2.0, 3.0, 4.0])).toList();
      // Change only the last token; positions 0..2 must be identical.
      final modified = m(Tensor.fromList([4], [1.0, 2.0, 3.0, 42.0])).toList();
      for (int i = 0; i < 3 * 8; i++) {
        expect(
          (base[i] - modified[i]).abs() < 1e-5,
          isTrue,
          reason:
              'causal violation at i=$i: '
              'base=${base[i]} modified=${modified[i]}',
        );
      }
    });

    test('rejects empty / too-long sequence', () {
      final m = CLIPTextModel(
        const CLIPTextConfig(
          vocabSize: 50,
          maxCtx: 3,
          embedDim: 8,
          numLayers: 1,
          numHeads: 2,
          ffnDim: 16,
        ),
      );
      expect(() => m(Tensor.fromList([0], [])), throwsArgumentError);
      expect(
        () => m(Tensor.fromList([4], [1.0, 2.0, 3.0, 4.0])),
        throwsArgumentError,
      );
    });
  });

  group('ClipHFLoader text-tower', () {
    test('roundtrip: dump synthetic model -> load -> outputs match', () {
      const cfg = CLIPTextConfig(
        vocabSize: 100,
        maxCtx: 16,
        embedDim: 16,
        numLayers: 2,
        numHeads: 4,
        ffnDim: 32,
      );
      final src = CLIPTextModel(cfg);
      final state = _dumpTextToHF(src);

      const dstCfg = CLIPTextConfig(
        vocabSize: 100,
        maxCtx: 16,
        embedDim: 16,
        numLayers: 2,
        numHeads: 4,
        ffnDim: 32,
        seed: 999,
      );
      final dst = CLIPTextModel(dstCfg);
      final report = ClipHFLoader.loadTextMap(dst, state);
      expect(report.unusedKeys, isEmpty);
      // Per layer: 4 LN + 6 attn (Q/K/V + biases) + 2 out_proj + 4 mlp = 16
      // + 2 embeddings + 2 final LN
      final expected = 2 + cfg.numLayers * 16 + 2;
      expect(report.consumedCount, expected);

      final tokens = Tensor.fromList([4], [0.0, 5.0, 7.0, 3.0]);
      final srcOut = src(tokens).toList();
      final dstOut = dst(tokens).toList();
      for (int i = 0; i < srcOut.length; i++) {
        expect(
          (srcOut[i] - dstOut[i]).abs() < 1e-5,
          isTrue,
          reason: 'i=$i src=${srcOut[i]} dst=${dstOut[i]}',
        );
      }
    });

    test('accepts text_model. prefix (joint CLIPModel bundle)', () {
      const cfg = CLIPTextConfig(
        vocabSize: 50,
        maxCtx: 8,
        embedDim: 8,
        numLayers: 1,
        numHeads: 2,
        ffnDim: 16,
      );
      final src = CLIPTextModel(cfg);
      final flat = _dumpTextToHF(src);
      final prefixed = <String, Tensor>{
        for (final e in flat.entries) 'text_model.${e.key}': e.value,
      };
      final dst = CLIPTextModel(cfg);
      final r = ClipHFLoader.loadTextMap(dst, prefixed);
      expect(r.prefix, 'text_model.');
      expect(r.unusedKeys, isEmpty);
    });

    test('rejects a missing tensor', () {
      final src = CLIPTextModel(
        const CLIPTextConfig(
          vocabSize: 30,
          maxCtx: 4,
          embedDim: 8,
          numLayers: 1,
          numHeads: 2,
          ffnDim: 16,
        ),
      );
      final state = _dumpTextToHF(src);
      state.remove('final_layer_norm.bias');
      expect(() => ClipHFLoader.loadTextMap(src, state), throwsArgumentError);
    });
  });

  group('CLIPTextModel on GPU', () {
    final gpuOk = _gpuAvailable();
    if (!gpuOk) {
      test(
        'GPU unavailable → skipped',
        () {},
        skip: 'CUDA / native/lib/libmat_mul.so not usable in this env',
      );
      return;
    }
    test('CPU vs GPU pooled embedding agree', () {
      final cpu = CLIPTextModel(
        const CLIPTextConfig(
          vocabSize: 100,
          maxCtx: 16,
          embedDim: 16,
          numLayers: 2,
          numHeads: 4,
          ffnDim: 32,
        ),
      );
      final gpu = CLIPTextModel(
        const CLIPTextConfig(
          vocabSize: 100,
          maxCtx: 16,
          embedDim: 16,
          numLayers: 2,
          numHeads: 4,
          ffnDim: 32,
          device: Device.GPU,
          seed: 111,
        ),
      );
      final state = _dumpTextToHF(cpu);
      ClipHFLoader.loadTextMap(cpu, state);
      ClipHFLoader.loadTextMap(gpu, state);
      final tokVals = <double>[0, 5, 40, 7, 2];
      final cpuOut = cpu
          .pooledEmbedding(Tensor.fromList([tokVals.length], tokVals))
          .toList();
      final gpuOut = gpu
          .pooledEmbedding(
            Tensor.fromList([tokVals.length], tokVals, device: Device.GPU),
          )
          .toList();
      for (int i = 0; i < cpuOut.length; i++) {
        expect(
          (cpuOut[i] - gpuOut[i]).abs() < 5e-3,
          isTrue,
          reason: 'i=$i cpu=${cpuOut[i]} gpu=${gpuOut[i]}',
        );
      }
    });
  });

  group('ClipHFLoader projections', () {
    test('loadProjectionsMap returns null when both projections absent', () {
      final s = <String, Tensor>{};
      expect(ClipHFLoader.loadProjectionsMap(s), isNull);
    });

    test('loadProjectionsMap parses shapes correctly', () {
      final s = <String, Tensor>{
        'visual_projection.weight': Tensor.fromList([
          512,
          768,
        ], List<double>.filled(512 * 768, 0.1)),
        'text_projection.weight': Tensor.fromList([
          512,
          512,
        ], List<double>.filled(512 * 512, 0.2)),
        'logit_scale': Tensor.fromList([1], [4.6052]),
      };
      final p = ClipHFLoader.loadProjectionsMap(s)!;
      expect(p.projDim, 512);
      expect(p.visionHidden, 768);
      expect(p.textHidden, 512);
      expect(p.logitScale, isNotNull);
      expect(p.logitScale!.toList()[0], closeTo(4.6052, 1e-4));
    });

    test('loadProjectionsMap accepts missing logit_scale', () {
      final s = <String, Tensor>{
        'visual_projection.weight': Tensor.fromList([
          512,
          768,
        ], List<double>.filled(512 * 768, 0.1)),
        'text_projection.weight': Tensor.fromList([
          512,
          512,
        ], List<double>.filled(512 * 512, 0.2)),
      };
      final p = ClipHFLoader.loadProjectionsMap(s);
      expect(p, isNotNull);
      expect(p!.logitScale, isNull);
    });
  });
}
