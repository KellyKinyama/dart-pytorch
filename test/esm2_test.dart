@Timeout(Duration(minutes: 3))
library;

import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

/// Serialise the given [ESM2Model] as an HF-style state_dict so the
/// loader roundtrip can be exercised without downloading real weights.
Map<String, Tensor> _dumpToHFMap(ESM2Model m) {
  final state = <String, Tensor>{};
  final cfg = m.config;
  final d = cfg.embedDim;
  final h = cfg.numHeads;
  final headDim = d ~/ h;
  final ffn = cfg.ffnDim;

  state['esm.embeddings.word_embeddings.weight'] = m.embedIn.weight;

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
    final layer = m.layers[i];
    final p = 'esm.encoder.layer.$i';

    state['$p.attention.LayerNorm.weight'] = layer.attnLn.gamma;
    state['$p.attention.LayerNorm.bias'] = layer.attnLn.beta;

    state['$p.attention.self.query.weight'] = concatRows(
      [for (final l in layer.attn.wq) l.weight],
      h * headDim,
      d,
    );
    state['$p.attention.self.query.bias'] = Tensor.fromList([
      h * headDim,
    ], concatBias(layer.attn.wq));
    state['$p.attention.self.key.weight'] = concatRows(
      [for (final l in layer.attn.wk) l.weight],
      h * headDim,
      d,
    );
    state['$p.attention.self.key.bias'] = Tensor.fromList([
      h * headDim,
    ], concatBias(layer.attn.wk));
    state['$p.attention.self.value.weight'] = concatRows(
      [for (final l in layer.attn.wv) l.weight],
      h * headDim,
      d,
    );
    state['$p.attention.self.value.bias'] = Tensor.fromList([
      h * headDim,
    ], concatBias(layer.attn.wv));
    state['$p.attention.output.dense.weight'] = layer.attn.wo.weight;
    state['$p.attention.output.dense.bias'] = Tensor.fromList([
      d,
    ], layer.attn.wo.bias!.toList());

    state['$p.LayerNorm.weight'] = layer.ffnLn.gamma;
    state['$p.LayerNorm.bias'] = layer.ffnLn.beta;
    state['$p.intermediate.dense.weight'] = layer.ffnIntermediate.weight;
    state['$p.intermediate.dense.bias'] = Tensor.fromList([
      ffn,
    ], layer.ffnIntermediate.bias!.toList());
    state['$p.output.dense.weight'] = layer.ffnOutput.weight;
    state['$p.output.dense.bias'] = Tensor.fromList([
      d,
    ], layer.ffnOutput.bias!.toList());
  }

  state['esm.encoder.emb_layer_norm_after.weight'] = m.finalLn.gamma;
  state['esm.encoder.emb_layer_norm_after.bias'] = m.finalLn.beta;
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
  group('ESM2Config', () {
    test('esm2_8m defaults', () {
      final cfg = ESM2HFLoader.esm2_8mConfig();
      expect(cfg.vocabSize, 33);
      expect(cfg.embedDim, 320);
      expect(cfg.numLayers, 6);
      expect(cfg.numHeads, 20);
      expect(cfg.ffnDim, 1280);
      expect(cfg.ropeBase, 10000.0);
    });

    test('esm2_35m defaults', () {
      final cfg = ESM2HFLoader.esm2_35mConfig();
      expect(cfg.embedDim, 480);
      expect(cfg.numLayers, 12);
      expect(cfg.ffnDim, 1920);
    });
  });

  group('encodeProteinSequence', () {
    test('wraps in <cls> ... <eos>', () {
      final ids = encodeProteinSequence('MK');
      // cls=0, M=20, K=15, eos=2
      expect(ids, equals([0, 20, 15, 2]));
    });

    test('unknown chars map to <unk>', () {
      final ids = encodeProteinSequence('J');
      // <cls>, J is not in vocab → <unk>=3, <eos>
      expect(ids, equals([0, 3, 2]));
    });
  });

  group('ESM2Model forward', () {
    test('output shape [N, D]', () {
      final m = ESM2Model(
        const ESM2Config(
          vocabSize: 33,
          maxCtx: 32,
          embedDim: 16,
          numLayers: 2,
          numHeads: 4,
          ffnDim: 32,
        ),
      );
      final ids = encodeProteinSequence('MSVAK');
      final tokens = Tensor.fromList([
        ids.length,
      ], ids.map((i) => i.toDouble()).toList());
      final h = m(tokens);
      expect(h.shape, equals([ids.length, 16]));
    });

    test('meanPool returns [D]', () {
      final m = ESM2Model(
        const ESM2Config(
          vocabSize: 33,
          maxCtx: 32,
          embedDim: 8,
          numLayers: 1,
          numHeads: 2,
          ffnDim: 16,
        ),
      );
      final tokens = Tensor.fromList([4], [0.0, 20.0, 15.0, 2.0]);
      final pool = m.meanPool(tokens);
      expect(pool.shape, equals([8]));
    });

    test('rejects empty and too-long sequences', () {
      final m = ESM2Model(
        const ESM2Config(
          vocabSize: 33,
          maxCtx: 3,
          embedDim: 8,
          numLayers: 1,
          numHeads: 2,
          ffnDim: 16,
        ),
      );
      expect(() => m(Tensor.fromList([0], [])), throwsArgumentError);
      expect(
        () => m(Tensor.fromList([4], [0.0, 1.0, 2.0, 3.0])),
        throwsArgumentError,
      );
    });
  });

  group('ESM2HFLoader', () {
    test('roundtrip: dump synthetic model -> load -> logits match', () {
      final cfg = const ESM2Config(
        vocabSize: 33,
        maxCtx: 16,
        embedDim: 16,
        numLayers: 2,
        numHeads: 4,
        ffnDim: 32,
      );
      final src = ESM2Model(cfg);
      final state = _dumpToHFMap(src);

      final dst = ESM2Model(
        const ESM2Config(
          vocabSize: 33,
          maxCtx: 16,
          embedDim: 16,
          numLayers: 2,
          numHeads: 4,
          ffnDim: 32,
          seed: 999, // different init so roundtrip is meaningful
        ),
      );
      final report = ESM2HFLoader.loadMap(dst, state);
      expect(report.unusedKeys, isEmpty);
      // 1 embed + 2 layers * (2+2+2+2+1+1+1+1+1+1 = 14 keys) + 2 final = 31
      // Keys per layer:
      //   attention.LayerNorm.{w,b}          (2)
      //   attention.self.{q,k,v}.{w,b}       (6)
      //   attention.output.dense.{w,b}       (2)
      //   LayerNorm.{w,b}                    (2)
      //   intermediate.dense.{w,b}           (2)
      //   output.dense.{w,b}                 (2)
      // = 16 keys/layer
      final expected = 1 + cfg.numLayers * 16 + 2;
      expect(report.consumedCount, expected);

      final tokens = Tensor.fromList([5], [0.0, 20.0, 15.0, 4.0, 2.0]);
      final srcOut = src(tokens).toList();
      final dstOut = dst(tokens).toList();
      expect(srcOut.length, dstOut.length);
      for (int i = 0; i < srcOut.length; i++) {
        expect(
          (srcOut[i] - dstOut[i]).abs() < 1e-5,
          isTrue,
          reason: 'i=$i src=${srcOut[i]} dst=${dstOut[i]}',
        );
      }
    });

    test('rejects missing tensor', () {
      final m = ESM2Model(
        const ESM2Config(
          vocabSize: 33,
          maxCtx: 8,
          embedDim: 8,
          numLayers: 1,
          numHeads: 2,
          ffnDim: 16,
        ),
      );
      final state = _dumpToHFMap(m);
      state.remove('esm.encoder.emb_layer_norm_after.weight');
      expect(() => ESM2HFLoader.loadMap(m, state), throwsArgumentError);
    });

    test(
      'silently absorbs unused HF keys (position embeds, lm_head, etc.)',
      () {
        final m = ESM2Model(
          const ESM2Config(
            vocabSize: 33,
            maxCtx: 8,
            embedDim: 8,
            numLayers: 1,
            numHeads: 2,
            ffnDim: 16,
          ),
        );
        final state = _dumpToHFMap(m);
        state['esm.embeddings.position_embeddings.weight'] = Tensor.fromList([
          1026,
          8,
        ], List<double>.filled(1026 * 8, 0));
        state['lm_head.dense.weight'] = Tensor.fromList([
          8,
          8,
        ], List<double>.filled(64, 0));
        state['esm.contact_head.regression.weight'] = Tensor.fromList([
          1,
          6,
        ], List<double>.filled(6, 0));
        final report = ESM2HFLoader.loadMap(m, state);
        expect(report.unusedKeys, isEmpty);
      },
    );
  });

  group('ESM2 on GPU', () {
    final gpuOk = _gpuAvailable();
    if (!gpuOk) {
      test(
        'GPU unavailable → skipped',
        () {},
        skip: 'CUDA / native/lib/libmat_mul.so not usable in this env',
      );
      return;
    }
    test('CPU vs GPU per-residue hidden state parity', () {
      final cpuCfg = const ESM2Config(
        vocabSize: 33,
        maxCtx: 16,
        embedDim: 16,
        numLayers: 2,
        numHeads: 4,
        ffnDim: 32,
      );
      final gpuCfg = const ESM2Config(
        vocabSize: 33,
        maxCtx: 16,
        embedDim: 16,
        numLayers: 2,
        numHeads: 4,
        ffnDim: 32,
        device: Device.GPU,
        seed: 42, // will be overwritten
      );
      final cpu = ESM2Model(cpuCfg);
      final gpu = ESM2Model(gpuCfg);
      final state = _dumpToHFMap(cpu);
      ESM2HFLoader.loadMap(cpu, state);
      ESM2HFLoader.loadMap(gpu, state);

      final ids = encodeProteinSequence('MSVAK');
      final tokVals = ids.map((i) => i.toDouble()).toList();
      final cpuOut = cpu(Tensor.fromList([ids.length], tokVals)).toList();
      final gpuOut = gpu(
        Tensor.fromList([ids.length], tokVals, device: Device.GPU),
      ).toList();
      double maxDiff = 0;
      for (int i = 0; i < cpuOut.length; i++) {
        final d = (cpuOut[i] - gpuOut[i]).abs();
        if (d > maxDiff) maxDiff = d;
      }
      expect(maxDiff, lessThan(5e-3), reason: 'max cpu/gpu diff = $maxDiff');
    });
  });
}
