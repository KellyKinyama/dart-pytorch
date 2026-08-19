@Timeout(Duration(minutes: 3))
library;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

/// GPU parity for [Llama] — covers both the plain Llama-style attention
/// (no biases) and the Qwen2.5-style variant (Q/K/V biases). Uses tiny
/// synthetic configs so it runs on any CUDA-capable GPU including 6 GB
/// laptop cards. Skipped automatically when the GPU backend is
/// unavailable (native lib missing, no CUDA runtime, etc.).
///
/// Requires `native/lib/libmat_mul.so` on WSL:
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib dart test test/llama_gpu_test.dart
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
    if (cfg.attentionBias) {
      List<double> concatBias(List<Linear> parts) {
        final out = <double>[];
        for (final l in parts) {
          out.addAll(l.bias!.toList());
        }
        return out;
      }

      state['$p.self_attn.q_proj.bias'] = Tensor.fromList([
        h * headDim,
      ], concatBias(blk.attn.wq));
      state['$p.self_attn.k_proj.bias'] = Tensor.fromList([
        kvH * headDim,
      ], concatBias(blk.attn.wk));
      state['$p.self_attn.v_proj.bias'] = Tensor.fromList([
        kvH * headDim,
      ], concatBias(blk.attn.wv));
    }
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

bool _gpuAvailable() {
  try {
    final probe = Tensor.fromList([2], [1.0, 2.0], device: Device.GPU);
    probe.toList();
    return true;
  } catch (_) {
    return false;
  }
}

void main() {
  final gpuOk = _gpuAvailable();
  group('Llama on GPU', () {
    if (!gpuOk) {
      test(
        'GPU unavailable → skipped',
        () {},
        skip: 'CUDA / native/lib/libmat_mul.so not usable in this env',
      );
      return;
    }

    test('Llama-style (no bias) CPU vs GPU logits agree', () {
      final cpuCfg = LlamaConfig(
        vocabSize: 32,
        maxCtx: 8,
        embedDim: 8,
        numLayers: 2,
        numHeads: 4,
        numKvHeads: 2,
        ffnDim: 16,
        seed: 123,
      );
      final gpuCfg = LlamaConfig(
        vocabSize: 32,
        maxCtx: 8,
        embedDim: 8,
        numLayers: 2,
        numHeads: 4,
        numKvHeads: 2,
        ffnDim: 16,
        device: Device.GPU,
        seed: 999,
      );
      final cpu = Llama(cpuCfg);
      final gpu = Llama(gpuCfg);

      final state = _dumpToHFMap(cpu);
      LlamaHFLoader.loadMap(cpu, state);
      LlamaHFLoader.loadMap(gpu, state);
      cpu.eval();
      gpu.eval();

      final tokVals = [0.0, 3.0, 1.0, 2.0];
      final cpuLogits = cpu(Tensor.fromList([4], tokVals)).toList();
      final gpuLogits = gpu(
        Tensor.fromList([4], tokVals, device: Device.GPU),
      ).toList();
      expect(cpuLogits.length, gpuLogits.length);
      for (int i = 0; i < cpuLogits.length; i++) {
        expect(
          (cpuLogits[i] - gpuLogits[i]).abs() < 5e-3,
          isTrue,
          reason: 'logit $i cpu=${cpuLogits[i]} gpu=${gpuLogits[i]}',
        );
      }
    });

    test('Qwen-style (attentionBias) CPU vs GPU logits agree', () {
      final cpuCfg = LlamaConfig(
        vocabSize: 32,
        maxCtx: 8,
        embedDim: 8,
        numLayers: 2,
        numHeads: 4,
        numKvHeads: 2,
        ffnDim: 16,
        attentionBias: true,
        seed: 321,
      );
      final gpuCfg = LlamaConfig(
        vocabSize: 32,
        maxCtx: 8,
        embedDim: 8,
        numLayers: 2,
        numHeads: 4,
        numKvHeads: 2,
        ffnDim: 16,
        attentionBias: true,
        device: Device.GPU,
        seed: 111,
      );
      final cpu = Llama(cpuCfg);
      final gpu = Llama(gpuCfg);

      final state = _dumpToHFMap(cpu);
      LlamaHFLoader.loadMap(cpu, state);
      LlamaHFLoader.loadMap(gpu, state);
      cpu.eval();
      gpu.eval();

      final tokVals = [0.0, 3.0, 1.0, 2.0];
      final cpuLogits = cpu(Tensor.fromList([4], tokVals)).toList();
      final gpuLogits = gpu(
        Tensor.fromList([4], tokVals, device: Device.GPU),
      ).toList();
      expect(cpuLogits.length, gpuLogits.length);
      for (int i = 0; i < cpuLogits.length; i++) {
        expect(
          (cpuLogits[i] - gpuLogits[i]).abs() < 5e-3,
          isTrue,
          reason: 'logit $i cpu=${cpuLogits[i]} gpu=${gpuLogits[i]}',
        );
      }
    });
  });
}
