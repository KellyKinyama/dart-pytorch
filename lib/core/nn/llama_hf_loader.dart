/// Loads HuggingFace **Llama** (`llama` / `LlamaForCausalLM` architecture)
/// weights from a safetensors file into a [Llama].
///
/// Supported checkpoints (all pre-Llama-4, use RMSNorm + SwiGLU + RoPE +
/// GQA — same recipe for Llama 1/2/3/3.1/3.2, Mistral 7B, Qwen 2, etc.):
///
///   * `meta-llama/Llama-3.2-1B-Instruct` — 16 layers, 2048 embed,
///     32 heads, 8 kv-heads, ffn 8192, ropeBase 500000, **tied** head.
///   * `meta-llama/Llama-3.2-3B-Instruct` — 28 layers, 3072 embed,
///     24 heads, 8 kv-heads, ffn 8192, ropeBase 500000, **tied** head.
///   * `meta-llama/Llama-3.1-8B-Instruct` — 32 layers, 4096 embed,
///     32 heads, 8 kv-heads, ffn 14336, ropeBase 500000, **untied**.
///
/// HF key conventions handled here (Llama's namespace has no
/// leading `gpt_neox.` — top-level `model.` prefix):
///
///   * `model.embed_tokens.weight` — `[V, D]`
///   * `model.layers.{i}.input_layernorm.weight` — `[D]`
///   * `model.layers.{i}.self_attn.q_proj.weight` — `[H·hd, D]`
///     (H = numHeads, hd = headDim). Row-split into per-head slices.
///   * `model.layers.{i}.self_attn.k_proj.weight` — `[Hkv·hd, D]`
///   * `model.layers.{i}.self_attn.v_proj.weight` — `[Hkv·hd, D]`
///   * `model.layers.{i}.self_attn.o_proj.weight` — `[D, D]`
///   * `model.layers.{i}.post_attention_layernorm.weight` — `[D]`
///   * `model.layers.{i}.mlp.gate_proj.weight` — `[F, D]`
///   * `model.layers.{i}.mlp.up_proj.weight` — `[F, D]`
///   * `model.layers.{i}.mlp.down_proj.weight` — `[D, F]`
///   * `model.norm.weight` — `[D]`
///   * `lm_head.weight` — `[V, D]` (present only when the model does
///     **not** tie its head, e.g. Llama 3.1 8B; ignored when tied).
///
/// Ignored (safe): per-layer `self_attn.rotary_emb.inv_freq` — we
/// recompute RoPE tables from `config.ropeBase`.
library;

import 'dart:typed_data';

import '../tensor/tensor.dart';
import 'llama.dart';
import 'safetensors.dart';

class LlamaHFLoader {
  /// Llama-3.2-1B-Instruct config. `tieWeights: true`.
  static LlamaConfig llama32_1BConfig({
    Device device = Device.CPU,
    int seed = 0,
    int? maxCtx,
  }) => LlamaConfig(
    vocabSize: 128256,
    maxCtx: maxCtx ?? 131072,
    embedDim: 2048,
    numLayers: 16,
    numHeads: 32,
    numKvHeads: 8,
    ffnDim: 8192,
    ropeBase: 500000.0,
    rmsNormEps: 1e-5,
    tieWeights: true,
    device: device,
    seed: seed,
  );

  /// Llama-3.2-3B-Instruct config. `tieWeights: true`.
  static LlamaConfig llama32_3BConfig({
    Device device = Device.CPU,
    int seed = 0,
    int? maxCtx,
  }) => LlamaConfig(
    vocabSize: 128256,
    maxCtx: maxCtx ?? 131072,
    embedDim: 3072,
    numLayers: 28,
    numHeads: 24,
    numKvHeads: 8,
    ffnDim: 8192,
    ropeBase: 500000.0,
    rmsNormEps: 1e-5,
    tieWeights: true,
    device: device,
    seed: seed,
  );

  /// Llama-3.1-8B-Instruct config. `tieWeights: false`.
  static LlamaConfig llama31_8BConfig({
    Device device = Device.CPU,
    int seed = 0,
    int? maxCtx,
  }) => LlamaConfig(
    vocabSize: 128256,
    maxCtx: maxCtx ?? 131072,
    embedDim: 4096,
    numLayers: 32,
    numHeads: 32,
    numKvHeads: 8,
    ffnDim: 14336,
    ropeBase: 500000.0,
    rmsNormEps: 1e-5,
    tieWeights: false,
    device: device,
    seed: seed,
  );

  static LlamaLoadReport loadFile(Llama model, String path) {
    final state = SafeTensors.loadFile(path);
    return loadMap(model, state);
  }

  static LlamaLoadReport loadMap(Llama model, Map<String, Tensor> state) {
    final consumed = <String>{};

    Tensor take(String name) {
      final t = state[name];
      if (t == null) {
        throw ArgumentError('llama loader: missing tensor "$name"');
      }
      consumed.add(name);
      return t;
    }

    final cfg = model.config;
    final d = cfg.embedDim;
    final h = cfg.numHeads;
    final kvH = cfg.numKvHeads;
    final headDim = d ~/ h;
    final ffn = cfg.ffnDim;

    // -------- token embedding --------
    _copy(
      model.embedIn.weight,
      _expectShape(take('model.embed_tokens.weight'), [
        cfg.vocabSize,
        d,
      ], 'model.embed_tokens.weight'),
    );

    // -------- per-layer --------
    for (int i = 0; i < cfg.numLayers; i++) {
      final block = model.blocks[i];
      final p = 'model.layers.$i';

      // input_layernorm (pre-attention RMSNorm).
      _copy(
        block.attnNorm.gamma,
        _expectShape(take('$p.input_layernorm.weight'), [
          d,
        ], '$p.input_layernorm.weight'),
      );

      // post_attention_layernorm (pre-FFN RMSNorm).
      _copy(
        block.ffnNorm.gamma,
        _expectShape(
          take('$p.post_attention_layernorm.weight'),
          [d],
          '$p.post_attention_layernorm.weight',
        ),
      );

      // q_proj — [H*hd, D], sliced by row into `h` per-head [hd, D]
      // slices.
      final qW = _expectShape(take('$p.self_attn.q_proj.weight'), [
        h * headDim,
        d,
      ], '$p.self_attn.q_proj.weight');
      for (int hh = 0; hh < h; hh++) {
        _copy(
          block.attn.wq[hh].weight,
          _sliceRows(qW, hh * headDim, (hh + 1) * headDim),
        );
      }

      // k_proj — [Hkv*hd, D], sliced by row into `kvH` per-kv-head
      // [hd, D] slices.
      final kW = _expectShape(take('$p.self_attn.k_proj.weight'), [
        kvH * headDim,
        d,
      ], '$p.self_attn.k_proj.weight');
      for (int hh = 0; hh < kvH; hh++) {
        _copy(
          block.attn.wk[hh].weight,
          _sliceRows(kW, hh * headDim, (hh + 1) * headDim),
        );
      }

      // v_proj — same shape/layout as k_proj.
      final vW = _expectShape(take('$p.self_attn.v_proj.weight'), [
        kvH * headDim,
        d,
      ], '$p.self_attn.v_proj.weight');
      for (int hh = 0; hh < kvH; hh++) {
        _copy(
          block.attn.wv[hh].weight,
          _sliceRows(vW, hh * headDim, (hh + 1) * headDim),
        );
      }

      // o_proj — [D, D].
      _copy(
        block.attn.wo.weight,
        _expectShape(take('$p.self_attn.o_proj.weight'), [
          d,
          d,
        ], '$p.self_attn.o_proj.weight'),
      );

      // SwiGLU FFN. gate/up: [F, D]; down: [D, F].
      _copy(
        block.ffn.gateProj.weight,
        _expectShape(take('$p.mlp.gate_proj.weight'), [
          ffn,
          d,
        ], '$p.mlp.gate_proj.weight'),
      );
      _copy(
        block.ffn.upProj.weight,
        _expectShape(take('$p.mlp.up_proj.weight'), [
          ffn,
          d,
        ], '$p.mlp.up_proj.weight'),
      );
      _copy(
        block.ffn.downProj.weight,
        _expectShape(take('$p.mlp.down_proj.weight'), [
          d,
          ffn,
        ], '$p.mlp.down_proj.weight'),
      );
    }

    // Final RMSNorm.
    _copy(
      model.finalNorm.gamma,
      _expectShape(take('model.norm.weight'), [d], 'model.norm.weight'),
    );

    // Untied lm_head (only present when tieWeights == false).
    if (!cfg.tieWeights) {
      _copy(
        model.untiedHead!.weight,
        _expectShape(take('lm_head.weight'), [
          cfg.vocabSize,
          d,
        ], 'lm_head.weight'),
      );
    } else if (state.containsKey('lm_head.weight')) {
      // Some tied checkpoints redundantly serialize the head — mark
      // it consumed so it doesn't show up in `unusedKeys`.
      consumed.add('lm_head.weight');
    }

    final unused = state.keys.where((k) => !consumed.contains(k)).toList()
      ..sort();
    return LlamaLoadReport(consumedCount: consumed.length, unusedKeys: unused);
  }

  // ---------------- tensor helpers (CPU only) ----------------

  static Tensor _expectShape(Tensor t, List<int> expected, String name) {
    if (t.shape.length != expected.length || !_shapesEqual(t.shape, expected)) {
      throw ArgumentError(
        'llama loader: "$name" expected shape $expected, got ${t.shape}',
      );
    }
    return t;
  }

  static bool _shapesEqual(List<int> a, List<int> b) {
    for (int i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  static void _copy(Tensor dst, Tensor src) {
    if (dst.length != src.length) {
      throw ArgumentError(
        'llama loader: copy length mismatch — dst=${dst.shape} '
        '(${dst.length}), src=${src.shape} (${src.length})',
      );
    }
    final vals = src.toList();
    final matched = Tensor.fromList(dst.shape, vals, device: dst.device);
    dst.assign(matched);
  }

  static Tensor _sliceRows(Tensor t, int start, int end) {
    if (t.shape.length != 2) {
      throw ArgumentError('_sliceRows: expected rank 2, got ${t.shape}');
    }
    final c = t.shape[1];
    final src = t.toList();
    final n = (end - start) * c;
    final out = Float32List(n);
    for (int i = 0; i < n; i++) {
      out[i] = src[start * c + i];
    }
    return Tensor.fromList([end - start, c], out, device: Device.CPU);
  }
}

class LlamaLoadReport {
  final int consumedCount;
  final List<String> unusedKeys;

  const LlamaLoadReport({
    required this.consumedCount,
    required this.unusedKeys,
  });

  @override
  String toString() =>
      'LlamaLoadReport(consumed=$consumedCount, '
      'unused=${unusedKeys.length})';
}
