/// Loads HuggingFace **Pythia** (`gpt_neox` architecture) weights from
/// a safetensors file into a [PythiaModel].
///
/// Supported checkpoints (all use the standard GPT-NeoX tokenizer,
/// vocab 50304, context 2048):
///
///   * `EleutherAI/pythia-14m`   — 6 layers, 128 embed, 8 heads
///   * `EleutherAI/pythia-70m`   — 6 layers, 512 embed, 8 heads
///   * `EleutherAI/pythia-160m`  — 12 layers, 768 embed, 12 heads
///   * `EleutherAI/pythia-410m`  — 24 layers, 1024 embed, 16 heads
///
/// HF key conventions handled here:
///
///   * `gpt_neox.embed_in.weight`
///   * `gpt_neox.layers.{i}.input_layernorm.{weight,bias}`
///   * `gpt_neox.layers.{i}.post_attention_layernorm.{weight,bias}`
///   * `gpt_neox.layers.{i}.attention.query_key_value.{weight,bias}`
///     — shape `[3*D, D]` / `[3*D]`; per-head Q|K|V interleaved as
///     `[num_heads, 3, head_dim, ...]` (**not** `[3, num_heads, ...]`
///     the way GPT-2 does it).
///   * `gpt_neox.layers.{i}.attention.dense.{weight,bias}` — output
///     projection, standard `[D, D]` PyTorch Linear.
///   * `gpt_neox.layers.{i}.mlp.dense_h_to_4h.{weight,bias}` — `[4D, D]`.
///   * `gpt_neox.layers.{i}.mlp.dense_4h_to_h.{weight,bias}` — `[D, 4D]`.
///   * `gpt_neox.final_layer_norm.{weight,bias}`
///   * `embed_out.weight` — untied `[V, D]`.
///
/// Ignored (safe): per-layer `attention.rotary_emb.inv_freq`,
/// `attention.bias`, `attention.masked_bias` — we don't need HF's
/// cached rotary freqs (RopeCache recomputes them) and we build
/// causal masks on the fly.
library;

import 'dart:typed_data';

import '../tensor/tensor.dart';
import '../tensor/dtype.dart';
import 'pythia.dart';
import 'safetensors.dart';

class PythiaHFLoader {
  /// Pythia-14m (deduped model). Vocab 50304, ctx 2048, 6 layers,
  /// embed 128, 8 heads.
  static PythiaConfig pythia14mConfig({
    Device device = Device.CPU,
    int seed = 0,
  }) => PythiaConfig(
    vocabSize: 50304,
    maxCtx: 2048,
    embedDim: 128,
    numLayers: 6,
    numHeads: 8,
    device: device,
    seed: seed,
  );

  /// Pythia-70m. Vocab 50304, ctx 2048, 6 layers, embed 512, 8 heads.
  static PythiaConfig pythia70mConfig({
    Device device = Device.CPU,
    int seed = 0,
  }) => PythiaConfig(
    vocabSize: 50304,
    maxCtx: 2048,
    embedDim: 512,
    numLayers: 6,
    numHeads: 8,
    device: device,
    seed: seed,
  );

  /// Pythia-160m. Vocab 50304, ctx 2048, 12 layers, embed 768, 12 heads.
  static PythiaConfig pythia160mConfig({
    Device device = Device.CPU,
    int seed = 0,
  }) => PythiaConfig(
    vocabSize: 50304,
    maxCtx: 2048,
    embedDim: 768,
    numLayers: 12,
    numHeads: 12,
    device: device,
    seed: seed,
  );

  /// Pythia-410m. Vocab 50304, ctx 2048, 24 layers, embed 1024,
  /// 16 heads.
  static PythiaConfig pythia410mConfig({
    Device device = Device.CPU,
    int seed = 0,
  }) => PythiaConfig(
    vocabSize: 50304,
    maxCtx: 2048,
    embedDim: 1024,
    numLayers: 24,
    numHeads: 16,
    device: device,
    seed: seed,
  );

  /// Pythia-1b. Vocab 50304, ctx 2048, 16 layers, embed 2048,
  /// 8 heads (headDim=256, rotaryDim=64). ~3.3 GB fp32 weights.
  static PythiaConfig pythia1bConfig({
    Device device = Device.CPU,
    int seed = 0,
  }) => PythiaConfig(
    vocabSize: 50304,
    maxCtx: 2048,
    embedDim: 2048,
    numLayers: 16,
    numHeads: 8,
    device: device,
    seed: seed,
  );

  /// Load a `.safetensors` checkpoint into [model]. When [keepFp16]
  /// is true, `F16` entries are held in fp16 CPU storage (read-only,
  /// halves resident RAM); the fp16 path is inference-only.
  static PythiaLoadReport loadFile(
    PythiaModel model,
    String path, {
    bool keepFp16 = false,
  }) {
    final state = SafeTensors.loadFile(path, keepFp16: keepFp16);
    return loadMap(model, state);
  }

  /// Load a sharded Pythia checkpoint via `model.safetensors.index.json`.
  static PythiaLoadReport loadSharded(
    PythiaModel model,
    String indexPath, {
    bool keepFp16 = false,
  }) {
    final state = SafeTensors.loadSharded(indexPath, keepFp16: keepFp16);
    return loadMap(model, state);
  }

  static PythiaLoadReport loadMap(
    PythiaModel model,
    Map<String, Tensor> state,
  ) {
    final consumed = <String>{};

    Tensor take(String name) {
      final t = state[name];
      if (t == null) {
        throw ArgumentError('pythia loader: missing tensor "$name"');
      }
      consumed.add(name);
      return t;
    }

    final cfg = model.config;
    final d = cfg.embedDim;
    final h = cfg.numHeads;
    final headDim = d ~/ h;
    final ffn = cfg.ffnDim;

    // -------- token embedding --------
    _copy(
      model.embedIn.weight,
      _expectShape(take('gpt_neox.embed_in.weight'), [
        cfg.vocabSize,
        d,
      ], 'gpt_neox.embed_in.weight'),
    );

    // -------- per-layer --------
    for (int i = 0; i < cfg.numLayers; i++) {
      final block = model.blocks[i];
      final p = 'gpt_neox.layers.$i';

      // input_layernorm
      _copy(
        block.inputLn.gamma,
        _expectShape(take('$p.input_layernorm.weight'), [
          d,
        ], '$p.input_layernorm.weight'),
      );
      _copy(
        block.inputLn.beta,
        _expectShape(take('$p.input_layernorm.bias'), [
          d,
        ], '$p.input_layernorm.bias'),
      );

      // post_attention_layernorm
      _copy(
        block.postAttnLn.gamma,
        _expectShape(
          take('$p.post_attention_layernorm.weight'),
          [d],
          '$p.post_attention_layernorm.weight',
        ),
      );
      _copy(
        block.postAttnLn.beta,
        _expectShape(
          take('$p.post_attention_layernorm.bias'),
          [d],
          '$p.post_attention_layernorm.bias',
        ),
      );

      // Fused Q|K|V — per-head interleaved:
      //   rows [h*3*hd .. h*3*hd + hd)         -> Q for head h
      //   rows [h*3*hd + hd .. h*3*hd + 2*hd)  -> K for head h
      //   rows [h*3*hd + 2*hd .. (h+1)*3*hd)   -> V for head h
      final qkvW = _expectShape(
        take('$p.attention.query_key_value.weight'),
        [3 * d, d],
        '$p.attention.query_key_value.weight',
      );
      final qkvB = _expectShape(
        take('$p.attention.query_key_value.bias'),
        [3 * d],
        '$p.attention.query_key_value.bias',
      );
      for (int hh = 0; hh < h; hh++) {
        final qStart = hh * 3 * headDim;
        final kStart = qStart + headDim;
        final vStart = qStart + 2 * headDim;
        _copy(
          block.attn.wq[hh].weight,
          _sliceRows(qkvW, qStart, qStart + headDim),
        );
        _copy(
          block.attn.wq[hh].bias!,
          _reshapeVectorTo1xN(_sliceVector(qkvB, qStart, qStart + headDim)),
        );
        _copy(
          block.attn.wk[hh].weight,
          _sliceRows(qkvW, kStart, kStart + headDim),
        );
        _copy(
          block.attn.wk[hh].bias!,
          _reshapeVectorTo1xN(_sliceVector(qkvB, kStart, kStart + headDim)),
        );
        _copy(
          block.attn.wv[hh].weight,
          _sliceRows(qkvW, vStart, vStart + headDim),
        );
        _copy(
          block.attn.wv[hh].bias!,
          _reshapeVectorTo1xN(_sliceVector(qkvB, vStart, vStart + headDim)),
        );
      }

      // Output projection.
      _copy(
        block.attn.wo.weight,
        _expectShape(take('$p.attention.dense.weight'), [
          d,
          d,
        ], '$p.attention.dense.weight'),
      );
      _copy(
        block.attn.wo.bias!,
        _reshapeVectorTo1xN(
          _expectShape(take('$p.attention.dense.bias'), [
            d,
          ], '$p.attention.dense.bias'),
        ),
      );

      // MLP: PyTorch Linear layouts, no transpose.
      _copy(
        block.ffn1.weight,
        _expectShape(take('$p.mlp.dense_h_to_4h.weight'), [
          ffn,
          d,
        ], '$p.mlp.dense_h_to_4h.weight'),
      );
      _copy(
        block.ffn1.bias!,
        _reshapeVectorTo1xN(
          _expectShape(take('$p.mlp.dense_h_to_4h.bias'), [
            ffn,
          ], '$p.mlp.dense_h_to_4h.bias'),
        ),
      );
      _copy(
        block.ffn2.weight,
        _expectShape(take('$p.mlp.dense_4h_to_h.weight'), [
          d,
          ffn,
        ], '$p.mlp.dense_4h_to_h.weight'),
      );
      _copy(
        block.ffn2.bias!,
        _reshapeVectorTo1xN(
          _expectShape(take('$p.mlp.dense_4h_to_h.bias'), [
            d,
          ], '$p.mlp.dense_4h_to_h.bias'),
        ),
      );
    }

    // Final LN + untied output head.
    _copy(
      model.finalLn.gamma,
      _expectShape(
        take('gpt_neox.final_layer_norm.weight'),
        [d],
        'gpt_neox.final_layer_norm.weight',
      ),
    );
    _copy(
      model.finalLn.beta,
      _expectShape(take('gpt_neox.final_layer_norm.bias'), [
        d,
      ], 'gpt_neox.final_layer_norm.bias'),
    );
    _copy(
      model.embedOut.weight,
      _expectShape(take('embed_out.weight'), [
        cfg.vocabSize,
        d,
      ], 'embed_out.weight'),
    );

    final unused = state.keys.where((k) => !consumed.contains(k)).toList()
      ..sort();
    return PythiaLoadReport(consumedCount: consumed.length, unusedKeys: unused);
  }

  // ---------------- tensor helpers (CPU only) ----------------

  static Tensor _expectShape(Tensor t, List<int> expected, String name) {
    if (t.shape.length != expected.length || !_shapesEqual(t.shape, expected)) {
      throw ArgumentError(
        'pythia loader: "$name" expected shape $expected, got ${t.shape}',
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
        'pythia loader: copy length mismatch — dst=${dst.shape} '
        '(${dst.length}), src=${src.shape} (${src.length})',
      );
    }
    if (src.dtype == DType.fp16 && dst.device == Device.CPU) {
      dst.adoptCpuStorageFrom(src);
      return;
    }
    final vals = src.toList();
    final matched = Tensor.fromList(dst.shape, vals, device: dst.device);
    dst.assign(matched);
  }

  // Delegates to Tensor.sliceRows so fp16 storage is preserved
  // through the per-head Q/K/V split.
  static Tensor _sliceRows(Tensor t, int start, int end) =>
      t.sliceRows(start, end);

  static Tensor _sliceVector(Tensor t, int start, int end) {
    if (t.shape.length != 1) {
      throw ArgumentError('_sliceVector: expected rank 1, got ${t.shape}');
    }
    final src = t.toList();
    final n = end - start;
    final out = Float32List(n);
    for (int i = 0; i < n; i++) {
      out[i] = src[start + i];
    }
    return Tensor.fromList([n], out, device: Device.CPU);
  }

  static Tensor _reshapeVectorTo1xN(Tensor v) {
    if (v.shape.length != 1) {
      throw ArgumentError(
        '_reshapeVectorTo1xN: expected rank 1, got ${v.shape}',
      );
    }
    final n = v.shape[0];
    return Tensor.fromList([1, n], v.toList(), device: Device.CPU);
  }
}

class PythiaLoadReport {
  final int consumedCount;
  final List<String> unusedKeys;

  const PythiaLoadReport({
    required this.consumedCount,
    required this.unusedKeys,
  });

  @override
  String toString() =>
      'PythiaLoadReport(consumed=$consumedCount, '
      'unused=${unusedKeys.length})';
}
