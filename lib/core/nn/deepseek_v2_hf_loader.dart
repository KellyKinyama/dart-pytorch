/// Loader for HuggingFace DeepSeek-V2 (`DeepseekV2ForCausalLM`)
/// safetensors into a [DeepSeekV2Model].
///
/// Supported checkpoints:
///
///   * `deepseek-ai/DeepSeek-V2-Lite` (`config = DeepSeekV2Config.lite()`)
///     — 16 B total (2.4 B active), no Q compression, 27 layers,
///     64 routed + 2 shared experts.
///   * `deepseek-ai/DeepSeek-V2` (`config = DeepSeekV2Config.full()`)
///     — 236 B total (21 B active), Q-lora 1536, 60 layers,
///     160 routed + 2 shared experts across 8 groups (top-3 group).
///
/// HF key layout (top-level `model.` prefix):
///
///   * `model.embed_tokens.weight`                    `[V, D]`
///   * `model.layers.{i}.input_layernorm.weight`       `[D]`
///   * `model.layers.{i}.self_attn.*` — MLA weights:
///     - **With Q compression** (`qLoraRank != null`):
///       - `q_a_proj.weight`                          `[qLoraRank, D]`
///       - `q_a_layernorm.weight`                     `[qLoraRank]`
///       - `q_b_proj.weight`                          `[H · qkHeadDim, qLoraRank]`
///     - **Without** (`qLoraRank == null`, Lite):
///       - `q_proj.weight`                            `[H · qkHeadDim, D]`
///     - Shared:
///       - `kv_a_proj_with_mqa.weight`               `[kvLoraRank + qkRopeHeadDim, D]`
///         (row-split into `kvDown` + `kRope`)
///       - `kv_a_layernorm.weight`                    `[kvLoraRank]`
///       - `kv_b_proj.weight`                         `[H · (qkNopeHeadDim + vHeadDim), kvLoraRank]`
///         (row-split per head into `kUpNope[h]` + `vUp[h]`)
///       - `o_proj.weight`                            `[D, H · vHeadDim]`
///   * `model.layers.{i}.post_attention_layernorm.weight`  `[D]`
///   * `model.layers.{i}.mlp.*` — FFN:
///     - Dense layer (`i < firstKDenseReplace`):
///       - `mlp.gate_proj.weight`                     `[denseFfnDim, D]`
///       - `mlp.up_proj.weight`                       `[denseFfnDim, D]`
///       - `mlp.down_proj.weight`                     `[D, denseFfnDim]`
///     - MoE layer:
///       - `mlp.gate.weight`                          `[numRoutedExperts, D]` (transposed on copy)
///       - `mlp.experts.{j}.{gate,up,down}_proj.weight` per routed expert
///       - `mlp.shared_experts.{gate,up,down}_proj.weight` — one fused
///         body of size `numSharedExperts · moeExpertHiddenDim`. Row/col-
///         split into per-expert `sharedExperts[k]` bodies.
///   * `model.norm.weight`                            `[D]`
///   * `lm_head.weight`                               `[V, D]` (untied only)
///
/// Ignored (safe): `model.layers.{i}.self_attn.rotary_emb.inv_freq`
/// — we recompute RoPE from `ropeBase`.
library;

import 'dart:typed_data';

import '../tensor/tensor.dart';
import 'deepseek_v2.dart';
import 'moe.dart';
import 'safetensors.dart';

class DeepSeekV2LoadReport {
  final int consumedCount;
  final List<String> unusedKeys;
  const DeepSeekV2LoadReport({
    required this.consumedCount,
    required this.unusedKeys,
  });

  @override
  String toString() =>
      'DeepSeekV2LoadReport(consumed=$consumedCount, '
      'unused=${unusedKeys.length})';
}

class DeepSeekV2HFLoader {
  static DeepSeekV2LoadReport loadFile(DeepSeekV2Model model, String path) {
    final state = SafeTensors.loadFile(path);
    return loadMap(model, state);
  }

  static DeepSeekV2LoadReport loadSharded(
    DeepSeekV2Model model,
    String indexPath,
  ) {
    final state = SafeTensors.loadSharded(indexPath);
    return loadMap(model, state);
  }

  static DeepSeekV2LoadReport loadMap(
    DeepSeekV2Model model,
    Map<String, Tensor> state,
  ) {
    final consumed = <String>{};

    Tensor take(String name) {
      final t = state[name];
      if (t == null) {
        throw ArgumentError('deepseek-v2 loader: missing tensor "$name"');
      }
      consumed.add(name);
      return t;
    }

    final cfg = model.config;
    final mla = cfg.mlaConfig;
    final d = cfg.embedDim;
    final h = mla.numHeads;
    final headDim = mla.qkHeadDim;
    final nopeD = mla.qkNopeHeadDim;
    final ropeD = mla.qkRopeHeadDim;
    final vD = mla.vHeadDim;
    final kvL = mla.kvLoraRank;

    // ---------- token embedding ----------
    _copy(
      model.embedIn.weight,
      _expectShape(take('model.embed_tokens.weight'), [cfg.vocabSize, d],
          'model.embed_tokens.weight'),
    );

    // ---------- per-layer ----------
    for (int i = 0; i < cfg.numLayers; i++) {
      final block = model.blocks[i];
      final p = 'model.layers.$i';

      _copy(
        block.attnLn.gamma,
        _expectShape(take('$p.input_layernorm.weight'), [d],
            '$p.input_layernorm.weight'),
      );
      _copy(
        block.ffnLn.gamma,
        _expectShape(take('$p.post_attention_layernorm.weight'), [d],
            '$p.post_attention_layernorm.weight'),
      );

      // ============ MLA ============
      final attn = block.attn;
      if (mla.qLoraRank != null) {
        // Compressed Q path.
        _copy(
          attn.qDown!.weight,
          _expectShape(take('$p.self_attn.q_a_proj.weight'),
              [mla.qLoraRank!, d], '$p.self_attn.q_a_proj.weight'),
        );
        _copy(
          attn.qLn!.gamma,
          _expectShape(take('$p.self_attn.q_a_layernorm.weight'),
              [mla.qLoraRank!], '$p.self_attn.q_a_layernorm.weight'),
        );
        final qBw = _expectShape(take('$p.self_attn.q_b_proj.weight'),
            [h * headDim, mla.qLoraRank!], '$p.self_attn.q_b_proj.weight');
        _loadPerHeadQ(attn, qBw, h, nopeD, ropeD, mla.qLoraRank!);
      } else {
        // Direct Q path (Lite).
        final qW = _expectShape(take('$p.self_attn.q_proj.weight'),
            [h * headDim, d], '$p.self_attn.q_proj.weight');
        _loadPerHeadQ(attn, qW, h, nopeD, ropeD, d);
      }

      // KV low-rank + shared K-rope (fused single Linear on HF side).
      final kvAw = _expectShape(
        take('$p.self_attn.kv_a_proj_with_mqa.weight'),
        [kvL + ropeD, d],
        '$p.self_attn.kv_a_proj_with_mqa.weight',
      );
      // Row-split: rows [0, kvL) → kvDown; rows [kvL, kvL+ropeD) → kRope.
      _copy(attn.kvDown.weight, _sliceRows(kvAw, 0, kvL));
      _copy(attn.kRope.weight, _sliceRows(kvAw, kvL, kvL + ropeD));

      _copy(
        attn.kvLn.gamma,
        _expectShape(take('$p.self_attn.kv_a_layernorm.weight'), [kvL],
            '$p.self_attn.kv_a_layernorm.weight'),
      );

      // kv_b_proj: [H · (nopeD + vD), kvL], per-head split into K_nope + V.
      final kvBw = _expectShape(take('$p.self_attn.kv_b_proj.weight'),
          [h * (nopeD + vD), kvL], '$p.self_attn.kv_b_proj.weight');
      for (int hh = 0; hh < h; hh++) {
        final base = hh * (nopeD + vD);
        _copy(attn.kUpNope[hh].weight, _sliceRows(kvBw, base, base + nopeD));
        _copy(attn.vUp[hh].weight,
            _sliceRows(kvBw, base + nopeD, base + nopeD + vD));
      }

      _copy(
        attn.oProj.weight,
        _expectShape(take('$p.self_attn.o_proj.weight'), [d, h * vD],
            '$p.self_attn.o_proj.weight'),
      );

      // Silently absorb the recomputed rotary buffer if HF included it.
      if (state.containsKey('$p.self_attn.rotary_emb.inv_freq')) {
        consumed.add('$p.self_attn.rotary_emb.inv_freq');
      }

      // ============ FFN ============
      if (!block.isMoE) {
        final ffn = block.denseFfn!;
        _copy(
          ffn.gateProj.weight,
          _expectShape(take('$p.mlp.gate_proj.weight'), [cfg.denseFfnDim, d],
              '$p.mlp.gate_proj.weight'),
        );
        _copy(
          ffn.upProj.weight,
          _expectShape(take('$p.mlp.up_proj.weight'), [cfg.denseFfnDim, d],
              '$p.mlp.up_proj.weight'),
        );
        _copy(
          ffn.downProj.weight,
          _expectShape(take('$p.mlp.down_proj.weight'), [d, cfg.denseFfnDim],
              '$p.mlp.down_proj.weight'),
        );
      } else {
        _loadMoE(block.moeFfn!, cfg, take, p);
      }
    }

    // ---------- final RMSNorm ----------
    _copy(
      model.finalNorm.gamma,
      _expectShape(take('model.norm.weight'), [d], 'model.norm.weight'),
    );

    // ---------- lm_head ----------
    if (!cfg.tieWordEmbeddings) {
      _copy(
        model.untiedHead!.weight,
        _expectShape(take('lm_head.weight'), [cfg.vocabSize, d],
            'lm_head.weight'),
      );
    } else if (state.containsKey('lm_head.weight')) {
      consumed.add('lm_head.weight');
    }

    final unused = state.keys.where((k) => !consumed.contains(k)).toList()
      ..sort();
    return DeepSeekV2LoadReport(
      consumedCount: consumed.length,
      unusedKeys: unused,
    );
  }

  /// Common Q-path per-head split — used for both the compressed
  /// (`q_b_proj`) and uncompressed (`q_proj`) fused Q weights. Layout
  /// is `[num_heads · (nopeD + ropeD), inDim]` with each per-head block
  /// laid out as `[nope | rope]`.
  static void _loadPerHeadQ(
    dynamic attn, // MultiHeadLatentAttention (avoid extra import)
    Tensor fused,
    int h,
    int nopeD,
    int ropeD,
    int inDim,
  ) {
    final headDim = nopeD + ropeD;
    for (int hh = 0; hh < h; hh++) {
      final base = hh * headDim;
      _copy(attn.qUpNope[hh].weight, _sliceRows(fused, base, base + nopeD));
      _copy(attn.qUpRope[hh].weight,
          _sliceRows(fused, base + nopeD, base + headDim));
    }
  }

  static void _loadMoE(
    MoEFeedForward moe,
    DeepSeekV2Config cfg,
    Tensor Function(String) take,
    String layerPrefix,
  ) {
    final d = cfg.embedDim;
    final ffn = cfg.moeExpertHiddenDim;

    // Router — HF stores `[numRoutedExperts, D]`; ours is `[D, E]`.
    final routerHF = _expectShape(
      take('$layerPrefix.mlp.gate.weight'),
      [cfg.numRoutedExperts, d],
      '$layerPrefix.mlp.gate.weight',
    );
    _copy(moe.gateW, _transpose2D(routerHF));

    // Routed experts.
    for (int j = 0; j < cfg.numRoutedExperts; j++) {
      final expert = moe.routedExperts[j];
      final ep = '$layerPrefix.mlp.experts.$j';
      _copy(
        expert.w1.weight,
        _expectShape(take('$ep.gate_proj.weight'), [ffn, d],
            '$ep.gate_proj.weight'),
      );
      _copy(
        expert.w3!.weight,
        _expectShape(take('$ep.up_proj.weight'), [ffn, d],
            '$ep.up_proj.weight'),
      );
      _copy(
        expert.w2.weight,
        _expectShape(take('$ep.down_proj.weight'), [d, ffn],
            '$ep.down_proj.weight'),
      );
      // HF's DeepseekV2MLP has no biases — zero the biases our
      // `Expert` allocates by default so they don't drift the
      // forward pass.
      _zeroBiases(expert);
    }

    // Shared experts — HF stores ONE fused body of intermediate size
    // `numSharedExperts * moeExpertHiddenDim`. Split into our N
    // per-expert bodies (equivalent under sum: each expert reads a
    // row-slice of gate/up and outputs a column-slice of down).
    final sharedFfn = cfg.numSharedExperts * ffn;
    final sp = '$layerPrefix.mlp.shared_experts';
    final gateFused = _expectShape(take('$sp.gate_proj.weight'), [sharedFfn, d],
        '$sp.gate_proj.weight');
    final upFused = _expectShape(take('$sp.up_proj.weight'), [sharedFfn, d],
        '$sp.up_proj.weight');
    final downFused = _expectShape(take('$sp.down_proj.weight'), [d, sharedFfn],
        '$sp.down_proj.weight');
    for (int k = 0; k < cfg.numSharedExperts; k++) {
      final e = moe.sharedExperts[k];
      final rStart = k * ffn;
      _copy(e.w1.weight, _sliceRows(gateFused, rStart, rStart + ffn));
      _copy(e.w3!.weight, _sliceRows(upFused, rStart, rStart + ffn));
      _copy(e.w2.weight, _sliceCols(downFused, rStart, rStart + ffn));
      _zeroBiases(e);
    }
  }

  /// Zero out the auto-allocated biases on an [Expert] (both routed
  /// and shared). HF's `DeepseekV2MLP` has no biases; leaving these
  /// at their init values would drift the forward pass.
  static void _zeroBiases(dynamic expert) {
    if (expert.w1.bias != null) {
      final shape = expert.w1.bias!.shape as List<int>;
      final n = shape.fold<int>(1, (a, b) => a * b);
      final zeros = Tensor.fromList(shape, List<double>.filled(n, 0.0),
          device: expert.w1.bias!.device);
      expert.w1.bias!.assign(zeros);
    }
    if (expert.w2.bias != null) {
      final shape = expert.w2.bias!.shape as List<int>;
      final n = shape.fold<int>(1, (a, b) => a * b);
      final zeros = Tensor.fromList(shape, List<double>.filled(n, 0.0),
          device: expert.w2.bias!.device);
      expert.w2.bias!.assign(zeros);
    }
    if (expert.w3 != null && expert.w3!.bias != null) {
      final shape = expert.w3!.bias!.shape as List<int>;
      final n = shape.fold<int>(1, (a, b) => a * b);
      final zeros = Tensor.fromList(shape, List<double>.filled(n, 0.0),
          device: expert.w3!.bias!.device);
      expert.w3!.bias!.assign(zeros);
    }
  }

  // -------------------- tensor helpers --------------------

  static Tensor _expectShape(Tensor t, List<int> expected, String name) {
    if (t.shape.length != expected.length ||
        !_shapesEqual(t.shape, expected)) {
      throw ArgumentError(
        'deepseek-v2 loader: "$name" expected shape $expected, got ${t.shape}',
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
        'deepseek-v2 loader: copy length mismatch — dst=${dst.shape} '
        '(${dst.length}), src=${src.shape} (${src.length})',
      );
    }
    final vals = src.toList();
    final matched = Tensor.fromList(dst.shape, vals, device: dst.device);
    dst.assign(matched);
  }

  static Tensor _sliceRows(Tensor t, int start, int end) =>
      t.sliceRows(start, end);

  /// Column slice for a `[rows, cols]` tensor. Used to split the fused
  /// `shared_experts.down_proj.weight` into per-expert bodies.
  static Tensor _sliceCols(Tensor t, int start, int end) {
    if (t.shape.length != 2) {
      throw ArgumentError('_sliceCols: expected rank 2, got ${t.shape}');
    }
    final rows = t.shape[0];
    final cols = t.shape[1];
    final n = end - start;
    final data = t.toList();
    final out = Float32List(rows * n);
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < n; c++) {
        out[r * n + c] = data[r * cols + start + c];
      }
    }
    return Tensor.fromFloat32List([rows, n], out, device: Device.CPU);
  }

  /// Transpose `[R, C]` -> `[C, R]`. HF's router is `[E, D]`; ours
  /// wants `[D, E]` to match the matmul `logits = x @ gateW`.
  static Tensor _transpose2D(Tensor t) {
    if (t.shape.length != 2) {
      throw ArgumentError('_transpose2D: expected rank 2, got ${t.shape}');
    }
    final r = t.shape[0];
    final c = t.shape[1];
    final data = t.toList();
    final out = Float32List(r * c);
    for (int i = 0; i < r; i++) {
      for (int j = 0; j < c; j++) {
        out[j * r + i] = data[i * c + j];
      }
    }
    return Tensor.fromFloat32List([c, r], out, device: Device.CPU);
  }
}
