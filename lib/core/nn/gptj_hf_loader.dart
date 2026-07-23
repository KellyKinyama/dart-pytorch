/// Loads HuggingFace **GPT-J** (`EleutherAI/gpt-j-6B`) safetensors
/// weights into a [GPTJModel].
///
/// GPT-J HF key layout (all under `transformer.*` plus a separate
/// `lm_head.*`):
///
///   * `transformer.wte.weight`                         `[V, D]`
///   * `transformer.h.{i}.ln_1.{weight,bias}`           `[D]`
///   * `transformer.h.{i}.attn.q_proj.weight`           `[D, D]`  (no bias)
///   * `transformer.h.{i}.attn.k_proj.weight`           `[D, D]`  (no bias)
///   * `transformer.h.{i}.attn.v_proj.weight`           `[D, D]`  (no bias)
///   * `transformer.h.{i}.attn.out_proj.weight`         `[D, D]`  (no bias)
///   * `transformer.h.{i}.mlp.fc_in.{weight,bias}`      `[4D, D]`, `[4D]`
///   * `transformer.h.{i}.mlp.fc_out.{weight,bias}`     `[D, 4D]`, `[D]`
///   * `transformer.ln_f.{weight,bias}`                 `[D]`
///   * `lm_head.{weight,bias}`                          `[V, D]`, `[V]`
///
/// Ignored (safe): `transformer.h.{i}.attn.bias` (causal buffer),
/// `transformer.h.{i}.attn.masked_bias` (scalar), any per-layer
/// rotary `inv_freq` if present.
///
/// ## Rotary layout — the interleaved→half-split trick
///
/// GPT-J applies rotary to the first `rotary_dim` (=64 for 6B) of
/// each head with the **interleaved pair** convention:
/// ```
///   q_rot[..., 2i]   = q[..., 2i]  *cos(θ_i) - q[..., 2i+1]*sin(θ_i)
///   q_rot[..., 2i+1] = q[..., 2i+1]*cos(θ_i) + q[..., 2i]  *sin(θ_i)
/// ```
/// Our shared [RopeCache] uses the "half-split" GPT-NeoX/LLaMA
/// convention:
/// ```
///   q_rot[..., i]         = q[..., i]        *cos(θ_i) - q[..., i+H]*sin(θ_i)
///   q_rot[..., i+H]       = q[..., i+H]      *cos(θ_i) + q[..., i]  *sin(θ_i)
/// ```
/// where `H = rotary_dim / 2`. If we define a permutation
/// `π: dim → dim'` on each head such that
///   `π(2i) = i` and `π(2i+1) = i + H`  for `i in [0, H)`,
/// and leave `π(j) = j` for `j >= rotary_dim`, then applying our
/// half-split rotary to `π(q)` produces exactly the same values as
/// GPT-J's interleaved rotary applied to `q`, only re-indexed by π.
///
/// Because attention scores are `Q @ K^T`, and we permute *both* Q
/// and K by the same π, the score matrix is invariant:
///   `π(Q) @ π(K)^T = Q P P^T K^T = Q K^T`  (P is orthogonal).
///
/// V is not touched (V doesn't get rotary), so the attention output
/// and everything downstream matches the reference exactly. So the
/// only load-time work is: for every head's slice of q_proj.weight
/// and k_proj.weight, apply π to the first `rotary_dim` output rows.
library;

import 'dart:typed_data';

import '../tensor/tensor.dart';
import 'gptj.dart';
import 'safetensors.dart';

class GPTJHFLoader {
  /// EleutherAI/gpt-j-6B. 28 layers, 4096 embed, 16 heads (headDim=256),
  /// rotary_dim=64, ffn=16384, vocab=50400, ctx=2048. ~24 GB fp32
  /// weights, ~12 GB fp16.
  static GPTJConfig gptJ6bConfig({Device device = Device.CPU, int seed = 0}) =>
      GPTJConfig(
        vocabSize: 50400,
        maxCtx: 2048,
        embedDim: 4096,
        numLayers: 28,
        numHeads: 16,
        rotaryDim: 64,
        device: device,
        seed: seed,
      );

  /// Hybrid split preset for GPT-J-6B: place the first [gpuLayers]
  /// transformer blocks on GPU, the remaining `28 - gpuLayers` blocks
  /// on CPU. The token embedding (`wte`) and output stack
  /// (`ln_f` + `lm_head`) stay on CPU by default — each is ~826 MB
  /// fp32 and doesn't leave much room for transformer blocks
  /// alongside it on a 6 GB GPU.
  ///
  /// Per-block VRAM cost (fp32) is roughly
  /// `12 * embedDim^2 * 4 bytes ≈ 805 MB`, so budget ~5 blocks max
  /// for a 6 GB GPU (leaving room for activations, KV cache, rope
  /// tables, and the CUDA runtime itself).
  ///
  /// Set [embedOnGpu] / [lmHeadOnGpu] to override the CPU default
  /// for those pieces when you have spare VRAM.
  static GPTJConfig gptJ6bHybridConfig({
    required int gpuLayers,
    bool embedOnGpu = false,
    bool lmHeadOnGpu = false,
    int seed = 0,
  }) {
    const numLayers = 28;
    if (gpuLayers < 0 || gpuLayers > numLayers) {
      throw ArgumentError(
        'gptJ6bHybridConfig: gpuLayers ($gpuLayers) must be in [0, $numLayers]',
      );
    }
    final layerDevices = List<Device>.generate(
      numLayers,
      (i) => i < gpuLayers ? Device.GPU : Device.CPU,
    );
    return GPTJConfig(
      vocabSize: 50400,
      maxCtx: 2048,
      embedDim: 4096,
      numLayers: numLayers,
      numHeads: 16,
      rotaryDim: 64,
      device: Device.CPU, // primary/default; overridden per-block
      layerDevices: layerDevices,
      embeddingDevice: embedOnGpu ? Device.GPU : Device.CPU,
      lmHeadDevice: lmHeadOnGpu ? Device.GPU : Device.CPU,
      seed: seed,
    );
  }

  static GPTJLoadReport loadFile(GPTJModel model, String path) {
    final state = SafeTensors.loadFile(path);
    return loadMap(model, state);
  }

  static GPTJLoadReport loadMap(GPTJModel model, Map<String, Tensor> state) {
    final consumed = <String>{};

    Tensor take(String name) {
      final t = state[name];
      if (t == null) {
        throw ArgumentError('gpt-j loader: missing tensor "$name"');
      }
      consumed.add(name);
      return t;
    }

    final cfg = model.config;
    final d = cfg.embedDim;
    final h = cfg.numHeads;
    final headDim = d ~/ h;
    final ffn = cfg.ffnDim;
    final rDim = cfg.rotaryDim;

    // -------- token embedding --------
    _copy(
      model.wte.weight,
      _expectShape(take('transformer.wte.weight'), [
        cfg.vocabSize,
        d,
      ], 'transformer.wte.weight'),
    );

    // -------- per-layer --------
    for (int i = 0; i < cfg.numLayers; i++) {
      final block = model.blocks[i];
      final p = 'transformer.h.$i';

      // Shared LN (input to both attn and mlp).
      _copy(
        block.ln.gamma,
        _expectShape(take('$p.ln_1.weight'), [d], '$p.ln_1.weight'),
      );
      _copy(
        block.ln.beta,
        _expectShape(take('$p.ln_1.bias'), [d], '$p.ln_1.bias'),
      );

      // Q / K — need per-head interleaved→half-split permutation of
      // the first `rDim` output rows.
      final qW = _expectShape(take('$p.attn.q_proj.weight'), [
        d,
        d,
      ], '$p.attn.q_proj.weight');
      final kW = _expectShape(take('$p.attn.k_proj.weight'), [
        d,
        d,
      ], '$p.attn.k_proj.weight');
      // V — no rotary, no permutation.
      final vW = _expectShape(take('$p.attn.v_proj.weight'), [
        d,
        d,
      ], '$p.attn.v_proj.weight');
      for (int hh = 0; hh < h; hh++) {
        final start = hh * headDim;
        final qHead = _sliceRows(qW, start, start + headDim); // [headDim, D]
        final kHead = _sliceRows(kW, start, start + headDim);
        final vHead = _sliceRows(vW, start, start + headDim);
        _copy(block.attn.wq[hh].weight, _permuteRotaryRows(qHead, rDim));
        _copy(block.attn.wk[hh].weight, _permuteRotaryRows(kHead, rDim));
        _copy(block.attn.wv[hh].weight, vHead);
      }

      // Output projection — no bias, no permutation.
      _copy(
        block.attn.wo.weight,
        _expectShape(take('$p.attn.out_proj.weight'), [
          d,
          d,
        ], '$p.attn.out_proj.weight'),
      );

      // MLP.
      _copy(
        block.ffn1.weight,
        _expectShape(take('$p.mlp.fc_in.weight'), [
          ffn,
          d,
        ], '$p.mlp.fc_in.weight'),
      );
      _copy(
        block.ffn1.bias!,
        _reshape1xN(
          _expectShape(take('$p.mlp.fc_in.bias'), [ffn], '$p.mlp.fc_in.bias'),
        ),
      );
      _copy(
        block.ffn2.weight,
        _expectShape(take('$p.mlp.fc_out.weight'), [
          d,
          ffn,
        ], '$p.mlp.fc_out.weight'),
      );
      _copy(
        block.ffn2.bias!,
        _reshape1xN(
          _expectShape(take('$p.mlp.fc_out.bias'), [d], '$p.mlp.fc_out.bias'),
        ),
      );
    }

    // Final LN.
    _copy(
      model.finalLn.gamma,
      _expectShape(take('transformer.ln_f.weight'), [
        d,
      ], 'transformer.ln_f.weight'),
    );
    _copy(
      model.finalLn.beta,
      _expectShape(take('transformer.ln_f.bias'), [d], 'transformer.ln_f.bias'),
    );

    // Untied output head (with bias!).
    _copy(
      model.lmHead.weight,
      _expectShape(take('lm_head.weight'), [
        cfg.vocabSize,
        d,
      ], 'lm_head.weight'),
    );
    _copy(
      model.lmHead.bias!,
      _reshape1xN(
        _expectShape(take('lm_head.bias'), [cfg.vocabSize], 'lm_head.bias'),
      ),
    );

    final unused = state.keys.where((k) => !consumed.contains(k)).toList()
      ..sort();
    return GPTJLoadReport(consumedCount: consumed.length, unusedKeys: unused);
  }

  // ---------------- tensor helpers (CPU only) ----------------

  /// Apply the interleaved→half-split permutation to the first
  /// `rDim` rows of a `[headDim, D]` weight slice. Rows [rDim, headDim)
  /// are left untouched.
  ///
  ///   π(2i)   = i        for i in [0, H)
  ///   π(2i+1) = i + H    for i in [0, H)
  /// where `H = rDim / 2`. So `out[i, :] = src[2i, :]` and
  /// `out[i + H, :] = src[2i + 1, :]`.
  static Tensor _permuteRotaryRows(Tensor headSlice, int rDim) {
    if (headSlice.shape.length != 2) {
      throw ArgumentError(
        '_permuteRotaryRows: expected rank 2, got ${headSlice.shape}',
      );
    }
    final headDim = headSlice.shape[0];
    final d = headSlice.shape[1];
    if (rDim <= 0 || rDim > headDim || rDim.isOdd) {
      throw ArgumentError(
        '_permuteRotaryRows: rDim=$rDim invalid for headDim=$headDim',
      );
    }
    final halfR = rDim ~/ 2;
    final src = headSlice.toList();
    final out = Float32List(headDim * d);
    // rows in [0, rDim): apply π.
    for (int i = 0; i < halfR; i++) {
      // out[i, :]        <- src[2i, :]
      final srcBase0 = (2 * i) * d;
      final dstBase0 = i * d;
      for (int c = 0; c < d; c++) {
        out[dstBase0 + c] = src[srcBase0 + c];
      }
      // out[i + halfR, :] <- src[2i + 1, :]
      final srcBase1 = (2 * i + 1) * d;
      final dstBase1 = (i + halfR) * d;
      for (int c = 0; c < d; c++) {
        out[dstBase1 + c] = src[srcBase1 + c];
      }
    }
    // rows in [rDim, headDim): copy through.
    for (int r = rDim; r < headDim; r++) {
      final base = r * d;
      for (int c = 0; c < d; c++) {
        out[base + c] = src[base + c];
      }
    }
    return Tensor.fromList([headDim, d], out.toList(), device: Device.CPU);
  }

  static Tensor _expectShape(Tensor t, List<int> expected, String name) {
    if (t.shape.length != expected.length || !_shapesEqual(t.shape, expected)) {
      throw ArgumentError(
        'gpt-j loader: "$name" expected shape $expected, got ${t.shape}',
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
        'gpt-j loader: copy length mismatch — dst=${dst.shape} '
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
    return Tensor.fromList([end - start, c], out.toList(), device: Device.CPU);
  }

  static Tensor _reshape1xN(Tensor v) {
    if (v.shape.length != 1) {
      throw ArgumentError('_reshape1xN: expected rank 1, got ${v.shape}');
    }
    final n = v.shape[0];
    return Tensor.fromList([1, n], v.toList(), device: Device.CPU);
  }
}

class GPTJLoadReport {
  final int consumedCount;
  final List<String> unusedKeys;

  const GPTJLoadReport({required this.consumedCount, required this.unusedKeys});

  @override
  String toString() =>
      'GPTJLoadReport(consumed=$consumedCount, unused=${unusedKeys.length})';
}
