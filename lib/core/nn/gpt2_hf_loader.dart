/// Loads HuggingFace [gpt2 / gpt2-medium / gpt2-large / gpt2-xl]
/// weights into a [GPT] module.
///
/// The expected on-disk layout is a HuggingFace `safetensors` file
/// produced by e.g.:
///
/// ```python
/// from transformers import GPT2LMHeadModel
/// m = GPT2LMHeadModel.from_pretrained("gpt2")
/// m.save_pretrained("gpt2", safe_serialization=True)
/// # -> gpt2/model.safetensors
/// ```
///
/// or downloaded straight from the HF Hub (`model.safetensors`).
///
/// Weight conventions (HF → dart_pytorch):
///
///   * `wte.weight`               `[V, D]`      → `tokenEmb.weight`
///   * `wpe.weight`               `[maxCtx, D]` → `posEmb.table.weight`
///   * `h.{i}.ln_1.weight/bias`   `[D]`         → `block.ln1.gamma/beta`
///   * `h.{i}.attn.c_attn.weight` `[D, 3D]`     → block QKV per-head weights
///   * `h.{i}.attn.c_attn.bias`   `[3D]`        → block QKV per-head biases
///   * `h.{i}.attn.c_proj.weight` `[D, D]`      → block `mha.wo.weight`
///   * `h.{i}.attn.c_proj.bias`   `[D]`         → block `mha.wo.bias`
///   * `h.{i}.ln_2.weight/bias`   `[D]`         → `block.ln2.gamma/beta`
///   * `h.{i}.mlp.c_fc.weight`    `[D, 4D]`     → `block.ffn1.weight`
///   * `h.{i}.mlp.c_fc.bias`      `[4D]`        → `block.ffn1.bias`
///   * `h.{i}.mlp.c_proj.weight`  `[4D, D]`     → `block.ffn2.weight`
///   * `h.{i}.mlp.c_proj.bias`    `[D]`         → `block.ffn2.bias`
///   * `ln_f.weight/bias`         `[D]`         → `encoder.finalNorm.gamma/beta`
///
/// GPT-2 uses `Conv1D` layers (weight shape `[in, out]`), so every
/// `_.weight` matrix is transposed on load.
///
/// The `lm_head.weight` is tied to `wte.weight` in HF; when the target
/// [GPT] has `tieWeights: true` (the default) it is loaded implicitly
/// via `wte`. When untied, the loader falls back to `wte` for the
/// head as well (HF does not save a separate head).
///
/// GPT-2 uses GELU (tanh approx) and biased attention projections;
/// callers **must** build the target [GPT] with
/// `attnBias: true, activation: Activation.geluTanh` or the outputs
/// will diverge from the reference model.
library;

import 'dart:typed_data';

import '../tensor/tensor.dart';
import '../tensor/dtype.dart';
import 'gpt.dart';
import 'safetensors.dart';
import 'transformer.dart';

class GPT2HFLoader {
  /// Standard GPT-2 small config (117M params). Vocab is 50257, ctx is
  /// 1024, embed 768, 12 layers, 12 heads. Matches HF `gpt2`.
  static GPTConfig gpt2SmallConfig({
    Device device = Device.CPU,
    int seed = 0,
  }) => GPTConfig(
    vocabSize: 50257,
    maxCtx: 1024,
    embedDim: 768,
    numLayers: 12,
    numHeads: 12,
    dropoutP: 0.0,
    tieWeights: true,
    attnBias: true,
    activation: Activation.geluTanh,
    device: device,
    seed: seed,
  );

  /// HuggingFace `distilgpt2` config (82M params). Same shape as GPT-2
  /// small except only **6** transformer blocks instead of 12. Loads
  /// with the same [loadFile] / [loadMap] used for regular GPT-2.
  static GPTConfig distilGpt2Config({
    Device device = Device.CPU,
    int seed = 0,
  }) => GPTConfig(
    vocabSize: 50257,
    maxCtx: 1024,
    embedDim: 768,
    numLayers: 6,
    numHeads: 12,
    dropoutP: 0.0,
    tieWeights: true,
    attnBias: true,
    activation: Activation.geluTanh,
    device: device,
    seed: seed,
  );

  /// GPT-2 medium (345M).
  static GPTConfig gpt2MediumConfig({
    Device device = Device.CPU,
    int seed = 0,
  }) => GPTConfig(
    vocabSize: 50257,
    maxCtx: 1024,
    embedDim: 1024,
    numLayers: 24,
    numHeads: 16,
    dropoutP: 0.0,
    tieWeights: true,
    attnBias: true,
    activation: Activation.geluTanh,
    device: device,
    seed: seed,
  );

  /// GPT-2 large (774M).
  static GPTConfig gpt2LargeConfig({
    Device device = Device.CPU,
    int seed = 0,
  }) => GPTConfig(
    vocabSize: 50257,
    maxCtx: 1024,
    embedDim: 1280,
    numLayers: 36,
    numHeads: 20,
    dropoutP: 0.0,
    tieWeights: true,
    attnBias: true,
    activation: Activation.geluTanh,
    device: device,
    seed: seed,
  );

  /// GPT-2 XL (1558M).
  static GPTConfig gpt2XLConfig({Device device = Device.CPU, int seed = 0}) =>
      GPTConfig(
        vocabSize: 50257,
        maxCtx: 1024,
        embedDim: 1600,
        numLayers: 48,
        numHeads: 25,
        dropoutP: 0.0,
        tieWeights: true,
        attnBias: true,
        activation: Activation.geluTanh,
        device: device,
        seed: seed,
      );

  /// Load HuggingFace GPT-2 weights from a safetensors file into an
  /// existing [GPT] module.
  ///
  /// `gpt` must have been built with a config that matches the weight
  /// dimensions — the loader validates every shape it touches and
  /// throws [ArgumentError] on any mismatch (no silent partial loads).
  ///
  /// Returns a report describing what was loaded, which HF keys were
  /// ignored, and which of `gpt`'s parameters had no matching HF key.
  /// Load a `.safetensors` checkpoint into [gpt]. When [keepFp16]
  /// is true, `F16` entries are held in fp16 CPU storage (read-only,
  /// halves resident RAM); the fp16 path is inference-only.
  static GPT2LoadReport loadFile(
    GPT gpt,
    String path, {
    bool keepFp16 = false,
  }) {
    final state = SafeTensors.loadFile(path, keepFp16: keepFp16);
    return loadMap(gpt, state);
  }

  /// Load a sharded GPT-2 checkpoint via `model.safetensors.index.json`.
  static GPT2LoadReport loadSharded(
    GPT gpt,
    String indexPath, {
    bool keepFp16 = false,
  }) {
    final state = SafeTensors.loadSharded(indexPath, keepFp16: keepFp16);
    return loadMap(gpt, state);
  }

  /// Same as [loadFile] but takes a pre-parsed `state_dict` map.
  /// Useful for tests and for callers that want to preview / mutate
  /// the map before loading.
  static GPT2LoadReport loadMap(GPT gpt, Map<String, Tensor> stateIn) {
    // HF `GPT2LMHeadModel` saves under a `transformer.` prefix (this
    // is what `distilgpt2/model.safetensors` uses). Raw `gpt2` from
    // the hub has no prefix. Strip if *every* key starts with it so
    // both layouts load with the same code.
    Map<String, Tensor> state = stateIn;
    if (stateIn.isNotEmpty &&
        stateIn.keys.every(
          (k) => k.startsWith('transformer.') || k.startsWith('lm_head.'),
        )) {
      state = <String, Tensor>{
        for (final e in stateIn.entries)
          if (e.key.startsWith('transformer.'))
            e.key.substring('transformer.'.length): e.value
          else
            e.key: e.value,
      };
    }
    final consumed = <String>{};
    final missing = <String>[];

    Tensor take(String name) {
      final t = state[name];
      if (t == null) {
        missing.add(name);
        throw ArgumentError('gpt2 loader: missing tensor "$name"');
      }
      consumed.add(name);
      return t;
    }

    final cfg = gpt.config;
    final d = cfg.embedDim;
    final h = cfg.numHeads;
    final headDim = d ~/ h;
    final ffn = cfg.ffnDim ?? 4 * d;

    // -------- token embedding --------
    final wte = take('wte.weight');
    _expectShape(wte, [cfg.vocabSize, d], 'wte.weight');
    _copy(gpt.tokenEmb.weight, wte);

    // -------- positional embedding --------
    final wpe = take('wpe.weight');
    _expectShape(wpe, [cfg.maxCtx, d], 'wpe.weight');
    _copy(gpt.posEmb.table.weight, wpe);

    // -------- per-block --------
    for (int i = 0; i < cfg.numLayers; i++) {
      final block = gpt.encoder.blocks[i];
      final p = 'h.$i';

      // LN1
      _copy(
        block.ln1.gamma,
        _expectShape(take('$p.ln_1.weight'), [d], '$p.ln_1.weight'),
      );
      _copy(
        block.ln1.beta,
        _expectShape(take('$p.ln_1.bias'), [d], '$p.ln_1.bias'),
      );

      // c_attn: Conv1D weight shape [D, 3D] -> logical [3D, D] fused
      // Q/K/V @ x.
      final cAttnW = _expectShape(take('$p.attn.c_attn.weight'), [
        d,
        3 * d,
      ], '$p.attn.c_attn.weight');
      final cAttnB = _expectShape(take('$p.attn.c_attn.bias'), [
        3 * d,
      ], '$p.attn.c_attn.bias');

      // Transpose [D, 3D] -> [3D, D], then split rows into Q, K, V
      // (each [D, D]).
      final cAttnWT = _transpose2D(cAttnW); // shape [3D, D]
      final qW = _sliceRows(cAttnWT, 0, d); // [D, D]
      final kW = _sliceRows(cAttnWT, d, 2 * d); // [D, D]
      final vW = _sliceRows(cAttnWT, 2 * d, 3 * d); // [D, D]

      // Split biases the same way.
      final qB = _sliceVector(cAttnB, 0, d);
      final kB = _sliceVector(cAttnB, d, 2 * d);
      final vB = _sliceVector(cAttnB, 2 * d, 3 * d);

      // Distribute across heads. Our MHA has per-head Linears with
      // weight [headDim, D]. HF fuses all heads into the [D, D] matrix
      // above, so slicing rows [h*headDim : (h+1)*headDim] yields the
      // per-head weight, and the same for the bias.
      for (int hi = 0; hi < h; hi++) {
        _copy(
          block.mha.wq[hi].weight,
          _sliceRows(qW, hi * headDim, (hi + 1) * headDim),
        );
        _copy(
          block.mha.wk[hi].weight,
          _sliceRows(kW, hi * headDim, (hi + 1) * headDim),
        );
        _copy(
          block.mha.wv[hi].weight,
          _sliceRows(vW, hi * headDim, (hi + 1) * headDim),
        );
        if (cfg.attnBias) {
          _copy(
            block.mha.wq[hi].bias!,
            _reshapeVectorTo1xN(
              _sliceVector(qB, hi * headDim, (hi + 1) * headDim),
            ),
          );
          _copy(
            block.mha.wk[hi].bias!,
            _reshapeVectorTo1xN(
              _sliceVector(kB, hi * headDim, (hi + 1) * headDim),
            ),
          );
          _copy(
            block.mha.wv[hi].bias!,
            _reshapeVectorTo1xN(
              _sliceVector(vB, hi * headDim, (hi + 1) * headDim),
            ),
          );
        }
      }

      // c_proj (output projection): Conv1D [D, D] -> Linear weight [D, D].
      final cProjW = _expectShape(take('$p.attn.c_proj.weight'), [
        d,
        d,
      ], '$p.attn.c_proj.weight');
      final cProjB = _expectShape(take('$p.attn.c_proj.bias'), [
        d,
      ], '$p.attn.c_proj.bias');
      _copy(block.mha.wo.weight, _transpose2D(cProjW));
      if (cfg.attnBias) {
        _copy(block.mha.wo.bias!, _reshapeVectorTo1xN(cProjB));
      }

      // LN2
      _copy(
        block.ln2.gamma,
        _expectShape(take('$p.ln_2.weight'), [d], '$p.ln_2.weight'),
      );
      _copy(
        block.ln2.beta,
        _expectShape(take('$p.ln_2.bias'), [d], '$p.ln_2.bias'),
      );

      // MLP c_fc: Conv1D [D, 4D] -> Linear weight [4D, D].
      final cFcW = _expectShape(take('$p.mlp.c_fc.weight'), [
        d,
        ffn,
      ], '$p.mlp.c_fc.weight');
      final cFcB = _expectShape(take('$p.mlp.c_fc.bias'), [
        ffn,
      ], '$p.mlp.c_fc.bias');
      _copy(block.ffn1.weight, _transpose2D(cFcW));
      _copy(block.ffn1.bias!, _reshapeVectorTo1xN(cFcB));

      // MLP c_proj: Conv1D [4D, D] -> Linear weight [D, 4D].
      final cProj2W = _expectShape(take('$p.mlp.c_proj.weight'), [
        ffn,
        d,
      ], '$p.mlp.c_proj.weight');
      final cProj2B = _expectShape(take('$p.mlp.c_proj.bias'), [
        d,
      ], '$p.mlp.c_proj.bias');
      _copy(block.ffn2.weight, _transpose2D(cProj2W));
      _copy(block.ffn2.bias!, _reshapeVectorTo1xN(cProj2B));
    }

    // -------- final LN --------
    _copy(
      gpt.encoder.finalNorm!.gamma,
      _expectShape(take('ln_f.weight'), [d], 'ln_f.weight'),
    );
    _copy(
      gpt.encoder.finalNorm!.beta,
      _expectShape(take('ln_f.bias'), [d], 'ln_f.bias'),
    );

    // If GPT is built with `tieWeights: false`, HF still doesn't ship
    // a separate `lm_head.weight` — reuse `wte`.
    if (!cfg.tieWeights) {
      _copy(gpt.untiedHead!.weight, wte);
    }

    final unused = state.keys.where((k) => !consumed.contains(k)).toList()
      ..sort();
    return GPT2LoadReport(
      consumedCount: consumed.length,
      unusedKeys: unused,
      missingKeys: missing,
    );
  }

  // ---------------- tensor helpers (CPU only) ----------------

  static Tensor _expectShape(Tensor t, List<int> expected, String name) {
    if (t.shape.length != expected.length || !_shapesEqual(t.shape, expected)) {
      throw ArgumentError(
        'gpt2 loader: "$name" expected shape $expected, got ${t.shape}',
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

  /// Copy `src`'s values into `dst` in place (host-side). If `dst`
  /// lives on GPU, uploads via a temporary CPU tensor. Shapes must
  /// match exactly in terms of total element count; we only reshape,
  /// never broadcast.
  static void _copy(Tensor dst, Tensor src) {
    if (dst.length != src.length) {
      throw ArgumentError(
        'gpt2 loader: copy length mismatch — dst=${dst.shape} '
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

  /// Transpose a 2-D CPU tensor `[R, C]` -> `[C, R]`, returning a new
  /// CPU tensor. Cheap even for GPT-2 XL (1600×1600 = 2.5M floats).
  static Tensor _transpose2D(Tensor t) {
    if (t.shape.length != 2) {
      throw ArgumentError('_transpose2D: expected rank 2, got ${t.shape}');
    }
    final r = t.shape[0];
    final c = t.shape[1];
    final src = t.toList();
    final out = Float32List(r * c);
    for (int i = 0; i < r; i++) {
      for (int j = 0; j < c; j++) {
        out[j * r + i] = src[i * c + j];
      }
    }
    return Tensor.fromList([c, r], out, device: Device.CPU);
  }

  /// Slice contiguous rows `[start, end)` out of a 2-D tensor.
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

  /// Slice a contiguous span `[start, end)` out of a 1-D tensor.
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

  /// Reshape a length-N vector `[N]` into `[1, N]` for compatibility
  /// with our `Linear.bias` storage convention.
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

/// Diagnostic summary of a single [GPT2HFLoader.loadFile] / `loadMap`
/// call. `unusedKeys` typically contains extra HF conveniences like
/// `attn.bias` (the causal mask buffer) or `masked_bias`; these are
/// safe to ignore because we build the causal mask on the fly.
class GPT2LoadReport {
  final int consumedCount;
  final List<String> unusedKeys;
  final List<String> missingKeys;

  const GPT2LoadReport({
    required this.consumedCount,
    required this.unusedKeys,
    required this.missingKeys,
  });

  @override
  String toString() =>
      'GPT2LoadReport(consumed=$consumedCount, '
      'unused=${unusedKeys.length}, missing=${missingKeys.length})';
}
