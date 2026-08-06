/// Loads HuggingFace **CLIP vision** weights (`CLIPVisionModel` /
/// `CLIPModel` safetensors) into a [CLIPVisionModel].
///
/// Supported checkpoints (fp32 weight sizes; fp16 safetensors are
/// promoted to fp32 by [SafeTensors]):
///
///   * `openai/clip-vit-base-patch32` — 12 layers, 768 embed,
///     12 heads, ffn 3072, image 224, patch 32 → 49 patches.
///     ~350 MB fp32 (~150 MB fp16 on disk).
///   * `openai/clip-vit-base-patch16` — same config as B/32 but
///     patch 16 → 196 patches. Same weight count.
///   * `openai/clip-vit-large-patch14` — 24 layers, 1024 embed,
///     16 heads, ffn 4096, image 224, patch 14 → 256 patches.
///     ~1.2 GB fp32.
///
/// HF key conventions handled here (the top-level namespace is
/// `vision_model.` for `CLIPVisionModel` safetensors, or
/// `vision_model.` under the joint `CLIPModel` bundle; the loader
/// auto-detects the prefix by probing for both):
///
///   * `vision_model.embeddings.patch_embedding.weight` —
///     `[D, C, P, P]` **Conv2d weight**, permuted to
///     `[D, P*P*C]` (channels-last per pixel) to match our
///     patchify layout, then assigned to the [Linear] projector.
///   * `vision_model.embeddings.class_embedding` — `[D]`
///   * `vision_model.embeddings.position_embedding.weight` —
///     `[numPatches + 1, D]`
///   * `vision_model.pre_layrnorm.{weight,bias}` — `[D]` each
///     (HF *does* misspell it — the typo is in `transformers`).
///   * `vision_model.encoder.layers.{i}.self_attn.{q,k,v}_proj.{weight,bias}`
///     — `[D, D]` / `[D]`. Sliced row-wise into per-head `[hd, D]` / `[hd]`.
///   * `vision_model.encoder.layers.{i}.self_attn.out_proj.{weight,bias}`
///     — `[D, D]` / `[D]`. Directly copied to `wo`.
///   * `vision_model.encoder.layers.{i}.layer_norm{1,2}.{weight,bias}`
///     — `[D]` each; mapped to `ln1` (pre-attn) and `ln2` (pre-FFN).
///   * `vision_model.encoder.layers.{i}.mlp.fc1.{weight,bias}` —
///     `[F, D]` / `[F]`.
///   * `vision_model.encoder.layers.{i}.mlp.fc2.{weight,bias}` —
///     `[D, F]` / `[D]`.
///   * `vision_model.post_layernorm.{weight,bias}` — `[D]` each.
///
/// Ignored (safe): every key that doesn't start with `vision_model.`
/// (text tower, `logit_scale`, `visual_projection`, etc.). They are
/// returned in [ClipLoadReport.unusedKeys] so callers can log them.
library;

import 'dart:typed_data';

import '../tensor/tensor.dart';
import 'safetensors.dart';
import 'vision/clip_vision_model.dart';

class ClipHFLoader {
  /// `openai/clip-vit-base-patch32` config.
  static CLIPVisionConfig base32Config({
    Device device = Device.CPU,
    int seed = 0,
  }) => CLIPVisionConfig(
    imageSize: 224,
    patchSize: 32,
    embedDim: 768,
    numLayers: 12,
    numHeads: 12,
    ffnDim: 3072,
    layerNormEps: 1e-5,
    device: device,
    seed: seed,
  );

  /// `openai/clip-vit-base-patch16` config.
  static CLIPVisionConfig base16Config({
    Device device = Device.CPU,
    int seed = 0,
  }) => CLIPVisionConfig(
    imageSize: 224,
    patchSize: 16,
    embedDim: 768,
    numLayers: 12,
    numHeads: 12,
    ffnDim: 3072,
    layerNormEps: 1e-5,
    device: device,
    seed: seed,
  );

  /// `openai/clip-vit-large-patch14` config.
  static CLIPVisionConfig large14Config({
    Device device = Device.CPU,
    int seed = 0,
  }) => CLIPVisionConfig(
    imageSize: 224,
    patchSize: 14,
    embedDim: 1024,
    numLayers: 24,
    numHeads: 16,
    ffnDim: 4096,
    layerNormEps: 1e-5,
    device: device,
    seed: seed,
  );

  static ClipLoadReport loadFile(CLIPVisionModel model, String path) {
    final state = SafeTensors.loadFile(path);
    return loadMap(model, state);
  }

  static ClipLoadReport loadMap(
    CLIPVisionModel model,
    Map<String, Tensor> state,
  ) {
    final prefix = _detectPrefix(state);
    final consumed = <String>{};

    Tensor take(String name) {
      final key = '$prefix$name';
      final t = state[key];
      if (t == null) {
        throw ArgumentError('clip loader: missing tensor "$key"');
      }
      consumed.add(key);
      return t;
    }

    final cfg = model.config;
    final d = cfg.embedDim;
    final h = cfg.numHeads;
    final headDim = d ~/ h;
    final ffn = cfg.ffnDim;

    // -------- patch embedding --------
    // HF: [D, C, P, P] (Conv2d). Ours: Linear([D, P*P*C]) with pixels
    // ordered channels-last per pixel (row, col, channel). Permute.
    final patchConv = _expectShape(
      take('embeddings.patch_embedding.weight'),
      [d, cfg.numChannels, cfg.patchSize, cfg.patchSize],
      'embeddings.patch_embedding.weight',
    );
    final patchLinear = _permuteConvToLinear(
      patchConv,
      outDim: d,
      channels: cfg.numChannels,
      patchSize: cfg.patchSize,
    );
    _copy(model.patchProjection.weight, patchLinear);

    // -------- class + position embeddings --------
    _copy(
      model.clsToken,
      _reshape(
        _expectShape(take('embeddings.class_embedding'), [
          d,
        ], 'embeddings.class_embedding'),
        [1, d],
      ),
    );
    _copy(
      model.posEmbeddings,
      _expectShape(
        take('embeddings.position_embedding.weight'),
        [cfg.numPatches + 1, d],
        'embeddings.position_embedding.weight',
      ),
    );

    // -------- pre_layrnorm (HF typo preserved) --------
    _copy(
      model.preLayerNorm.gamma,
      _expectShape(take('pre_layrnorm.weight'), [d], 'pre_layrnorm.weight'),
    );
    _copy(
      model.preLayerNorm.beta,
      _expectShape(take('pre_layrnorm.bias'), [d], 'pre_layrnorm.bias'),
    );

    // -------- transformer blocks --------
    for (int i = 0; i < cfg.numLayers; i++) {
      final block = model.encoder.blocks[i];
      final p = 'encoder.layers.$i';

      // layer_norm1 → ln1 (pre-attn).
      _copy(
        block.ln1.gamma,
        _expectShape(take('$p.layer_norm1.weight'), [
          d,
        ], '$p.layer_norm1.weight'),
      );
      _copy(
        block.ln1.beta,
        _expectShape(take('$p.layer_norm1.bias'), [d], '$p.layer_norm1.bias'),
      );

      // layer_norm2 → ln2 (pre-FFN).
      _copy(
        block.ln2.gamma,
        _expectShape(take('$p.layer_norm2.weight'), [
          d,
        ], '$p.layer_norm2.weight'),
      );
      _copy(
        block.ln2.beta,
        _expectShape(take('$p.layer_norm2.bias'), [d], '$p.layer_norm2.bias'),
      );

      // q_proj — [D, D] weight, [D] bias. Row-split into `h` heads.
      final qW = _expectShape(take('$p.self_attn.q_proj.weight'), [
        d,
        d,
      ], '$p.self_attn.q_proj.weight');
      final qB = _expectShape(take('$p.self_attn.q_proj.bias'), [
        d,
      ], '$p.self_attn.q_proj.bias');
      for (int hh = 0; hh < h; hh++) {
        _copy(
          block.mha.wq[hh].weight,
          _sliceRows(qW, hh * headDim, (hh + 1) * headDim),
        );
        _copy(
          block.mha.wq[hh].bias!,
          _slice1D(qB, hh * headDim, (hh + 1) * headDim),
        );
      }

      // k_proj — same shape/layout as q_proj.
      final kW = _expectShape(take('$p.self_attn.k_proj.weight'), [
        d,
        d,
      ], '$p.self_attn.k_proj.weight');
      final kB = _expectShape(take('$p.self_attn.k_proj.bias'), [
        d,
      ], '$p.self_attn.k_proj.bias');
      for (int hh = 0; hh < h; hh++) {
        _copy(
          block.mha.wk[hh].weight,
          _sliceRows(kW, hh * headDim, (hh + 1) * headDim),
        );
        _copy(
          block.mha.wk[hh].bias!,
          _slice1D(kB, hh * headDim, (hh + 1) * headDim),
        );
      }

      // v_proj — same again.
      final vW = _expectShape(take('$p.self_attn.v_proj.weight'), [
        d,
        d,
      ], '$p.self_attn.v_proj.weight');
      final vB = _expectShape(take('$p.self_attn.v_proj.bias'), [
        d,
      ], '$p.self_attn.v_proj.bias');
      for (int hh = 0; hh < h; hh++) {
        _copy(
          block.mha.wv[hh].weight,
          _sliceRows(vW, hh * headDim, (hh + 1) * headDim),
        );
        _copy(
          block.mha.wv[hh].bias!,
          _slice1D(vB, hh * headDim, (hh + 1) * headDim),
        );
      }

      // out_proj → wo.
      _copy(
        block.mha.wo.weight,
        _expectShape(take('$p.self_attn.out_proj.weight'), [
          d,
          d,
        ], '$p.self_attn.out_proj.weight'),
      );
      _copy(
        block.mha.wo.bias!,
        _expectShape(take('$p.self_attn.out_proj.bias'), [
          d,
        ], '$p.self_attn.out_proj.bias'),
      );

      // MLP fc1 [F, D] + bias [F].
      _copy(
        block.ffn1.weight,
        _expectShape(take('$p.mlp.fc1.weight'), [ffn, d], '$p.mlp.fc1.weight'),
      );
      _copy(
        block.ffn1.bias!,
        _expectShape(take('$p.mlp.fc1.bias'), [ffn], '$p.mlp.fc1.bias'),
      );

      // MLP fc2 [D, F] + bias [D].
      _copy(
        block.ffn2.weight,
        _expectShape(take('$p.mlp.fc2.weight'), [d, ffn], '$p.mlp.fc2.weight'),
      );
      _copy(
        block.ffn2.bias!,
        _expectShape(take('$p.mlp.fc2.bias'), [d], '$p.mlp.fc2.bias'),
      );
    }

    // -------- post_layernorm --------
    _copy(
      model.encoder.finalNorm!.gamma,
      _expectShape(take('post_layernorm.weight'), [d], 'post_layernorm.weight'),
    );
    _copy(
      model.encoder.finalNorm!.beta,
      _expectShape(take('post_layernorm.bias'), [d], 'post_layernorm.bias'),
    );

    final unused = state.keys.where((k) => !consumed.contains(k)).toList()
      ..sort();
    return ClipLoadReport(
      prefix: prefix,
      consumedCount: consumed.length,
      unusedKeys: unused,
    );
  }

  // ---------------------------------------------------------------------
  // Prefix detection.
  //
  // Standalone `CLIPVisionModel` safetensors use no prefix
  // (`embeddings.patch_embedding.weight`).
  // Full `CLIPModel` bundles use `vision_model.` prefix everywhere.
  // Some HF re-exports strip the prefix. We probe for the patch
  // embedding key with and without the prefix.
  // ---------------------------------------------------------------------
  static String _detectPrefix(Map<String, Tensor> state) {
    const candidates = ['vision_model.', ''];
    for (final p in candidates) {
      if (state.containsKey('${p}embeddings.patch_embedding.weight')) {
        return p;
      }
    }
    throw ArgumentError(
      'clip loader: cannot find "embeddings.patch_embedding.weight" '
      'in state dict (checked prefixes: $candidates). Is this a CLIP '
      'vision safetensors?',
    );
  }

  // ---------------------------------------------------------------------
  // Tensor helpers.
  // ---------------------------------------------------------------------
  static Tensor _expectShape(Tensor t, List<int> expected, String name) {
    if (t.shape.length != expected.length || !_shapesEqual(t.shape, expected)) {
      throw ArgumentError(
        'clip loader: "$name" expected shape $expected, got ${t.shape}',
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
        'clip loader: copy length mismatch — dst=${dst.shape} '
        '(${dst.length}), src=${src.shape} (${src.length})',
      );
    }
    final vals = src.toList();
    final matched = Tensor.fromList(dst.shape, vals, device: dst.device);
    dst.assign(matched);
  }

  static Tensor _reshape(Tensor t, List<int> shape) {
    final n = shape.fold<int>(1, (a, b) => a * b);
    if (t.length != n) {
      throw ArgumentError('_reshape: length mismatch');
    }
    final vals = t.toList();
    return Tensor.fromList(shape, vals, device: Device.CPU);
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

  static Tensor _slice1D(Tensor t, int start, int end) {
    if (t.shape.length != 1) {
      throw ArgumentError('_slice1D: expected rank 1, got ${t.shape}');
    }
    final src = t.toList();
    final n = end - start;
    final out = Float32List(n);
    for (int i = 0; i < n; i++) {
      out[i] = src[start + i];
    }
    return Tensor.fromList([n], out, device: Device.CPU);
  }

  /// Permute a Conv2d weight `[out, C, H, W]` into a Linear weight
  /// `[out, H*W*C]` such that applying the Linear on a flattened
  /// `[H*W*C]` patch (channels-last per pixel, `(row, col, channel)`
  /// order) matches the numeric result of the original Conv2d on the
  /// `[C, H, W]` patch.
  ///
  /// Conv computes `out[o] = sum_{c,h,w} in[c,h,w] * W[o,c,h,w]`. Our
  /// flat patch order is `x[i] = in[h, w, c]` with
  /// `i = h*W*C + w*C + c`. Requiring
  /// `sum_i W_lin[o, i] * x[i]` == the conv sum forces
  /// `W_lin[o, h*W*C + w*C + c] = W[o, c, h, w]`.
  static Tensor _permuteConvToLinear(
    Tensor conv, {
    required int outDim,
    required int channels,
    required int patchSize,
  }) {
    final src = conv.toList();
    final H = patchSize, W = patchSize, C = channels;
    final total = outDim * H * W * C;
    final out = Float32List(total);
    // src layout: (o, c, h, w) with w fastest.
    // dst layout: (o, h, w, c) with c fastest.
    for (int o = 0; o < outDim; o++) {
      for (int c = 0; c < C; c++) {
        for (int hh = 0; hh < H; hh++) {
          for (int ww = 0; ww < W; ww++) {
            final srcIdx = ((o * C + c) * H + hh) * W + ww;
            final dstIdx = ((o * H + hh) * W + ww) * C + c;
            out[dstIdx] = src[srcIdx].toDouble();
          }
        }
      }
    }
    return Tensor.fromList([outDim, H * W * C], out, device: Device.CPU);
  }
}

class ClipLoadReport {
  final String prefix;
  final int consumedCount;
  final List<String> unusedKeys;

  const ClipLoadReport({
    required this.prefix,
    required this.consumedCount,
    required this.unusedKeys,
  });

  @override
  String toString() =>
      'ClipLoadReport(prefix="$prefix", consumed=$consumedCount, '
      'unused=${unusedKeys.length})';
}
