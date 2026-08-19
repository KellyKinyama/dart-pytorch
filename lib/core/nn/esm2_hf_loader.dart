/// Loader for HuggingFace ESM-2 (`facebook/esm2_t{L}_{M}M_UR50D`)
/// safetensors into an [ESM2Model].
///
/// HF key layout (as of `transformers` v4.x, `EsmModel`):
///
///   * `esm.embeddings.word_embeddings.weight` — `[V, D]`
///   * `esm.encoder.layer.{i}.attention.self.{query,key,value}.{weight,bias}`
///     weight: `[D, D]`; bias: `[D]`. We split by row into per-head
///     `[headDim, D]` slices (matches BERT/Llama loaders' convention).
///   * `esm.encoder.layer.{i}.attention.output.dense.{weight,bias}`
///     — `[D, D]` / `[D]` — output projection (biased).
///   * `esm.encoder.layer.{i}.attention.LayerNorm.{weight,bias}` — the
///     **pre-attention** LayerNorm (pre-LN convention).
///   * `esm.encoder.layer.{i}.intermediate.dense.{weight,bias}`
///     — `[FF, D]` / `[FF]`.
///   * `esm.encoder.layer.{i}.output.dense.{weight,bias}`
///     — `[D, FF]` / `[D]`.
///   * `esm.encoder.layer.{i}.LayerNorm.{weight,bias}` — the
///     **pre-FFN** LayerNorm.
///   * `esm.encoder.emb_layer_norm_after.{weight,bias}` — final LN.
///
/// Ignored (safe): `esm.embeddings.position_embeddings.weight` — the
/// HF checkpoint carries an unused learned pos embedding since ESM-2
/// uses `position_embedding_type: rotary`. Also ignored:
/// `esm.contact_head.*`, `lm_head.*`, `esm.pooler.*`,
/// `esm.embeddings.position_ids`.
library;

import 'dart:typed_data';

import '../tensor/tensor.dart';
import 'esm2.dart';
import 'safetensors.dart';

class ESM2LoadReport {
  final int consumedCount;
  final List<String> unusedKeys;
  const ESM2LoadReport({required this.consumedCount, required this.unusedKeys});

  @override
  String toString() =>
      'ESM2LoadReport(consumed=$consumedCount, unused=${unusedKeys.length})';
}

class ESM2HFLoader {
  /// `facebook/esm2_t6_8M_UR50D` config — 6 layers, 320 hidden, 20 heads.
  static ESM2Config esm2_8mConfig({Device device = Device.CPU, int seed = 0}) =>
      ESM2Config(
        vocabSize: 33,
        maxCtx: 1026,
        embedDim: 320,
        numLayers: 6,
        numHeads: 20,
        ffnDim: 1280,
        device: device,
        seed: seed,
      );

  /// `facebook/esm2_t12_35M_UR50D` config — 12 layers, 480 hidden, 20 heads.
  static ESM2Config esm2_35mConfig({
    Device device = Device.CPU,
    int seed = 0,
  }) => ESM2Config(
    vocabSize: 33,
    maxCtx: 1026,
    embedDim: 480,
    numLayers: 12,
    numHeads: 20,
    ffnDim: 1920,
    device: device,
    seed: seed,
  );

  static ESM2LoadReport loadFile(ESM2Model model, String path) {
    final state = SafeTensors.loadFile(path);
    return loadMap(model, state);
  }

  static ESM2LoadReport loadMap(ESM2Model model, Map<String, Tensor> state) {
    final consumed = <String>{};
    final cfg = model.config;
    final d = cfg.embedDim;
    final h = cfg.numHeads;
    final headDim = d ~/ h;
    final ffn = cfg.ffnDim;

    Tensor take(String name) {
      final t = state[name];
      if (t == null) {
        throw ArgumentError('esm2 loader: missing tensor "$name"');
      }
      consumed.add(name);
      return t;
    }

    // ---------- Token embedding ----------
    _copy(
      model.embedIn.weight,
      _expectShape(
        take('esm.embeddings.word_embeddings.weight'),
        [cfg.vocabSize, d],
        'esm.embeddings.word_embeddings.weight',
      ),
    );

    // ---------- Per-layer ----------
    for (int i = 0; i < cfg.numLayers; i++) {
      final layer = model.layers[i];
      final p = 'esm.encoder.layer.$i';

      // Pre-attention LN.
      _copy(
        layer.attnLn.gamma,
        _expectShape(take('$p.attention.LayerNorm.weight'), [
          d,
        ], '$p.attention.LayerNorm.weight'),
      );
      _copy(
        layer.attnLn.beta,
        _expectShape(take('$p.attention.LayerNorm.bias'), [
          d,
        ], '$p.attention.LayerNorm.bias'),
      );

      // Q/K/V weights + biases — per-head slice.
      final qW = _expectShape(take('$p.attention.self.query.weight'), [
        d,
        d,
      ], '$p.attention.self.query.weight');
      final qB = _expectShape(take('$p.attention.self.query.bias'), [
        d,
      ], '$p.attention.self.query.bias');
      final kW = _expectShape(take('$p.attention.self.key.weight'), [
        d,
        d,
      ], '$p.attention.self.key.weight');
      final kB = _expectShape(take('$p.attention.self.key.bias'), [
        d,
      ], '$p.attention.self.key.bias');
      final vW = _expectShape(take('$p.attention.self.value.weight'), [
        d,
        d,
      ], '$p.attention.self.value.weight');
      final vB = _expectShape(take('$p.attention.self.value.bias'), [
        d,
      ], '$p.attention.self.value.bias');
      for (int hh = 0; hh < h; hh++) {
        final start = hh * headDim;
        final end = start + headDim;
        _copy(layer.attn.wq[hh].weight, _sliceRows(qW, start, end));
        _copy(
          layer.attn.wq[hh].bias!,
          _reshapeVectorTo1xN(_sliceVector(qB, start, end)),
        );
        _copy(layer.attn.wk[hh].weight, _sliceRows(kW, start, end));
        _copy(
          layer.attn.wk[hh].bias!,
          _reshapeVectorTo1xN(_sliceVector(kB, start, end)),
        );
        _copy(layer.attn.wv[hh].weight, _sliceRows(vW, start, end));
        _copy(
          layer.attn.wv[hh].bias!,
          _reshapeVectorTo1xN(_sliceVector(vB, start, end)),
        );
      }

      // Attention output projection (biased).
      _copy(
        layer.attn.wo.weight,
        _expectShape(
          take('$p.attention.output.dense.weight'),
          [d, d],
          '$p.attention.output.dense.weight',
        ),
      );
      _copy(
        layer.attn.wo.bias!,
        _reshapeVectorTo1xN(
          _expectShape(
            take('$p.attention.output.dense.bias'),
            [d],
            '$p.attention.output.dense.bias',
          ),
        ),
      );

      // Pre-FFN LN.
      _copy(
        layer.ffnLn.gamma,
        _expectShape(take('$p.LayerNorm.weight'), [d], '$p.LayerNorm.weight'),
      );
      _copy(
        layer.ffnLn.beta,
        _expectShape(take('$p.LayerNorm.bias'), [d], '$p.LayerNorm.bias'),
      );

      // FFN: intermediate + output.
      _copy(
        layer.ffnIntermediate.weight,
        _expectShape(take('$p.intermediate.dense.weight'), [
          ffn,
          d,
        ], '$p.intermediate.dense.weight'),
      );
      _copy(
        layer.ffnIntermediate.bias!,
        _reshapeVectorTo1xN(
          _expectShape(take('$p.intermediate.dense.bias'), [
            ffn,
          ], '$p.intermediate.dense.bias'),
        ),
      );
      _copy(
        layer.ffnOutput.weight,
        _expectShape(take('$p.output.dense.weight'), [
          d,
          ffn,
        ], '$p.output.dense.weight'),
      );
      _copy(
        layer.ffnOutput.bias!,
        _reshapeVectorTo1xN(
          _expectShape(take('$p.output.dense.bias'), [
            d,
          ], '$p.output.dense.bias'),
        ),
      );
    }

    // ---------- Final post-encoder LN ----------
    _copy(
      model.finalLn.gamma,
      _expectShape(
        take('esm.encoder.emb_layer_norm_after.weight'),
        [d],
        'esm.encoder.emb_layer_norm_after.weight',
      ),
    );
    _copy(
      model.finalLn.beta,
      _expectShape(
        take('esm.encoder.emb_layer_norm_after.bias'),
        [d],
        'esm.encoder.emb_layer_norm_after.bias',
      ),
    );

    // ---------- Silently absorb known-ignored HF keys ----------
    const ignored = <String>[
      'esm.embeddings.position_embeddings.weight',
      'esm.embeddings.position_ids',
      'esm.embeddings.LayerNorm.weight',
      'esm.embeddings.LayerNorm.bias',
      'esm.pooler.dense.weight',
      'esm.pooler.dense.bias',
      'lm_head.dense.weight',
      'lm_head.dense.bias',
      'lm_head.layer_norm.weight',
      'lm_head.layer_norm.bias',
      'lm_head.decoder.weight',
      'lm_head.decoder.bias',
      'lm_head.bias',
    ];
    for (final k in ignored) {
      if (state.containsKey(k)) consumed.add(k);
    }
    for (final k in state.keys) {
      if (k.startsWith('esm.contact_head.')) consumed.add(k);
    }

    final unused = state.keys.where((k) => !consumed.contains(k)).toList()
      ..sort();
    return ESM2LoadReport(consumedCount: consumed.length, unusedKeys: unused);
  }

  static Tensor _expectShape(Tensor t, List<int> expected, String name) {
    if (t.shape.length != expected.length || !_shapesEqual(t.shape, expected)) {
      throw ArgumentError(
        'esm2 loader: "$name" expected shape $expected, got ${t.shape}',
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
        'esm2 loader: copy length mismatch — dst=${dst.shape} '
        '(${dst.length}), src=${src.shape} (${src.length})',
      );
    }
    final vals = src.toList();
    final matched = Tensor.fromList(dst.shape, vals, device: dst.device);
    dst.assign(matched);
  }

  static Tensor _sliceRows(Tensor t, int start, int end) =>
      t.sliceRows(start, end);

  static Tensor _sliceVector(Tensor t, int start, int end) {
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
        'esm2 loader: expected rank 1 for bias vector, got ${v.shape}',
      );
    }
    return Tensor.fromList([1, v.shape[0]], v.toList(), device: Device.CPU);
  }
}
