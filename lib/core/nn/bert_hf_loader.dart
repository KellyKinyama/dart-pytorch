/// Loads HuggingFace BERT-style safetensors (bert-base, MiniLM,
/// sentence-transformers/all-MiniLM-*, all-mpnet-*) into a
/// [BertModel]. Pooler tensors (`pooler.dense.*`), position_ids and
/// buffers are ignored — sentence-transformer models don't use the
/// [CLS] pooler.
///
/// HF key conventions handled here:
///
///   * embeddings.word_embeddings.weight        [V, D]
///   * embeddings.position_embeddings.weight    [maxPos, D]
///   * embeddings.token_type_embeddings.weight  [typeVocab, D]
///     — collapsed to a `[1, D]` bias (row 0 only).
///   * embeddings.LayerNorm.{weight,bias}       [D]
///   * encoder.layer.i.attention.self.{query,key,value}.{weight,bias}
///     — weight `[D, D]` sliced row-wise into 12 per-head chunks
///     `[headDim, D]`; bias `[D]` sliced into `[headDim]` per head.
///   * encoder.layer.i.attention.output.dense.{weight,bias}
///   * encoder.layer.i.attention.output.LayerNorm.{weight,bias}
///   * encoder.layer.i.intermediate.dense.{weight,bias}
///   * encoder.layer.i.output.dense.{weight,bias}
///   * encoder.layer.i.output.LayerNorm.{weight,bias}
library;

import 'dart:typed_data';

import '../tensor/tensor.dart';
import '../tensor/dtype.dart';
import 'bert.dart';
import 'safetensors.dart';

class BertLoadReport {
  final int consumedCount;
  final List<String> unusedKeys;
  const BertLoadReport({required this.consumedCount, required this.unusedKeys});

  @override
  String toString() =>
      'BertLoadReport(consumed=$consumedCount, unused=${unusedKeys.length})';
}

class BertHFLoader {
  /// all-MiniLM-L6-v2 (`nreimers/MiniLM-L6-H384-uncased` backbone):
  /// 6 layers, hidden=384, heads=12, ffn=1536, vocab=30522.
  static BertConfig miniLmL6V2Config({
    Device device = Device.CPU,
    int seed = 0,
  }) => BertConfig(
    vocabSize: 30522,
    maxPositionEmbeddings: 512,
    embedDim: 384,
    numLayers: 6,
    numHeads: 12,
    intermediateSize: 1536,
    layerNormEps: 1e-12,
    typeVocabSize: 2,
    device: device,
    seed: seed,
  );

  /// bert-base-uncased: 12 layers, hidden=768, heads=12, ffn=3072.
  static BertConfig bertBaseUncasedConfig({
    Device device = Device.CPU,
    int seed = 0,
  }) => BertConfig(
    vocabSize: 30522,
    maxPositionEmbeddings: 512,
    embedDim: 768,
    numLayers: 12,
    numHeads: 12,
    intermediateSize: 3072,
    layerNormEps: 1e-12,
    typeVocabSize: 2,
    device: device,
    seed: seed,
  );

  static BertLoadReport loadFile(
    BertModel model,
    String path, {
    bool keepFp16 = false,
  }) {
    final state = SafeTensors.loadFile(path, keepFp16: keepFp16);
    return loadMap(model, state);
  }

  static BertLoadReport loadMap(BertModel model, Map<String, Tensor> state) {
    final consumed = <String>{};

    Tensor take(String name) {
      final t = state[name];
      if (t == null) {
        throw ArgumentError('bert loader: missing tensor "$name"');
      }
      consumed.add(name);
      return t;
    }

    final cfg = model.config;
    final d = cfg.embedDim;
    final h = cfg.numHeads;
    final headDim = d ~/ h;
    final ffn = cfg.intermediateSize;

    // ---- embeddings ----
    _copy(
      model.embeddings.wordEmbeddings.weight,
      _expectShape(
        take('embeddings.word_embeddings.weight'),
        [cfg.vocabSize, d],
        'embeddings.word_embeddings.weight',
      ),
    );
    _copy(
      model.embeddings.positionEmbeddings.weight,
      _expectShape(
        take('embeddings.position_embeddings.weight'),
        [cfg.maxPositionEmbeddings, d],
        'embeddings.position_embeddings.weight',
      ),
    );
    final typeTable = _expectShape(
      take('embeddings.token_type_embeddings.weight'),
      [cfg.typeVocabSize, d],
      'embeddings.token_type_embeddings.weight',
    );
    _copy(
      model.embeddings.tokenTypeBias,
      _reshapeVectorTo1xN(_sliceVector(_flattenRow0(typeTable, d), 0, d)),
    );
    _copy(
      model.embeddings.layerNorm.gamma,
      _expectShape(take('embeddings.LayerNorm.weight'), [
        d,
      ], 'embeddings.LayerNorm.weight'),
    );
    _copy(
      model.embeddings.layerNorm.beta,
      _expectShape(take('embeddings.LayerNorm.bias'), [
        d,
      ], 'embeddings.LayerNorm.bias'),
    );

    // ---- per-layer ----
    for (int i = 0; i < cfg.numLayers; i++) {
      final layer = model.layers[i];
      final p = 'encoder.layer.$i';

      // Self-attention: query/key/value are separate [D, D] weights.
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
        _copy(layer.attention.wq[hh].weight, _sliceRows(qW, start, end));
        _copy(
          layer.attention.wq[hh].bias!,
          _reshapeVectorTo1xN(_sliceVector(qB, start, end)),
        );
        _copy(layer.attention.wk[hh].weight, _sliceRows(kW, start, end));
        _copy(
          layer.attention.wk[hh].bias!,
          _reshapeVectorTo1xN(_sliceVector(kB, start, end)),
        );
        _copy(layer.attention.wv[hh].weight, _sliceRows(vW, start, end));
        _copy(
          layer.attention.wv[hh].bias!,
          _reshapeVectorTo1xN(_sliceVector(vB, start, end)),
        );
      }

      // Attention output projection.
      _copy(
        layer.attention.wo.weight,
        _expectShape(
          take('$p.attention.output.dense.weight'),
          [d, d],
          '$p.attention.output.dense.weight',
        ),
      );
      _copy(
        layer.attention.wo.bias!,
        _reshapeVectorTo1xN(
          _expectShape(
            take('$p.attention.output.dense.bias'),
            [d],
            '$p.attention.output.dense.bias',
          ),
        ),
      );

      // Post-attention LayerNorm.
      _copy(
        layer.attentionOutputLn.gamma,
        _expectShape(
          take('$p.attention.output.LayerNorm.weight'),
          [d],
          '$p.attention.output.LayerNorm.weight',
        ),
      );
      _copy(
        layer.attentionOutputLn.beta,
        _expectShape(
          take('$p.attention.output.LayerNorm.bias'),
          [d],
          '$p.attention.output.LayerNorm.bias',
        ),
      );

      // FFN.
      _copy(
        layer.intermediate.weight,
        _expectShape(take('$p.intermediate.dense.weight'), [
          ffn,
          d,
        ], '$p.intermediate.dense.weight'),
      );
      _copy(
        layer.intermediate.bias!,
        _reshapeVectorTo1xN(
          _expectShape(take('$p.intermediate.dense.bias'), [
            ffn,
          ], '$p.intermediate.dense.bias'),
        ),
      );
      _copy(
        layer.output.weight,
        _expectShape(take('$p.output.dense.weight'), [
          d,
          ffn,
        ], '$p.output.dense.weight'),
      );
      _copy(
        layer.output.bias!,
        _reshapeVectorTo1xN(
          _expectShape(take('$p.output.dense.bias'), [
            d,
          ], '$p.output.dense.bias'),
        ),
      );

      // Post-FFN LayerNorm.
      _copy(
        layer.outputLn.gamma,
        _expectShape(take('$p.output.LayerNorm.weight'), [
          d,
        ], '$p.output.LayerNorm.weight'),
      );
      _copy(
        layer.outputLn.beta,
        _expectShape(take('$p.output.LayerNorm.bias'), [
          d,
        ], '$p.output.LayerNorm.bias'),
      );
    }

    // Ignore keys we intentionally don't wire: pooler, position_ids.
    const ignored = {
      'pooler.dense.weight',
      'pooler.dense.bias',
      'embeddings.position_ids',
    };
    final unused =
        state.keys
            .where((k) => !consumed.contains(k) && !ignored.contains(k))
            .toList()
          ..sort();
    return BertLoadReport(consumedCount: consumed.length, unusedKeys: unused);
  }

  // ------------ tensor helpers ------------

  static Tensor _expectShape(Tensor t, List<int> expected, String name) {
    if (t.shape.length != expected.length || !_shapesEqual(t.shape, expected)) {
      throw ArgumentError(
        'bert loader: "$name" expected shape $expected, got ${t.shape}',
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
        'bert loader: copy length mismatch — dst=${dst.shape} '
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

  // Downloads a `[R, D]` tensor's first row into a `[D]` vector; used
  // to fold `token_type_embeddings[0]` into a single bias.
  static Tensor _flattenRow0(Tensor t, int d) {
    final vals = t.toList();
    final out = Float32List(d);
    for (int i = 0; i < d; i++) {
      out[i] = vals[i];
    }
    return Tensor.fromList([d], out, device: Device.CPU);
  }

  static Tensor _reshapeVectorTo1xN(Tensor v) {
    if (v.shape.length != 1) {
      throw ArgumentError(
        '_reshapeVectorTo1xN: expected rank 1, got ${v.shape}',
      );
    }
    return Tensor.fromList([1, v.shape[0]], v.toList(), device: Device.CPU);
  }
}
