/// Loader for HuggingFace MarianMT safetensors.
///
/// Targets the `Helsinki-NLP/opus-mt-*` family. HF key layout:
///
///   model.shared.weight                             [vocab, dModel]
///   model.encoder.embed_positions.weight            [max_pos, dModel]
///   model.encoder.layers.{i}.self_attn.{q,k,v,out}_proj.{weight,bias}
///   model.encoder.layers.{i}.self_attn_layer_norm.{weight,bias}
///   model.encoder.layers.{i}.fc1.{weight,bias}      [ffn, dModel]
///   model.encoder.layers.{i}.fc2.{weight,bias}      [dModel, ffn]
///   model.encoder.layers.{i}.final_layer_norm.{weight,bias}
///   model.decoder.embed_positions.weight
///   model.decoder.layers.{i}.self_attn.*            (same shape)
///   model.decoder.layers.{i}.encoder_attn.*         (cross-attn)
///   model.decoder.layers.{i}.encoder_attn_layer_norm.*
///   model.decoder.layers.{i}.fc1.*  fc2.*
///   model.decoder.layers.{i}.self_attn_layer_norm.*
///   model.decoder.layers.{i}.final_layer_norm.*
///   final_logits_bias                               [1, vocab]
///
/// Marian ships the same embedding weight under several aliases
/// (`model.shared.weight`, `model.encoder.embed_tokens.weight`,
/// `model.decoder.embed_tokens.weight`, `lm_head.weight`). The
/// conversion script drops the aliases; we take just `model.shared`.
library;

import '../tensor/tensor.dart';
import 'marian.dart';
import 'safetensors.dart';

class MarianLoadReport {
  final int consumedCount;
  final List<String> unusedKeys;
  const MarianLoadReport({
    required this.consumedCount,
    required this.unusedKeys,
  });

  @override
  String toString() =>
      'MarianLoadReport(consumed=$consumedCount, unused=${unusedKeys.length})';
}

class MarianHFLoader {
  /// Generic Opus-MT config factory. Every `Helsinki-NLP/opus-mt-*`
  /// language pair we've tested shares the same 6+6-layer, 512-dim,
  /// 8-head, ffn=2048, SiLU-activation architecture — only the vocab
  /// size (and therefore pad / decoder-start ids) varies per pair.
  ///
  /// Pass `vocabSize` from the pair's `config.json` and this returns
  /// a fully-populated [MarianConfig]. Convention (as used by
  /// Helsinki-NLP): `padTokenId = vocabSize - 1`,
  /// `decoderStartTokenId = padTokenId`, `eosTokenId = 0`.
  static MarianConfig opusMtConfig({
    required int vocabSize,
    Device device = Device.CPU,
    int seed = 0,
  }) => MarianConfig(
    vocabSize: vocabSize,
    dModel: 512,
    ffnDim: 2048,
    numLayers: 6,
    numDecoderLayers: 6,
    numHeads: 8,
    maxPositionEmbeddings: 512,
    padTokenId: vocabSize - 1,
    eosTokenId: 0,
    decoderStartTokenId: vocabSize - 1,
    scaleEmbeddings: true,
    activation: MarianActivation.silu,
    device: device,
    seed: seed,
  );

  /// `Helsinki-NLP/opus-mt-en-de` config (~74 M dense).
  static MarianConfig opusMtEnDeConfig({
    Device device = Device.CPU,
    int seed = 0,
  }) => opusMtConfig(vocabSize: 58101, device: device, seed: seed);

  /// `Helsinki-NLP/opus-mt-en-zh` — English -> Simplified Chinese.
  static MarianConfig opusMtEnZhConfig({
    Device device = Device.CPU,
    int seed = 0,
  }) => opusMtConfig(vocabSize: 65001, device: device, seed: seed);

  /// `Helsinki-NLP/opus-mt-zh-en` — Chinese -> English.
  static MarianConfig opusMtZhEnConfig({
    Device device = Device.CPU,
    int seed = 0,
  }) => opusMtConfig(vocabSize: 65001, device: device, seed: seed);

  /// `Helsinki-NLP/opus-mt-{en-bem, bem-en}` — English ↔ Bemba
  /// (most widely spoken Zambian language, ~4.1 M speakers).
  static MarianConfig opusMtEnBemConfig({
    Device device = Device.CPU,
    int seed = 0,
  }) => opusMtConfig(vocabSize: 59828, device: device, seed: seed);
  static MarianConfig opusMtBemEnConfig({
    Device device = Device.CPU,
    int seed = 0,
  }) => opusMtConfig(vocabSize: 59828, device: device, seed: seed);

  /// `Helsinki-NLP/opus-mt-{en-ny, ny-en}` — English ↔ Chichewa /
  /// Nyanja (widely used across Zambia + Malawi, ~14 M speakers).
  static MarianConfig opusMtEnNyConfig({
    Device device = Device.CPU,
    int seed = 0,
  }) => opusMtConfig(vocabSize: 59811, device: device, seed: seed);
  static MarianConfig opusMtNyEnConfig({
    Device device = Device.CPU,
    int seed = 0,
  }) => opusMtConfig(vocabSize: 59811, device: device, seed: seed);

  /// `Helsinki-NLP/opus-mt-{en-toi, toi-en}` — English ↔ Tonga
  /// (Zambia's Southern Province, ~1.5 M speakers).
  static MarianConfig opusMtEnToiConfig({
    Device device = Device.CPU,
    int seed = 0,
  }) => opusMtConfig(vocabSize: 61051, device: device, seed: seed);
  static MarianConfig opusMtToiEnConfig({
    Device device = Device.CPU,
    int seed = 0,
  }) => opusMtConfig(vocabSize: 61051, device: device, seed: seed);

  /// `Helsinki-NLP/opus-mt-{en-loz, loz-en}` — English ↔ Lozi
  /// (Zambia's Western Province, ~700 K speakers).
  static MarianConfig opusMtEnLozConfig({
    Device device = Device.CPU,
    int seed = 0,
  }) => opusMtConfig(vocabSize: 57974, device: device, seed: seed);
  static MarianConfig opusMtLozEnConfig({
    Device device = Device.CPU,
    int seed = 0,
  }) => opusMtConfig(vocabSize: 57974, device: device, seed: seed);

  static MarianLoadReport loadFile(MarianModel model, String path) {
    final state = SafeTensors.loadFile(path);
    return loadMap(model, state);
  }

  static MarianLoadReport loadMap(
    MarianModel model,
    Map<String, Tensor> state,
  ) {
    final consumed = <String>{};
    Tensor take(String name) {
      final t = state[name];
      if (t == null) {
        throw ArgumentError('marian loader: missing tensor "$name"');
      }
      consumed.add(name);
      return t;
    }

    final cfg = model.config;

    // Shared token embedding.
    _assign(
      model.sharedEmbedding.weight,
      take('model.shared.weight'),
      expectShape: [cfg.vocabSize, cfg.dModel],
    );

    // Positional embeddings (loaded verbatim from the file).
    _assign(
      model.encoder.positionEmbeddings,
      take('model.encoder.embed_positions.weight'),
      expectShape: [cfg.maxPositionEmbeddings, cfg.dModel],
    );
    _assign(
      model.decoder.positionEmbeddings,
      take('model.decoder.embed_positions.weight'),
      expectShape: [cfg.maxPositionEmbeddings, cfg.dModel],
    );

    // Encoder blocks.
    for (int i = 0; i < cfg.numLayers; i++) {
      final b = model.encoder.blocks[i];
      _loadAttention(
        b.selfAttn,
        take,
        base: 'model.encoder.layers.$i.self_attn',
        cfg: cfg,
      );
      _loadLayerNorm(
        b.selfAttnLn,
        take,
        base: 'model.encoder.layers.$i.self_attn_layer_norm',
        dim: cfg.dModel,
      );
      _loadFfn(b.ffn, take, base: 'model.encoder.layers.$i', cfg: cfg);
      _loadLayerNorm(
        b.finalLn,
        take,
        base: 'model.encoder.layers.$i.final_layer_norm',
        dim: cfg.dModel,
      );
    }

    // Decoder blocks.
    for (int i = 0; i < cfg.numDecoderLayers; i++) {
      final b = model.decoder.blocks[i];
      _loadAttention(
        b.selfAttn,
        take,
        base: 'model.decoder.layers.$i.self_attn',
        cfg: cfg,
      );
      _loadLayerNorm(
        b.selfAttnLn,
        take,
        base: 'model.decoder.layers.$i.self_attn_layer_norm',
        dim: cfg.dModel,
      );
      _loadAttention(
        b.crossAttn,
        take,
        base: 'model.decoder.layers.$i.encoder_attn',
        cfg: cfg,
      );
      _loadLayerNorm(
        b.crossAttnLn,
        take,
        base: 'model.decoder.layers.$i.encoder_attn_layer_norm',
        dim: cfg.dModel,
      );
      _loadFfn(b.ffn, take, base: 'model.decoder.layers.$i', cfg: cfg);
      _loadLayerNorm(
        b.finalLn,
        take,
        base: 'model.decoder.layers.$i.final_layer_norm',
        dim: cfg.dModel,
      );
    }

    // final_logits_bias.
    _assign(
      model.finalLogitsBias,
      take('final_logits_bias'),
      expectShape: [1, cfg.vocabSize],
    );

    final unused = state.keys.where((k) => !consumed.contains(k)).toList()
      ..sort();
    return MarianLoadReport(consumedCount: consumed.length, unusedKeys: unused);
  }

  static void _loadAttention(
    MarianAttention attn,
    Tensor Function(String) take, {
    required String base,
    required MarianConfig cfg,
  }) {
    // Per-head row slicing of the fused Q/K/V weight+bias.
    final qw = take('$base.q_proj.weight');
    final qb = take('$base.q_proj.bias');
    final kw = take('$base.k_proj.weight');
    final kb = take('$base.k_proj.bias');
    final vw = take('$base.v_proj.weight');
    final vb = take('$base.v_proj.bias');
    _sliceHeadWeightsInto(
      qw,
      qb,
      attn.wq,
      cfg.numHeads,
      cfg.headDim,
      cfg.dModel,
    );
    _sliceHeadWeightsInto(
      kw,
      kb,
      attn.wk,
      cfg.numHeads,
      cfg.headDim,
      cfg.dModel,
    );
    _sliceHeadWeightsInto(
      vw,
      vb,
      attn.wv,
      cfg.numHeads,
      cfg.headDim,
      cfg.dModel,
    );
    _assign(
      attn.wo.weight,
      take('$base.out_proj.weight'),
      expectShape: [cfg.dModel, cfg.dModel],
    );
    _assign(
      attn.wo.bias!,
      _reshape1xN(take('$base.out_proj.bias'), cfg.dModel),
      expectShape: [1, cfg.dModel],
    );
  }

  static void _loadFfn(
    MarianFfn ffn,
    Tensor Function(String) take, {
    required String base,
    required MarianConfig cfg,
  }) {
    _assign(
      ffn.fc1.weight,
      take('$base.fc1.weight'),
      expectShape: [cfg.ffnDim, cfg.dModel],
    );
    _assign(
      ffn.fc1.bias!,
      _reshape1xN(take('$base.fc1.bias'), cfg.ffnDim),
      expectShape: [1, cfg.ffnDim],
    );
    _assign(
      ffn.fc2.weight,
      take('$base.fc2.weight'),
      expectShape: [cfg.dModel, cfg.ffnDim],
    );
    _assign(
      ffn.fc2.bias!,
      _reshape1xN(take('$base.fc2.bias'), cfg.dModel),
      expectShape: [1, cfg.dModel],
    );
  }

  static void _loadLayerNorm(
    dynamic ln,
    Tensor Function(String) take, {
    required String base,
    required int dim,
  }) {
    _assign(ln.gamma, take('$base.weight'), expectShape: [dim]);
    _assign(ln.beta, take('$base.bias'), expectShape: [dim]);
  }

  /// HF stores fused Q/K/V weights as `[numHeads * headDim, dModel]`
  /// and biases as `[numHeads * headDim]`. Per-head weight goes to
  /// rows `[h*headDim, (h+1)*headDim)`, bias to slice
  /// `[h*headDim, (h+1)*headDim)`.
  static void _sliceHeadWeightsInto(
    Tensor fullW,
    Tensor fullB,
    List<dynamic> heads,
    int numHeads,
    int headDim,
    int dModel,
  ) {
    _expectShape(fullW, [numHeads * headDim, dModel], 'fused attn W');
    _expectShape(fullB, [numHeads * headDim], 'fused attn B');
    final wData = fullW.toList();
    final bData = fullB.toList();
    for (int h = 0; h < numHeads; h++) {
      final wVals = List<double>.filled(headDim * dModel, 0);
      final srcW = h * headDim * dModel;
      for (int i = 0; i < headDim * dModel; i++) {
        wVals[i] = wData[srcW + i];
      }
      final bVals = List<double>.filled(headDim, 0);
      for (int i = 0; i < headDim; i++) {
        bVals[i] = bData[h * headDim + i];
      }
      final head = heads[h];
      _assign(
        head.weight,
        Tensor.fromList([headDim, dModel], wVals, device: fullW.device),
        expectShape: [headDim, dModel],
      );
      _assign(
        head.bias!,
        Tensor.fromList([1, headDim], bVals, device: fullB.device),
        expectShape: [1, headDim],
      );
    }
  }

  /// HF stores biases as `[N]`; our `Linear.bias` is `[1, N]`.
  static Tensor _reshape1xN(Tensor t, int n) {
    if (t.shape.length == 2 && t.shape[0] == 1 && t.shape[1] == n) return t;
    if (t.shape.length == 1 && t.shape[0] == n) {
      return Tensor.fromList([1, n], t.toList(), device: t.device);
    }
    throw ArgumentError('marian loader: cannot reshape ${t.shape} to [1, $n]');
  }

  static void _assign(Tensor dst, Tensor src, {List<int>? expectShape}) {
    if (expectShape != null) _expectShape(src, expectShape, 'assign');
    if (dst.shape.length != src.shape.length ||
        !_shapesEqual(dst.shape, src.shape)) {
      throw ArgumentError(
        'marian loader: shape mismatch — dst=${dst.shape} src=${src.shape}',
      );
    }
    if (src.device != dst.device) {
      final vals = src.toList();
      final matched = Tensor.fromList(dst.shape, vals, device: dst.device);
      dst.assign(matched);
    } else {
      dst.assign(src);
    }
  }

  static void _expectShape(Tensor t, List<int> shape, String label) {
    if (!_shapesEqual(t.shape, shape)) {
      throw ArgumentError(
        'marian loader: $label expected shape $shape; got ${t.shape}',
      );
    }
  }

  static bool _shapesEqual(List<int> a, List<int> b) {
    if (a.length != b.length) return false;
    for (int i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }
}
