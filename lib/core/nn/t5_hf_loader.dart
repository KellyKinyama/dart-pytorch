/// Loader for HuggingFace T5 safetensors.
///
/// Targets `google/t5-{v1_1-,flan-t5-,}small` and the base/large
/// variants of the same family. HF key layout:
///
///   shared.weight
///   encoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}.weight
///   encoder.block.{i}.layer.0.SelfAttention.relative_attention_bias.weight
///     (only on i == 0; other blocks omit this key and reuse block-0's
///      table at forward time.)
///   encoder.block.{i}.layer.0.layer_norm.weight
///   encoder.block.{i}.layer.1.DenseReluDense.wi.weight          (T5 v1.0)
///   encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight        (v1.1 / FLAN)
///   encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight        (v1.1 / FLAN)
///   encoder.block.{i}.layer.1.DenseReluDense.wo.weight
///   encoder.block.{i}.layer.1.layer_norm.weight
///   encoder.final_layer_norm.weight
///   decoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}.weight
///   decoder.block.{i}.layer.0.SelfAttention.relative_attention_bias.weight
///     (i == 0 only)
///   decoder.block.{i}.layer.0.layer_norm.weight
///   decoder.block.{i}.layer.1.EncDecAttention.{q,k,v,o}.weight
///   decoder.block.{i}.layer.1.layer_norm.weight
///   decoder.block.{i}.layer.2.DenseReluDense.*  (same shape as encoder FFN)
///   decoder.block.{i}.layer.2.layer_norm.weight
///   decoder.final_layer_norm.weight
///   lm_head.weight  (present only when tie_word_embeddings=False)
///
/// T5 attention weights are stored **untransposed relative to how our
/// Linear stores them.** In HF the tensors are shaped
/// `[num_heads * d_kv, d_model]` (for q/k/v) and `[d_model, num_heads *
/// d_kv]` (for o), which happens to match our `[out, in]` convention
/// exactly — so we just slice per-head rows and drop them into our
/// per-head Linear list.
library;

import '../tensor/tensor.dart';
import 't5.dart';
import 'rms_norm.dart';
import 'safetensors.dart';

class T5LoadReport {
  final int consumedCount;
  final List<String> unusedKeys;
  const T5LoadReport({required this.consumedCount, required this.unusedKeys});

  @override
  String toString() =>
      'T5LoadReport(consumed=$consumedCount, unused=${unusedKeys.length})';
}

class T5HFLoader {
  /// `google/t5-small` — 60 M params, ReLU FFN.
  static T5Config t5SmallConfig({
    Device device = Device.CPU,
    int seed = 0,
    int? maxCtx,
  }) => T5Config(
    vocabSize: 32128,
    dModel: 512,
    dFf: 2048,
    dKv: 64,
    numLayers: 6,
    numDecoderLayers: 6,
    numHeads: 8,
    feedForwardProj: T5FfnActivation.relu,
    maxCtx: maxCtx ?? 512,
    device: device,
    seed: seed,
  );

  /// `google/t5-v1_1-small` — 60 M params, gated-GELU FFN.
  static T5Config t5V11SmallConfig({
    Device device = Device.CPU,
    int seed = 0,
    int? maxCtx,
  }) => T5Config(
    vocabSize: 32128,
    dModel: 512,
    dFf: 1024,
    dKv: 64,
    numLayers: 8,
    numDecoderLayers: 8,
    numHeads: 6,
    feedForwardProj: T5FfnActivation.gatedGelu,
    maxCtx: maxCtx ?? 512,
    device: device,
    seed: seed,
  );

  /// `google/flan-t5-small` — same arch as `t5-v1_1-small`, just
  /// instruction-tuned weights.
  static T5Config flanT5SmallConfig({
    Device device = Device.CPU,
    int seed = 0,
    int? maxCtx,
  }) => t5V11SmallConfig(device: device, seed: seed, maxCtx: maxCtx);

  /// `google/flan-t5-base` — same as `t5-v1_1-base` (12 layers,
  /// dModel=768, dFf=2048, numHeads=12, dKv=64).
  static T5Config flanT5BaseConfig({
    Device device = Device.CPU,
    int seed = 0,
    int? maxCtx,
  }) => T5Config(
    vocabSize: 32128,
    dModel: 768,
    dFf: 2048,
    dKv: 64,
    numLayers: 12,
    numDecoderLayers: 12,
    numHeads: 12,
    feedForwardProj: T5FfnActivation.gatedGelu,
    maxCtx: maxCtx ?? 512,
    device: device,
    seed: seed,
  );

  /// `Salesforce/codet5p-220m` — CodeT5+ 220M. T5-v1.1-base
  /// architecture (12 encoder + 12 decoder layers, dModel=768,
  /// gated-GELU FFN with dFf=2048), fine-tuned for code. Uses the
  /// same 32100-token vocab as regular T5.
  static T5Config codeT5pBaseConfig({
    Device device = Device.CPU,
    int seed = 0,
    int? maxCtx,
  }) => flanT5BaseConfig(device: device, seed: seed, maxCtx: maxCtx);

  static T5LoadReport loadFile(T5Model model, String path) {
    final state = SafeTensors.loadFile(path);
    return loadMap(model, state);
  }

  static T5LoadReport loadMap(T5Model model, Map<String, Tensor> state) {
    final consumed = <String>{};
    Tensor take(String name) {
      final t = state[name];
      if (t == null) {
        throw ArgumentError('t5 loader: missing tensor "$name"');
      }
      consumed.add(name);
      return t;
    }

    Tensor? takeOptional(String name) {
      final t = state[name];
      if (t != null) consumed.add(name);
      return t;
    }

    // shared embedding
    _assign(
      model.sharedEmbedding.weight,
      take('shared.weight'),
      expectShape: [model.config.vocabSize, model.config.dModel],
    );

    // encoder blocks
    _loadStack(
      state: state,
      take: take,
      takeOptional: takeOptional,
      cfg: model.config,
      prefix: 'encoder.block',
      selfAttnKey: 'SelfAttention',
      selfAttnLayerIdx: 0,
      ffnLayerIdx: 1,
      isDecoder: false,
      relativeBias: model.encoder.relativeBias,
      selfAttns: [for (final b in model.encoder.blocks) b.selfAttn],
      selfAttnNorms: [for (final b in model.encoder.blocks) b.selfAttnNorm],
      crossAttns: null,
      crossAttnNorms: null,
      ffns: [for (final b in model.encoder.blocks) b.ffn],
      ffnNorms: [for (final b in model.encoder.blocks) b.ffnNorm],
    );
    _assign(
      model.encoder.finalNorm.gamma,
      take('encoder.final_layer_norm.weight'),
      expectShape: [model.config.dModel],
    );

    // decoder blocks
    _loadStack(
      state: state,
      take: take,
      takeOptional: takeOptional,
      cfg: model.config,
      prefix: 'decoder.block',
      selfAttnKey: 'SelfAttention',
      selfAttnLayerIdx: 0,
      ffnLayerIdx: 2,
      isDecoder: true,
      relativeBias: model.decoder.relativeBias,
      selfAttns: [for (final b in model.decoder.blocks) b.selfAttn],
      selfAttnNorms: [for (final b in model.decoder.blocks) b.selfAttnNorm],
      crossAttns: [for (final b in model.decoder.blocks) b.crossAttn],
      crossAttnNorms: [for (final b in model.decoder.blocks) b.crossAttnNorm],
      ffns: [for (final b in model.decoder.blocks) b.ffn],
      ffnNorms: [for (final b in model.decoder.blocks) b.ffnNorm],
    );
    _assign(
      model.decoder.finalNorm.gamma,
      take('decoder.final_layer_norm.weight'),
      expectShape: [model.config.dModel],
    );

    // LM head. HF checkpoints for T5-v1.1 and FLAN ship a distinct
    // `lm_head.weight` even when config.tie_word_embeddings is True
    // — Google's official pretraining stored them separately. When
    // both are present we honor the checkpoint (untied) and skip
    // the `1/sqrt(dModel)` rescale HF would otherwise apply.
    final lmHeadTensor = takeOptional('lm_head.weight');
    if (lmHeadTensor != null) {
      _assign(
        model.lmHead.weight,
        lmHeadTensor,
        expectShape: [model.config.vocabSize, model.config.dModel],
      );
      model.useUntiedLmHead = true;
    } else {
      // Tied path: copy shared.weight into the pre-allocated lmHead.
      model.lmHead.weight.assign(model.sharedEmbedding.weight);
      model.useUntiedLmHead = false;
    }

    final unused = state.keys.where((k) => !consumed.contains(k)).toList()
      ..sort();
    return T5LoadReport(consumedCount: consumed.length, unusedKeys: unused);
  }

  static void _loadStack({
    required Map<String, Tensor> state,
    required Tensor Function(String) take,
    required Tensor? Function(String) takeOptional,
    required T5Config cfg,
    required String prefix,
    required String selfAttnKey,
    required int selfAttnLayerIdx,
    required int ffnLayerIdx,
    required bool isDecoder,
    required T5RelativeBias relativeBias,
    required List<T5Attention> selfAttns,
    required List<RMSNorm> selfAttnNorms,
    required List<T5Attention>? crossAttns,
    required List<RMSNorm>? crossAttnNorms,
    required List<T5Ffn> ffns,
    required List<RMSNorm> ffnNorms,
  }) {
    final n = selfAttns.length;
    for (int i = 0; i < n; i++) {
      final base = '$prefix.$i.layer';

      // Self-attention block.
      _loadT5Attention(
        selfAttns[i],
        take,
        base: '$base.$selfAttnLayerIdx.$selfAttnKey',
        cfg: cfg,
      );
      _assign(
        selfAttnNorms[i].gamma,
        take('$base.$selfAttnLayerIdx.layer_norm.weight'),
        expectShape: [cfg.dModel],
      );

      // Relative bias only on block 0.
      if (i == 0) {
        final biasKey =
            '$base.$selfAttnLayerIdx.$selfAttnKey.relative_attention_bias.weight';
        _assign(
          relativeBias.table.weight,
          take(biasKey),
          expectShape: [cfg.relativeAttentionNumBuckets, cfg.numHeads],
        );
        relativeBias.invalidateCache();
      }

      // Decoder-only: cross-attention.
      if (isDecoder) {
        _loadT5Attention(
          crossAttns![i],
          take,
          base: '$base.1.EncDecAttention',
          cfg: cfg,
        );
        _assign(
          crossAttnNorms![i].gamma,
          take('$base.1.layer_norm.weight'),
          expectShape: [cfg.dModel],
        );
      }

      // FFN.
      _loadT5Ffn(
        ffns[i],
        take,
        base: '$base.$ffnLayerIdx.DenseReluDense',
        cfg: cfg,
      );
      _assign(
        ffnNorms[i].gamma,
        take('$base.$ffnLayerIdx.layer_norm.weight'),
        expectShape: [cfg.dModel],
      );
    }
  }

  static void _loadT5Attention(
    T5Attention attn,
    Tensor Function(String) take, {
    required String base,
    required T5Config cfg,
  }) {
    // HF stores q/k/v as [num_heads * d_kv, d_model], o as
    // [d_model, num_heads * d_kv]. Slice per-head rows for Q/K/V,
    // and per-head columns for O.
    final qFull = take('$base.q.weight');
    final kFull = take('$base.k.weight');
    final vFull = take('$base.v.weight');
    final oFull = take('$base.o.weight');
    final h = cfg.numHeads;
    final d = cfg.dKv;
    _sliceHeadRowsInto(qFull, attn.wq, h, d, cfg.dModel);
    _sliceHeadRowsInto(kFull, attn.wk, h, d, attn.kvDim);
    _sliceHeadRowsInto(vFull, attn.wv, h, d, attn.kvDim);
    _assign(attn.wo.weight, oFull, expectShape: [cfg.dModel, h * d]);
  }

  static void _loadT5Ffn(
    T5Ffn ffn,
    Tensor Function(String) take, {
    required String base,
    required T5Config cfg,
  }) {
    if (ffn.wi1 == null) {
      _assign(
        ffn.wi0.weight,
        take('$base.wi.weight'),
        expectShape: [cfg.dFf, cfg.dModel],
      );
    } else {
      _assign(
        ffn.wi0.weight,
        take('$base.wi_0.weight'),
        expectShape: [cfg.dFf, cfg.dModel],
      );
      _assign(
        ffn.wi1!.weight,
        take('$base.wi_1.weight'),
        expectShape: [cfg.dFf, cfg.dModel],
      );
    }
    _assign(
      ffn.wo.weight,
      take('$base.wo.weight'),
      expectShape: [cfg.dModel, cfg.dFf],
    );
  }

  /// HF QKV weight is `[numHeads * dKv, inDim]`. Slice head `h` as
  /// rows `[h*dKv, (h+1)*dKv)` -> our per-head Linear `weight` is
  /// `[dKv, inDim]`.
  static void _sliceHeadRowsInto(
    Tensor full,
    List<dynamic> heads,
    int numHeads,
    int dKv,
    int inDim,
  ) {
    _expectShape(full, [numHeads * dKv, inDim], 'attention fused Q/K/V');
    final data = full.toList();
    for (int h = 0; h < numHeads; h++) {
      final rowVals = List<double>.filled(dKv * inDim, 0);
      final srcBase = h * dKv * inDim;
      for (int i = 0; i < dKv * inDim; i++) {
        rowVals[i] = data[srcBase + i];
      }
      final headLinear = heads[h];
      _assign(
        headLinear.weight,
        Tensor.fromList([dKv, inDim], rowVals, device: full.device),
        expectShape: [dKv, inDim],
      );
    }
  }

  static void _assign(Tensor dst, Tensor src, {List<int>? expectShape}) {
    if (expectShape != null) _expectShape(src, expectShape, 'assign');
    if (dst.shape.length != src.shape.length ||
        !_shapesEqual(dst.shape, src.shape)) {
      throw ArgumentError(
        't5 loader: shape mismatch — dst=${dst.shape} src=${src.shape}',
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
        't5 loader: $label expected shape $shape; got ${t.shape}',
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
