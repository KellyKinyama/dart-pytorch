/// Loader for HuggingFace `openai/whisper-*` safetensors weights.
///
/// Maps HuggingFace's fused `q_proj`, `k_proj`, `v_proj` `[C, C]`
/// weights onto our per-head `qHeads`, `kHeads`, `vHeads` lists
/// (each head: `[headDim, C]`). Row split: rows `[h*headDim,
/// (h+1)*headDim)` of the fused weight go to head `h`.
///
/// HF key -> our field:
///
///   model.encoder.conv1.{weight,bias}                 -> encoder.conv1
///   model.encoder.conv2.{weight,bias}                 -> encoder.conv2
///   model.encoder.embed_positions.weight              -> encoder.positionalEmbedding
///   model.encoder.layer_norm.{weight,bias}            -> encoder.lnPost
///   model.encoder.layers.{i}.self_attn.q_proj.*       -> block.qHeads
///   model.encoder.layers.{i}.self_attn.k_proj.weight  -> block.kHeads
///   model.encoder.layers.{i}.self_attn.v_proj.*       -> block.vHeads
///   model.encoder.layers.{i}.self_attn.out_proj.*     -> block.outProj
///   model.encoder.layers.{i}.self_attn_layer_norm.*   -> block.attnLn
///   model.encoder.layers.{i}.fc1.*                    -> block.mlp0
///   model.encoder.layers.{i}.fc2.*                    -> block.mlp2
///   model.encoder.layers.{i}.final_layer_norm.*       -> block.mlpLn
///
/// Decoder shares the same layout with `model.decoder.*`, plus
/// `encoder_attn.*` and `embed_tokens` / `embed_positions`.
library;

import 'dart:typed_data';

import '../tensor/tensor.dart';
import 'safetensors.dart';
import 'whisper.dart';
import 'whisper_decoder.dart';

class WhisperLoadReport {
  final int consumedCount;
  final List<String> unusedKeys;
  const WhisperLoadReport({
    required this.consumedCount,
    required this.unusedKeys,
  });

  @override
  String toString() =>
      'WhisperLoadReport(consumed=$consumedCount, unused=${unusedKeys.length})';
}

class WhisperHFLoader {
  static WhisperLoadReport loadFile(
    WhisperEncoder encoder,
    String path, {
    bool keepFp16 = false,
  }) {
    final state = SafeTensors.loadFile(path, keepFp16: keepFp16);
    return loadMap(encoder, state);
  }

  static WhisperLoadReport loadDecoderFile(
    WhisperDecoder decoder,
    String path, {
    bool keepFp16 = false,
  }) {
    final state = SafeTensors.loadFile(path, keepFp16: keepFp16);
    return loadDecoderMap(decoder, state);
  }

  static WhisperLoadReport loadMap(
    WhisperEncoder encoder,
    Map<String, Tensor> state,
  ) {
    final consumed = <String>{};
    Tensor take(String name) {
      final t = state[name];
      if (t == null) {
        throw ArgumentError('whisper loader: missing tensor "$name"');
      }
      consumed.add(name);
      return t;
    }

    _loadConv(
      encoder.conv1,
      take('model.encoder.conv1.weight'),
      take('model.encoder.conv1.bias'),
      inC: encoder.nMels,
      outC: encoder.embedDim,
      k: 3,
    );
    _loadConv(
      encoder.conv2,
      take('model.encoder.conv2.weight'),
      take('model.encoder.conv2.bias'),
      inC: encoder.embedDim,
      outC: encoder.embedDim,
      k: 3,
    );

    final pe = take('model.encoder.embed_positions.weight');
    _expectShape(pe, [
      encoder.nCtx,
      encoder.embedDim,
    ], 'model.encoder.embed_positions.weight');
    _assign(encoder.positionalEmbedding, pe);

    for (int i = 0; i < encoder.numLayers; i++) {
      final blk = encoder.blocks[i];
      final p = 'model.encoder.layers.$i';

      _loadFusedHeads(
        blk.qHeads,
        weight: take('$p.self_attn.q_proj.weight'),
        bias: take('$p.self_attn.q_proj.bias'),
        embedDim: encoder.embedDim,
        numHeads: encoder.numHeads,
      );
      _loadFusedHeads(
        blk.kHeads,
        weight: take('$p.self_attn.k_proj.weight'),
        bias: null,
        embedDim: encoder.embedDim,
        numHeads: encoder.numHeads,
      );
      _loadFusedHeads(
        blk.vHeads,
        weight: take('$p.self_attn.v_proj.weight'),
        bias: take('$p.self_attn.v_proj.bias'),
        embedDim: encoder.embedDim,
        numHeads: encoder.numHeads,
      );
      _loadLinear(
        blk.outProj,
        weight: take('$p.self_attn.out_proj.weight'),
        bias: take('$p.self_attn.out_proj.bias'),
        outF: encoder.embedDim,
        inF: encoder.embedDim,
      );
      _loadLayerNorm(
        blk.attnLn,
        weight: take('$p.self_attn_layer_norm.weight'),
        bias: take('$p.self_attn_layer_norm.bias'),
        dim: encoder.embedDim,
      );

      _loadLinear(
        blk.mlp0,
        weight: take('$p.fc1.weight'),
        bias: take('$p.fc1.bias'),
        outF: encoder.embedDim * 4,
        inF: encoder.embedDim,
      );
      _loadLinear(
        blk.mlp2,
        weight: take('$p.fc2.weight'),
        bias: take('$p.fc2.bias'),
        outF: encoder.embedDim,
        inF: encoder.embedDim * 4,
      );
      _loadLayerNorm(
        blk.mlpLn,
        weight: take('$p.final_layer_norm.weight'),
        bias: take('$p.final_layer_norm.bias'),
        dim: encoder.embedDim,
      );
    }

    _loadLayerNorm(
      encoder.lnPost,
      weight: take('model.encoder.layer_norm.weight'),
      bias: take('model.encoder.layer_norm.bias'),
      dim: encoder.embedDim,
    );

    final unused =
        state.keys
            .where(
              (k) =>
                  !consumed.contains(k) &&
                  !k.startsWith('model.decoder.') &&
                  k != 'proj_out.weight',
            )
            .toList()
          ..sort();
    return WhisperLoadReport(
      consumedCount: consumed.length,
      unusedKeys: unused,
    );
  }

  static WhisperLoadReport loadDecoderMap(
    WhisperDecoder decoder,
    Map<String, Tensor> state,
  ) {
    final consumed = <String>{};
    Tensor take(String name) {
      final t = state[name];
      if (t == null) {
        throw ArgumentError('whisper decoder loader: missing tensor "$name"');
      }
      consumed.add(name);
      return t;
    }

    final tokE = take('model.decoder.embed_tokens.weight');
    _expectShape(tokE, [
      decoder.vocabSize,
      decoder.embedDim,
    ], 'model.decoder.embed_tokens.weight');
    _assign(decoder.tokenEmbedding.weight, tokE);

    final posE = take('model.decoder.embed_positions.weight');
    _expectShape(posE, [
      decoder.nCtx,
      decoder.embedDim,
    ], 'model.decoder.embed_positions.weight');
    _assign(decoder.positionalEmbedding, posE);

    for (int i = 0; i < decoder.numLayers; i++) {
      final blk = decoder.blocks[i];
      final p = 'model.decoder.layers.$i';

      // Self-attention.
      _loadFusedHeads(
        blk.qHeads,
        weight: take('$p.self_attn.q_proj.weight'),
        bias: take('$p.self_attn.q_proj.bias'),
        embedDim: decoder.embedDim,
        numHeads: decoder.numHeads,
      );
      _loadFusedHeads(
        blk.kHeads,
        weight: take('$p.self_attn.k_proj.weight'),
        bias: null,
        embedDim: decoder.embedDim,
        numHeads: decoder.numHeads,
      );
      _loadFusedHeads(
        blk.vHeads,
        weight: take('$p.self_attn.v_proj.weight'),
        bias: take('$p.self_attn.v_proj.bias'),
        embedDim: decoder.embedDim,
        numHeads: decoder.numHeads,
      );
      _loadLinear(
        blk.outProj,
        weight: take('$p.self_attn.out_proj.weight'),
        bias: take('$p.self_attn.out_proj.bias'),
        outF: decoder.embedDim,
        inF: decoder.embedDim,
      );
      _loadLayerNorm(
        blk.attnLn,
        weight: take('$p.self_attn_layer_norm.weight'),
        bias: take('$p.self_attn_layer_norm.bias'),
        dim: decoder.embedDim,
      );

      // Cross-attention.
      _loadFusedHeads(
        blk.crossQHeads,
        weight: take('$p.encoder_attn.q_proj.weight'),
        bias: take('$p.encoder_attn.q_proj.bias'),
        embedDim: decoder.embedDim,
        numHeads: decoder.numHeads,
      );
      _loadFusedHeads(
        blk.crossKHeads,
        weight: take('$p.encoder_attn.k_proj.weight'),
        bias: null,
        embedDim: decoder.embedDim,
        numHeads: decoder.numHeads,
      );
      _loadFusedHeads(
        blk.crossVHeads,
        weight: take('$p.encoder_attn.v_proj.weight'),
        bias: take('$p.encoder_attn.v_proj.bias'),
        embedDim: decoder.embedDim,
        numHeads: decoder.numHeads,
      );
      _loadLinear(
        blk.crossOutProj,
        weight: take('$p.encoder_attn.out_proj.weight'),
        bias: take('$p.encoder_attn.out_proj.bias'),
        outF: decoder.embedDim,
        inF: decoder.embedDim,
      );
      _loadLayerNorm(
        blk.crossAttnLn,
        weight: take('$p.encoder_attn_layer_norm.weight'),
        bias: take('$p.encoder_attn_layer_norm.bias'),
        dim: decoder.embedDim,
      );

      // MLP.
      _loadLinear(
        blk.mlp0,
        weight: take('$p.fc1.weight'),
        bias: take('$p.fc1.bias'),
        outF: decoder.embedDim * 4,
        inF: decoder.embedDim,
      );
      _loadLinear(
        blk.mlp2,
        weight: take('$p.fc2.weight'),
        bias: take('$p.fc2.bias'),
        outF: decoder.embedDim,
        inF: decoder.embedDim * 4,
      );
      _loadLayerNorm(
        blk.mlpLn,
        weight: take('$p.final_layer_norm.weight'),
        bias: take('$p.final_layer_norm.bias'),
        dim: decoder.embedDim,
      );
    }

    _loadLayerNorm(
      decoder.ln,
      weight: take('model.decoder.layer_norm.weight'),
      bias: take('model.decoder.layer_norm.bias'),
      dim: decoder.embedDim,
    );

    final unused =
        state.keys
            .where(
              (k) =>
                  !consumed.contains(k) &&
                  !k.startsWith('model.encoder.') &&
                  k != 'proj_out.weight',
            )
            .toList()
          ..sort();
    return WhisperLoadReport(
      consumedCount: consumed.length,
      unusedKeys: unused,
    );
  }

  // ------------ per-module helpers ------------

  static void _loadConv(
    dynamic conv,
    Tensor weight,
    Tensor bias, {
    required int inC,
    required int outC,
    required int k,
  }) {
    _expectShape(weight, [outC, inC, k], 'conv.weight');
    _expectShape(bias, [outC], 'conv.bias');
    conv.loadFromPytorch(_toF32(weight), _toF32(bias));
  }

  static void _loadLinear(
    dynamic linear, {
    required Tensor weight,
    required Tensor? bias,
    required int outF,
    required int inF,
  }) {
    _expectShape(weight, [outF, inF], 'linear.weight');
    _assign(linear.weight, weight);
    if (bias != null) {
      _expectShape(bias, [outF], 'linear.bias');
      _assignBias1xN(linear.bias, bias);
    }
  }

  static void _loadLayerNorm(
    dynamic ln, {
    required Tensor weight,
    required Tensor bias,
    required int dim,
  }) {
    _expectShape(weight, [dim], 'ln.weight');
    _expectShape(bias, [dim], 'ln.bias');
    _assign(ln.gamma, weight);
    _assign(ln.beta, bias);
  }

  /// Split HF fused `weight [C, C]` (and optional `bias [C]`) row-wise
  /// into `numHeads` per-head Linear projections of shape
  /// `[headDim, C]` (weight) and `[1, headDim]` (bias).
  static void _loadFusedHeads(
    List<dynamic> heads, {
    required Tensor weight,
    required Tensor? bias,
    required int embedDim,
    required int numHeads,
  }) {
    final headDim = embedDim ~/ numHeads;
    _expectShape(weight, [embedDim, embedDim], 'fused.weight');
    final wSrc = _toF32(weight);
    if (bias != null) {
      _expectShape(bias, [embedDim], 'fused.bias');
    }
    final bSrc = bias == null ? null : _toF32(bias);
    for (int h = 0; h < numHeads; h++) {
      final headLinear = heads[h];
      // Slice rows [h*headDim, (h+1)*headDim) into a [headDim, C] chunk.
      final wChunk = Float32List(headDim * embedDim);
      wChunk.setRange(
        0,
        headDim * embedDim,
        wSrc,
        h * headDim * embedDim,
      );
      final wT = Tensor.fromFloat32List(
        [headDim, embedDim],
        wChunk,
        device: Device.CPU,
      );
      _assign(headLinear.weight, wT);
      if (bSrc != null) {
        final bChunk = Float32List(headDim);
        bChunk.setRange(0, headDim, bSrc, h * headDim);
        final bT = Tensor.fromFloat32List(
          [headDim],
          bChunk,
          device: Device.CPU,
        );
        _assignBias1xN(headLinear.bias, bT);
      }
    }
  }

  // ------------ tensor helpers ------------

  static Tensor _expectShape(Tensor t, List<int> expected, String name) {
    if (t.shape.length != expected.length ||
        !_shapesEqual(t.shape, expected)) {
      throw ArgumentError(
        'whisper loader: "$name" expected shape $expected, got ${t.shape}',
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

  static Float32List _toF32(Tensor t) {
    final data = t.toList();
    final out = Float32List(data.length);
    for (int i = 0; i < data.length; i++) {
      out[i] = data[i];
    }
    return out;
  }

  /// Element-count-preserving copy of `src` into `dst`, honouring
  /// `dst`'s device (uploads to GPU as needed).
  static void _assign(Tensor dst, Tensor src) {
    if (dst.length != src.length) {
      throw ArgumentError(
        'whisper loader: _assign length mismatch — dst=${dst.shape}, '
        'src=${src.shape}',
      );
    }
    final vals = src.toList();
    final matched = Tensor.fromList(dst.shape, vals, device: dst.device);
    dst.assign(matched);
  }

  /// Bias variant: source is [outF]; destination is [1, outF].
  static void _assignBias1xN(Tensor dst, Tensor src) {
    if (dst.shape.length != 2 ||
        dst.shape[0] != 1 ||
        dst.shape[1] != src.length) {
      throw ArgumentError(
        'whisper loader: _assignBias1xN shape mismatch — dst=${dst.shape}, '
        'src=${src.shape}',
      );
    }
    final vals = src.toList();
    final matched = Tensor.fromList([1, src.length], vals, device: dst.device);
    dst.assign(matched);
  }
}
