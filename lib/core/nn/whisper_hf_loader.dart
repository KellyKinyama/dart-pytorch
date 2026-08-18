/// Loader for HuggingFace `openai/whisper-*` safetensors weights.
///
/// Maps HuggingFace's `model.encoder.*` naming onto the openai-whisper
/// layout expected by [WhisperEncoder]:
///
///   HF                                                   openai
///   -----------------------------------------------------------------
///   model.encoder.conv1.{weight,bias}                    conv1
///   model.encoder.conv2.{weight,bias}                    conv2
///   model.encoder.embed_positions.weight                 positional_embedding
///   model.encoder.layer_norm.{weight,bias}               ln_post
///   model.encoder.layers.{i}.self_attn.q_proj.*          blocks.{i}.attn.query
///   model.encoder.layers.{i}.self_attn.k_proj.weight     blocks.{i}.attn.key   (no bias)
///   model.encoder.layers.{i}.self_attn.v_proj.*          blocks.{i}.attn.value
///   model.encoder.layers.{i}.self_attn.out_proj.*        blocks.{i}.attn.out
///   model.encoder.layers.{i}.self_attn_layer_norm.*      blocks.{i}.attn_ln
///   model.encoder.layers.{i}.fc1.*                       blocks.{i}.mlp.0
///   model.encoder.layers.{i}.fc2.*                       blocks.{i}.mlp.2
///   model.encoder.layers.{i}.final_layer_norm.*          blocks.{i}.mlp_ln
///
/// Decoder loading is handled by [WhisperHFLoader.loadDecoderFile] /
/// [loadDecoderMap], with an analogous key map for `model.decoder.*`.
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
  /// Load `openai/whisper-tiny.en` weights into [encoder].
  ///
  /// Decoder weights present in the file are recorded as "ignored"
  /// (not "unused") so the report stays clean.
  static WhisperLoadReport loadFile(
    WhisperEncoder encoder,
    String path, {
    bool keepFp16 = false,
  }) {
    final state = SafeTensors.loadFile(path, keepFp16: keepFp16);
    return loadMap(encoder, state);
  }

  /// Load `openai/whisper-tiny.en` decoder weights from the same
  /// safetensors file. Encoder tensors present in the file are
  /// ignored.
  static WhisperLoadReport loadDecoderFile(
    WhisperDecoder decoder,
    String path, {
    bool keepFp16 = false,
  }) {
    final state = SafeTensors.loadFile(path, keepFp16: keepFp16);
    return loadDecoderMap(decoder, state);
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

    // ---- token + positional embedding ----
    final tokE = take('model.decoder.embed_tokens.weight');
    _expectShape(tokE, [
      decoder.vocabSize,
      decoder.embedDim,
    ], 'model.decoder.embed_tokens.weight');
    _assign1d(decoder.tokenEmbedding.weight, tokE);

    final posE = take('model.decoder.embed_positions.weight');
    _expectShape(posE, [
      decoder.nCtx,
      decoder.embedDim,
    ], 'model.decoder.embed_positions.weight');
    _assign1d(decoder.positionalEmbedding, posE);

    // ---- per block ----
    for (int i = 0; i < decoder.numLayers; i++) {
      final blk = decoder.blocks[i];
      final p = 'model.decoder.layers.$i';

      // Self-attention.
      _loadLinear(
        blk.qProj,
        weight: take('$p.self_attn.q_proj.weight'),
        bias: take('$p.self_attn.q_proj.bias'),
        outF: decoder.embedDim,
        inF: decoder.embedDim,
      );
      _loadLinear(
        blk.kProj,
        weight: take('$p.self_attn.k_proj.weight'),
        bias: null,
        outF: decoder.embedDim,
        inF: decoder.embedDim,
      );
      _loadLinear(
        blk.vProj,
        weight: take('$p.self_attn.v_proj.weight'),
        bias: take('$p.self_attn.v_proj.bias'),
        outF: decoder.embedDim,
        inF: decoder.embedDim,
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
      _loadLinear(
        blk.crossQProj,
        weight: take('$p.encoder_attn.q_proj.weight'),
        bias: take('$p.encoder_attn.q_proj.bias'),
        outF: decoder.embedDim,
        inF: decoder.embedDim,
      );
      _loadLinear(
        blk.crossKProj,
        weight: take('$p.encoder_attn.k_proj.weight'),
        bias: null,
        outF: decoder.embedDim,
        inF: decoder.embedDim,
      );
      _loadLinear(
        blk.crossVProj,
        weight: take('$p.encoder_attn.v_proj.weight'),
        bias: take('$p.encoder_attn.v_proj.bias'),
        outF: decoder.embedDim,
        inF: decoder.embedDim,
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

    // ---- final LayerNorm ----
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

    // ---------- conv1 / conv2 ----------
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

    // ---------- positional embedding (HF stores the sinusoids as a
    // learned matrix). Overwriting our computed sinusoids guarantees
    // bit-close agreement with the reference implementation.
    final pe = take('model.encoder.embed_positions.weight');
    _expectShape(pe, [
      encoder.nCtx,
      encoder.embedDim,
    ], 'model.encoder.embed_positions.weight');
    _assign1d(encoder.positionalEmbedding, pe);

    // ---------- per-block ----------
    for (int i = 0; i < encoder.numLayers; i++) {
      final blk = encoder.blocks[i];
      final p = 'model.encoder.layers.$i';

      _loadLinear(
        blk.qProj,
        weight: take('$p.self_attn.q_proj.weight'),
        bias: take('$p.self_attn.q_proj.bias'),
        outF: encoder.embedDim,
        inF: encoder.embedDim,
      );
      _loadLinear(
        blk.kProj,
        weight: take('$p.self_attn.k_proj.weight'),
        bias: null, // Whisper's k_proj has no bias.
        outF: encoder.embedDim,
        inF: encoder.embedDim,
      );
      _loadLinear(
        blk.vProj,
        weight: take('$p.self_attn.v_proj.weight'),
        bias: take('$p.self_attn.v_proj.bias'),
        outF: encoder.embedDim,
        inF: encoder.embedDim,
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

  // ------------ per-module helpers ------------

  static void _loadConv(
    dynamic conv, // Conv1d — dyn typed to avoid re-import
    Tensor weight,
    Tensor bias, {
    required int inC,
    required int outC,
    required int k,
  }) {
    _expectShape(weight, [outC, inC, k], 'conv.weight');
    _expectShape(bias, [outC], 'conv.bias');
    final w = _toF32(weight);
    final b = _toF32(bias);
    conv.loadFromPytorch(w, b);
  }

  static void _loadLinear(
    dynamic linear, {
    required Tensor weight,
    required Tensor? bias,
    required int outF,
    required int inF,
  }) {
    _expectShape(weight, [outF, inF], 'linear.weight');
    _assign2d(linear.weight, weight);
    if (bias != null) {
      _expectShape(bias, [outF], 'linear.bias');
      // Our Linear stores bias as [1, outF]; source is [outF].
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
    _assign1d(ln.gamma, weight);
    _assign1d(ln.beta, bias);
  }

  // ------------ tensor helpers ------------

  static Tensor _expectShape(Tensor t, List<int> expected, String name) {
    if (t.shape.length != expected.length || !_shapesEqual(t.shape, expected)) {
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

  /// Assign a rank-1 tensor `src` into a rank-1 destination `dst`
  /// (LayerNorm gamma/beta, positional embedding rank-2 too).
  static void _assign1d(Tensor dst, Tensor src) {
    if (dst.length != src.length) {
      throw ArgumentError(
        'whisper loader: _assign1d length mismatch — dst=${dst.shape}, '
        'src=${src.shape}',
      );
    }
    final vals = src.toList();
    final matched = Tensor.fromList(dst.shape, vals, device: dst.device);
    dst.assign(matched);
  }

  /// Assign a rank-2 weight [outF, inF] into a Linear weight of the
  /// same shape (device-adapted).
  static void _assign2d(Tensor dst, Tensor src) {
    if (dst.length != src.length) {
      throw ArgumentError(
        'whisper loader: _assign2d length mismatch — dst=${dst.shape}, '
        'src=${src.shape}',
      );
    }
    final vals = src.toList();
    final matched = Tensor.fromList(dst.shape, vals, device: dst.device);
    dst.assign(matched);
  }

  /// Assign a rank-1 bias [outF] into a Linear bias stored as [1, outF].
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
