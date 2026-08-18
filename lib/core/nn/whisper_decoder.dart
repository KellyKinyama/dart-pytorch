/// Whisper (openai) text decoder architecture.
///
/// Layout mirrors `openai-whisper/whisper/model.py::TextDecoder`:
///
///   x = token_embedding(tokens) + positional_embedding[offset:offset+T]
///   for block in N:
///     x = x + attn(attn_ln(x), mask=causal)
///     x = x + cross_attn(cross_attn_ln(x), xa)
///     x = x + mlp(mlp_ln(x))
///   x = ln(x)
///   logits = x @ token_embedding.weight.T
///
/// Whisper tiny.en: d_model=384, n_head=6, n_layer=4, n_ctx=448,
/// vocab=51864.
///
/// Device-agnostic (CPU or GPU). Multi-head attention is expressed as
/// per-head `Linear` projections + `Tensor.scaledDotProductAttention`,
/// so every step stays on-device. Cross-attention K/V of the encoder
/// memory is cached per-block (as device tensors) via
/// [WhisperDecoderBlock.primeCrossAttn], so each decode step pays
/// O(T_text) rather than O(T_audio).
library;

import 'dart:typed_data';

import '../tensor/tensor.dart';
import 'embedding.dart';
import 'layer_norm.dart';
import 'linear.dart';
import 'module.dart';
import 'whisper.dart' show WhisperEncoderBlock;

class WhisperDecoderBlock extends Module {
  final int embedDim;
  final int numHeads;
  final int headDim;

  final LayerNorm attnLn;
  final List<Linear> qHeads;
  final List<Linear> kHeads;
  final List<Linear> vHeads;
  final Linear outProj;

  final LayerNorm crossAttnLn;
  final List<Linear> crossQHeads;
  final List<Linear> crossKHeads;
  final List<Linear> crossVHeads;
  final Linear crossOutProj;

  final LayerNorm mlpLn;
  final Linear mlp0;
  final Linear mlp2;

  // Cached cross-attn K/V of encoder memory, kept on the block's device.
  List<Tensor>? _crossK;
  List<Tensor>? _crossV;

  WhisperDecoderBlock(
    this.embedDim,
    this.numHeads, {
    Device device = Device.CPU,
  }) : headDim = embedDim ~/ numHeads,
       attnLn = LayerNorm(embedDim, device: device),
       qHeads = _mkHeads(embedDim, numHeads, bias: true, seed: 100, device: device),
       kHeads = _mkHeads(embedDim, numHeads, bias: false, seed: 200, device: device),
       vHeads = _mkHeads(embedDim, numHeads, bias: true, seed: 300, device: device),
       outProj = Linear(embedDim, embedDim, bias: true, device: device),
       crossAttnLn = LayerNorm(embedDim, device: device),
       crossQHeads = _mkHeads(embedDim, numHeads, bias: true, seed: 400, device: device),
       crossKHeads = _mkHeads(embedDim, numHeads, bias: false, seed: 500, device: device),
       crossVHeads = _mkHeads(embedDim, numHeads, bias: true, seed: 600, device: device),
       crossOutProj = Linear(embedDim, embedDim, bias: true, device: device),
       mlpLn = LayerNorm(embedDim, device: device),
       mlp0 = Linear(embedDim, embedDim * 4, bias: true, device: device),
       mlp2 = Linear(embedDim * 4, embedDim, bias: true, device: device);

  static List<Linear> _mkHeads(
    int embedDim,
    int numHeads, {
    required bool bias,
    required int seed,
    required Device device,
  }) {
    return List<Linear>.generate(
      numHeads,
      (h) => Linear(
        embedDim,
        embedDim ~/ numHeads,
        bias: bias,
        device: device,
        seed: seed + h,
      ),
    );
  }

  /// Pre-project encoder memory `xa: [B, T_audio, C]` (B == 1 in
  /// practice) into this block's cross-attention K and V. Must be
  /// called once per audio clip before [call].
  void primeCrossAttn(Tensor xa) {
    final b = xa.shape[0];
    final t = xa.shape[1];
    final c = xa.shape[2];
    final flat = xa.reshape([b * t, c]);
    _crossK = [for (final h in crossKHeads) h(flat)];
    _crossV = [for (final h in crossVHeads) h(flat)];
  }

  Tensor call(Tensor x) {
    if (_crossK == null) {
      throw StateError('WhisperDecoderBlock: call primeCrossAttn(xa) first.');
    }
    final normed = attnLn(x);
    final attn = _causalSelfAttn(normed);
    var h = x + attn;

    final crossNormed = crossAttnLn(h);
    final cross = _crossAttn(crossNormed);
    h = h + cross;

    final mlpNormed = mlpLn(h);
    var mlp = mlp0(mlpNormed);
    mlp = WhisperEncoderBlock.geluTanh(mlp);
    mlp = mlp2(mlp);
    return h + mlp;
  }

  Tensor _causalSelfAttn(Tensor x) {
    final b = x.shape[0];
    final t = x.shape[1];
    final c = x.shape[2];
    final xFlat = x.reshape([b * t, c]);
    final mask = _causalMask(t, x.device);
    final heads = <Tensor>[];
    for (int h = 0; h < numHeads; h++) {
      final qh = qHeads[h](xFlat);
      final kh = kHeads[h](xFlat);
      final vh = vHeads[h](xFlat);
      heads.add(qh.scaledDotProductAttention(kh, vh, mask: mask));
    }
    final concat = TensorConcat.concat(heads, axis: 1);
    final projected = outProj(concat);
    return projected.reshape([b, t, c]);
  }

  Tensor _crossAttn(Tensor x) {
    final b = x.shape[0];
    final t = x.shape[1];
    final c = x.shape[2];
    final xFlat = x.reshape([b * t, c]);
    final heads = <Tensor>[];
    for (int h = 0; h < numHeads; h++) {
      final qh = crossQHeads[h](xFlat);
      heads.add(qh.scaledDotProductAttention(_crossK![h], _crossV![h]));
    }
    final concat = TensorConcat.concat(heads, axis: 1);
    final projected = crossOutProj(concat);
    return projected.reshape([b, t, c]);
  }

  static Tensor _causalMask(int t, Device device) {
    // Upper triangle (strictly above diagonal) = -1e9; on/below = 0.
    final data = Float32List(t * t);
    const neg = -1e9;
    for (int i = 0; i < t; i++) {
      for (int j = i + 1; j < t; j++) {
        data[i * t + j] = neg;
      }
    }
    return Tensor.fromFloat32List([t, t], data, device: device);
  }

  @override
  List<Tensor> parameters() => [
    ...attnLn.parameters(),
    for (final l in qHeads) ...l.parameters(),
    for (final l in kHeads) ...l.parameters(),
    for (final l in vHeads) ...l.parameters(),
    ...outProj.parameters(),
    ...crossAttnLn.parameters(),
    for (final l in crossQHeads) ...l.parameters(),
    for (final l in crossKHeads) ...l.parameters(),
    for (final l in crossVHeads) ...l.parameters(),
    ...crossOutProj.parameters(),
    ...mlpLn.parameters(),
    ...mlp0.parameters(),
    ...mlp2.parameters(),
  ];
}

class WhisperDecoder extends Module {
  final int vocabSize;
  final int embedDim;
  final int numHeads;
  final int numLayers;
  final int nCtx;
  final Device device;

  final Embedding tokenEmbedding;
  final Tensor positionalEmbedding;
  final List<WhisperDecoderBlock> blocks;
  final LayerNorm ln;

  WhisperDecoder({
    required this.vocabSize,
    required this.embedDim,
    required this.numHeads,
    required this.numLayers,
    this.nCtx = 448,
    this.device = Device.CPU,
  }) : tokenEmbedding = Embedding(vocabSize, embedDim, device: device),
       positionalEmbedding = Tensor.fill(
         [nCtx, embedDim],
         0.0,
         device: device,
         requiresGrad: true,
       ),
       blocks = List.generate(
         numLayers,
         (_) => WhisperDecoderBlock(embedDim, numHeads, device: device),
       ),
       ln = LayerNorm(embedDim, device: device);

  void primeCrossAttn(Tensor encoderMemory) {
    for (final b in blocks) {
      b.primeCrossAttn(encoderMemory);
    }
  }

  /// Run the decoder on the *full* token prefix `[B, T]`. Returns
  /// hidden states `[B, T, embedDim]`.
  Tensor forward(Tensor tokens) {
    if (tokens.shape.length != 2) {
      throw ArgumentError(
        'WhisperDecoder.forward: expected [B, T]; got ${tokens.shape}',
      );
    }
    final b = tokens.shape[0];
    final t = tokens.shape[1];
    if (t > nCtx) {
      throw ArgumentError('WhisperDecoder: T=$t exceeds nCtx=$nCtx');
    }
    var x = tokenEmbedding(tokens); // [B, T, C]
    x = _addPositional(x, b, t);
    for (final block in blocks) {
      x = block(x);
    }
    return ln(x);
  }

  Tensor _addPositional(Tensor btc, int b, int t) {
    final c = btc.shape[2];
    // Positional slice [T, C] via embedding lookup (device-native).
    final posIdx = Tensor.fromList(
      [t],
      List<double>.generate(t, (i) => i.toDouble()),
      device: device,
    );
    final peSlice = positionalEmbedding.embedding(posIdx); // [T, C]
    if (b == 1) {
      final flat = btc.reshape([t, c]);
      return (flat + peSlice).reshape([1, t, c]);
    }
    // Fallback for B > 1: host-side broadcast add.
    final xData = btc.toFloat32List();
    final peData = peSlice.toFloat32List();
    final out = Float32List(b * t * c);
    for (int bi = 0; bi < b; bi++) {
      for (int ti = 0; ti < t; ti++) {
        for (int ci = 0; ci < c; ci++) {
          out[bi * t * c + ti * c + ci] =
              xData[bi * t * c + ti * c + ci] + peData[ti * c + ci];
        }
      }
    }
    return Tensor.fromFloat32List([b, t, c], out, device: btc.device);
  }

  /// Given decoder hidden `[B, T, C]`, project only the last position
  /// into vocab logits `[B, V]` using the tied token-embedding matrix.
  Tensor logitsLastToken(Tensor hidden) {
    final b = hidden.shape[0];
    final t = hidden.shape[1];
    final c = hidden.shape[2];
    if (b < 1) throw ArgumentError('empty batch');
    // Slice the last position as a fresh [B, C] tensor on-device via
    // host round-trip (tiny: B*C floats).
    final all = hidden.toFloat32List();
    final last = Float32List(b * c);
    for (int bi = 0; bi < b; bi++) {
      for (int ci = 0; ci < c; ci++) {
        last[bi * c + ci] = all[bi * t * c + (t - 1) * c + ci];
      }
    }
    final lastT = Tensor.fromFloat32List(
      [b, c],
      last,
      device: hidden.device,
    );
    // [B, C] @ [C, V] = [B, V] on-device.
    return lastT.matmul(tokenEmbedding.weight.transpose());
  }

  /// Greedy decode from `startTokens` up to `maxLen`, stopping when
  /// `eot` is emitted. Caller must have called [primeCrossAttn] first.
  List<int> greedyDecode({
    required List<int> startTokens,
    required int eot,
    int maxLen = 100,
    List<int> initialSuppress = const [],
  }) {
    final tokens = List<int>.of(startTokens);
    while (tokens.length < maxLen) {
      final tensor = Tensor.fromList(
        [1, tokens.length],
        List<double>.generate(tokens.length, (i) => tokens[i].toDouble()),
        device: device,
      );
      final hidden = forward(tensor);
      final logitsT = logitsLastToken(hidden);
      final logits = logitsT.toFloat32List();

      final isFirstSample = tokens.length == startTokens.length;
      double best = -double.infinity;
      int bestId = -1;
      for (int v = 0; v < vocabSize; v++) {
        if (isFirstSample && initialSuppress.contains(v)) continue;
        final lv = logits[v];
        if (lv > best) {
          best = lv;
          bestId = v;
        }
      }
      if (bestId == eot) break;
      tokens.add(bestId);
    }
    return tokens;
  }

  @override
  List<Tensor> parameters() => [
    ...tokenEmbedding.parameters(),
    positionalEmbedding,
    for (final b in blocks) ...b.parameters(),
    ...ln.parameters(),
  ];
}
