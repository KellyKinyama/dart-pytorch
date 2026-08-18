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
/// Hand-rolled attention (same style as `WhisperEncoder`). Cross-
/// attention K/V projections of the encoder memory are cached
/// per-block via [WhisperDecoderBlock.primeCrossAttn], so each
/// decode step only pays the O(T_text) cost, not O(T_audio).
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../tensor/tensor.dart';
import 'embedding.dart';
import 'layer_norm.dart';
import 'linear.dart';
import 'module.dart';
import 'whisper.dart' show WhisperEncoder;

class WhisperDecoderBlock extends Module {
  final int embedDim;
  final int numHeads;

  // Self-attention.
  final LayerNorm attnLn;
  final Linear qProj;
  final Linear kProj; // no bias
  final Linear vProj;
  final Linear outProj;

  // Cross-attention.
  final LayerNorm crossAttnLn;
  final Linear crossQProj;
  final Linear crossKProj; // no bias
  final Linear crossVProj;
  final Linear crossOutProj;

  // MLP.
  final LayerNorm mlpLn;
  final Linear mlp0;
  final Linear mlp2;

  // Cached cross-attn K/V of encoder memory (set by primeCrossAttn).
  Float32List? _crossK; // flattened [B, T_audio, C]
  Float32List? _crossV;
  int _crossB = 0;
  int _crossT = 0;

  WhisperDecoderBlock(
    this.embedDim,
    this.numHeads, {
    Device device = Device.CPU,
  })  : attnLn = LayerNorm(embedDim, device: device),
        qProj = Linear(embedDim, embedDim, bias: true, device: device),
        kProj = Linear(embedDim, embedDim, bias: false, device: device),
        vProj = Linear(embedDim, embedDim, bias: true, device: device),
        outProj = Linear(embedDim, embedDim, bias: true, device: device),
        crossAttnLn = LayerNorm(embedDim, device: device),
        crossQProj = Linear(embedDim, embedDim, bias: true, device: device),
        crossKProj = Linear(embedDim, embedDim, bias: false, device: device),
        crossVProj = Linear(embedDim, embedDim, bias: true, device: device),
        crossOutProj = Linear(embedDim, embedDim, bias: true, device: device),
        mlpLn = LayerNorm(embedDim, device: device),
        mlp0 = Linear(embedDim, embedDim * 4, bias: true, device: device),
        mlp2 = Linear(embedDim * 4, embedDim, bias: true, device: device);

  /// Pre-project encoder memory `xa: [B, T_audio, C]` into this
  /// block's cross-attention K and V. Must be called once per audio
  /// clip before [call]; subsequent decoding steps reuse the cache.
  void primeCrossAttn(Tensor xa) {
    final b = xa.shape[0];
    final t = xa.shape[1];
    final c = xa.shape[2];
    final flat = xa.reshape([b * t, c]);
    final k = crossKProj(flat);
    final v = crossVProj(flat);
    _crossK = k.toFloat32List();
    _crossV = v.toFloat32List();
    _crossB = b;
    _crossT = t;
  }

  /// Forward on `[B, T_text, C]` decoder hidden state. Applies causal
  /// self-attn, cross-attn over the primed encoder memory, then MLP.
  Tensor call(Tensor x) {
    if (_crossK == null) {
      throw StateError('WhisperDecoderBlock: call primeCrossAttn(xa) first.');
    }
    // Self-attention residual.
    final normed = attnLn(x);
    final attn = _causalSelfAttn(normed);
    var h = x + attn;

    // Cross-attention residual.
    final crossNormed = crossAttnLn(h);
    final cross = _crossAttn(crossNormed);
    h = h + cross;

    // MLP residual.
    final mlpNormed = mlpLn(h);
    var mlp = mlp0(mlpNormed);
    mlp = _geluErf(mlp);
    mlp = mlp2(mlp);
    return h + mlp;
  }

  Tensor _causalSelfAttn(Tensor x) {
    final b = x.shape[0];
    final t = x.shape[1];
    final c = x.shape[2];
    final headDim = c ~/ numHeads;
    final scale = 1.0 / math.sqrt(headDim);

    final xFlat = x.reshape([b * t, c]);
    final q = qProj(xFlat);
    final k = kProj(xFlat);
    final v = vProj(xFlat);

    final qHost = q.toFloat32List();
    final kHost = k.toFloat32List();
    final vHost = v.toFloat32List();
    final outHost = Float32List(b * t * c);

    for (int bi = 0; bi < b; bi++) {
      for (int h = 0; h < numHeads; h++) {
        for (int i = 0; i < t; i++) {
          final qBase = bi * t * c + i * c + h * headDim;
          double maxS = -double.infinity;
          final scores = Float32List(i + 1);
          for (int j = 0; j <= i; j++) {
            final kBase = bi * t * c + j * c + h * headDim;
            double s = 0.0;
            for (int d = 0; d < headDim; d++) {
              s += qHost[qBase + d] * kHost[kBase + d];
            }
            s *= scale;
            scores[j] = s;
            if (s > maxS) maxS = s;
          }
          double sum = 0.0;
          for (int j = 0; j <= i; j++) {
            final e = math.exp(scores[j] - maxS);
            scores[j] = e;
            sum += e;
          }
          final invSum = 1.0 / sum;
          for (int d = 0; d < headDim; d++) {
            double acc = 0.0;
            for (int j = 0; j <= i; j++) {
              final vBase = bi * t * c + j * c + h * headDim;
              acc += scores[j] * invSum * vHost[vBase + d];
            }
            outHost[bi * t * c + i * c + h * headDim + d] = acc;
          }
        }
      }
    }

    final concat =
        Tensor.fromFloat32List([b * t, c], outHost, device: x.device);
    final projected = outProj(concat);
    return projected.reshape([b, t, c]);
  }

  Tensor _crossAttn(Tensor x) {
    final b = x.shape[0];
    final t = x.shape[1];
    final c = x.shape[2];
    final headDim = c ~/ numHeads;
    final scale = 1.0 / math.sqrt(headDim);

    if (b != _crossB) {
      throw StateError(
        'cross-attn batch mismatch: dec B=$b vs enc B=$_crossB',
      );
    }
    final ta = _crossT;

    final xFlat = x.reshape([b * t, c]);
    final q = crossQProj(xFlat);
    final qHost = q.toFloat32List();
    final kHost = _crossK!;
    final vHost = _crossV!;
    final outHost = Float32List(b * t * c);

    for (int bi = 0; bi < b; bi++) {
      for (int h = 0; h < numHeads; h++) {
        for (int i = 0; i < t; i++) {
          final qBase = bi * t * c + i * c + h * headDim;
          double maxS = -double.infinity;
          final scores = Float32List(ta);
          for (int j = 0; j < ta; j++) {
            final kBase = bi * ta * c + j * c + h * headDim;
            double s = 0.0;
            for (int d = 0; d < headDim; d++) {
              s += qHost[qBase + d] * kHost[kBase + d];
            }
            s *= scale;
            scores[j] = s;
            if (s > maxS) maxS = s;
          }
          double sum = 0.0;
          for (int j = 0; j < ta; j++) {
            final e = math.exp(scores[j] - maxS);
            scores[j] = e;
            sum += e;
          }
          final invSum = 1.0 / sum;
          for (int d = 0; d < headDim; d++) {
            double acc = 0.0;
            for (int j = 0; j < ta; j++) {
              final vBase = bi * ta * c + j * c + h * headDim;
              acc += scores[j] * invSum * vHost[vBase + d];
            }
            outHost[bi * t * c + i * c + h * headDim + d] = acc;
          }
        }
      }
    }

    final concat =
        Tensor.fromFloat32List([b * t, c], outHost, device: x.device);
    final projected = crossOutProj(concat);
    return projected.reshape([b, t, c]);
  }

  static Tensor _geluErf(Tensor x) {
    final data = x.toFloat32List();
    final out = Float32List(data.length);
    const invSqrt2 = 0.7071067811865475;
    for (int i = 0; i < data.length; i++) {
      final v = data[i];
      out[i] = 0.5 * v * (1.0 + _erf(v * invSqrt2));
    }
    return Tensor.fromFloat32List(x.shape, out, device: x.device);
  }

  static double _erf(double x) {
    final sign = x < 0 ? -1.0 : 1.0;
    x = x.abs();
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;
    final t = 1.0 / (1.0 + p * x);
    final y = 1.0 -
        (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) *
            t *
            math.exp(-x * x);
    return sign * y;
  }

  @override
  List<Tensor> parameters() => [
        ...attnLn.parameters(),
        ...qProj.parameters(),
        ...kProj.parameters(),
        ...vProj.parameters(),
        ...outProj.parameters(),
        ...crossAttnLn.parameters(),
        ...crossQProj.parameters(),
        ...crossKProj.parameters(),
        ...crossVProj.parameters(),
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

  final Embedding tokenEmbedding;
  final Tensor positionalEmbedding; // [nCtx, embedDim], learned
  final List<WhisperDecoderBlock> blocks;
  final LayerNorm ln;

  WhisperDecoder({
    required this.vocabSize,
    required this.embedDim,
    required this.numHeads,
    required this.numLayers,
    this.nCtx = 448,
    Device device = Device.CPU,
  })  : tokenEmbedding = Embedding(vocabSize, embedDim, device: device),
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

  /// Prime all cross-attention blocks with the encoder memory. Must
  /// be called once per audio clip before [call].
  void primeCrossAttn(Tensor encoderMemory) {
    for (final b in blocks) {
      b.primeCrossAttn(encoderMemory);
    }
  }

  /// Run the decoder on the *full* token prefix `[B, T]`. Returns
  /// hidden states `[B, T, embedDim]` (call [logitsLastToken] to
  /// project only the final position into vocab space).
  Tensor forward(Tensor tokens) {
    if (tokens.shape.length != 2) {
      throw ArgumentError(
        'WhisperDecoder.forward: expected [B, T]; got ${tokens.shape}',
      );
    }
    final t = tokens.shape[1];
    if (t > nCtx) {
      throw ArgumentError('WhisperDecoder: T=$t exceeds nCtx=$nCtx');
    }
    var x = tokenEmbedding(tokens); // [B, T, C]
    x = _addPositional(x);
    for (final block in blocks) {
      x = block(x);
    }
    return ln(x);
  }

  Tensor _addPositional(Tensor btc) {
    final b = btc.shape[0];
    final t = btc.shape[1];
    final c = btc.shape[2];
    final xData = btc.toFloat32List();
    final peData = positionalEmbedding.toFloat32List();
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
    // Extract last row per batch: [B, C].
    final all = hidden.toFloat32List();
    final last = Float32List(b * c);
    for (int bi = 0; bi < b; bi++) {
      for (int ci = 0; ci < c; ci++) {
        last[bi * c + ci] = all[bi * t * c + (t - 1) * c + ci];
      }
    }
    // Multiply [B, C] @ E.T where E is [V, C].
    final embed = tokenEmbedding.weight.toFloat32List();
    final logits = Float32List(b * vocabSize);
    for (int bi = 0; bi < b; bi++) {
      for (int v = 0; v < vocabSize; v++) {
        double s = 0.0;
        for (int d = 0; d < c; d++) {
          s += last[bi * c + d] * embed[v * c + d];
        }
        logits[bi * vocabSize + v] = s;
      }
    }
    return Tensor.fromFloat32List(
      [b, vocabSize],
      logits,
      device: hidden.device,
    );
  }

  /// Convenience: greedy decode from `startTokens` up to `maxLen`,
  /// stopping when `eot` is emitted. Returns the token stream
  /// including the prefix. `suppress` are token ids never sampled
  /// (used e.g. for the `begin_suppress_tokens` list at t=0).
  ///
  /// `encoder` is only used for shape info; the caller must have
  /// already called [primeCrossAttn] with the actual audio memory.
  List<int> greedyDecode({
    required List<int> startTokens,
    required int eot,
    int maxLen = 100,
    List<int> initialSuppress = const [],
    // ignore: unused_element
    WhisperEncoder? encoder,
  }) {
    final tokens = List<int>.of(startTokens);
    final device = positionalEmbedding.device;
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
