/// Whisper (openai) encoder architecture.
///
/// Follows `openai-whisper/whisper/model.py::AudioEncoder` exactly:
///
///   x = gelu(conv1(mel))          conv1: Conv1d(n_mels, d_model, k=3, p=1)
///   x = gelu(conv2(x))            conv2: Conv1d(d_model, d_model, k=3, s=2, p=1)
///   x = x.permute(B, T, C)
///   x = x + sinusoidal_pe(T, C)
///   for block in N: x = attn_ln → mha residual → mlp_ln → gelu-mlp residual
///   x = ln_post(x)                → [B, T = n_ctx = 1500, d_model]
///
/// Whisper tiny.en config: d_model=384, n_head=6, n_layer=4, n_ctx=1500.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../tensor/tensor.dart';
import 'conv1d.dart';
import 'layer_norm.dart';
import 'linear.dart';
import 'module.dart';

/// A single Whisper encoder block.
///
///   x + mha(attn_ln(x))
///   x + mlp(mlp_ln(x))       mlp = Linear → GELU → Linear
class WhisperEncoderBlock extends Module {
  final int embedDim;
  final int numHeads;
  final LayerNorm attnLn;
  final LayerNorm mlpLn;
  final Linear qProj;
  final Linear kProj; // no bias
  final Linear vProj;
  final Linear outProj;
  final Linear mlp0;
  final Linear mlp2;

  WhisperEncoderBlock(
    this.embedDim,
    this.numHeads, {
    Device device = Device.CPU,
  })  : attnLn = LayerNorm(embedDim, device: device),
        mlpLn = LayerNorm(embedDim, device: device),
        qProj = Linear(embedDim, embedDim, bias: true, device: device),
        kProj = Linear(embedDim, embedDim, bias: false, device: device),
        vProj = Linear(embedDim, embedDim, bias: true, device: device),
        outProj = Linear(embedDim, embedDim, bias: true, device: device),
        mlp0 = Linear(embedDim, embedDim * 4, bias: true, device: device),
        mlp2 = Linear(embedDim * 4, embedDim, bias: true, device: device);

  /// Forward on `[B, T, C]` input. Returns `[B, T, C]`.
  Tensor call(Tensor x) {
    // Self-attention residual.
    final normed = attnLn(x);
    final attn = _mha(normed);
    var h = x + attn;

    // MLP residual.
    final mlpNormed = mlpLn(h);
    var mlp = mlp0(mlpNormed);
    mlp = _geluErf(mlp);
    mlp = mlp2(mlp);
    return h + mlp;
  }

  /// Multi-head self-attention with pre-split q/k/v.
  ///
  /// Splits `[B, T, C]` linear projections into `[B, numHeads, T, headDim]`
  /// tensors, applies scaled dot-product attention per head, concatenates
  /// heads, then applies `outProj`.
  Tensor _mha(Tensor x) {
    final b = x.shape[0];
    final t = x.shape[1];
    final c = x.shape[2];
    final headDim = c ~/ numHeads;
    final scale = 1.0 / math.sqrt(headDim);

    // Project.  We use `Linear`s that expect `[.., C]` → `[.., C]`.
    // Reshape to `[B*T, C]` so `Linear` works on any device.
    final xFlat = x.reshape([b * t, c]);
    final q = qProj(xFlat); // [B*T, C]
    final k = kProj(xFlat);
    final v = vProj(xFlat);

    // Reshape to [B, T, H, headDim], then compute attention per head.
    final qHost = q.toFloat32List();
    final kHost = k.toFloat32List();
    final vHost = v.toFloat32List();
    final outHost = Float32List(b * t * c);

    for (int bi = 0; bi < b; bi++) {
      for (int h = 0; h < numHeads; h++) {
        // Extract per-head slices.
        final qh = Float32List(t * headDim);
        final kh = Float32List(t * headDim);
        final vh = Float32List(t * headDim);
        for (int ti = 0; ti < t; ti++) {
          final srcBase = bi * t * c + ti * c + h * headDim;
          final dstBase = ti * headDim;
          for (int d = 0; d < headDim; d++) {
            qh[dstBase + d] = qHost[srcBase + d];
            kh[dstBase + d] = kHost[srcBase + d];
            vh[dstBase + d] = vHost[srcBase + d];
          }
        }
        // Scores [T, T] = qh @ kh^T
        final scores = Float32List(t * t);
        for (int i = 0; i < t; i++) {
          for (int j = 0; j < t; j++) {
            double s = 0.0;
            for (int d = 0; d < headDim; d++) {
              s += qh[i * headDim + d] * kh[j * headDim + d];
            }
            scores[i * t + j] = s * scale;
          }
        }
        // Softmax over j per row i.
        for (int i = 0; i < t; i++) {
          double maxS = -double.infinity;
          for (int j = 0; j < t; j++) {
            final v = scores[i * t + j];
            if (v > maxS) maxS = v;
          }
          double sum = 0.0;
          for (int j = 0; j < t; j++) {
            final e = math.exp(scores[i * t + j] - maxS);
            scores[i * t + j] = e;
            sum += e;
          }
          for (int j = 0; j < t; j++) {
            scores[i * t + j] /= sum;
          }
        }
        // Attn output [T, headDim] = scores @ vh.
        for (int i = 0; i < t; i++) {
          for (int d = 0; d < headDim; d++) {
            double s = 0.0;
            for (int j = 0; j < t; j++) {
              s += scores[i * t + j] * vh[j * headDim + d];
            }
            outHost[bi * t * c + i * c + h * headDim + d] = s;
          }
        }
      }
    }

    final concat = Tensor.fromFloat32List(
      [b * t, c],
      outHost,
      device: x.device,
    );
    final projected = outProj(concat); // [B*T, C]
    return projected.reshape([b, t, c]);
  }

  static Tensor _geluErf(Tensor x) {
    // erf-based GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    final data = x.toFloat32List();
    final out = Float32List(data.length);
    const invSqrt2 = 0.7071067811865475;
    for (int i = 0; i < data.length; i++) {
      final v = data[i];
      out[i] = 0.5 * v * (1.0 + _erf(v * invSqrt2));
    }
    return Tensor.fromFloat32List(x.shape, out, device: x.device);
  }

  /// Abramowitz & Stegun approximation of erf, max error ~1.5e-7.
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
        (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x);
    return sign * y;
  }

  @override
  List<Tensor> parameters() => [
        ...attnLn.parameters(),
        ...mlpLn.parameters(),
        ...qProj.parameters(),
        ...kProj.parameters(),
        ...vProj.parameters(),
        ...outProj.parameters(),
        ...mlp0.parameters(),
        ...mlp2.parameters(),
      ];
}

class WhisperEncoder extends Module {
  final int nMels;
  final int embedDim;
  final int numHeads;
  final int numLayers;
  final int nCtx;

  final Conv1d conv1;
  final Conv1d conv2;
  final List<WhisperEncoderBlock> blocks;
  final LayerNorm lnPost;
  final Tensor positionalEmbedding; // [nCtx, embedDim], sinusoidal, non-trainable

  WhisperEncoder({
    required this.nMels,
    required this.embedDim,
    required this.numHeads,
    required this.numLayers,
    this.nCtx = 1500,
    Device device = Device.CPU,
  })  : conv1 = Conv1d(
          inChannels: nMels,
          outChannels: embedDim,
          kernelSize: 3,
          padding: 1,
          device: device,
        ),
        conv2 = Conv1d(
          inChannels: embedDim,
          outChannels: embedDim,
          kernelSize: 3,
          stride: 2,
          padding: 1,
          device: device,
        ),
        blocks = List.generate(
          numLayers,
          (_) => WhisperEncoderBlock(embedDim, numHeads, device: device),
        ),
        lnPost = LayerNorm(embedDim, device: device),
        positionalEmbedding = _sinusoids(nCtx, embedDim, device);

  /// Sinusoidal positional embedding used by Whisper: length `nCtx`,
  /// channels `embedDim`. Matches openai-whisper's `sinusoids()`.
  static Tensor _sinusoids(int length, int channels, Device device) {
    assert(channels % 2 == 0, 'sinusoids: channels must be even');
    const maxTimescale = 10000.0;
    final logTimescaleIncrement =
        math.log(maxTimescale) / (channels ~/ 2 - 1);
    final invTimescales = Float32List(channels ~/ 2);
    for (int i = 0; i < invTimescales.length; i++) {
      invTimescales[i] = math.exp(-logTimescaleIncrement * i);
    }
    final data = Float32List(length * channels);
    for (int t = 0; t < length; t++) {
      for (int i = 0; i < channels ~/ 2; i++) {
        final scaled = t * invTimescales[i];
        data[t * channels + i] = math.sin(scaled);
        data[t * channels + channels ~/ 2 + i] = math.cos(scaled);
      }
    }
    return Tensor.fromFloat32List([length, channels], data, device: device);
  }

  /// Run the encoder on a log-mel input `[B, nMels, T=3000]`.
  /// Returns encoder hidden states `[B, nCtx=1500, embedDim]`.
  Tensor call(Tensor mel) {
    if (mel.shape.length != 3) {
      throw ArgumentError(
        'WhisperEncoder: expected [B, nMels, T]; got ${mel.shape}',
      );
    }
    // Conv1 + GELU
    var x = conv1(mel);
    x = WhisperEncoderBlock._geluErf(x);
    // Conv2 (stride 2) + GELU
    x = conv2(x);
    x = WhisperEncoderBlock._geluErf(x);
    // x: [B, C, T/2]. Permute to [B, T/2, C] and add positional.
    x = _permuteBCTtoBTC(x);
    final t = x.shape[1];
    if (t != nCtx) {
      throw StateError(
        'WhisperEncoder: post-conv T=$t, expected nCtx=$nCtx',
      );
    }
    // Broadcast-add: positionalEmbedding is [nCtx, C], x is [B, T, C].
    x = _addPositionalPerRow(x);

    for (final block in blocks) {
      x = block(x);
    }
    // Final layernorm applied per token.
    return lnPost(x);
  }

  Tensor _permuteBCTtoBTC(Tensor bct) {
    final b = bct.shape[0];
    final c = bct.shape[1];
    final t = bct.shape[2];
    final data = bct.toFloat32List();
    final out = Float32List(b * t * c);
    for (int bi = 0; bi < b; bi++) {
      for (int ti = 0; ti < t; ti++) {
        for (int ci = 0; ci < c; ci++) {
          out[bi * t * c + ti * c + ci] = data[bi * c * t + ci * t + ti];
        }
      }
    }
    return Tensor.fromFloat32List([b, t, c], out, device: bct.device);
  }

  Tensor _addPositionalPerRow(Tensor btc) {
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

  @override
  List<Tensor> parameters() {
    return [
      ...conv1.parameters(),
      ...conv2.parameters(),
      for (final b in blocks) ...b.parameters(),
      ...lnPost.parameters(),
    ];
  }
}
