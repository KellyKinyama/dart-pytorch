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
///
/// Device-agnostic: constructing with `device: Device.GPU` keeps every
/// matmul / softmax / layernorm / add on GPU. Multi-head attention is
/// expressed as per-head Q/K/V `Linear` projections + `Tensor.scaledDot`.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../tensor/tensor.dart';
import 'conv1d.dart';
import 'layer_norm.dart';
import 'linear.dart';
import 'module.dart';

class WhisperEncoderBlock extends Module {
  final int embedDim;
  final int numHeads;
  final int headDim;

  final LayerNorm attnLn;
  final LayerNorm mlpLn;
  final List<Linear> qHeads; // per-head, [C -> headDim], bias=true
  final List<Linear> kHeads; // per-head, [C -> headDim], bias=false
  final List<Linear> vHeads; // per-head, [C -> headDim], bias=true
  final Linear outProj; // [C -> C], bias=true
  final Linear mlp0;
  final Linear mlp2;

  WhisperEncoderBlock(
    this.embedDim,
    this.numHeads, {
    Device device = Device.CPU,
  }) : headDim = embedDim ~/ numHeads,
       attnLn = LayerNorm(embedDim, device: device),
       mlpLn = LayerNorm(embedDim, device: device),
       qHeads = List<Linear>.generate(
         numHeads,
         (h) => Linear(
           embedDim,
           embedDim ~/ numHeads,
           bias: true,
           device: device,
           seed: h,
         ),
       ),
       kHeads = List<Linear>.generate(
         numHeads,
         (h) => Linear(
           embedDim,
           embedDim ~/ numHeads,
           bias: false,
           device: device,
           seed: 1000 + h,
         ),
       ),
       vHeads = List<Linear>.generate(
         numHeads,
         (h) => Linear(
           embedDim,
           embedDim ~/ numHeads,
           bias: true,
           device: device,
           seed: 2000 + h,
         ),
       ),
       outProj = Linear(embedDim, embedDim, bias: true, device: device),
       mlp0 = Linear(embedDim, embedDim * 4, bias: true, device: device),
       mlp2 = Linear(embedDim * 4, embedDim, bias: true, device: device);

  Tensor call(Tensor x) {
    final normed = attnLn(x);
    final attn = _mha(normed);
    var h = x + attn;

    final mlpNormed = mlpLn(h);
    var mlp = mlp0(mlpNormed);
    mlp = geluTanh(mlp);
    mlp = mlp2(mlp);
    return h + mlp;
  }

  Tensor _mha(Tensor x) {
    final b = x.shape[0];
    final t = x.shape[1];
    final c = x.shape[2];
    final xFlat = x.reshape([b * t, c]);
    final heads = <Tensor>[];
    for (int h = 0; h < numHeads; h++) {
      final qh = qHeads[h](xFlat);
      final kh = kHeads[h](xFlat);
      final vh = vHeads[h](xFlat);
      heads.add(qh.scaledDotProductAttention(kh, vh));
    }
    final concat = TensorConcat.concat(heads, axis: 1);
    final projected = outProj(concat);
    return projected.reshape([b, t, c]);
  }

  /// Tanh-approximate GELU (bit-identical to the GPT-2 / BERT flavour
  /// used across the rest of dart_pytorch). Max abs diff from Whisper's
  /// erf-GELU is < 2e-4.
  static Tensor geluTanh(Tensor x) {
    const c = 0.7978845608028654; // sqrt(2 / pi)
    final inner = (x + x.pow(3) * 0.044715) * c;
    return x * (inner.tanh() + 1.0) * 0.5;
  }

  @override
  List<Tensor> parameters() => [
    ...attnLn.parameters(),
    ...mlpLn.parameters(),
    for (final l in qHeads) ...l.parameters(),
    for (final l in kHeads) ...l.parameters(),
    for (final l in vHeads) ...l.parameters(),
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
  final Device device;

  final Conv1d conv1;
  final Conv1d conv2;
  final List<WhisperEncoderBlock> blocks;
  final LayerNorm lnPost;
  final Tensor positionalEmbedding;

  WhisperEncoder({
    required this.nMels,
    required this.embedDim,
    required this.numHeads,
    required this.numLayers,
    this.nCtx = 1500,
    this.device = Device.CPU,
  }) : conv1 = Conv1d(
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

  static Tensor _sinusoids(int length, int channels, Device device) {
    assert(channels % 2 == 0, 'sinusoids: channels must be even');
    const maxTimescale = 10000.0;
    final logTimescaleIncrement = math.log(maxTimescale) / (channels ~/ 2 - 1);
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
    var x = conv1(mel);
    x = WhisperEncoderBlock.geluTanh(x);
    x = conv2(x);
    x = WhisperEncoderBlock.geluTanh(x);
    x = _permuteBCTtoBTC(x);
    final t = x.shape[1];
    if (t != nCtx) {
      throw StateError('WhisperEncoder: post-conv T=$t, expected nCtx=$nCtx');
    }
    x = _addPositional(x);
    for (final block in blocks) {
      x = block(x);
    }
    return lnPost(x);
  }

  /// [B, C, T'] -> [B, T', C]. For B==1 this is a plain 2D transpose,
  /// which stays on-device; larger B falls back to a host scatter.
  Tensor _permuteBCTtoBTC(Tensor bct) {
    final b = bct.shape[0];
    final c = bct.shape[1];
    final t = bct.shape[2];
    if (b == 1) {
      final ct = bct.reshape([c, t]);
      final tc = ct.transpose();
      return tc.reshape([1, t, c]);
    }
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

  Tensor _addPositional(Tensor btc) {
    final b = btc.shape[0];
    final t = btc.shape[1];
    final c = btc.shape[2];
    if (b == 1) {
      final flat = btc.reshape([t, c]);
      return (flat + positionalEmbedding).reshape([1, t, c]);
    }
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
