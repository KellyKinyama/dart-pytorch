/// Full encoder-decoder Transformer for seq2seq tasks
/// (translation, summarization, etc.).
///
/// Composes two token pipelines:
///
///   * **Encoder** (bidirectional, no causal mask): source token
///     embedding + sinusoidal PE + [TransformerEncoder] stack. Yields
///     a "memory" tensor of shape `[Se, embedDim]` (or `[B, Se, D]`)
///     that summarises the source sequence.
///   * **Decoder** (causal self-attention + cross-attention over
///     memory): target token embedding + sinusoidal PE +
///     [TransformerDecoder] stack + `Linear` LM head. Yields logits
///     `[St, targetVocabSize]` (or `[B, St, V]`).
///
/// Both sides use their own embeddings and vocab; this is the classic
/// "Attention Is All You Need" architecture.
///
/// Same 1D/2D-token convention as `TransformerLM` / `GPT`:
/// `srcTokens` and `tgtTokens` are `[N]` for a single sequence or
/// `[B, N]` batched — mixed rank between src and tgt is rejected.
library;

import '../tensor/tensor.dart';
import 'embedding.dart';
import 'linear.dart';
import 'masks.dart';
import 'module.dart';
import 'positional.dart';
import 'transformer_decoder.dart';
import 'transformer_encoder.dart';

class EncoderDecoderTransformer extends Module {
  final int sourceVocabSize;
  final int targetVocabSize;
  final int embedDim;
  final int numLayers;
  final int numHeads;
  final int maxSourceLen;
  final int maxTargetLen;

  final Embedding sourceEmb;
  final Embedding targetEmb;
  final SinusoidalPositionalEncoding srcPosEnc;
  final SinusoidalPositionalEncoding tgtPosEnc;
  final TransformerEncoder encoder;
  final TransformerDecoder decoder;
  final Linear head;

  EncoderDecoderTransformer({
    required this.sourceVocabSize,
    required this.targetVocabSize,
    required this.embedDim,
    required this.numLayers,
    required this.numHeads,
    required this.maxSourceLen,
    required this.maxTargetLen,
    int? ffnDim,
    double dropoutP = 0.0,
    Device device = Device.CPU,
    int seed = 0,
  })  : sourceEmb =
            Embedding(sourceVocabSize, embedDim, device: device, seed: seed),
        targetEmb = Embedding(targetVocabSize, embedDim,
            device: device, seed: seed + 500000),
        srcPosEnc = SinusoidalPositionalEncoding(embedDim),
        tgtPosEnc = SinusoidalPositionalEncoding(embedDim),
        encoder = TransformerEncoder(
          numLayers,
          embedDim,
          numHeads,
          ffnDim: ffnDim,
          dropoutP: dropoutP,
          device: device,
          seed: seed + 100000,
        ),
        decoder = TransformerDecoder(
          numLayers,
          embedDim,
          numHeads,
          kvEmbedDim: embedDim,
          ffnDim: ffnDim,
          dropoutP: dropoutP,
          device: device,
          seed: seed + 700000,
        ),
        head = Linear(embedDim, targetVocabSize,
            device: device, seed: seed + 900000);

  /// Encoder-only pass returning "memory" for the decoder.
  Tensor encode(Tensor srcTokens) {
    if (srcTokens.shape.length != 1 && srcTokens.shape.length != 2) {
      throw ArgumentError(
        'EncoderDecoderTransformer.encode: srcTokens must be 1D or 2D; '
        'got ${srcTokens.shape}',
      );
    }
    final n = srcTokens.shape.last;
    if (n > maxSourceLen) {
      throw ArgumentError(
        'EncoderDecoderTransformer.encode: seqLen $n exceeds '
        'maxSourceLen $maxSourceLen',
      );
    }
    var x = sourceEmb(srcTokens);
    x = srcPosEnc(x);
    return encoder(x); // no causal mask on the encoder side
  }

  /// Decoder-side forward given a precomputed memory (encoder output).
  /// Applies causal masking to the decoder self-attention.
  Tensor decode(Tensor tgtTokens, Tensor memory) {
    if (tgtTokens.shape.length != 1 && tgtTokens.shape.length != 2) {
      throw ArgumentError(
        'EncoderDecoderTransformer.decode: tgtTokens must be 1D or 2D; '
        'got ${tgtTokens.shape}',
      );
    }
    final n = tgtTokens.shape.last;
    if (n > maxTargetLen) {
      throw ArgumentError(
        'EncoderDecoderTransformer.decode: seqLen $n exceeds '
        'maxTargetLen $maxTargetLen',
      );
    }
    var y = targetEmb(tgtTokens);
    y = tgtPosEnc(y);
    final mask = causalMask(n, device: y.device);
    y = decoder(y, memory, selfMask: mask);
    return head(y);
  }

  /// Full forward: encode source, decode target against memory,
  /// return logits `[St, targetVocabSize]` or `[B, St, V]`.
  Tensor call(Tensor srcTokens, Tensor tgtTokens) {
    if (srcTokens.shape.length != tgtTokens.shape.length) {
      throw ArgumentError(
        'EncoderDecoderTransformer: srcTokens and tgtTokens must share '
        'rank; got src=${srcTokens.shape} tgt=${tgtTokens.shape}',
      );
    }
    if (srcTokens.shape.length == 2 &&
        srcTokens.shape[0] != tgtTokens.shape[0]) {
      throw ArgumentError(
        'EncoderDecoderTransformer: batched src/tgt must share batch '
        'size; got src=${srcTokens.shape} tgt=${tgtTokens.shape}',
      );
    }
    final memory = encode(srcTokens);
    return decode(tgtTokens, memory);
  }

  @override
  List<Tensor> parameters() => [
        ...sourceEmb.parameters(),
        ...targetEmb.parameters(),
        ...srcPosEnc.parameters(),
        ...tgtPosEnc.parameters(),
        ...encoder.parameters(),
        ...decoder.parameters(),
        ...head.parameters(),
      ];

  @override
  List<Module> submodules() => [
        sourceEmb,
        targetEmb,
        srcPosEnc,
        tgtPosEnc,
        encoder,
        decoder,
        head,
      ];
}
