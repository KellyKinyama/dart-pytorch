/// Stack of `TransformerDecoderBlock`s with an optional final
/// `LayerNorm`.
///
/// This is the "seq2seq decoder" (masked self-attention + cross-
/// attention + FFN, per block). It takes a decoder-side hidden state
/// `x` and an encoder-side `memory` and returns contextualized
/// decoder representations of the same shape as `x`.
///
/// No token/positional embeddings and no LM head — the caller is
/// expected to prepend those. See `EncoderDecoderTransformer` for a
/// full seq2seq wrapper.
library;

import '../tensor/tensor.dart';
import 'layer_norm.dart';
import 'module.dart';
import 'transformer_decoder_block.dart';

class TransformerDecoder extends Module {
  final int numLayers;
  final int embedDim;
  final int kvEmbedDim;
  final int numHeads;
  final int ffnDim;

  final List<TransformerDecoderBlock> blocks;
  final LayerNorm? finalNorm;

  TransformerDecoder(
    this.numLayers,
    this.embedDim,
    this.numHeads, {
    int? kvEmbedDim,
    int? ffnDim,
    double dropoutP = 0.0,
    bool finalNorm = true,
    Device device = Device.CPU,
    int seed = 0,
  }) : kvEmbedDim = kvEmbedDim ?? embedDim,
       ffnDim = ffnDim ?? embedDim * 4,
       blocks = List<TransformerDecoderBlock>.generate(
         numLayers,
         (i) => TransformerDecoderBlock(
           embedDim,
           numHeads,
           kvEmbedDim: kvEmbedDim,
           ffnDim: ffnDim,
           dropoutP: dropoutP,
           device: device,
           seed: seed + i * 10000,
         ),
       ),
       finalNorm = finalNorm ? LayerNorm(embedDim, device: device) : null;

  Tensor call(Tensor x, Tensor memory, {Tensor? selfMask}) {
    var h = x;
    for (final b in blocks) {
      h = b(h, memory, selfMask: selfMask);
    }
    if (finalNorm != null) {
      h = finalNorm!(h);
    }
    return h;
  }

  @override
  List<Tensor> parameters() => [
    for (final b in blocks) ...b.parameters(),
    if (finalNorm != null) ...finalNorm!.parameters(),
  ];

  @override
  List<Module> submodules() => [...blocks, if (finalNorm != null) finalNorm!];
}
