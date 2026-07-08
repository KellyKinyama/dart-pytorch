/// Pre-LayerNorm Transformer decoder block (seq2seq style).
///
/// Layout (three pre-norm sub-layers, standard for encoder-decoder
/// Transformers):
///
///     h1 = x     + dropout(selfAttn(ln1(x), causalMask))
///     h2 = h1    + dropout(crossAttn(ln2(h1), memory))
///     out = h2   + dropout(ffn(ln3(h2)))
///
/// The self-attention is masked (causal) — the decoder can only look
/// at earlier positions. The cross-attention is unmasked — the whole
/// encoder output (memory) is visible at every decoder step.
///
/// Accepts either 2D (`[Sd, embedDim]` decoder, `[Se, kvEmbedDim]`
/// memory) or 3D (`[B, Sd, embedDim]` and `[B, Se, kvEmbedDim]`).
library;

import '../tensor/tensor.dart';
import 'dropout.dart';
import 'layer_norm.dart';
import 'linear.dart';
import 'module.dart';
import 'attention/multi_head_attention.dart';
import 'attention/multi_head_cross_attention.dart';

class TransformerDecoderBlock extends Module {
  final int embedDim;
  final int kvEmbedDim;
  final int numHeads;
  final int ffnDim;

  final LayerNorm ln1;
  final LayerNorm ln2;
  final LayerNorm ln3;
  final MultiHeadAttention selfAttn;
  final MultiHeadCrossAttention crossAttn;
  final Linear ffn1;
  final Linear ffn2;
  final Dropout dropout;

  TransformerDecoderBlock(
    this.embedDim,
    this.numHeads, {
    int? kvEmbedDim,
    int? ffnDim,
    double dropoutP = 0.0,
    Device device = Device.CPU,
    int seed = 0,
  }) : kvEmbedDim = kvEmbedDim ?? embedDim,
       ffnDim = ffnDim ?? embedDim * 4,
       ln1 = LayerNorm(embedDim, device: device),
       ln2 = LayerNorm(embedDim, device: device),
       ln3 = LayerNorm(embedDim, device: device),
       selfAttn = MultiHeadAttention(
         embedDim,
         numHeads,
         dropoutP: dropoutP,
         device: device,
         seed: seed,
       ),
       crossAttn = MultiHeadCrossAttention(
         embedDim,
         kvEmbedDim ?? embedDim,
         numHeads,
         dropoutP: dropoutP,
         device: device,
         seed: seed + 6000,
       ),
       ffn1 = Linear(
         embedDim,
         ffnDim ?? embedDim * 4,
         device: device,
         seed: seed + 4000,
       ),
       ffn2 = Linear(
         ffnDim ?? embedDim * 4,
         embedDim,
         device: device,
         seed: seed + 5000,
       ),
       dropout = Dropout(dropoutP);

  /// Forward. `x` is the decoder input (target-side hidden state).
  /// `memory` is the encoder output. `selfMask` is the causal mask
  /// used by the self-attention sub-layer.
  Tensor call(Tensor x, Tensor memory, {Tensor? selfMask}) {
    final h1 = x + dropout(selfAttn(ln1(x), mask: selfMask));
    final h2 = h1 + dropout(crossAttn(ln2(h1), memory));
    final ff = ffn2(ffn1(ln3(h2)).relu());
    return h2 + dropout(ff);
  }

  @override
  List<Tensor> parameters() => [
    ...ln1.parameters(),
    ...ln2.parameters(),
    ...ln3.parameters(),
    ...selfAttn.parameters(),
    ...crossAttn.parameters(),
    ...ffn1.parameters(),
    ...ffn2.parameters(),
  ];

  @override
  List<Module> submodules() => [
    ln1,
    ln2,
    ln3,
    selfAttn,
    crossAttn,
    ffn1,
    ffn2,
    dropout,
  ];
}
