/// Pre-LayerNorm Transformer encoder block.
///
/// Layout (per Vaswani et al. with pre-norm; more stable to train):
///
///     h = x + dropout(mha(ln1(x)))
///     y = h + dropout(ffn(ln2(h)))
///
/// Where `ffn(z) = W2(relu(W1(z)))` — the classic 2-layer feed-forward
/// with an inner width of `ffnDim` (defaults to `4 * embedDim`).
library;

import '../tensor/tensor.dart';
import 'dropout.dart';
import 'kv_cache.dart';
import 'layer_norm.dart';
import 'linear.dart';
import 'module.dart';
import 'multi_head_attention.dart';

class TransformerBlock extends Module {
  final int embedDim;
  final int numHeads;
  final int ffnDim;

  final LayerNorm ln1;
  final LayerNorm ln2;
  final MultiHeadAttention mha;
  final Linear ffn1;
  final Linear ffn2;
  final Dropout dropout;

  TransformerBlock(
    this.embedDim,
    this.numHeads, {
    int? ffnDim,
    double dropoutP = 0.0,
    Device device = Device.CPU,
    int seed = 0,
  }) : ffnDim = ffnDim ?? embedDim * 4,
       ln1 = LayerNorm(embedDim),
       ln2 = LayerNorm(embedDim),
       mha = MultiHeadAttention(
         embedDim,
         numHeads,
         dropoutP: dropoutP,
         device: device,
         seed: seed,
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

  Tensor call(Tensor x, {Tensor? mask, MHACache? cache}) {
    final h = x + dropout(mha(ln1(x), mask: mask, cache: cache));
    final ff = ffn2(ffn1(ln2(h)).relu());
    return h + dropout(ff);
  }

  @override
  List<Tensor> parameters() => [
    ...ln1.parameters(),
    ...ln2.parameters(),
    ...mha.parameters(),
    ...ffn1.parameters(),
    ...ffn2.parameters(),
  ];

  @override
  List<Module> submodules() => [ln1, ln2, mha, ffn1, ffn2, dropout];
}
