/// Attention-Free Transformer (AFT) block + language-model stack.
///
/// Same pre-LN skeleton as the vanilla `TransformerBlock`, but the
/// self-attention sublayer is [AFTAttention] rather than
/// [MultiHeadAttention]. No cross-attention (decoder-only).
///
/// `AFTLanguageModel` glues this together with token/positional
/// embeddings and a linear LM head — mirrors `TransformerLM`'s API so
/// the two are drop-in comparable in training loops and demos.
///
/// CPU-only for now (transitively — [AFTAttention] is CPU-only).
library;

import '../tensor/tensor.dart';
import 'attention/aft_attention.dart';
import 'dropout.dart';
import 'embedding.dart';
import 'layer_norm.dart';
import 'linear.dart';
import 'module.dart';
import 'positional.dart';

class AFTBlock extends Module {
  final int embedDim;
  final int maxSeqLen;
  final int ffnDim;

  final LayerNorm ln1;
  final LayerNorm ln2;
  final AFTAttention attn;
  final Linear ffn1;
  final Linear ffn2;
  final Dropout dropout;

  AFTBlock(
    this.embedDim, {
    required this.maxSeqLen,
    bool masked = false,
    int? ffnDim,
    double dropoutP = 0.0,
    Device device = Device.CPU,
    int seed = 0,
  }) : ffnDim = ffnDim ?? embedDim * 4,
       ln1 = LayerNorm(embedDim),
       ln2 = LayerNorm(embedDim),
       attn = AFTAttention(
         embedDim,
         maxSeqLen: maxSeqLen,
         masked: masked,
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

  Tensor call(Tensor x) {
    final h = x + dropout(attn(ln1(x)));
    final ff = ffn2(ffn1(ln2(h)).relu());
    return h + dropout(ff);
  }

  @override
  List<Tensor> parameters() => [
    ...ln1.parameters(),
    ...ln2.parameters(),
    ...attn.parameters(),
    ...ffn1.parameters(),
    ...ffn2.parameters(),
  ];

  @override
  List<Module> submodules() => [ln1, ln2, attn, ffn1, ffn2, dropout];
}

/// Decoder-only language model built from AFT blocks. Analogous to
/// [TransformerLM] but with attention-free self-attention.
class AFTLanguageModel extends Module {
  final int vocabSize;
  final int embedDim;
  final int numLayers;
  final int maxLen;

  final Embedding tokenEmb;
  final SinusoidalPositionalEncoding posEnc;
  final List<AFTBlock> blocks;
  final LayerNorm finalLn;
  final Linear head;

  AFTLanguageModel({
    required this.vocabSize,
    required this.embedDim,
    required this.numLayers,
    required this.maxLen,
    int? ffnDim,
    double dropoutP = 0.0,
    Device device = Device.CPU,
    int seed = 0,
  }) : tokenEmb = Embedding(vocabSize, embedDim, device: device, seed: seed),
       posEnc = SinusoidalPositionalEncoding(embedDim),
       blocks = List<AFTBlock>.generate(
         numLayers,
         (i) => AFTBlock(
           embedDim,
           maxSeqLen: maxLen,
           masked: true,
           ffnDim: ffnDim,
           dropoutP: dropoutP,
           device: device,
           seed: seed + 100000 + i * 10000,
         ),
       ),
       finalLn = LayerNorm(embedDim),
       head = Linear(embedDim, vocabSize, device: device, seed: seed + 900000);

  /// Forward pass. `tokens` is 1D `[seqLen]` — returns logits
  /// `[seqLen, vocabSize]`. No batched 2D path yet (AFT attention
  /// module is 2D-only in this implementation).
  Tensor call(Tensor tokens) {
    if (tokens.shape.length != 1) {
      throw ArgumentError(
        'AFTLanguageModel: tokens must be 1D [seqLen]; got ${tokens.shape}',
      );
    }
    final n = tokens.shape[0];
    if (n > maxLen) {
      throw ArgumentError('AFTLanguageModel: seqLen $n exceeds maxLen $maxLen');
    }
    var x = tokenEmb(tokens);
    x = posEnc(x);
    for (final b in blocks) {
      x = b(x);
    }
    x = finalLn(x);
    return head(x);
  }

  @override
  List<Tensor> parameters() => [
    ...tokenEmb.parameters(),
    ...posEnc.parameters(),
    for (final b in blocks) ...b.parameters(),
    ...finalLn.parameters(),
    ...head.parameters(),
  ];

  @override
  List<Module> submodules() => [tokenEmb, posEnc, ...blocks, finalLn, head];
}
