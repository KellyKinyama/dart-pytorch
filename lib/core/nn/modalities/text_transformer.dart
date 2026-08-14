/// Text Transformer — token-index encoder.
///
/// Input:  `[seqLen]` (1D tensor of token indices as floats — matches
///         the convention used everywhere else in this repo since we
///         don't yet have an int tensor dtype).
/// Output: `[seqLen, embedDim]` — per-position contextualized features.
///
/// Recipe: [`Embedding`] + [`LearnedPositionalEmbedding`] +
/// [`TransformerEncoder`] (no causal mask — this is a bidirectional
/// encoder, not a decoder). Use [`poolMean`] to collapse to a single
/// `[1, embedDim]` sentence embedding.
library;

import '../../tensor/tensor.dart';
import '../embedding.dart';
import '../module.dart';
import '../positional.dart';
import '../sentence/sentence_encoder.dart' show TokenEncoder;
import '../transformer_encoder.dart';
import 'audio_transformer.dart' show meanRows;

class TextTransformer extends Module implements TokenEncoder {
  final int vocabSize;
  final int maxSeqLen;
  final int embedDim;
  final int numLayers;
  final int numHeads;

  final Embedding tokenEmbed;
  final LearnedPositionalEmbedding posEmbed;
  final TransformerEncoder encoder;

  TextTransformer({
    required this.vocabSize,
    required this.maxSeqLen,
    required this.embedDim,
    this.numLayers = 4,
    this.numHeads = 4,
    double dropoutP = 0.0,
    Device device = Device.CPU,
    int seed = 0,
  }) : tokenEmbed = Embedding(
         vocabSize,
         embedDim,
         device: device,
         seed: seed + 1,
       ),
       posEmbed = LearnedPositionalEmbedding(
         maxSeqLen,
         embedDim,
         device: device,
         seed: seed + 2,
       ),
       encoder = TransformerEncoder(
         numLayers,
         embedDim,
         numHeads,
         dropoutP: dropoutP,
         device: device,
         seed: seed + 100,
       );

  /// Encode a variable-length token sequence.
  ///
  /// `tokens` — 1D `[seqLen]` of token indices as floats.
  Tensor call(Tensor tokens) {
    if (tokens.shape.length != 1) {
      throw ArgumentError(
        'TextTransformer: expected 1D [seqLen]; got ${tokens.shape}',
      );
    }
    final embedded = tokenEmbed(tokens); // [seqLen, embedDim]
    final withPos = posEmbed(embedded);
    return encoder(withPos);
  }

  Tensor poolMean(Tensor tokens) => meanRows(this(tokens));

  @override
  List<Tensor> parameters() => [
    ...tokenEmbed.parameters(),
    ...posEmbed.parameters(),
    ...encoder.parameters(),
  ];

  @override
  List<Module> submodules() => [tokenEmbed, posEmbed, encoder];
}
