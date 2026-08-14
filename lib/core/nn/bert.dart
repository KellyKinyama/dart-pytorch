/// HuggingFace BERT-style encoder. Post-LN, learned absolute
/// positions, WordPiece vocab, token-type embeddings folded into a
/// single bias since sentence-transformer inference always uses type
/// id 0. Matches the layout of `bert-base-*`, `sentence-transformers/
/// all-MiniLM-L6-v2`, etc.
///
/// Input is a 1D `[seqLen]` tensor of token indices (as floats, since
/// this repo doesn't have an int tensor). Output is `[seqLen,
/// embedDim]` — feed it into a [SentenceEncoder] (or a [CrossEncoder])
/// with mean pooling + L2 norm to reproduce the sentence-BERT recipe.
library;

import '../tensor/tensor.dart';
import 'attention/multi_head_attention.dart';
import 'embedding.dart';
import 'layer_norm.dart';
import 'linear.dart';
import 'module.dart';
import 'sentence/sentence_encoder.dart' show TokenEncoder;

class BertConfig {
  final int vocabSize;
  final int maxPositionEmbeddings;
  final int embedDim;
  final int numLayers;
  final int numHeads;
  final int intermediateSize;
  final double layerNormEps;
  final int typeVocabSize;
  final Device device;
  final int seed;

  const BertConfig({
    required this.vocabSize,
    required this.maxPositionEmbeddings,
    required this.embedDim,
    required this.numLayers,
    required this.numHeads,
    required this.intermediateSize,
    this.layerNormEps = 1e-12,
    this.typeVocabSize = 2,
    this.device = Device.CPU,
    this.seed = 0,
  });
}

class BertEmbeddings extends Module {
  final Embedding wordEmbeddings;
  final Embedding positionEmbeddings;
  // BERT also has a `token_type_embeddings` [type_vocab, D] table.
  // Sentence-transformer inference always uses type id 0, so we
  // collapse the whole table to its row-0 vector, add it as a
  // constant bias, and skip the extra lookup entirely.
  final Tensor tokenTypeBias;
  final LayerNorm layerNorm;

  final int embedDim;
  final int maxLen;

  BertEmbeddings({
    required int vocabSize,
    required int maxPositionEmbeddings,
    required this.embedDim,
    double layerNormEps = 1e-12,
    Device device = Device.CPU,
    int seed = 0,
  }) : wordEmbeddings = Embedding(
         vocabSize,
         embedDim,
         device: device,
         seed: seed + 1,
       ),
       positionEmbeddings = Embedding(
         maxPositionEmbeddings,
         embedDim,
         device: device,
         seed: seed + 2,
       ),
       tokenTypeBias = Tensor.fill(
         [1, embedDim],
         0.0,
         requiresGrad: true,
         device: device,
       ),
       layerNorm = LayerNorm(embedDim, eps: layerNormEps, device: device),
       maxLen = maxPositionEmbeddings;

  Tensor call(Tensor tokens) {
    if (tokens.shape.length != 1) {
      throw ArgumentError(
        'BertEmbeddings: expected 1D [seqLen]; got ${tokens.shape}',
      );
    }
    final s = tokens.shape[0];
    if (s == 0) {
      throw ArgumentError('BertEmbeddings: empty sequence');
    }
    if (s > maxLen) {
      throw ArgumentError(
        'BertEmbeddings: seqLen $s exceeds maxPositionEmbeddings $maxLen',
      );
    }
    final positions = Tensor.fromList(
      [s],
      List<double>.generate(s, (i) => i.toDouble()),
      device: tokens.device,
    );
    var h = wordEmbeddings(tokens) + positionEmbeddings(positions);
    h = h + tokenTypeBias;
    return layerNorm(h);
  }

  @override
  List<Tensor> parameters() => [
    ...wordEmbeddings.parameters(),
    ...positionEmbeddings.parameters(),
    tokenTypeBias,
    ...layerNorm.parameters(),
  ];

  @override
  List<Module> submodules() => [wordEmbeddings, positionEmbeddings, layerNorm];
}

class BertLayer extends Module {
  final MultiHeadAttention attention;
  final LayerNorm attentionOutputLn;
  final Linear intermediate;
  final Linear output;
  final LayerNorm outputLn;

  BertLayer({
    required int embedDim,
    required int numHeads,
    required int intermediateSize,
    double layerNormEps = 1e-12,
    Device device = Device.CPU,
    int seed = 0,
  }) : attention = MultiHeadAttention(
         embedDim,
         numHeads,
         bias: true,
         device: device,
         seed: seed,
       ),
       attentionOutputLn = LayerNorm(
         embedDim,
         eps: layerNormEps,
         device: device,
       ),
       intermediate = Linear(
         embedDim,
         intermediateSize,
         bias: true,
         device: device,
         seed: seed + 4000,
       ),
       output = Linear(
         intermediateSize,
         embedDim,
         bias: true,
         device: device,
         seed: seed + 5000,
       ),
       outputLn = LayerNorm(embedDim, eps: layerNormEps, device: device);

  Tensor call(Tensor x) {
    final attnOut = attention(x);
    final h = attentionOutputLn(x + attnOut);
    // Tanh-approx GELU matches exact GELU to < 2e-4 abs on [-6, 6];
    // fine for sentence-embedding retrieval.
    final ffnOut = output(_geluTanh(intermediate(h)));
    return outputLn(h + ffnOut);
  }

  static Tensor _geluTanh(Tensor x) {
    const c = 0.7978845608028654; // sqrt(2 / pi)
    final inner = (x + x.pow(3.0) * 0.044715) * c;
    return x * (inner.tanh() + 1.0) * 0.5;
  }

  @override
  List<Tensor> parameters() => [
    ...attention.parameters(),
    ...attentionOutputLn.parameters(),
    ...intermediate.parameters(),
    ...output.parameters(),
    ...outputLn.parameters(),
  ];

  @override
  List<Module> submodules() => [
    attention,
    attentionOutputLn,
    intermediate,
    output,
    outputLn,
  ];
}

class BertModel extends Module implements TokenEncoder {
  final BertConfig config;
  final BertEmbeddings embeddings;
  final List<BertLayer> layers;

  @override
  int get embedDim => config.embedDim;

  BertModel(this.config)
    : embeddings = BertEmbeddings(
        vocabSize: config.vocabSize,
        maxPositionEmbeddings: config.maxPositionEmbeddings,
        embedDim: config.embedDim,
        layerNormEps: config.layerNormEps,
        device: config.device,
        seed: config.seed,
      ),
      layers = List<BertLayer>.generate(
        config.numLayers,
        (i) => BertLayer(
          embedDim: config.embedDim,
          numHeads: config.numHeads,
          intermediateSize: config.intermediateSize,
          layerNormEps: config.layerNormEps,
          device: config.device,
          seed: config.seed + 100 + i * 10000,
        ),
      );

  @override
  Tensor call(Tensor tokens) {
    var h = embeddings(tokens);
    for (final layer in layers) {
      h = layer(h);
    }
    return h;
  }

  @override
  List<Tensor> parameters() => [
    ...embeddings.parameters(),
    for (final l in layers) ...l.parameters(),
  ];

  @override
  List<Module> submodules() => [embeddings, ...layers];
}
