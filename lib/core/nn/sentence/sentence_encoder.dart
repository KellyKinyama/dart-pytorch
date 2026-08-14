/// Sentence-Transformers style embedder.
///
/// A [SentenceEncoder] wraps a [TextTransformer] backbone with a
/// pooling step (mean over tokens or the first "[CLS]"-style token)
/// and an optional row-wise L2 normalization, producing a single
/// `[1, embedDim]` sentence embedding per input.
///
/// Matches the recipe from
/// <https://github.com/huggingface/sentence-transformers>: backbone
/// output `[seqLen, D]` → pool → optional normalize → dense
/// embedding suitable for cosine-similarity search and reranking.
library;

import '../../tensor/tensor.dart';
import '../modalities/text_transformer.dart';
import '../module.dart';

enum PoolingMode { mean, cls }

/// Anything that maps `[seqLen]` token indices to a `[seqLen, embedDim]`
/// feature matrix and reports its output width. Both [TextTransformer]
/// and the BERT-style `BertModel` implement this so [SentenceEncoder]
/// and [CrossEncoder] can wrap either backbone.
abstract class TokenEncoder implements Module {
  int get embedDim;
  Tensor call(Tensor tokens);
}

/// Pool a token feature matrix `[seqLen, embedDim]` to a sentence
/// vector `[1, embedDim]`.
///
/// * [PoolingMode.mean] — arithmetic mean over the sequence axis.
///   Standard sentence-BERT default.
/// * [PoolingMode.cls] — the first token's features. Useful when the
///   backbone was trained with a "[CLS]" token at index 0.
Tensor poolTokens(Tensor tokenFeatures, PoolingMode mode) {
  if (tokenFeatures.shape.length != 2) {
    throw ArgumentError(
      'poolTokens: expected [seqLen, D]; got ${tokenFeatures.shape}',
    );
  }
  final s = tokenFeatures.shape[0];
  if (s == 0) {
    throw ArgumentError('poolTokens: cannot pool empty sequence');
  }
  switch (mode) {
    case PoolingMode.mean:
      // `[1, S] @ [S, D]` = `[1, D]`, differentiable through both operands.
      final ones = Tensor.fill([1, s], 1.0 / s, device: tokenFeatures.device);
      return ones.matmul(tokenFeatures);
    case PoolingMode.cls:
      return tokenFeatures.sliceRows(0, 1);
  }
}

/// Row-wise L2 normalize a `[1, D]` tensor.
///
/// Uses `y = x / sqrt(sum(x*x) + eps)`. The `[1, 1]` scalar reduction
/// broadcasts back over the `[1, D]` numerator via the built-in scalar
/// broadcast, so the whole op stays differentiable end-to-end.
Tensor l2NormalizeRow(Tensor x, {double eps = 1e-12}) {
  if (x.shape.length != 2 || x.shape[0] != 1) {
    throw ArgumentError('l2NormalizeRow: expected [1, D]; got ${x.shape}');
  }
  final sq = x * x;
  final sumSq = sq.sum();
  final norm = (sumSq + eps).pow(0.5);
  return x / norm;
}

class SentenceEncoder extends Module {
  final TokenEncoder backbone;
  final PoolingMode pooling;
  final bool normalize;
  final double eps;

  int get embedDim => backbone.embedDim;

  SentenceEncoder({
    required int vocabSize,
    required int maxSeqLen,
    required int embedDim,
    int numLayers = 4,
    int numHeads = 4,
    double dropoutP = 0.0,
    this.pooling = PoolingMode.mean,
    this.normalize = true,
    this.eps = 1e-12,
    Device device = Device.CPU,
    int seed = 0,
  }) : backbone = TextTransformer(
         vocabSize: vocabSize,
         maxSeqLen: maxSeqLen,
         embedDim: embedDim,
         numLayers: numLayers,
         numHeads: numHeads,
         dropoutP: dropoutP,
         device: device,
         seed: seed,
       );

  /// Wrap an already-constructed backbone. Useful for loading a
  /// pretrained transformer and swapping the pooling head.
  SentenceEncoder.wrap(
    this.backbone, {
    this.pooling = PoolingMode.mean,
    this.normalize = true,
    this.eps = 1e-12,
  });

  /// Encode a single sequence of token indices `[seqLen]` to a
  /// `[1, embedDim]` sentence embedding.
  Tensor call(Tensor tokens) {
    final features = backbone(tokens);
    final pooled = poolTokens(features, pooling);
    return normalize ? l2NormalizeRow(pooled, eps: eps) : pooled;
  }

  /// Encode a batch of token sequences and stack the results into a
  /// single `[N, embedDim]` tensor. Each sequence still runs through
  /// the backbone independently (the transformer here does not batch
  /// over a leading dimension), so this is a convenience wrapper —
  /// still differentiable when called under a live grad tape.
  Tensor encodeBatch(List<Tensor> tokenSequences) {
    if (tokenSequences.isEmpty) {
      throw ArgumentError('encodeBatch: input list is empty');
    }
    final rows = [for (final t in tokenSequences) this(t)];
    return TensorConcat.concat(rows, axis: 0);
  }

  @override
  List<Tensor> parameters() => backbone.parameters();

  @override
  List<Module> submodules() => [backbone];
}
