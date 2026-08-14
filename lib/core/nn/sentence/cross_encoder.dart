/// Cross-encoder reranker.
///
/// Mirrors the `CrossEncoder` class from
/// <https://github.com/huggingface/sentence-transformers>: it takes a
/// single joint tokenization of a (query, passage) pair and produces a
/// scalar relevance score. Slower than a bi-encoder [SentenceEncoder]
/// because the transformer sees both sides at once, but much more
/// accurate for reranking a shortlist retrieved by a bi-encoder.
///
/// Input tokens: the caller is responsible for concatenating the
/// query and passage token ids into one 1D `[seqLen]` sequence,
/// usually as `[CLS] q0 q1 ... [SEP] d0 d1 ... [SEP]`.
///
/// Output: a `[1, numLabels]` logits tensor (defaults to `numLabels =
/// 1` for a single relevance score). For binary classification use
/// `score()` which passes the logit through a sigmoid.
library;

import '../../tensor/tensor.dart';
import '../linear.dart';
import '../modalities/text_transformer.dart';
import '../module.dart';
import 'sentence_encoder.dart';class CrossEncoder extends Module {
  final TokenEncoder backbone;
  final Linear head;
  final PoolingMode pooling;
  final int numLabels;

  int get embedDim => backbone.embedDim;

  CrossEncoder({
    required int vocabSize,
    required int maxSeqLen,
    required int embedDim,
    int numLayers = 4,
    int numHeads = 4,
    double dropoutP = 0.0,
    this.pooling = PoolingMode.cls,
    this.numLabels = 1,
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
       ),
       head = Linear(
         embedDim,
         numLabels,
         bias: true,
         device: device,
         seed: seed + 424242,
       );

  /// Wrap a pretrained backbone. Adds a fresh classification head.
  CrossEncoder.wrap(
    this.backbone, {
    this.pooling = PoolingMode.cls,
    this.numLabels = 1,
    Device device = Device.CPU,
    int seed = 424242,
  }) : head = Linear(
         backbone.embedDim,
         numLabels,
         bias: true,
         device: device,
         seed: seed,
       );

  /// Forward pass on a single tokenized pair. Returns `[1, numLabels]`
  /// raw logits.
  Tensor call(Tensor pairTokens) {
    final features = backbone(pairTokens);
    final pooled = poolTokens(features, pooling);
    return head(pooled);
  }

  /// Convenience: run under `noGrad`, return the first-label logit
  /// squashed through a sigmoid. Useful when `numLabels == 1`.
  double score(Tensor pairTokens) {
    return Tensor.noGrad(() {
      final logits = call(pairTokens);
      final probs = logits.sigmoid();
      return probs.toList()[0];
    });
  }

  @override
  List<Tensor> parameters() => [...backbone.parameters(), ...head.parameters()];

  @override
  List<Module> submodules() => [backbone, head];
}
