/// Training losses for sentence-transformers style bi-encoders.
///
/// Mirrors the three most-used losses from
/// <https://github.com/huggingface/sentence-transformers/tree/master/sentence_transformers/losses>:
///
/// * [multipleNegativesRankingLoss] — in-batch contrastive. Every
///   other positive in the batch is treated as a negative for the
///   current anchor. Fits well with a [SentenceEncoder] because it
///   needs only aligned (anchor, positive) pairs — no explicit
///   negative mining.
/// * [cosineSimilarityLoss] — MSE between the pair's cosine
///   similarity and a supervised label in `[-1, 1]`. Used for STS-
///   style regression.
/// * [tripletLoss] — margin ranking on cosine distance:
///   `max(0, d(a, p) - d(a, n) + margin)`.
///
/// All three assume inputs are `[N, D]` row-wise L2-normalized
/// embeddings (i.e. produced by a [SentenceEncoder] with
/// `normalize: true`, which is the default).
library;

import '../../tensor/tensor.dart';

class SentenceLosses {
  SentenceLosses._();

  /// Multiple Negatives Ranking Loss.
  ///
  /// `anchors [N, D]` and `positives [N, D]` are row-aligned pairs.
  /// The scaled similarity matrix `anchors @ positives^T * scale` is
  /// treated as `N`-way classification logits: the correct class for
  /// row `i` is column `i` (its true positive). Cross-entropy over
  /// this drives each anchor towards its positive and away from every
  /// other batch element.
  ///
  /// `scale` (aka temperature) defaults to 20.0 = 1 / 0.05, matching
  /// the sentence-transformers default. Higher values sharpen the
  /// softmax; too high and gradients vanish, too low and negatives
  /// stop pushing.
  ///
  /// Returns a `[1, 1]` scalar loss ready for `.backward()`.
  static Tensor multipleNegativesRankingLoss(
    Tensor anchors,
    Tensor positives, {
    double scale = 20.0,
  }) {
    _requireMatrix(anchors, 'multipleNegativesRankingLoss.anchors');
    _requireMatrix(positives, 'multipleNegativesRankingLoss.positives');
    if (anchors.shape[0] != positives.shape[0] ||
        anchors.shape[1] != positives.shape[1]) {
      throw ArgumentError(
        'multipleNegativesRankingLoss: shape mismatch ${anchors.shape} '
        'vs ${positives.shape}',
      );
    }
    final n = anchors.shape[0];
    final scores = anchors.matmul(positives.transpose()) * scale;
    final targets = Tensor.fromList(
      [n],
      List<double>.generate(n, (i) => i.toDouble()),
      device: anchors.device,
    );
    return scores.crossEntropy(targets);
  }

  /// Cosine Similarity Loss.
  ///
  /// `a [N, D]` and `b [N, D]` — row-aligned pairs, assumed L2-
  /// normalized (so per-row dot product = cosine similarity).
  /// `targets [N]` — labels in `[-1, 1]`.
  ///
  /// Loss = mean over the batch of `(cosSim(a_i, b_i) - targets_i)^2`.
  static Tensor cosineSimilarityLoss(Tensor a, Tensor b, Tensor targets) {
    _requireMatrix(a, 'cosineSimilarityLoss.a');
    _requireMatrix(b, 'cosineSimilarityLoss.b');
    if (a.shape[0] != b.shape[0] || a.shape[1] != b.shape[1]) {
      throw ArgumentError(
        'cosineSimilarityLoss: shape mismatch ${a.shape} vs ${b.shape}',
      );
    }
    final n = a.shape[0];
    if (targets.shape.length != 1 || targets.shape[0] != n) {
      throw ArgumentError(
        'cosineSimilarityLoss: targets must be [$n]; got ${targets.shape}',
      );
    }
    final d = a.shape[1];
    // Per-row dot product: elementwise then reduce over the last axis
    // via `(a*b) @ ones([D, 1])` → `[N, 1]`.
    final rowOnes = Tensor.fill([d, 1], 1.0, device: a.device);
    final sims = (a * b).matmul(rowOnes); // [N, 1]
    final tgt = targets.reshape([n, 1]);
    final diff = sims - tgt;
    return (diff * diff).mean();
  }

  /// Triplet Loss on cosine distance.
  ///
  /// `anchors [N, D]`, `positives [N, D]`, `negatives [N, D]` —
  /// row-aligned triplets of L2-normalized embeddings. Uses cosine
  /// distance `d = 1 - cosSim`, so
  ///
  ///   loss_i = max(0, d(a_i, p_i) - d(a_i, n_i) + margin)
  ///          = max(0, cosSim(a_i, n_i) - cosSim(a_i, p_i) + margin)
  ///
  /// Returns the mean over the batch.
  static Tensor tripletLoss(
    Tensor anchors,
    Tensor positives,
    Tensor negatives, {
    double margin = 0.5,
  }) {
    _requireMatrix(anchors, 'tripletLoss.anchors');
    _requireMatrix(positives, 'tripletLoss.positives');
    _requireMatrix(negatives, 'tripletLoss.negatives');
    if (anchors.shape[0] != positives.shape[0] ||
        anchors.shape[0] != negatives.shape[0] ||
        anchors.shape[1] != positives.shape[1] ||
        anchors.shape[1] != negatives.shape[1]) {
      throw ArgumentError(
        'tripletLoss: shape mismatch a=${anchors.shape} '
        'p=${positives.shape} n=${negatives.shape}',
      );
    }
    final d = anchors.shape[1];
    final rowOnes = Tensor.fill([d, 1], 1.0, device: anchors.device);
    final posSim = (anchors * positives).matmul(rowOnes); // [N, 1]
    final negSim = (anchors * negatives).matmul(rowOnes); // [N, 1]
    final gap = (negSim - posSim) + margin;
    return gap.relu().mean();
  }

  static void _requireMatrix(Tensor t, String name) {
    if (t.shape.length != 2) {
      throw ArgumentError('$name: expected 2D [N, D]; got ${t.shape}');
    }
  }
}
