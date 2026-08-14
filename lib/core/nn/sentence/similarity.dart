/// Similarity + semantic-search helpers for sentence embeddings.
///
/// Mirrors the utility functions on the Python
/// [`sentence_transformers.util`](
/// https://github.com/huggingface/sentence-transformers) module:
/// pairwise cosine similarity / dot score matrices, and a batched
/// semantic-search top-k over a corpus.
///
/// All inputs are expected to be 2D `[N, D]` embedding matrices. For
/// cosine similarity, callers should pass L2-normalized embeddings
/// (which is what [SentenceEncoder] emits by default); the "cosine"
/// helpers then reduce to a plain matrix product.
library;

import '../../tensor/tensor.dart';

/// A single search hit: the corpus row index and its raw score.
class SemanticSearchHit {
  final int corpusIndex;
  final double score;
  const SemanticSearchHit(this.corpusIndex, this.score);

  @override
  String toString() =>
      'SemanticSearchHit(idx=$corpusIndex, score=${score.toStringAsFixed(4)})';
}

class Similarity {
  Similarity._();

  /// Cosine similarity between two `[1, D]` embeddings. Assumes rows
  /// are already L2-normalized (the default from [SentenceEncoder]).
  /// Returns a plain double.
  static double cosSim(Tensor a, Tensor b) {
    _requireRow(a, 'cosSim.a');
    _requireRow(b, 'cosSim.b');
    if (a.shape[1] != b.shape[1]) {
      throw ArgumentError(
        'cosSim: dim mismatch ${a.shape[1]} vs ${b.shape[1]}',
      );
    }
    return Tensor.noGrad(() {
      final s = (a * b).sum();
      return s.toList()[0];
    });
  }

  /// Dot-product score between two `[1, D]` embeddings — same math as
  /// [cosSim] but named to make the assumption explicit when inputs
  /// are not normalized.
  static double dotScore(Tensor a, Tensor b) => cosSim(a, b);

  /// Pairwise similarity matrix `A [N, D] @ B [M, D]^T` → `[N, M]`.
  ///
  /// For cosine similarity, both `a` and `b` should already be row-
  /// wise L2-normalized. Runs under `noGrad` — meant for evaluation
  /// / retrieval, not training (training-time similarity should stay
  /// on the tape).
  static Tensor pairwise(Tensor a, Tensor b) {
    if (a.shape.length != 2 || b.shape.length != 2) {
      throw ArgumentError(
        'pairwise: expected 2D [N, D]; got ${a.shape} and ${b.shape}',
      );
    }
    if (a.shape[1] != b.shape[1]) {
      throw ArgumentError(
        'pairwise: dim mismatch ${a.shape[1]} vs ${b.shape[1]}',
      );
    }
    return Tensor.noGrad(() => a.matmul(b.transpose()));
  }

  /// For each query row, return the top-[topK] corpus indices by
  /// similarity (descending).
  ///
  /// `queries` — `[Q, D]`, `corpus` — `[N, D]`. Both should be L2-
  /// normalized for cosine ranking. Output is a `Q`-long list, each
  /// entry a length-`topK` list of [SemanticSearchHit].
  static List<List<SemanticSearchHit>> semanticSearch(
    Tensor queries,
    Tensor corpus, {
    int topK = 10,
  }) {
    if (queries.shape.length != 2 || corpus.shape.length != 2) {
      throw ArgumentError(
        'semanticSearch: expected 2D [Q, D] and [N, D]; got '
        '${queries.shape} and ${corpus.shape}',
      );
    }
    if (queries.shape[1] != corpus.shape[1]) {
      throw ArgumentError(
        'semanticSearch: dim mismatch ${queries.shape[1]} vs '
        '${corpus.shape[1]}',
      );
    }
    final q = queries.shape[0];
    final n = corpus.shape[0];
    final k = topK > n ? n : topK;

    final scores = pairwise(queries, corpus).toList();
    final out = <List<SemanticSearchHit>>[];
    for (int i = 0; i < q; i++) {
      // Materialize (idx, score) pairs for row `i`, sort desc, take k.
      final row = List<SemanticSearchHit>.generate(
        n,
        (j) => SemanticSearchHit(j, scores[i * n + j]),
      );
      row.sort((a, b) => b.score.compareTo(a.score));
      out.add(row.sublist(0, k));
    }
    return out;
  }

  static void _requireRow(Tensor t, String name) {
    if (t.shape.length != 2 || t.shape[0] != 1) {
      throw ArgumentError('$name: expected [1, D]; got ${t.shape}');
    }
  }
}
