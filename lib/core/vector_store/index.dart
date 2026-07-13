/// Dart port of a subset of FAISS' vector-index toolkit.
///
/// Mirrors the FAISS Python/C++ API: an [Index] holds a collection of
/// fixed-dimensionality `float32` vectors and supports [add], optional
/// [train], and k-NN [search]. All indexes are CPU-only for now and
/// operate on `Float32List`s for cache-friendly numerics.
library;

import 'dart:typed_data';

/// Distance metric used by an [Index].
///
/// * [Metric.l2] — squared Euclidean distance (the FAISS default).
/// * [Metric.innerProduct] — dot product. Cosine similarity is obtained
///   by pre-normalising both database and query vectors.
enum Metric { l2, innerProduct }

/// Result of a k-NN [Index.search] call.
///
/// `distances[i][j]` is the distance between query `i` and its `j`th
/// neighbour; `ids[i][j]` is that neighbour's integer id (assigned at
/// [Index.add] time — sequential from `0` unless the concrete index
/// supports custom ids).
///
/// Missing slots (fewer than `k` results available) are filled with
/// `double.infinity` for L2 / `-double.infinity` for inner product and
/// `-1` for ids.
class SearchResult {
  SearchResult(this.distances, this.ids);
  final List<Float32List> distances;
  final List<Int32List> ids;

  int get nq => distances.length;
  int get k => nq == 0 ? 0 : distances[0].length;
}

/// Convenience: convert a row-major `List<List<double>>` to a list of
/// `Float32List` rows (validating dimensionality).
List<Float32List> toFloat32Rows(List<List<double>> rows, int d) {
  final out = List<Float32List>.generate(rows.length, (_) => Float32List(d));
  for (var i = 0; i < rows.length; i++) {
    if (rows[i].length != d) {
      throw ArgumentError('Row $i has length ${rows[i].length}, expected $d.');
    }
    for (var j = 0; j < d; j++) {
      out[i][j] = rows[i][j];
    }
  }
  return out;
}

/// Base class for all FAISS-style indexes.
///
/// The subclass contract is:
///   1. Set [d], [metric] in the constructor.
///   2. Override [train] (no-op for indexes that don't need training).
///   3. Override [add] to encode+store vectors and bump [ntotal].
///   4. Override [search] to return the top-`k` neighbours per query.
abstract class Index {
  Index(this.d, this.metric);

  /// Vector dimensionality.
  final int d;

  /// Distance metric.
  final Metric metric;

  /// Number of vectors currently indexed.
  int ntotal = 0;

  /// Whether the index has been trained (or does not need training).
  bool isTrained = true;

  /// Train the index on a representative sample of vectors.
  /// Default is a no-op for indexes that don't need training.
  void train(List<Float32List> xs) {
    isTrained = true;
  }

  /// Add vectors to the index. Ids are assigned sequentially starting
  /// at [ntotal].
  void add(List<Float32List> xs);

  /// k-NN search. Returns `nq × k` distance/id matrices.
  SearchResult search(List<Float32List> queries, int k);

  /// Convenience wrapper for `List<List<double>>` inputs.
  void addRows(List<List<double>> xs) => add(toFloat32Rows(xs, d));

  /// Convenience wrapper for `List<List<double>>` inputs.
  SearchResult searchRows(List<List<double>> queries, int k) =>
      search(toFloat32Rows(queries, d), k);
}

/// Squared L2 distance between two `d`-length float vectors.
double l2sq(Float32List a, Float32List b) {
  assert(a.length == b.length);
  var s = 0.0;
  for (var i = 0; i < a.length; i++) {
    final diff = a[i] - b[i];
    s += diff * diff;
  }
  return s;
}

/// Dot product between two `d`-length float vectors.
double dot(Float32List a, Float32List b) {
  assert(a.length == b.length);
  var s = 0.0;
  for (var i = 0; i < a.length; i++) {
    s += a[i] * b[i];
  }
  return s;
}

/// Bounded-size max-heap of `(score, id)` pairs, used by every index
/// to accumulate the top-`k` neighbours in a single pass.
///
/// For [Metric.l2] the caller pushes squared distances and asks for the
/// [k] smallest by setting [maxIsWorst] = true. For [Metric.innerProduct]
/// pushes similarities and asks for the [k] largest by setting
/// [maxIsWorst] = false (i.e. worst-so-far is the *smallest* score).
///
/// Internally a binary heap is kept where the root is always the
/// *worst* accepted entry, so pushes can be rejected in O(1) once the
/// heap is full.
class TopK {
  TopK(this.k, {required this.maxIsWorst})
    : _score = Float32List(k),
      _id = Int32List(k);

  final int k;

  /// If true, larger score = worse (i.e. we want the smallest k scores,
  /// L2 mode). If false, smaller score = worse (inner-product mode).
  final bool maxIsWorst;

  final Float32List _score;
  final Int32List _id;
  int _size = 0;

  int get size => _size;

  /// The worst score currently accepted (root of the heap), or the
  /// sentinel [double.infinity] / `-double.infinity` if the heap is
  /// not yet full.
  double get worst {
    if (_size < k)
      return maxIsWorst ? double.infinity : double.negativeInfinity;
    return _score[0];
  }

  /// True iff `s` is worse than the current worst-accepted score.
  /// A push with this predicate can be skipped.
  bool worseThanWorst(double s) {
    if (_size < k) return false;
    return maxIsWorst ? s >= _score[0] : s <= _score[0];
  }

  /// Push a candidate. No-op if the heap is full and the new score is
  /// not strictly better than the current worst.
  void push(double score, int id) {
    if (_size < k) {
      _score[_size] = score;
      _id[_size] = id;
      _size++;
      _siftUp(_size - 1);
    } else if ((maxIsWorst && score < _score[0]) ||
        (!maxIsWorst && score > _score[0])) {
      _score[0] = score;
      _id[0] = id;
      _siftDown(0);
    }
  }

  bool _worse(int i, int j) =>
      maxIsWorst ? _score[i] > _score[j] : _score[i] < _score[j];

  void _siftUp(int i) {
    while (i > 0) {
      final p = (i - 1) >> 1;
      if (_worse(i, p)) {
        final ts = _score[i];
        _score[i] = _score[p];
        _score[p] = ts;
        final ti = _id[i];
        _id[i] = _id[p];
        _id[p] = ti;
        i = p;
      } else {
        break;
      }
    }
  }

  void _siftDown(int i) {
    final n = _size;
    while (true) {
      final l = 2 * i + 1;
      final r = 2 * i + 2;
      var w = i;
      if (l < n && _worse(l, w)) w = l;
      if (r < n && _worse(r, w)) w = r;
      if (w == i) break;
      final ts = _score[i];
      _score[i] = _score[w];
      _score[w] = ts;
      final ti = _id[i];
      _id[i] = _id[w];
      _id[w] = ti;
      i = w;
    }
  }

  /// Return the accepted candidates sorted best-first.
  ///
  /// The two output buffers are `k`-long; slots past [size] are filled
  /// with sentinel values ([double.infinity] / [double.negativeInfinity]
  /// for score, `-1` for id) so the caller can safely index them.
  ({Float32List scores, Int32List ids}) sorted() {
    final n = _size;
    final scores = Float32List(k);
    final ids = Int32List(k);
    for (var i = 0; i < k; i++) {
      scores[i] = maxIsWorst ? double.infinity : double.negativeInfinity;
      ids[i] = -1;
    }
    // Copy heap into arrays and sort.
    final idx = List<int>.generate(n, (i) => i);
    idx.sort((a, b) {
      final sa = _score[a];
      final sb = _score[b];
      if (maxIsWorst) {
        return sa.compareTo(sb); // ascending distance
      } else {
        return sb.compareTo(sa); // descending similarity
      }
    });
    for (var i = 0; i < n; i++) {
      scores[i] = _score[idx[i]];
      ids[i] = _id[idx[i]];
    }
    return (scores: scores, ids: ids);
  }
}
