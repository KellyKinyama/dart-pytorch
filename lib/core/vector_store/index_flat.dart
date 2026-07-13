/// Flat (brute-force) indexes — `IndexFlatL2` and `IndexFlatIP`.
///
/// These are the ground-truth reference implementations. Every other
/// index in the toolkit is measured against them via recall@k.
library;

import 'dart:typed_data';

import 'index.dart';

/// Brute-force exact k-NN over raw float32 vectors.
///
/// Storage is a single contiguous `Float32List` of length `ntotal * d`;
/// vector `i` occupies bytes `[i*d, (i+1)*d)`. This keeps the search
/// hot loop cache-friendly.
class IndexFlat extends Index {
  IndexFlat(super.d, super.metric);

  Float32List _storage = Float32List(0);
  int _capacity = 0; // in vectors

  @override
  void add(List<Float32List> xs) {
    if (xs.isEmpty) return;
    final n = xs.length;
    if (ntotal + n > _capacity) {
      var newCap = _capacity == 0 ? 1024 : _capacity;
      while (newCap < ntotal + n) {
        newCap *= 2;
      }
      final grown = Float32List(newCap * d);
      for (var i = 0; i < ntotal * d; i++) {
        grown[i] = _storage[i];
      }
      _storage = grown;
      _capacity = newCap;
    }
    for (var i = 0; i < n; i++) {
      if (xs[i].length != d) {
        throw ArgumentError(
          'vector $i has length ${xs[i].length}, expected $d',
        );
      }
      final base = (ntotal + i) * d;
      for (var j = 0; j < d; j++) {
        _storage[base + j] = xs[i][j];
      }
    }
    ntotal += n;
  }

  /// Direct access to the encoded vector at position `id`.
  /// Returns a view; do not mutate.
  Float32List reconstruct(int id) {
    if (id < 0 || id >= ntotal) {
      throw RangeError('id $id out of range [0, $ntotal)');
    }
    return Float32List.sublistView(_storage, id * d, (id + 1) * d);
  }

  @override
  SearchResult search(List<Float32List> queries, int k) {
    final nq = queries.length;
    final distances = List<Float32List>.generate(nq, (_) => Float32List(k));
    final ids = List<Int32List>.generate(nq, (_) => Int32List(k));

    for (var qi = 0; qi < nq; qi++) {
      final q = queries[qi];
      if (q.length != d) {
        throw ArgumentError('query $qi length ${q.length} != $d');
      }
      final heap = TopK(k, maxIsWorst: metric == Metric.l2);

      // Hot loop: iterate every stored vector.
      for (var vi = 0; vi < ntotal; vi++) {
        final base = vi * d;
        var s = 0.0;
        if (metric == Metric.l2) {
          for (var j = 0; j < d; j++) {
            final diff = q[j] - _storage[base + j];
            s += diff * diff;
          }
        } else {
          for (var j = 0; j < d; j++) {
            s += q[j] * _storage[base + j];
          }
        }
        heap.push(s, vi);
      }

      final sorted = heap.sorted();
      distances[qi] = sorted.scores;
      ids[qi] = sorted.ids;
    }
    return SearchResult(distances, ids);
  }
}

/// FAISS `IndexFlatL2` shortcut.
IndexFlat IndexFlatL2(int d) => IndexFlat(d, Metric.l2);

/// FAISS `IndexFlatIP` shortcut. For cosine similarity, L2-normalise
/// both the database and query vectors before adding/searching.
IndexFlat IndexFlatIP(int d) => IndexFlat(d, Metric.innerProduct);
