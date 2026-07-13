/// Cell-probe IVF index with a flat encoding (`IndexIVFFlat`).
///
/// Splits the feature space into `nlist` k-means cells; each database
/// vector is assigned to its closest centroid and stored (uncompressed)
/// in that cell's inverted list. At query time we visit `nprobe` of the
/// closest cells and brute-force within them.
library;

import 'dart:typed_data';

import 'index.dart';
import 'index_flat.dart';
import 'index_io.dart';
import 'kmeans.dart';

class IndexIVFFlat extends Index {
  IndexIVFFlat({
    required int d,
    required this.nlist,
    Metric metric = Metric.l2,
    this.nprobe = 1,
    this.kmeansIters = 20,
    this.seed = 1234,
  }) : quantizer = IndexFlat(d, metric),
       super(d, metric) {
    isTrained = false;
    _invLists = List<List<int>>.generate(nlist, (_) => <int>[]);
  }

  /// Coarse quantizer holding the `nlist` k-means centroids.
  final IndexFlat quantizer;

  /// Number of cells (k-means clusters).
  final int nlist;

  /// Number of cells to probe at search time.
  int nprobe;

  final int kmeansIters;
  final int seed;

  /// Encoded database vectors, contiguous `[ntotal, d]`.
  Float32List _storage = Float32List(0);
  int _capacity = 0;

  /// `_invLists[cell]` = ordered ids of vectors assigned to that cell.
  late List<List<int>> _invLists;

  @override
  void train(List<Float32List> xs) {
    if (xs.length < nlist) {
      throw ArgumentError(
        'IndexIVFFlat.train: need >= nlist=$nlist vectors, got ${xs.length}',
      );
    }
    final km = Kmeans(d: d, k: nlist, niter: kmeansIters, seed: seed);
    final res = km.train(xs);
    // Load centroids into the flat coarse quantizer.
    quantizer.add(res.centroids);
    isTrained = true;
  }

  @override
  void add(List<Float32List> xs) {
    if (!isTrained) {
      throw StateError('IndexIVFFlat.add called before train()');
    }
    if (xs.isEmpty) return;
    final n = xs.length;
    // Grow storage.
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
    // Assign each vector to its closest centroid via the coarse quantizer.
    final assign = quantizer.search(xs, 1);
    for (var i = 0; i < n; i++) {
      final id = ntotal + i;
      final base = id * d;
      final row = xs[i];
      for (var j = 0; j < d; j++) {
        _storage[base + j] = row[j];
      }
      _invLists[assign.ids[i][0]].add(id);
    }
    ntotal += n;
  }

  @override
  SearchResult search(List<Float32List> queries, int k) {
    if (!isTrained) throw StateError('IndexIVFFlat.search before train()');
    final nq = queries.length;
    // 1. Find nprobe closest cells for each query.
    final coarse = quantizer.search(queries, nprobe);
    final distances = List<Float32List>.generate(nq, (_) => Float32List(k));
    final ids = List<Int32List>.generate(nq, (_) => Int32List(k));

    for (var qi = 0; qi < nq; qi++) {
      final q = queries[qi];
      final heap = TopK(k, maxIsWorst: metric == Metric.l2);
      final probes = coarse.ids[qi];
      for (var p = 0; p < probes.length; p++) {
        final cell = probes[p];
        if (cell < 0) continue;
        final list = _invLists[cell];
        for (var li = 0; li < list.length; li++) {
          final vid = list[li];
          final base = vid * d;
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
          heap.push(s, vid);
        }
      }
      final sorted = heap.sorted();
      distances[qi] = sorted.scores;
      ids[qi] = sorted.ids;
    }
    return SearchResult(distances, ids);
  }

  /// For debugging: number of vectors in each cell.
  List<int> listSizes() => _invLists.map((l) => l.length).toList();

  // --- persistence --------------------------------------------------------

  void writeTo(IoWriter w) {
    w.writeU32(d);
    w.writeU32(metricToU32(metric));
    w.writeU32(ntotal);
    w.writeU8(isTrained ? 1 : 0);
    w.writeU32(nlist);
    w.writeU32(nprobe);
    w.writeU32(kmeansIters);
    w.writeU32(seed);
    // Embed the coarse quantizer as a self-contained child blob.
    writeChild(w, quantizer);
    // Raw storage (may be empty when untrained).
    if (ntotal > 0) {
      w.writeF32List(Float32List.sublistView(_storage, 0, ntotal * d));
    }
    // Inverted lists.
    for (var c = 0; c < nlist; c++) {
      final list = _invLists[c];
      w.writeU32(list.length);
      for (final id in list) {
        w.writeI32(id);
      }
    }
  }

  static IndexIVFFlat readFrom(IoReader r) {
    final d = r.readU32();
    final metric = metricFromU32(r.readU32());
    final ntotal = r.readU32();
    final isTrained = r.readU8() != 0;
    final nlist = r.readU32();
    final nprobe = r.readU32();
    final kmeansIters = r.readU32();
    final seed = r.readU32();
    final quantizerAny = readChild(r);
    if (quantizerAny is! IndexFlat) {
      throw FormatException(
        'IndexIVFFlat: embedded quantizer is ${quantizerAny.runtimeType}, '
        'expected IndexFlat',
      );
    }
    final idx = IndexIVFFlat(
      d: d,
      nlist: nlist,
      metric: metric,
      nprobe: nprobe,
      kmeansIters: kmeansIters,
      seed: seed,
    );
    // The constructor built a fresh quantizer; swap in the loaded one.
    idx._replaceQuantizer(quantizerAny);
    if (ntotal > 0) {
      idx._storage = r.readF32List(ntotal * d);
      idx._capacity = ntotal;
    }
    for (var c = 0; c < nlist; c++) {
      final len = r.readU32();
      final list = idx._invLists[c];
      for (var i = 0; i < len; i++) {
        list.add(r.readI32());
      }
    }
    idx.ntotal = ntotal;
    idx.isTrained = isTrained;
    return idx;
  }

  void _replaceQuantizer(IndexFlat q) {
    // Copy the loaded quantizer's centroids into our own so that the
    // `final` field remains untouched (constructor built an empty one).
    if (q.ntotal > 0) {
      final rows = List<Float32List>.generate(q.ntotal, (i) => q.reconstruct(i));
      quantizer.add(rows);
    }
  }
}
