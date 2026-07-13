/// `IndexIVFPQ` — a.k.a. IVFADC. FAISS' workhorse for billion-scale
/// search.
///
/// Pipeline:
///   1. k-means over the raw vectors → `nlist` coarse centroids.
///   2. Each database vector's **residual** (`x - centroid`) is encoded
///      with a Product Quantizer of `m` subquantizers.
///   3. At query time, probe `nprobe` cells; for each probed cell,
///      compute the query's residual w.r.t. that cell's centroid,
///      build a PQ distance LUT, and scan the inverted list via ADC.
library;

import 'dart:typed_data';

import 'index.dart';
import 'index_flat.dart';
import 'index_pq.dart';
import 'kmeans.dart';

class IndexIVFPQ extends Index {
  IndexIVFPQ({
    required int d,
    required this.nlist,
    required this.m,
    int nbits = 8,
    this.nprobe = 1,
    this.kmeansIters = 20,
    this.pqKmeansIters = 25,
    this.seed = 1234,
    Metric metric = Metric.l2,
  }) : quantizer = IndexFlat(d, metric),
       pq = ProductQuantizer(
         d: d,
         m: m,
         nbits: nbits,
         kmeansIters: pqKmeansIters,
         seed: seed + 42,
       ),
       super(d, metric) {
    isTrained = false;
    _invListsIds = List<List<int>>.generate(nlist, (_) => <int>[]);
    _invListsCodes = List<List<int>>.generate(nlist, (_) => <int>[]);
  }

  final IndexFlat quantizer;
  final ProductQuantizer pq;
  final int nlist;
  final int m;
  int nprobe;
  final int kmeansIters;
  final int pqKmeansIters;
  final int seed;

  /// `_invListsIds[cell][slot]` = database id of that stored vector.
  late final List<List<int>> _invListsIds;

  /// `_invListsCodes[cell]` = flat list of PQ code bytes for the cell,
  /// length `m * len(_invListsIds[cell])`.
  late final List<List<int>> _invListsCodes;

  @override
  void train(List<Float32List> xs) {
    if (xs.length < nlist) {
      throw ArgumentError(
        'IndexIVFPQ.train: need >= nlist=$nlist vectors, got ${xs.length}',
      );
    }
    // 1. Coarse quantizer over full vectors.
    final km = Kmeans(d: d, k: nlist, niter: kmeansIters, seed: seed);
    final coarse = km.train(xs);
    quantizer.add(coarse.centroids);

    // 2. Compute residuals of training vectors and train the PQ on them.
    final resid = List<Float32List>.generate(xs.length, (_) => Float32List(d));
    for (var i = 0; i < xs.length; i++) {
      final cen = coarse.centroids[coarse.assignments[i]];
      final src = xs[i];
      final dst = resid[i];
      for (var j = 0; j < d; j++) {
        dst[j] = src[j] - cen[j];
      }
    }
    pq.train(resid);
    isTrained = true;
  }

  @override
  void add(List<Float32List> xs) {
    if (!isTrained) throw StateError('IndexIVFPQ.add before train()');
    if (xs.isEmpty) return;
    // Assign each vector to its coarse cell.
    final assign = quantizer.search(xs, 1);
    // Compute residuals and encode.
    final resid = List<Float32List>.generate(xs.length, (_) => Float32List(d));
    for (var i = 0; i < xs.length; i++) {
      final cell = assign.ids[i][0];
      final cen = quantizer.reconstruct(cell);
      final src = xs[i];
      final dst = resid[i];
      for (var j = 0; j < d; j++) {
        dst[j] = src[j] - cen[j];
      }
    }
    final codes = pq.encode(resid);
    for (var i = 0; i < xs.length; i++) {
      final id = ntotal + i;
      final cell = assign.ids[i][0];
      _invListsIds[cell].add(id);
      final off = i * m;
      final dst = _invListsCodes[cell];
      for (var sub = 0; sub < m; sub++) {
        dst.add(codes[off + sub]);
      }
    }
    ntotal += xs.length;
  }

  @override
  SearchResult search(List<Float32List> queries, int k) {
    if (!isTrained) throw StateError('IndexIVFPQ.search before train()');
    final nq = queries.length;
    final coarse = quantizer.search(queries, nprobe);
    final distances = List<Float32List>.generate(nq, (_) => Float32List(k));
    final ids = List<Int32List>.generate(nq, (_) => Int32List(k));

    final ksub = pq.ksub;

    for (var qi = 0; qi < nq; qi++) {
      final q = queries[qi];
      final heap = TopK(k, maxIsWorst: metric == Metric.l2);
      final probes = coarse.ids[qi];

      for (var p = 0; p < probes.length; p++) {
        final cell = probes[p];
        if (cell < 0) continue;
        // Query residual for this cell.
        final cen = quantizer.reconstruct(cell);
        final qr = Float32List(d);
        for (var j = 0; j < d; j++) {
          qr[j] = q[j] - cen[j];
        }
        final lut = metric == Metric.l2 ? pq.buildL2LUT(qr) : pq.buildIPLUT(qr);
        final cellIds = _invListsIds[cell];
        final cellCodes = _invListsCodes[cell];
        for (var li = 0; li < cellIds.length; li++) {
          final base = li * m;
          var s = 0.0;
          for (var sub = 0; sub < m; sub++) {
            s += lut[sub * ksub + cellCodes[base + sub]];
          }
          heap.push(s, cellIds[li]);
        }
      }

      final sorted = heap.sorted();
      distances[qi] = sorted.scores;
      ids[qi] = sorted.ids;
    }
    return SearchResult(distances, ids);
  }

  List<int> listSizes() => _invListsIds.map((l) => l.length).toList();
}
