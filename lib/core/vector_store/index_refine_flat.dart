/// `IndexRefineFlat` — pair a fast approximate index with an exact
/// `IndexFlat` for post-verification.
///
/// At search time:
///   1. Ask the base (approximate) index for `k * kFactor` candidates.
///   2. Re-rank those candidates against the stored uncompressed
///      vectors using exact L2 or inner-product distance.
///   3. Return the top-`k` after re-ranking.
///
/// This is FAISS' standard recall-boost trick: pay for a larger
/// candidate list at the fast index, then a small exact pass to fix
/// ordering. Typically gets an approximate index like `IVFPQ` from
/// ~50 % recall to > 95 % at a fraction of a flat scan's cost.
///
/// The refine index stores the raw fp32 vectors — so memory cost is
/// `4 * d * ntotal` bytes on top of whatever the base index uses.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'index.dart';
import 'index_flat.dart';
import 'index_io.dart';

class IndexRefineFlat extends Index {
  IndexRefineFlat(this.base, {this.kFactor = 4})
    : refine = IndexFlat(base.d, base.metric),
      super(base.d, base.metric) {
    isTrained = base.isTrained;
  }

  /// Fast, approximate index. Any [Index] with a working [search].
  final Index base;

  /// Exact fp32 store used for re-ranking.
  final IndexFlat refine;

  /// Candidate pool multiplier — search retrieves `k * kFactor` from
  /// [base] before re-ranking. Higher = better recall, slower.
  int kFactor;

  @override
  void train(List<Float32List> xs) {
    base.train(xs);
    isTrained = base.isTrained;
  }

  @override
  void add(List<Float32List> xs) {
    if (!isTrained) throw StateError('IndexRefineFlat.add before train()');
    base.add(xs);
    refine.add(xs);
    ntotal = refine.ntotal;
  }

  @override
  SearchResult search(List<Float32List> queries, int k) {
    if (!isTrained) throw StateError('IndexRefineFlat.search before train()');
    final nq = queries.length;
    // Pull an over-full candidate list from the base index.
    final coarseK = math.min(k * kFactor, math.max(ntotal, 1));
    final coarse = base.search(queries, coarseK);
    final distances = List<Float32List>.generate(nq, (_) => Float32List(k));
    final ids = List<Int32List>.generate(nq, (_) => Int32List(k));

    for (var qi = 0; qi < nq; qi++) {
      final q = queries[qi];
      final heap = TopK(k, maxIsWorst: metric == Metric.l2);
      final cand = coarse.ids[qi];
      for (var j = 0; j < coarseK; j++) {
        final vid = cand[j];
        if (vid < 0) continue;
        final vec = refine.reconstruct(vid);
        var s = 0.0;
        if (metric == Metric.l2) {
          for (var jj = 0; jj < d; jj++) {
            final diff = q[jj] - vec[jj];
            s += diff * diff;
          }
        } else {
          for (var jj = 0; jj < d; jj++) {
            s += q[jj] * vec[jj];
          }
        }
        heap.push(s, vid);
      }
      final sorted = heap.sorted();
      distances[qi] = sorted.scores;
      ids[qi] = sorted.ids;
    }
    return SearchResult(distances, ids);
  }

  // --- persistence --------------------------------------------------------

  void writeTo(IoWriter w) {
    w.writeU32(d);
    w.writeU32(metricToU32(metric));
    w.writeU32(ntotal);
    w.writeU8(isTrained ? 1 : 0);
    w.writeU32(kFactor);
    writeChild(w, base);
    writeChild(w, refine);
  }

  static IndexRefineFlat readFrom(IoReader r) {
    final d = r.readU32();
    final metric = metricFromU32(r.readU32());
    final ntotal = r.readU32();
    final isTrained = r.readU8() != 0;
    final kFactor = r.readU32();
    final base = readChild(r);
    final refineAny = readChild(r);
    if (refineAny is! IndexFlat) {
      throw FormatException(
        'IndexRefineFlat: embedded refine is ${refineAny.runtimeType}, '
        'expected IndexFlat',
      );
    }
    if (base.d != d || base.metric != metric) {
      throw FormatException(
        'IndexRefineFlat: header (d=$d, metric=$metric) disagrees with '
        'base (d=${base.d}, metric=${base.metric})',
      );
    }
    // Bypass the constructor so we can plug in loaded children directly.
    final idx = IndexRefineFlat._forLoad(
      base: base,
      refine: refineAny,
      kFactor: kFactor,
    );
    idx.ntotal = ntotal;
    idx.isTrained = isTrained;
    return idx;
  }

  IndexRefineFlat._forLoad({
    required this.base,
    required this.refine,
    required this.kFactor,
  }) : super(base.d, base.metric);

  /// I/O hook: builds an [IndexRefineFlat] wrapping already-loaded
  /// sub-indexes, bypassing the normal constructor (which would
  /// allocate a fresh empty refine). Used by `faiss_io.dart` to
  /// rehydrate `IxRF` payloads.
  static IndexRefineFlat ioLoad({
    required Index base,
    required IndexFlat refine,
    required int kFactor,
    required int ntotal,
    required bool isTrained,
  }) {
    if (base.d != refine.d || base.metric != refine.metric) {
      throw ArgumentError(
        'IndexRefineFlat.ioLoad: base (d=${base.d}, metric=${base.metric}) '
        'differs from refine (d=${refine.d}, metric=${refine.metric})',
      );
    }
    final idx = IndexRefineFlat._forLoad(
      base: base,
      refine: refine,
      kFactor: kFactor,
    );
    idx.ntotal = ntotal;
    idx.isTrained = isTrained;
    return idx;
  }
}
