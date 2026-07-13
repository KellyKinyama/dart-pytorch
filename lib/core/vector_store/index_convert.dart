/// Cross-index conversion helpers.
///
/// FAISS-style utilities for turning one populated index into another
/// of a different structure — e.g. take a fully-loaded `IndexFlat` and
/// produce a trained, populated `IndexIVFFlat` sharing the same
/// vectors. Every helper here:
///
///   * Reads vectors out of the source via [IndexFlat.reconstruct] so
///     they can be re-encoded through any target pipeline.
///   * Delegates to the target index's own `train` / `add` — no direct
///     poking at private state.
///   * Preserves [Metric] (rejects at the API boundary if the source
///     and target disagree).
///
/// These are convenience wrappers, not zero-copy hand-offs — plan on
/// paying one full `d * ntotal` fp32 pass per conversion.
library;

import 'dart:typed_data';

import 'index.dart';
import 'index_flat.dart';
import 'index_hnsw.dart';
import 'index_ivf_flat.dart';
import 'index_ivf_pq.dart';
import 'index_refine_flat.dart';

/// Materialise every vector in an [IndexFlat] as a fresh
/// `List<Float32List>` (one owned row per id). Used internally by the
/// conversion helpers and exposed because it's useful on its own.
List<Float32List> reconstructAll(IndexFlat src) {
  final n = src.ntotal;
  final d = src.d;
  final out = List<Float32List>.generate(n, (i) {
    final row = Float32List(d);
    final view = src.reconstruct(i);
    for (var j = 0; j < d; j++) {
      row[j] = view[j];
    }
    return row;
  });
  return out;
}

/// Convert an [IndexFlat] into a trained + populated [IndexIVFFlat].
///
/// The IVF is trained on the flat's full vector set (or a random
/// subsample of `trainSize` rows if that's smaller) and then every
/// vector is added.
///
/// * [nlist] — number of coarse cells (defaults to `sqrt(ntotal)` bounded
///   to `[1, ntotal]`).
/// * [nprobe] — search-time cells to probe (defaults to `min(nlist, 8)`).
/// * [trainSize] — cap on training-set size. `null` = use everything.
IndexIVFFlat flatToIvfFlat(
  IndexFlat src, {
  int? nlist,
  int? nprobe,
  int? trainSize,
  int kmeansIters = 20,
  int seed = 1234,
}) {
  final n = src.ntotal;
  if (n == 0) {
    throw ArgumentError('flatToIvfFlat: source is empty (ntotal=0)');
  }
  final k = nlist ?? _autoNlist(n);
  final p = nprobe ?? (k < 8 ? k : 8);
  final xs = reconstructAll(src);
  final trainer = _maybeSubsample(xs, trainSize, seed);
  final ivf = IndexIVFFlat(
    d: src.d,
    nlist: k,
    metric: src.metric,
    nprobe: p,
    kmeansIters: kmeansIters,
    seed: seed,
  )..train(trainer);
  ivf.add(xs);
  return ivf;
}

/// Convert an [IndexFlat] into a trained + populated [IndexIVFPQ].
///
/// * [m] — number of PQ subquantizers (must divide `d`).
/// * [nbits] — bits per code (FAISS supports 8 in the port).
/// * [nlist], [nprobe], [trainSize], [kmeansIters], [seed] — see
///   [flatToIvfFlat]; PQ codebooks reuse `kmeansIters` via the
///   `pqKmeansIters` constructor default of the port.
IndexIVFPQ flatToIvfPq(
  IndexFlat src, {
  required int m,
  int nbits = 8,
  int? nlist,
  int? nprobe,
  int? trainSize,
  int kmeansIters = 20,
  int seed = 1234,
}) {
  final n = src.ntotal;
  if (n == 0) {
    throw ArgumentError('flatToIvfPq: source is empty (ntotal=0)');
  }
  if (src.d % m != 0) {
    throw ArgumentError('flatToIvfPq: m=$m does not divide d=${src.d}');
  }
  final k = nlist ?? _autoNlist(n);
  final p = nprobe ?? (k < 8 ? k : 8);
  final xs = reconstructAll(src);
  final trainer = _maybeSubsample(xs, trainSize, seed);
  final ivf = IndexIVFPQ(
    d: src.d,
    nlist: k,
    m: m,
    nbits: nbits,
    metric: src.metric,
    nprobe: p,
    kmeansIters: kmeansIters,
    seed: seed,
  )..train(trainer);
  ivf.add(xs);
  return ivf;
}

/// Convert an [IndexFlat] into a populated [IndexHNSW].
///
/// HNSW needs no separate training step — construction is the graph
/// build.
IndexHNSW flatToHnsw(
  IndexFlat src, {
  int m = 16,
  int efConstruction = 100,
  int efSearch = 32,
  int seed = 1234,
}) {
  final xs = reconstructAll(src);
  final hnsw = IndexHNSW(
    d: src.d,
    metric: src.metric,
    M: m,
    efConstruction: efConstruction,
    efSearch: efSearch,
    seed: seed,
  );
  if (xs.isNotEmpty) hnsw.add(xs);
  return hnsw;
}

/// Wrap any trained + populated approximate index with an
/// [IndexRefineFlat] using [src] as the exact re-ranker.
///
/// Handy for the "IVFPQ → RefineFlat(IVFPQ)" recall-recovery pattern:
///
/// ```dart
/// final ivfpq = flatToIvfPq(flat, m: 8);
/// final refined = wrapWithRefine(ivfpq, flat, kFactor: 4);
/// ```
///
/// Requires the base to be already trained and populated to the same
/// `ntotal` as `src`, and both to agree on `d` / `metric`.
IndexRefineFlat wrapWithRefine(
  Index base,
  IndexFlat src, {
  int kFactor = 4,
}) {
  if (base.d != src.d) {
    throw ArgumentError(
      'wrapWithRefine: base.d=${base.d} != src.d=${src.d}',
    );
  }
  if (base.metric != src.metric) {
    throw ArgumentError(
      'wrapWithRefine: base.metric=${base.metric} != '
      'src.metric=${src.metric}',
    );
  }
  if (base.ntotal != src.ntotal) {
    throw ArgumentError(
      'wrapWithRefine: base.ntotal=${base.ntotal} != '
      'src.ntotal=${src.ntotal}',
    );
  }
  if (!base.isTrained) {
    throw StateError('wrapWithRefine: base is not trained');
  }
  final xs = reconstructAll(src);
  final refined = IndexRefineFlat(base, kFactor: kFactor);
  // Skip re-adding to base; only need to populate the internal
  // `refine` slot. `IndexRefineFlat.add` would double-add to base, so
  // populate `refine` directly and sync `ntotal`.
  refined.isTrained = true;
  refined.refine.add(xs);
  refined.ntotal = refined.refine.ntotal;
  return refined;
}

// ---------------------------------------------------------------------
// Internal helpers.
// ---------------------------------------------------------------------

/// Round-of-thumb default for IVF cell count: `ceil(sqrt(ntotal))`,
/// clamped to `[1, ntotal]`.
int _autoNlist(int ntotal) {
  if (ntotal <= 1) return 1;
  var k = 1;
  while (k * k < ntotal) {
    k++;
  }
  return k > ntotal ? ntotal : k;
}

/// Deterministic uniform subsample of `xs` down to `cap` rows. Returns
/// `xs` unchanged if `cap` is null or `>= xs.length`.
List<Float32List> _maybeSubsample(
  List<Float32List> xs,
  int? cap,
  int seed,
) {
  if (cap == null || cap >= xs.length) return xs;
  // Fisher-Yates prefix — pick `cap` indices without replacement.
  final n = xs.length;
  final perm = List<int>.generate(n, (i) => i);
  final rng = _LinearRng(seed);
  for (var i = 0; i < cap; i++) {
    final j = i + rng.nextInt(n - i);
    final tmp = perm[i];
    perm[i] = perm[j];
    perm[j] = tmp;
  }
  return List<Float32List>.generate(cap, (i) => xs[perm[i]]);
}

/// Tiny deterministic LCG used only for `_maybeSubsample`. Keeps
/// `dart:math`-based helpers off this file's import surface.
class _LinearRng {
  _LinearRng(int seed) : _state = (seed == 0 ? 1 : seed) & 0x7fffffff;
  int _state;
  int nextInt(int bound) {
    // Numerical Recipes LCG.
    _state = (_state * 1664525 + 1013904223) & 0x7fffffff;
    return _state % bound;
  }
}
