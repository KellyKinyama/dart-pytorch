/// Auto-tuning for approximate index parameters.
///
/// FAISS ships a `ParameterSpace / OperatingPoints` machine that
/// walks the recall-vs-latency Pareto frontier — this file provides
/// a smaller, pragmatic subset of that idea:
///
///   * [OperatingPoint] — one (recall, mean_us) point, tagged with
///     the parameter value that produced it.
///   * [OperatingPoints] — a set of points with a helper to extract
///     the Pareto frontier and to pick the smallest / fastest point
///     that meets a recall floor (or the highest-recall point that
///     meets a latency ceiling).
///   * [autoTuneNprobe] — sweep `nprobe` on an IVF-flavoured index
///     (IndexIVFFlat / IndexIVFPQ, incl. wrapped in IndexRefineFlat)
///     and pick the smallest value that clears a recall target.
///   * [autoTuneEfSearch] — same idea, for IndexHNSW.
///
/// All tuners drive the shared bench harness so timings share the
/// same warmup / repeat semantics as manual benchmarks.
library;

import 'dart:typed_data';

import 'bench.dart';
import 'index.dart';
import 'index_flat.dart';
import 'index_hnsw.dart';
import 'index_ivf_flat.dart';
import 'index_ivf_pq.dart';
import 'index_refine_flat.dart';

/// One `(recall, latency)` measurement tagged with the parameter that
/// produced it.
class OperatingPoint {
  const OperatingPoint({
    required this.paramValue,
    required this.paramLabel,
    required this.recall,
    required this.meanUs,
  });

  /// Raw parameter value (e.g. `nprobe = 8`, `efSearch = 64`).
  final int paramValue;

  /// Short human-readable label ("nprobe=8").
  final String paramLabel;

  final double recall;
  final double meanUs;

  @override
  String toString() =>
      '$paramLabel  recall=${recall.toStringAsFixed(3)}  '
      'mean_us=${meanUs.toStringAsFixed(1)}';
}

/// Collection of operating points with Pareto / selection helpers.
class OperatingPoints {
  OperatingPoints(List<OperatingPoint> points)
    : points = List<OperatingPoint>.unmodifiable(points);

  /// All points, in the order they were measured.
  final List<OperatingPoint> points;

  /// Pareto frontier: points that are not dominated by any other
  /// (higher recall AND lower latency). Returned sorted by recall
  /// ascending.
  List<OperatingPoint> pareto() {
    final sorted = List<OperatingPoint>.from(points)
      ..sort((a, b) => a.recall.compareTo(b.recall));
    final out = <OperatingPoint>[];
    var bestUs = double.infinity;
    for (var i = sorted.length - 1; i >= 0; i--) {
      final p = sorted[i];
      if (p.meanUs < bestUs) {
        out.add(p);
        bestUs = p.meanUs;
      }
    }
    return out.reversed.toList();
  }

  /// Pick the smallest-parameter (fastest) point whose recall is at
  /// least [minRecall]. Returns `null` if no point clears the bar.
  OperatingPoint? pickForRecall(double minRecall) {
    OperatingPoint? best;
    for (final p in points) {
      if (p.recall >= minRecall) {
        if (best == null || p.meanUs < best.meanUs) {
          best = p;
        }
      }
    }
    return best;
  }

  /// Pick the highest-recall point whose latency is at most
  /// [maxMeanUs]. Returns `null` if no point is fast enough.
  OperatingPoint? pickForLatency(double maxMeanUs) {
    OperatingPoint? best;
    for (final p in points) {
      if (p.meanUs <= maxMeanUs) {
        if (best == null || p.recall > best.recall) {
          best = p;
        }
      }
    }
    return best;
  }
}

/// Sweep `nprobe` on an IVF-flavoured index and return the operating
/// points. Also mutates the index's `nprobe` in place — after the
/// tuner returns, the index is left at the *chosen* nprobe if
/// [applyBest] + [minRecall] are set; otherwise it is left at the
/// last value in [values].
///
/// [target] receives an IVF index (or an [IndexRefineFlat] whose base
/// is IVF); the tuner detects the type and pokes the underlying
/// `nprobe` field either way.
OperatingPoints autoTuneNprobe({
  required Index target,
  required List<Float32List> queries,
  required int k,
  required SearchResult truth,
  List<int> values = const <int>[1, 2, 4, 8, 16, 32, 64],
  double? minRecall,
  bool applyBest = true,
  BenchOptions options = const BenchOptions(),
}) {
  final setNprobe = _nprobeSetter(target);
  final nlist = _nprobeNlist(target);
  final rows = <OperatingPoint>[];
  final seen = <int>{};
  for (final raw in values) {
    final np = raw < 1 ? 1 : (raw > nlist ? nlist : raw);
    if (!seen.add(np)) continue; // dedupe after clamping
    setNprobe(np);
    final r = benchIndex(
      index: target,
      queries: queries,
      k: k,
      truth: truth,
      label: 'nprobe=$np',
      options: options,
    );
    rows.add(
      OperatingPoint(
        paramValue: np,
        paramLabel: 'nprobe=$np',
        recall: r.recall,
        meanUs: r.meanUs,
      ),
    );
  }
  final points = OperatingPoints(rows);
  if (applyBest && minRecall != null) {
    final chosen = points.pickForRecall(minRecall);
    if (chosen != null) setNprobe(chosen.paramValue);
  }
  return points;
}

/// Sweep `efSearch` on an [IndexHNSW] and return the operating points.
/// Mutates `target.efSearch` in place — see [autoTuneNprobe] for the
/// [applyBest] / [minRecall] semantics.
OperatingPoints autoTuneEfSearch({
  required IndexHNSW target,
  required List<Float32List> queries,
  required int k,
  required SearchResult truth,
  List<int> values = const <int>[8, 16, 32, 64, 128, 256],
  double? minRecall,
  bool applyBest = true,
  BenchOptions options = const BenchOptions(),
}) {
  final rows = <OperatingPoint>[];
  for (final ef in values) {
    if (ef < 1) continue;
    target.efSearch = ef;
    final r = benchIndex(
      index: target,
      queries: queries,
      k: k,
      truth: truth,
      label: 'efSearch=$ef',
      options: options,
    );
    rows.add(
      OperatingPoint(
        paramValue: ef,
        paramLabel: 'efSearch=$ef',
        recall: r.recall,
        meanUs: r.meanUs,
      ),
    );
  }
  final points = OperatingPoints(rows);
  if (applyBest && minRecall != null) {
    final chosen = points.pickForRecall(minRecall);
    if (chosen != null) target.efSearch = chosen.paramValue;
  }
  return points;
}

/// Result of an [autoTuneM] sweep.
///
/// Because each `m` value requires a fresh k-means-trained PQ (m is
/// baked into the codebooks), the sweep can't mutate a single index
/// in place — it builds one [IndexIVFPQ] per candidate and returns the
/// winning [chosen] alongside the full set of [points] and every
/// [built] index (so callers can pick a runner-up without paying the
/// build cost again).
class TuneMResult {
  TuneMResult({
    required this.points,
    required this.built,
    required this.chosen,
    required this.chosenIndex,
  });

  /// One point per candidate `m`.
  final OperatingPoints points;

  /// All built indexes, keyed by `m`. Includes those that produced
  /// [OperatingPoint]s; disposal is the caller's responsibility.
  final Map<int, IndexIVFPQ> built;

  /// The best point under the selection rule passed to [autoTuneM],
  /// or `null` when no candidate cleared the recall floor.
  final OperatingPoint? chosen;

  /// The [IndexIVFPQ] behind [chosen], or `null` when [chosen] is
  /// `null`.
  final IndexIVFPQ? chosenIndex;
}

/// Sweep the PQ subquantiser count `m` for an [IndexIVFPQ] built from
/// the given source vectors. Each candidate `m` triggers a fresh
/// train + add on a new [IndexIVFPQ], so this is markedly more
/// expensive than [autoTuneNprobe] — expect O(len(values)) k-means
/// runs plus O(len(values)) full-corpus PQ encoding passes.
///
/// * [values] — candidate `m`s. Values that do not divide `d` are
///   skipped with a warning printed to stderr-equivalent (dropped from
///   the result silently — callers that care should validate up
///   front).
/// * [nlist] — shared across all candidates (same coarse quantizer
///   configuration). Defaults to `ceil(sqrt(ntotal))`.
/// * [nprobe] — search-time cells to probe. Defaults to
///   `min(nlist, 8)`. Held constant across the sweep so `m` is the
///   only varying axis in the recall table.
/// * [minRecall] — when set, [TuneMResult.chosen] is the point with
///   the smallest `m` (i.e. best compression) that clears the floor.
///   When unset, [TuneMResult.chosen] falls back to the highest-`m`
///   candidate (the one most likely to have the strongest recall).
TuneMResult autoTuneM({
  required List<Float32List> sourceVectors,
  required int d,
  required Metric metric,
  required List<Float32List> queries,
  required int k,
  required SearchResult truth,
  required List<int> values,
  int nbits = 8,
  int? nlist,
  int? nprobe,
  int kmeansIters = 20,
  int seed = 1234,
  double? minRecall,
  BenchOptions options = const BenchOptions(),
}) {
  if (sourceVectors.isEmpty) {
    throw ArgumentError('autoTuneM: sourceVectors is empty');
  }
  final k0 = nlist ?? _autoNlist(sourceVectors.length);
  final p0 = nprobe ?? (k0 < 8 ? k0 : 8);

  final built = <int, IndexIVFPQ>{};
  final rows = <OperatingPoint>[];
  for (final m in values) {
    if (m < 1 || d % m != 0) continue; // skip invalid divisors
    final ivfpq =
        IndexIVFPQ(
            d: d,
            nlist: k0,
            m: m,
            nbits: nbits,
            metric: metric,
            nprobe: p0,
            kmeansIters: kmeansIters,
            seed: seed,
          )
          ..train(sourceVectors)
          ..add(sourceVectors);
    built[m] = ivfpq;
    final r = benchIndex(
      index: ivfpq,
      queries: queries,
      k: k,
      truth: truth,
      label: 'm=$m',
      options: options,
    );
    rows.add(
      OperatingPoint(
        paramValue: m,
        paramLabel: 'm=$m',
        recall: r.recall,
        meanUs: r.meanUs,
      ),
    );
  }

  final points = OperatingPoints(rows);
  OperatingPoint? chosen;
  if (minRecall != null) {
    // Smallest m (best compression) that clears the recall floor.
    OperatingPoint? best;
    for (final p in points.points) {
      if (p.recall >= minRecall) {
        if (best == null || p.paramValue < best.paramValue) {
          best = p;
        }
      }
    }
    chosen = best;
  } else if (rows.isNotEmpty) {
    // Fall back to the largest `m` (typically strongest recall).
    chosen = rows.reduce((a, b) => a.paramValue >= b.paramValue ? a : b);
  }
  final chosenIndex = chosen == null ? null : built[chosen.paramValue];
  return TuneMResult(
    points: points,
    built: built,
    chosen: chosen,
    chosenIndex: chosenIndex,
  );
}

/// Convenience: same as [autoTuneM] but takes an already-populated
/// [IndexFlat] as the source (matching the [flatToIvfPq] convention).
TuneMResult autoTuneMFromFlat({
  required IndexFlat source,
  required List<Float32List> queries,
  required int k,
  required SearchResult truth,
  required List<int> values,
  int nbits = 8,
  int? nlist,
  int? nprobe,
  int kmeansIters = 20,
  int seed = 1234,
  double? minRecall,
  BenchOptions options = const BenchOptions(),
}) {
  // Materialise the flat's storage as a plain vector list.
  final xs = <Float32List>[
    for (var i = 0; i < source.ntotal; i++)
      Float32List.fromList(source.reconstruct(i)),
  ];
  return autoTuneM(
    sourceVectors: xs,
    d: source.d,
    metric: source.metric,
    queries: queries,
    k: k,
    truth: truth,
    values: values,
    nbits: nbits,
    nlist: nlist,
    nprobe: nprobe,
    kmeansIters: kmeansIters,
    seed: seed,
    minRecall: minRecall,
    options: options,
  );
}

// ---------------------------------------------------------------------
// Internals.
// ---------------------------------------------------------------------

/// Extract the underlying IVF index from a candidate target so the
/// tuner can poke `nprobe`. Accepts `IndexIVFFlat`, `IndexIVFPQ`, or
/// `IndexRefineFlat` wrapping either.
void Function(int) _nprobeSetter(Index target) {
  if (target is IndexIVFFlat) {
    return (np) => target.nprobe = np;
  }
  if (target is IndexIVFPQ) {
    return (np) => target.nprobe = np;
  }
  if (target is IndexRefineFlat) {
    final base = target.base;
    if (base is IndexIVFFlat) return (np) => base.nprobe = np;
    if (base is IndexIVFPQ) return (np) => base.nprobe = np;
  }
  throw ArgumentError(
    'autoTuneNprobe: no nprobe setter for ${target.runtimeType}',
  );
}

int _nprobeNlist(Index target) {
  if (target is IndexIVFFlat) return target.nlist;
  if (target is IndexIVFPQ) return target.nlist;
  if (target is IndexRefineFlat) {
    final base = target.base;
    if (base is IndexIVFFlat) return base.nlist;
    if (base is IndexIVFPQ) return base.nlist;
  }
  throw ArgumentError('autoTuneNprobe: no nlist for ${target.runtimeType}');
}

/// `ceil(sqrt(ntotal))` clamped to `[1, ntotal]` — matches
/// `index_convert.dart`'s default IVF-cell-count heuristic so
/// [autoTuneM] and [flatToIvfPq] agree on defaults.
int _autoNlist(int ntotal) {
  if (ntotal <= 1) return 1;
  var k = 1;
  while (k * k < ntotal) {
    k++;
  }
  return k > ntotal ? ntotal : k;
}
