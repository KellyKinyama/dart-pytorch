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
