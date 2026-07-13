/// Reusable recall / latency benchmark harness for vector indexes.
///
/// The core primitive is [benchIndex], which times a set of query
/// batches against a candidate index, computes recall@k against a
/// ground-truth [SearchResult], and returns a [BenchResult]. A tiny
/// CSV writer ([writeBenchCsv]) and a "sweep" helper
/// ([benchIndexSweep]) are layered on top so callers can build
/// recall-vs-latency tables with a few lines of glue.
///
/// This file intentionally has no CLI surface — see
/// `bin/bench_faiss.dart` for a runnable frontend.
library;

import 'dart:typed_data';

import 'index.dart';

/// Result of a single [benchIndex] run.
class BenchResult {
  BenchResult({
    required this.label,
    required this.nq,
    required this.k,
    required this.recall,
    required this.meanUs,
    required this.p50Us,
    required this.p95Us,
    required this.p99Us,
    required this.ntotal,
  });

  /// Free-form label the caller passed in (e.g. `"IVFFlat nprobe=8"`).
  final String label;

  /// Query count (rows) of the batch this result summarises.
  final int nq;

  /// `k` used for the search.
  final int k;

  /// Recall@k against the ground truth passed to [benchIndex].
  final double recall;

  /// Mean per-query latency, in microseconds.
  final double meanUs;

  /// p50 / p95 / p99 per-query latency (microseconds). Computed from
  /// the per-query wall clock across the search batch — the batch is
  /// run once per warmup and then [BenchOptions.repeats] times for the
  /// timing pass; the reported latencies are the best (min) of the
  /// timed passes to reduce noise.
  final double p50Us;
  final double p95Us;
  final double p99Us;

  /// `index.ntotal` at bench time — useful when comparing runs of
  /// different sizes.
  final int ntotal;

  @override
  String toString() {
    return 'BenchResult($label: recall@$k=${recall.toStringAsFixed(3)}, '
        'mean=${meanUs.toStringAsFixed(1)}µs, '
        'p50=${p50Us.toStringAsFixed(1)}µs, '
        'p95=${p95Us.toStringAsFixed(1)}µs, '
        'ntotal=$ntotal, nq=$nq)';
  }
}

/// Knobs for [benchIndex]. Sensible defaults are provided for one-off
/// use; the sweep helpers reuse this class to keep argument lists sane.
class BenchOptions {
  const BenchOptions({
    this.warmup = 1,
    this.repeats = 3,
    this.perQuery = true,
  });

  /// Number of warmup passes (results discarded).
  final int warmup;

  /// Number of timed passes; percentiles are drawn from the best pass
  /// (min per-batch total).
  final int repeats;

  /// If true, latencies are measured per-query (the loop calls
  /// `search([q], k)` once per query). If false, one batch call is
  /// timed and the per-query numbers are `batchTotal / nq` for all
  /// percentiles — cheaper but hides tail latency. Default: true.
  final bool perQuery;
}

/// Time [index] on [queries] and compute recall@[k] against [truth].
BenchResult benchIndex({
  required Index index,
  required List<Float32List> queries,
  required int k,
  required SearchResult truth,
  String label = 'index',
  BenchOptions options = const BenchOptions(),
}) {
  if (queries.isEmpty) {
    throw ArgumentError('benchIndex: queries is empty');
  }
  if (truth.nq != queries.length) {
    throw ArgumentError(
      'benchIndex: truth.nq=${truth.nq} != queries.length=${queries.length}',
    );
  }
  if (truth.k < k) {
    throw ArgumentError(
      'benchIndex: truth.k=${truth.k} is smaller than requested k=$k',
    );
  }

  // Warmup passes — discard results.
  for (var w = 0; w < options.warmup; w++) {
    index.search(queries, k);
  }

  // Timed passes. For each pass, collect per-query timings (or a
  // single batched timing) and keep the pass with the smallest total.
  final nq = queries.length;
  List<int>? bestUsPerQuery;
  SearchResult? lastResult;
  var bestTotalUs = 1 << 62;

  for (var r = 0; r < options.repeats; r++) {
    final passUs = List<int>.filled(nq, 0);
    late SearchResult res;
    var passTotal = 0;
    if (options.perQuery) {
      // One-shot list reused per query.
      final buf = <Float32List>[Float32List(0)];
      final rowDistances = List<Float32List>.filled(nq, Float32List(0));
      final rowIds = List<Int32List>.filled(nq, Int32List(0));
      for (var qi = 0; qi < nq; qi++) {
        buf[0] = queries[qi];
        final sw = Stopwatch()..start();
        final one = index.search(buf, k);
        sw.stop();
        final us = sw.elapsedMicroseconds;
        passUs[qi] = us;
        passTotal += us;
        rowDistances[qi] = one.distances[0];
        rowIds[qi] = one.ids[0];
      }
      res = SearchResult(rowDistances, rowIds);
    } else {
      final sw = Stopwatch()..start();
      res = index.search(queries, k);
      sw.stop();
      final total = sw.elapsedMicroseconds;
      passTotal = total;
      final per = total / nq;
      for (var qi = 0; qi < nq; qi++) {
        passUs[qi] = per.round();
      }
    }
    if (passTotal < bestTotalUs) {
      bestTotalUs = passTotal;
      bestUsPerQuery = passUs;
      lastResult = res;
    }
  }

  final usList = bestUsPerQuery!;
  final result = lastResult!;
  final mean = usList.fold<int>(0, (a, b) => a + b) / usList.length;
  final sorted = List<int>.from(usList)..sort();
  double pct(double q) {
    if (sorted.isEmpty) return 0.0;
    final idx = ((sorted.length - 1) * q).round();
    return sorted[idx].toDouble();
  }

  return BenchResult(
    label: label,
    nq: nq,
    k: k,
    recall: _recallAtK(result, truth, k),
    meanUs: mean,
    p50Us: pct(0.5),
    p95Us: pct(0.95),
    p99Us: pct(0.99),
    ntotal: index.ntotal,
  );
}

/// Sweep an [Index] across a list of configurations. [configure] is
/// called once per point with the raw value from [values] and should
/// mutate the index in place (e.g. bump `nprobe` or `efSearch`); its
/// return value is used as the point's label.
///
/// Returns one [BenchResult] per value in insertion order.
List<BenchResult> benchIndexSweep<T>({
  required Index index,
  required List<Float32List> queries,
  required int k,
  required SearchResult truth,
  required List<T> values,
  required String Function(T value) configure,
  BenchOptions options = const BenchOptions(),
}) {
  final out = <BenchResult>[];
  for (final v in values) {
    final label = configure(v);
    out.add(benchIndex(
      index: index,
      queries: queries,
      k: k,
      truth: truth,
      label: label,
      options: options,
    ));
  }
  return out;
}

/// Serialise a list of [BenchResult]s to CSV. Column order:
/// `label,ntotal,nq,k,recall,mean_us,p50_us,p95_us,p99_us`. The label
/// column is CSV-escaped (commas / quotes / newlines) per RFC 4180.
String toBenchCsv(Iterable<BenchResult> rows) {
  final buf = StringBuffer(
    'label,ntotal,nq,k,recall,mean_us,p50_us,p95_us,p99_us\n',
  );
  for (final r in rows) {
    buf.write(_csvEscape(r.label));
    buf.write(',');
    buf.write(r.ntotal);
    buf.write(',');
    buf.write(r.nq);
    buf.write(',');
    buf.write(r.k);
    buf.write(',');
    buf.write(r.recall.toStringAsFixed(6));
    buf.write(',');
    buf.write(r.meanUs.toStringAsFixed(3));
    buf.write(',');
    buf.write(r.p50Us.toStringAsFixed(3));
    buf.write(',');
    buf.write(r.p95Us.toStringAsFixed(3));
    buf.write(',');
    buf.write(r.p99Us.toStringAsFixed(3));
    buf.write('\n');
  }
  return buf.toString();
}

/// Pretty-print a bench table to stdout-style text (fixed-width
/// columns). Useful for CLI or test output.
String formatBenchTable(Iterable<BenchResult> rows) {
  final rowsList = rows.toList();
  if (rowsList.isEmpty) return '(no rows)\n';
  final labelWidth =
      rowsList.map((r) => r.label.length).reduce((a, b) => a > b ? a : b);
  final lw = labelWidth < 12 ? 12 : labelWidth;
  final buf = StringBuffer();
  buf.write(_pad('label', lw));
  buf.write('  recall   mean_us    p50_us    p95_us    p99_us  ntotal   nq   k\n');
  buf.write('-' * (lw + 68));
  buf.write('\n');
  for (final r in rowsList) {
    buf.write(_pad(r.label, lw));
    buf.write('  ');
    buf.write(_rpad(r.recall.toStringAsFixed(3), 7));
    buf.write(_rpad(r.meanUs.toStringAsFixed(1), 10));
    buf.write(_rpad(r.p50Us.toStringAsFixed(1), 10));
    buf.write(_rpad(r.p95Us.toStringAsFixed(1), 10));
    buf.write(_rpad(r.p99Us.toStringAsFixed(1), 10));
    buf.write(_rpad(r.ntotal.toString(), 8));
    buf.write(_rpad(r.nq.toString(), 5));
    buf.write(r.k.toString());
    buf.write('\n');
  }
  return buf.toString();
}

// ---------------------------------------------------------------------
// Internals.
// ---------------------------------------------------------------------

double _recallAtK(SearchResult result, SearchResult truth, int k) {
  var hit = 0;
  var total = 0;
  for (var qi = 0; qi < result.nq; qi++) {
    final gold = <int>{};
    for (var j = 0; j < k; j++) {
      final id = truth.ids[qi][j];
      if (id >= 0) gold.add(id);
    }
    for (var j = 0; j < k; j++) {
      if (gold.contains(result.ids[qi][j])) hit++;
    }
    total += k;
  }
  return total == 0 ? 0.0 : hit / total;
}

String _csvEscape(String s) {
  if (s.contains(',') || s.contains('"') || s.contains('\n')) {
    return '"${s.replaceAll('"', '""')}"';
  }
  return s;
}

String _pad(String s, int w) {
  if (s.length >= w) return s;
  return s + ' ' * (w - s.length);
}

String _rpad(String s, int w) {
  if (s.length >= w) return s;
  return ' ' * (w - s.length) + s;
}
