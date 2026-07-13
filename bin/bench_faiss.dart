/// Recall-vs-latency bench harness for the vector-store surface.
///
/// Builds a synthetic corpus with a small first-axis smear so nearby
/// ids cluster in space (mirrors the FAISS tutorial), then runs every
/// index type at a handful of tuning points and prints a table of
/// recall vs. per-query latency vs. ground-truth. Optional CSV output
/// via `--csv path`.
///
/// Run:
///
///     dart run bin/bench_faiss.dart
///     dart run bin/bench_faiss.dart --nb 20000 --d 128 --csv /tmp/bench.csv
///
/// Flags:
///   `--nb N`             database size (default 5000)
///   `--nq N`             query count (default 200)
///   `--d  D`             dimensionality (default 64)
///   `--k  K`             neighbours per query (default 10)
///   `--seed S`           RNG seed (default 0)
///   `--csv P`            append CSV to path P (writes header if new)
///   `--md  P`            write Markdown table to path P (overwrites)
///   `--pareto`           restrict CSV / Markdown output to the Pareto
///                        frontier (recall-vs-latency)
///   `--tune-target T`    pick one built index to save as a tuned IxDT
///                        blob. One of: ivfflat, ivfpq, ivfpq+rf, hnsw.
///   `--tune-out P`       output path for the IxDT-wrapped tuned blob
///                        (required with `--tune-target`)
///   `--tune-recall F`    target min recall for `pickForRecall` (default
///                        0.9); falls back to the highest-recall point
///                        in the sweep when no candidate clears the bar
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

int main(List<String> args) {
  final opts = _Options.parse(args);
  if (opts == null) return 1;

  final xs = _sample(opts.nb, opts.d, seed: opts.seed);
  final queries = _sample(opts.nq, opts.d, seed: opts.seed + 1);

  // Ground truth.
  final flat = IndexFlatL2(opts.d)..add(xs);
  final truth = flat.search(queries, opts.k);

  final results = <BenchResult>[];

  // -----------------------------------------------------------------
  // Flat (baseline).
  // -----------------------------------------------------------------
  results.add(
    benchIndex(
      index: flat,
      queries: queries,
      k: opts.k,
      truth: truth,
      label: 'IndexFlatL2',
    ),
  );

  // -----------------------------------------------------------------
  // GPU-backed flat. Wraps the same IndexFlat so the vectors are
  // shared byte-for-byte with the CPU baseline. Warmed once so the
  // first timing pass does not eat the DB upload cost.
  // -----------------------------------------------------------------
  final gpuFlat = GpuIndexFlat.wrap(flat)..warmup();
  results.add(
    benchIndex(
      index: gpuFlat,
      queries: queries,
      k: opts.k,
      truth: truth,
      label: 'GpuIndexFlat',
    ),
  );

  // Batched-mode rows for the flat comparators — GPU's Q @ D^T
  // amortises upload + FFI cost across the whole batch, so this is
  // where the GPU speedup (if any) actually shows up. The per-query
  // number reported for these rows is (batch_us / nq) so it is
  // directly comparable to the per-query rows above.
  const batched = BenchOptions(perQuery: false);
  results.add(
    benchIndex(
      index: flat,
      queries: queries,
      k: opts.k,
      truth: truth,
      label: 'IndexFlatL2 [batched]',
      options: batched,
    ),
  );
  results.add(
    benchIndex(
      index: gpuFlat,
      queries: queries,
      k: opts.k,
      truth: truth,
      label: 'GpuIndexFlat [batched]',
      options: batched,
    ),
  );

  // -----------------------------------------------------------------
  // IVFFlat sweep on nprobe.
  // -----------------------------------------------------------------
  final ivf = flatToIvfFlat(flat, nlist: _autoNlist(opts.nb), nprobe: 1);
  results.addAll(
    benchIndexSweep(
      index: ivf,
      queries: queries,
      k: opts.k,
      truth: truth,
      values: <int>[1, 2, 4, 8, 16, 32],
      configure: (nprobe) {
        final np = nprobe.clamp(1, ivf.nlist);
        ivf.nprobe = np;
        return 'IVFFlat  nprobe=${np.toString().padLeft(2)}';
      },
    ),
  );

  // -----------------------------------------------------------------
  // IVFPQ + IVFPQ+Refine sweep on nprobe.
  // -----------------------------------------------------------------
  final m = _pickPqM(opts.d);
  final ivfpq = flatToIvfPq(flat, m: m, nlist: _autoNlist(opts.nb), nprobe: 1);
  final refined = wrapWithRefine(ivfpq, flat, kFactor: 4);
  for (final nprobe in <int>[1, 4, 8, 16]) {
    final np = nprobe.clamp(1, ivfpq.nlist);
    ivfpq.nprobe = np;
    results.add(
      benchIndex(
        index: ivfpq,
        queries: queries,
        k: opts.k,
        truth: truth,
        label: 'IVFPQ    m=$m nprobe=${np.toString().padLeft(2)}',
      ),
    );
    results.add(
      benchIndex(
        index: refined,
        queries: queries,
        k: opts.k,
        truth: truth,
        label: 'IVFPQ+RF m=$m nprobe=${np.toString().padLeft(2)}',
      ),
    );
  }

  // -----------------------------------------------------------------
  // HNSW sweep on efSearch.
  // -----------------------------------------------------------------
  final hnsw = flatToHnsw(flat, m: 16, efConstruction: 80);
  results.addAll(
    benchIndexSweep(
      index: hnsw,
      queries: queries,
      k: opts.k,
      truth: truth,
      values: <int>[8, 16, 32, 64, 128],
      configure: (ef) {
        hnsw.efSearch = ef;
        return 'HNSW    efSearch=${ef.toString().padLeft(3)}';
      },
    ),
  );

  stdout.write(formatBenchTable(results));

  // Rows to hand off to CSV / Markdown exports. --pareto keeps only
  // the recall-vs-latency frontier (dominated points dropped).
  final exportRows = opts.pareto ? paretoFrontier(results) : results;
  if (opts.pareto) {
    stdout.writeln(
      '\nPareto frontier (${exportRows.length} of '
      '${results.length} rows):',
    );
    stdout.write(formatBenchTable(exportRows));
  }

  if (opts.csvPath != null) {
    final file = File(opts.csvPath!);
    final existed = file.existsSync();
    final sink = file.openWrite(mode: FileMode.append);
    final csv = toBenchCsv(exportRows);
    // Skip the header row on append if the file already had content.
    if (existed && file.lengthSync() > 0) {
      final nl = csv.indexOf('\n');
      sink.write(csv.substring(nl + 1));
    } else {
      sink.write(csv);
    }
    sink.close();
    stdout.writeln('wrote CSV → ${opts.csvPath}');
  }

  if (opts.mdPath != null) {
    File(opts.mdPath!).writeAsStringSync(toBenchMarkdown(exportRows));
    stdout.writeln('wrote Markdown → ${opts.mdPath}');
  }

  if (opts.tuneTarget != null) {
    final ok = _writeTunedBlob(
      target: opts.tuneTarget!,
      outPath: opts.tuneOut!,
      minRecall: opts.tuneRecall,
      results: results,
      ivf: ivf,
      ivfpq: ivfpq,
      refined: refined,
      hnsw: hnsw,
    );
    if (!ok) return 2;
  }

  return 0;
}

// ---------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------

List<Float32List> _sample(int n, int d, {required int seed}) {
  final rng = math.Random(seed);
  return List<Float32List>.generate(n, (i) {
    final v = Float32List(d);
    for (var j = 0; j < d; j++) {
      v[j] = rng.nextDouble();
    }
    v[0] += i / 1000.0;
    return v;
  });
}

int _autoNlist(int ntotal) {
  if (ntotal <= 1) return 1;
  var k = 1;
  while (k * k < ntotal) {
    k++;
  }
  return k;
}

/// Pick the largest divisor of `d` that is ≤ 8. Falls back to 1 when
/// `d` is prime.
int _pickPqM(int d) {
  for (var m = 8; m >= 1; m--) {
    if (d % m == 0) return m;
  }
  return 1;
}

/// Extract the sweep points that match [target] from the bench
/// results, pick a winner via `OperatingPoints.pickForRecall(minRecall)`
/// (falling back to the highest-recall point if the target isn't met),
/// and write an IxDT-wrapped tuned blob to [outPath]. Prints a short
/// summary. Returns `false` on error.
bool _writeTunedBlob({
  required String target,
  required String outPath,
  required double minRecall,
  required List<BenchResult> results,
  required IndexIVFFlat ivf,
  required IndexIVFPQ ivfpq,
  required IndexRefineFlat refined,
  required IndexHNSW hnsw,
}) {
  final spec = _tuneSpecs[target];
  if (spec == null) {
    stderr.writeln(
      'bench_faiss: --tune-target must be one of '
      '${_tuneSpecs.keys.join(", ")} (got "$target")',
    );
    return false;
  }

  final points = <OperatingPoint>[];
  for (final r in results) {
    if (!r.label.startsWith(spec.labelPrefix)) continue;
    final m = spec.paramPattern.firstMatch(r.label);
    if (m == null) continue;
    final value = int.parse(m.group(1)!);
    points.add(
      OperatingPoint(
        paramValue: value,
        paramLabel: '${spec.paramName}=$value',
        recall: r.recall,
        meanUs: r.meanUs,
      ),
    );
  }
  if (points.isEmpty) {
    stderr.writeln(
      'bench_faiss: no sweep rows matched target "$target" '
      '(expected label prefix "${spec.labelPrefix}")',
    );
    return false;
  }

  final ops = OperatingPoints(points);
  var chosen = ops.pickForRecall(minRecall);
  final fellBack = chosen == null;
  if (chosen == null) {
    // Fallback: highest recall in the sweep, tie-break by smallest latency.
    final sorted = List<OperatingPoint>.from(points)
      ..sort((a, b) {
        final c = b.recall.compareTo(a.recall);
        return c != 0 ? c : a.meanUs.compareTo(b.meanUs);
      });
    chosen = sorted.first;
  }

  final meta = TuningMetadata.fromOperatingPoints(
    points: ops,
    metric: Metric.l2,
    chosenParamValue: chosen.paramValue,
  );

  final inner = spec.pick(ivf, ivfpq, refined, hnsw);
  saveTunedFaissIndex(outPath, inner, meta);

  stdout.writeln(
    'wrote tuned IxDT blob → $outPath\n'
    '  target=$target  chosen=${chosen.paramLabel}  '
    'recall=${chosen.recall.toStringAsFixed(3)}  '
    'mean_us=${chosen.meanUs.toStringAsFixed(1)}'
    '${fellBack ? "  (fallback: no point hit min_recall=$minRecall)" : ""}',
  );
  return true;
}

class _TuneSpec {
  const _TuneSpec({
    required this.labelPrefix,
    required this.paramName,
    required this.paramPattern,
    required this.pick,
  });
  final String labelPrefix;
  final String paramName;
  final RegExp paramPattern;
  final Index Function(IndexIVFFlat, IndexIVFPQ, IndexRefineFlat, IndexHNSW)
  pick;
}

final Map<String, _TuneSpec> _tuneSpecs = <String, _TuneSpec>{
  'ivfflat': _TuneSpec(
    labelPrefix: 'IVFFlat',
    paramName: 'nprobe',
    paramPattern: RegExp(r'nprobe=\s*(\d+)'),
    pick: (ivf, _, _, _) => ivf,
  ),
  'ivfpq': _TuneSpec(
    labelPrefix: 'IVFPQ    ',
    paramName: 'nprobe',
    paramPattern: RegExp(r'nprobe=\s*(\d+)'),
    pick: (_, ivfpq, _, _) => ivfpq,
  ),
  'ivfpq+rf': _TuneSpec(
    labelPrefix: 'IVFPQ+RF',
    paramName: 'nprobe',
    paramPattern: RegExp(r'nprobe=\s*(\d+)'),
    pick: (_, _, refined, _) => refined,
  ),
  'hnsw': _TuneSpec(
    labelPrefix: 'HNSW',
    paramName: 'efSearch',
    paramPattern: RegExp(r'efSearch=\s*(\d+)'),
    pick: (_, _, _, hnsw) => hnsw,
  ),
};

class _Options {
  _Options({
    required this.nb,
    required this.nq,
    required this.d,
    required this.k,
    required this.seed,
    required this.csvPath,
    required this.mdPath,
    required this.pareto,
    required this.tuneTarget,
    required this.tuneOut,
    required this.tuneRecall,
  });

  final int nb;
  final int nq;
  final int d;
  final int k;
  final int seed;
  final String? csvPath;
  final String? mdPath;
  final bool pareto;
  final String? tuneTarget;
  final String? tuneOut;
  final double tuneRecall;

  static _Options? parse(List<String> args) {
    var nb = 5000;
    var nq = 200;
    var d = 64;
    var k = 10;
    var seed = 0;
    String? csvPath;
    String? mdPath;
    var pareto = false;
    String? tuneTarget;
    String? tuneOut;
    var tuneRecall = 0.9;
    for (var i = 0; i < args.length; i++) {
      final a = args[i];
      String next() {
        if (i + 1 >= args.length) {
          stderr.writeln('bench_faiss: missing value for $a');
          throw _UsageError();
        }
        return args[++i];
      }

      try {
        switch (a) {
          case '-h':
          case '--help':
            stderr.writeln(_usage);
            return null;
          case '--nb':
            nb = int.parse(next());
          case '--nq':
            nq = int.parse(next());
          case '--d':
            d = int.parse(next());
          case '--k':
            k = int.parse(next());
          case '--seed':
            seed = int.parse(next());
          case '--csv':
            csvPath = next();
          case '--md':
            mdPath = next();
          case '--pareto':
            pareto = true;
          case '--tune-target':
            tuneTarget = next().toLowerCase();
          case '--tune-out':
            tuneOut = next();
          case '--tune-recall':
            tuneRecall = double.parse(next());
          default:
            stderr.writeln('bench_faiss: unknown arg $a\n$_usage');
            return null;
        }
      } on _UsageError {
        stderr.writeln(_usage);
        return null;
      }
    }
    if ((tuneTarget == null) != (tuneOut == null)) {
      stderr.writeln(
        'bench_faiss: --tune-target and --tune-out must be used together',
      );
      return null;
    }
    return _Options(
      nb: nb,
      nq: nq,
      d: d,
      k: k,
      seed: seed,
      csvPath: csvPath,
      mdPath: mdPath,
      pareto: pareto,
      tuneTarget: tuneTarget,
      tuneOut: tuneOut,
      tuneRecall: tuneRecall,
    );
  }
}

class _UsageError implements Exception {}

const String _usage =
    'Usage: dart run bin/bench_faiss.dart '
    '[--nb N] [--nq N] [--d D] [--k K] [--seed S] [--csv PATH] '
    '[--md PATH] [--pareto] '
    '[--tune-target ivfflat|ivfpq|ivfpq+rf|hnsw --tune-out PATH '
    '[--tune-recall F]]';
