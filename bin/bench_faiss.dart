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
///   `--nb N`    database size (default 5000)
///   `--nq N`    query count (default 200)
///   `--d  D`    dimensionality (default 64)
///   `--k  K`    neighbours per query (default 10)
///   `--seed S`  RNG seed (default 0)
///   `--csv P`   append CSV to path P (writes header if new)
///   `--md  P`   write Markdown table to path P (overwrites)
///   `--pareto`  restrict CSV / Markdown output to the Pareto
///               frontier (recall-vs-latency)
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
  });

  final int nb;
  final int nq;
  final int d;
  final int k;
  final int seed;
  final String? csvPath;
  final String? mdPath;
  final bool pareto;

  static _Options? parse(List<String> args) {
    var nb = 5000;
    var nq = 200;
    var d = 64;
    var k = 10;
    var seed = 0;
    String? csvPath;
    String? mdPath;
    var pareto = false;
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
          default:
            stderr.writeln('bench_faiss: unknown arg $a\n$_usage');
            return null;
        }
      } on _UsageError {
        stderr.writeln(_usage);
        return null;
      }
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
    );
  }
}

class _UsageError implements Exception {}

const String _usage =
    'Usage: dart run bin/bench_faiss.dart '
    '[--nb N] [--nq N] [--d D] [--k K] [--seed S] [--csv PATH] '
    '[--md PATH] [--pareto]';
