/// Index-type benchmark: same vectors, same queries, every index type.
///
/// Chapters 3-8 of docs/vectors explain **why** ANN indexes exist —
/// Flat is O(N·d) per query and doesn't scale, PQ compresses the
/// storage, IVF partitions the search, HNSW gives graph-based logarithmic
/// probes. This demo turns that theory into one table so the tradeoff
/// becomes concrete:
///
/// ```
///   Index               build_ms   qps   us/query   recall@10  bytes/vec
///   ------------------------------------------------------------------
///   IndexFlatL2 (GT)      ~0        ~2k  ~500       1.000      512
///   IndexScalarQuantizer  ...       ...  ...        ~0.99      128
///   IndexHNSW (M=16)      ...       ...  ...        ~0.98      512+graph
///   IndexIVFFlat          ...       ...  ...        ~0.95      512
///   IndexIVFPQ            ...       ...  ...        ~0.75      m bytes
///   IndexPQ               ...       ...  ...        ~0.65      m bytes
/// ```
///
/// The demo uses **synthetic Gaussian vectors** rather than real
/// embeddings so it runs in seconds regardless of hardware. What matters
/// for the ANN tradeoff is `(N, d)` and the intrinsic clusterability of
/// the data, not what the vectors "mean". Ground truth is the exact
/// top-k from `IndexFlatL2`; recall@k is the fraction of that ground-
/// truth set each ANN index recovers.
///
/// Tunable knobs (see `_help`): `--n`, `--d`, `--nq`, `--k`, `--seed`.
///
/// Run:
///
/// ```sh
///   dart run bin/vector_index_benchmark_demo.dart
/// ```
///
/// Larger sweep:
///
/// ```sh
///   dart run bin/vector_index_benchmark_demo.dart --n 20000 --d 256 --nq 500
/// ```
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

class _Opts {
  _Opts({
    required this.n,
    required this.d,
    required this.nq,
    required this.k,
    required this.seed,
  });
  final int n;
  final int d;
  final int nq;
  final int k;
  final int seed;
}

_Opts _parseArgs(List<String> args) {
  var n = 5000;
  var d = 128;
  var nq = 200;
  var k = 10;
  var seed = 1234;
  for (var i = 0; i < args.length; i++) {
    final a = args[i];
    String next() {
      if (i + 1 >= args.length) {
        stderr.writeln('missing value for $a');
        exit(64);
      }
      return args[++i];
    }

    switch (a) {
      case '--n':
        n = int.parse(next());
        break;
      case '--d':
        d = int.parse(next());
        break;
      case '--nq':
        nq = int.parse(next());
        break;
      case '--k':
        k = int.parse(next());
        break;
      case '--seed':
        seed = int.parse(next());
        break;
      case '-h':
      case '--help':
        stdout.writeln(_help);
        exit(0);
      default:
        stderr.writeln('unknown arg: $a');
        stderr.writeln(_help);
        exit(64);
    }
  }
  return _Opts(n: n, d: d, nq: nq, k: k, seed: seed);
}

const String _help = '''
Index-type benchmark on synthetic Gaussian vectors.

Usage:
  dart run bin/vector_index_benchmark_demo.dart [flags]

Flags:
  --n N       number of database vectors (default: 5000)
  --d D       vector dimension           (default: 128)
  --nq NQ     number of queries          (default: 200)
  --k K       top-k                      (default: 10)
  --seed S    RNG seed                   (default: 1234)
  -h, --help  print this message
''';

// ---------------------------------------------------------------------------

void main(List<String> args) {
  final opts = _parseArgs(args);
  stdout.writeln(
    'Benchmark: N=${opts.n} d=${opts.d} nq=${opts.nq} k=${opts.k}',
  );

  // ---- 1. Generate synthetic data --------------------------------

  final rng = math.Random(opts.seed);
  final xb = List<Float32List>.generate(opts.n, (_) {
    final v = Float32List(opts.d);
    for (var j = 0; j < opts.d; j++) {
      v[j] = _gaussian(rng);
    }
    return v;
  });
  final xq = List<Float32List>.generate(opts.nq, (_) {
    final v = Float32List(opts.d);
    for (var j = 0; j < opts.d; j++) {
      v[j] = _gaussian(rng);
    }
    return v;
  });

  // ---- 2. Ground truth via exact IndexFlatL2 --------------------

  final gtSw = Stopwatch()..start();
  final gtIdx = IndexFlatL2(opts.d)..add(xb);
  final gtRes = gtIdx.search(xq, opts.k);
  gtSw.stop();

  final gtSets = List<Set<int>>.generate(
    opts.nq,
    (qi) => gtRes.ids[qi].toSet(),
  );

  // ---- 3. Benchmark every index type ----------------------------

  final rows = <_Row>[];

  // 3a. IndexFlatL2 (ground truth timing — for the row)
  rows.add(
    _benchmark(
      name: 'IndexFlatL2 (baseline)',
      build: () => IndexFlatL2(opts.d)..add(xb),
      xq: xq,
      k: opts.k,
      gtSets: gtSets,
      bytesPerVec: 4 * opts.d,
    ),
  );

  // 3b. IndexScalarQuantizer (8-bit per dim, still exact search)
  rows.add(
    _benchmark(
      name: 'IndexScalarQuantizer',
      build: () {
        final idx = IndexScalarQuantizer(opts.d);
        idx.train(xb);
        idx.add(xb);
        return idx;
      },
      xq: xq,
      k: opts.k,
      gtSets: gtSets,
      bytesPerVec: opts.d, // 1 byte per dim
    ),
  );

  // 3c. IndexHNSW (graph, M=16, efSearch=32)
  rows.add(
    _benchmark(
      name: 'IndexHNSW (M=16, efSearch=32)',
      build: () {
        final idx = IndexHNSW(d: opts.d, M: 16, efSearch: 32);
        idx.add(xb);
        return idx;
      },
      xq: xq,
      k: opts.k,
      gtSets: gtSets,
      bytesPerVec: 4 * opts.d + 16 * 4 * 3, // approx: vec + ~3 layers × M ints
    ),
  );

  // 3d. IndexIVFFlat (partitioned exact, nlist ≈ sqrt(N), nprobe=8)
  final nlist = math.max(4, math.sqrt(opts.n).round());
  rows.add(
    _benchmark(
      name: 'IndexIVFFlat (nlist=$nlist, nprobe=8)',
      build: () {
        final idx = IndexIVFFlat(d: opts.d, nlist: nlist);
        idx.nprobe = math.min(8, nlist);
        idx.train(xb);
        idx.add(xb);
        return idx;
      },
      xq: xq,
      k: opts.k,
      gtSets: gtSets,
      bytesPerVec: 4 * opts.d,
    ),
  );

  // 3e. IndexIVFPQ (partitioned compressed, m=8, nprobe=8)
  final m = _pickM(opts.d);
  rows.add(
    _benchmark(
      name: 'IndexIVFPQ (nlist=$nlist, m=$m, nprobe=8)',
      build: () {
        final idx = IndexIVFPQ(d: opts.d, nlist: nlist, m: m);
        idx.nprobe = math.min(8, nlist);
        idx.train(xb);
        idx.add(xb);
        return idx;
      },
      xq: xq,
      k: opts.k,
      gtSets: gtSets,
      bytesPerVec: m, // m bytes per code
    ),
  );

  // 3f. IndexPQ (pure PQ, no IVF)
  rows.add(
    _benchmark(
      name: 'IndexPQ (m=$m)',
      build: () {
        final idx = IndexPQ(d: opts.d, m: m);
        idx.train(xb);
        idx.add(xb);
        return idx;
      },
      xq: xq,
      k: opts.k,
      gtSets: gtSets,
      bytesPerVec: m,
    ),
  );

  // ---- 4. Print table ------------------------------------------

  stdout.writeln('\nGround-truth (IndexFlatL2) search time: '
      '${gtSw.elapsedMilliseconds} ms for ${opts.nq} queries.');

  stdout.writeln('\n${'Index'.padRight(38)}  '
      '${'build_ms'.padLeft(9)}  '
      '${'us/query'.padLeft(9)}  '
      '${'recall@${opts.k}'.padLeft(10)}  '
      '${'bytes/vec'.padLeft(10)}');
  stdout.writeln('${'-' * 38}  '
      '${'-' * 9}  ${'-' * 9}  ${'-' * 10}  ${'-' * 10}');
  for (final row in rows) {
    stdout.writeln(
      '${row.name.padRight(38)}  '
      '${row.buildMs.toString().padLeft(9)}  '
      '${row.usPerQuery.toStringAsFixed(0).padLeft(9)}  '
      '${row.recall.toStringAsFixed(3).padLeft(10)}  '
      '${row.bytesPerVec.toString().padLeft(10)}',
    );
  }

  stdout.writeln('\nNotes:');
  stdout.writeln('  * "bytes/vec" is rough — HNSW also stores graph edges;');
  stdout.writeln('    IVF variants add centroid + inverted-list overhead.');
  stdout.writeln('  * Recall is measured against IndexFlatL2 (exact).');
  stdout.writeln('  * Random Gaussian data is the worst case for ANN — real');
  stdout.writeln('    embeddings have clustered structure and score higher.');
}

class _Row {
  _Row({
    required this.name,
    required this.buildMs,
    required this.usPerQuery,
    required this.recall,
    required this.bytesPerVec,
  });
  final String name;
  final int buildMs;
  final double usPerQuery;
  final double recall;
  final int bytesPerVec;
}

_Row _benchmark({
  required String name,
  required Index Function() build,
  required List<Float32List> xq,
  required int k,
  required List<Set<int>> gtSets,
  required int bytesPerVec,
}) {
  stdout.writeln('\nBuilding $name ...');
  final buildSw = Stopwatch()..start();
  final idx = build();
  buildSw.stop();

  final searchSw = Stopwatch()..start();
  final res = idx.search(xq, k);
  searchSw.stop();

  var hits = 0;
  var totalGt = 0;
  for (var qi = 0; qi < xq.length; qi++) {
    final row = res.ids[qi];
    for (final id in row) {
      if (gtSets[qi].contains(id)) hits++;
    }
    totalGt += gtSets[qi].length;
  }
  final recall = totalGt == 0 ? 0.0 : hits / totalGt;
  final usPerQ = searchSw.elapsedMicroseconds / xq.length;
  stdout.writeln(
    '  built in ${buildSw.elapsedMilliseconds} ms, '
    'searched ${xq.length}q in ${searchSw.elapsedMilliseconds} ms '
    '(${usPerQ.toStringAsFixed(0)} µs/q), recall@$k = '
    '${recall.toStringAsFixed(3)}',
  );
  return _Row(
    name: name,
    buildMs: buildSw.elapsedMilliseconds,
    usPerQuery: usPerQ,
    recall: recall,
    bytesPerVec: bytesPerVec,
  );
}

// Pick an `m` (# PQ subquantizers) that divides `d` evenly and gives
// reasonable compression. Prefer m=8 when d is divisible by 8.
int _pickM(int d) {
  for (final cand in [8, 4, 16, 2]) {
    if (d % cand == 0) return cand;
  }
  return 1;
}

// Box-Muller.
double _gaussian(math.Random rng) {
  final u1 = 1.0 - rng.nextDouble();
  final u2 = 1.0 - rng.nextDouble();
  return math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2);
}
