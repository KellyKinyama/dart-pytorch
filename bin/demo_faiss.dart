/// Dart port of the FAISS getting-started demo (extended with recall
/// vs. speed comparisons for every index type).
///
/// Generates a 5000×64 uniformly-random database with a small first-
/// axis smear so nearby ids cluster in space, then queries with the
/// first 100 database vectors. Reports recall@10 and per-query latency
/// against the [IndexFlatL2] ground truth.
///
/// Run:
///   dart run bin/demo_faiss.dart

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

const int _d = 64;
const int _nb = 5000;
const int _nq = 100;
const int _k = 10;

List<Float32List> _makeData(int n, {required int seed}) {
  final rng = math.Random(seed);
  return List<Float32List>.generate(n, (i) {
    final v = Float32List(_d);
    for (var j = 0; j < _d; j++) {
      v[j] = rng.nextDouble();
    }
    v[0] += i / 1000.0; // first-axis smear (mirrors the FAISS tutorial)
    return v;
  });
}

double _recallAtK(SearchResult r, SearchResult truth, int k) {
  var hit = 0;
  var total = 0;
  for (var qi = 0; qi < r.nq; qi++) {
    final gold = truth.ids[qi].sublist(0, k).toSet();
    for (var j = 0; j < k; j++) {
      if (gold.contains(r.ids[qi][j])) hit++;
    }
    total += k;
  }
  return hit / total;
}

({Duration wall, SearchResult result}) _timedSearch(
  Index idx,
  List<Float32List> queries,
  int k,
) {
  final sw = Stopwatch()..start();
  final r = idx.search(queries, k);
  sw.stop();
  return (wall: sw.elapsed, result: r);
}

void _report(String name, Duration build, Duration search, double recall) {
  final avgUs = search.inMicroseconds / _nq;
  print(
    '  ${name.padRight(20)}'
    'build ${build.inMilliseconds.toString().padLeft(5)} ms   '
    'search ${avgUs.toStringAsFixed(1).padLeft(7)} µs/q   '
    'recall@$_k ${(recall * 100).toStringAsFixed(1)}%',
  );
}

void main() {
  print('=== Dart FAISS demo — d=$_d  nb=$_nb  nq=$_nq  k=$_k ===\n');

  final xb = _makeData(_nb, seed: 1234);
  final xq = xb.sublist(0, _nq); // sanity: querying with the first nb vectors

  print('Ground truth: IndexFlatL2 brute force');
  final flat = IndexFlatL2(_d);
  var sw = Stopwatch()..start();
  flat.add(xb);
  final flatBuild = sw.elapsed;
  sw.stop();
  final truthTimed = _timedSearch(flat, xq, _k);
  final truth = truthTimed.result;
  _report('IndexFlatL2', flatBuild, truthTimed.wall, 1.0);

  // Sanity check per the FAISS getting-started tutorial: self-query
  // returns the vector itself at distance 0.
  print(
    '\n  sanity: query 0 → id ${truth.ids[0][0]}, '
    'dist ${truth.distances[0][0].toStringAsFixed(4)}',
  );
  print(
    '          query 1 → id ${truth.ids[1][0]}, '
    'dist ${truth.distances[1][0].toStringAsFixed(4)}',
  );

  print('\nApproximate indexes:');

  // IVFFlat
  {
    final ivf = IndexIVFFlat(d: _d, nlist: 64, nprobe: 8);
    sw = Stopwatch()..start();
    ivf.train(xb);
    ivf.add(xb);
    final buildT = sw.elapsed;
    sw.stop();
    final t = _timedSearch(ivf, xq, _k);
    _report(
      'IVFFlat nprobe=8',
      buildT,
      t.wall,
      _recallAtK(t.result, truth, _k),
    );
  }
  {
    final ivf = IndexIVFFlat(d: _d, nlist: 64, nprobe: 1);
    sw = Stopwatch()..start();
    ivf.train(xb);
    ivf.add(xb);
    final buildT = sw.elapsed;
    sw.stop();
    final t = _timedSearch(ivf, xq, _k);
    _report(
      'IVFFlat nprobe=1',
      buildT,
      t.wall,
      _recallAtK(t.result, truth, _k),
    );
  }

  // PQ
  {
    final pq = IndexPQ(d: _d, m: 8);
    sw = Stopwatch()..start();
    pq.train(xb);
    pq.add(xb);
    final buildT = sw.elapsed;
    sw.stop();
    final t = _timedSearch(pq, xq, _k);
    _report('PQ m=8', buildT, t.wall, _recallAtK(t.result, truth, _k));
  }

  // IVFPQ
  {
    final ivfpq = IndexIVFPQ(d: _d, nlist: 64, m: 8, nprobe: 8);
    sw = Stopwatch()..start();
    ivfpq.train(xb);
    ivfpq.add(xb);
    final buildT = sw.elapsed;
    sw.stop();
    final t = _timedSearch(ivfpq, xq, _k);
    _report('IVFPQ nprobe=8', buildT, t.wall, _recallAtK(t.result, truth, _k));
  }

  // HNSW
  {
    final hnsw = IndexHNSW(d: _d, M: 16, efConstruction: 100, efSearch: 32);
    sw = Stopwatch()..start();
    hnsw.add(xb);
    final buildT = sw.elapsed;
    sw.stop();
    final t = _timedSearch(hnsw, xq, _k);
    _report('HNSW ef=32', buildT, t.wall, _recallAtK(t.result, truth, _k));
  }
  {
    final hnsw = IndexHNSW(d: _d, M: 16, efConstruction: 100, efSearch: 128);
    sw = Stopwatch()..start();
    hnsw.add(xb);
    final buildT = sw.elapsed;
    sw.stop();
    final t = _timedSearch(hnsw, xq, _k);
    _report('HNSW ef=128', buildT, t.wall, _recallAtK(t.result, truth, _k));
  }

  // Scalar Quantizer — 4× compression versus flat, tiny recall loss.
  {
    final sq = IndexScalarQuantizer(_d);
    sw = Stopwatch()..start();
    sq.train(xb);
    sq.add(xb);
    final buildT = sw.elapsed;
    sw.stop();
    final t = _timedSearch(sq, xq, _k);
    _report('SQ8', buildT, t.wall, _recallAtK(t.result, truth, _k));
  }

  // Refine-Flat wrapper: IVFPQ candidates rescored by exact fp32.
  {
    final refined = IndexRefineFlat(
      IndexIVFPQ(d: _d, nlist: 64, m: 8, nprobe: 8),
      kFactor: 8,
    );
    sw = Stopwatch()..start();
    refined.train(xb);
    refined.add(xb);
    final buildT = sw.elapsed;
    sw.stop();
    final t = _timedSearch(refined, xq, _k);
    _report(
      'IVFPQ+RefineFlat',
      buildT,
      t.wall,
      _recallAtK(t.result, truth, _k),
    );
  }

  // IDMap around HNSW — round-trip custom int32 ids.
  {
    final ids = List<int>.generate(_nb, (i) => 100000 + i * 3);
    final idmap = IndexIDMap(
      IndexHNSW(d: _d, M: 16, efConstruction: 100, efSearch: 64),
    );
    sw = Stopwatch()..start();
    idmap.addWithIds(xb, ids);
    final buildT = sw.elapsed;
    sw.stop();
    final t = _timedSearch(idmap, xq, _k);
    // Sanity: first result for query 0 should be the id we assigned to xb[0].
    print(
      '  IDMap sanity: query 0 → id ${t.result.ids[0][0]} '
      '(expected ${ids[0]})',
    );
    // Recall in external-id space: translate truth to external ids first.
    final translatedTruth = SearchResult(
      truth.distances,
      List<Int32List>.generate(truth.nq, (qi) {
        final row = Int32List(truth.k);
        for (var j = 0; j < truth.k; j++) {
          final internal = truth.ids[qi][j];
          row[j] = internal >= 0 && internal < ids.length
              ? ids[internal]
              : -1;
        }
        return row;
      }),
    );
    _report(
      'IDMap(HNSW)',
      buildT,
      t.wall,
      _recallAtK(t.result, translatedTruth, _k),
    );
  }

  print(
    '\nDone. Larger corpora and higher-recall settings tighten the picture.',
  );
}
