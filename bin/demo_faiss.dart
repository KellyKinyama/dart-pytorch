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

import 'dart:io';
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
          row[j] = internal >= 0 && internal < ids.length ? ids[internal] : -1;
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

  // Persistence — save every kind of index to a temp file, reload, and
  // verify the top-10 search result is byte-identical.
  print('\nPersistence round-trip:');
  {
    final tmpDir = Directory.systemTemp.createTempSync('faiss_demo_');
    try {
      final samples = <String, Index>{
        'Flat.bin': IndexFlatL2(_d)..add(xb),
        'IVFFlat.bin': (IndexIVFFlat(d: _d, nlist: 64, nprobe: 8)
          ..train(xb)
          ..add(xb)),
        'PQ.bin': (IndexPQ(d: _d, m: 8)
          ..train(xb)
          ..add(xb)),
        'IVFPQ.bin': (IndexIVFPQ(d: _d, nlist: 64, m: 8, nprobe: 8)
          ..train(xb)
          ..add(xb)),
        'HNSW.bin': IndexHNSW(d: _d, M: 16, efConstruction: 100, efSearch: 32)
          ..add(xb),
        'SQ8.bin': (IndexScalarQuantizer(_d)
          ..train(xb)
          ..add(xb)),
      };
      for (final entry in samples.entries) {
        final path = '${tmpDir.path}/${entry.key}';
        final orig = entry.value;
        final blob = writeIndex(orig);
        File(path).writeAsBytesSync(blob);
        final onDisk = File(path).lengthSync();
        final loaded = readIndex(File(path).readAsBytesSync());
        final origResult = orig.search(xq, _k);
        final loadedResult = loaded.search(xq, _k);
        var ok = true;
        for (var qi = 0; qi < _nq && ok; qi++) {
          for (var j = 0; j < _k; j++) {
            if (origResult.ids[qi][j] != loadedResult.ids[qi][j]) {
              ok = false;
              break;
            }
          }
        }
        final sizeKb = (onDisk / 1024).toStringAsFixed(1);
        print(
          '  ${entry.key.padRight(14)}'
          '${sizeKb.padLeft(8)} KiB on disk   '
          'search parity: ${ok ? "OK" : "MISMATCH"}',
        );
      }
    } finally {
      tmpDir.deleteSync(recursive: true);
    }
  }

  // Range search — how many db vectors sit within radius `r` of each query.
  print('\nRange search (L2):');
  {
    // Pick a radius so that we typically capture a handful of neighbours.
    // The mean k=10 distance from the flat search gives a decent scale.
    var meanTopK = 0.0;
    for (var qi = 0; qi < _nq; qi++) {
      meanTopK += truth.distances[qi][_k - 1];
    }
    meanTopK /= _nq;
    final radius = meanTopK; // catch ≈ top-10 per query on average.

    void report(String name, RangeSearchResult r, Duration wall) {
      var total = 0;
      var minHits = 1 << 30;
      var maxHits = 0;
      for (var qi = 0; qi < _nq; qi++) {
        final n = r.lengthFor(qi);
        total += n;
        if (n < minHits) minHits = n;
        if (n > maxHits) maxHits = n;
      }
      final avg = total / _nq;
      final us = wall.inMicroseconds / _nq;
      print(
        '  ${name.padRight(20)}'
        'r=${radius.toStringAsFixed(3)}   '
        'hits/q: avg ${avg.toStringAsFixed(1)} min $minHits max $maxHits   '
        '${us.toStringAsFixed(1).padLeft(6)} µs/q',
      );
    }

    final flatRs = _timedRange(flat, xq, radius);
    report('IndexFlatL2', flatRs.result, flatRs.wall);

    final ivf = IndexIVFFlat(d: _d, nlist: 64, nprobe: 8)
      ..train(xb)
      ..add(xb);
    final ivfRs = _timedRange(ivf, xq, radius);
    report('IVFFlat nprobe=8', ivfRs.result, ivfRs.wall);

    final sq = IndexScalarQuantizer(_d)
      ..train(xb)
      ..add(xb);
    final sqRs = _timedRange(sq, xq, radius);
    report('SQ8', sqRs.result, sqRs.wall);
  }

  // removeIds — delete a slice, verify count and searchability.
  print('\nremoveIds:');
  {
    final ivf = IndexIVFFlat(d: _d, nlist: 64, nprobe: 8)
      ..train(xb)
      ..add(xb);
    print('  before  ntotal=${ivf.ntotal}');
    final toRemove = <int>{
      for (var i = 0; i < 1000; i++) i * 3, // 1000 evenly-spaced ids
    };
    sw = Stopwatch()..start();
    final removed = ivf.removeIds(toRemove);
    sw.stop();
    print(
      '  removed $removed vectors in ${sw.elapsed.inMilliseconds} ms   '
      'now ntotal=${ivf.ntotal}',
    );
    // Post-delete search: pick a query and confirm we still get k=10 hits.
    final r = ivf.search(xq.sublist(0, 5), _k);
    var minHits = 1 << 30;
    for (var qi = 0; qi < 5; qi++) {
      var n = 0;
      for (var j = 0; j < _k; j++) {
        if (r.ids[qi][j] >= 0) n++;
      }
      if (n < minHits) minHits = n;
    }
    print('  post-delete search: min $minHits/${_k} hits across 5 queries');
  }

  // Binary + LSH — Hamming-distance search over bit-packed vectors.
  print('\nBinary + LSH:');
  {
    // (1) IndexBinaryFlat over random 64-bit codes: exact Hamming
    //     ground truth, self-search should return distance 0.
    const codeSize = 8;
    final rngB = math.Random(11);
    final bcodes = List<Uint8List>.generate(_nb, (_) {
      final c = Uint8List(codeSize);
      for (var j = 0; j < codeSize; j++) {
        c[j] = rngB.nextInt(256);
      }
      return c;
    });
    final bqueries = bcodes.sublist(0, _nq);
    final bf = IndexBinaryFlat(codeSize)..add(bcodes);
    sw = Stopwatch()..start();
    final br = bf.search(bqueries, _k);
    sw.stop();
    var bhit = 0;
    for (var qi = 0; qi < _nq; qi++) {
      if (br.ids[qi][0] == qi) bhit++;
    }
    print(
      '  BinaryFlat ${codeSize * 8}-bit   '
      'search ${(sw.elapsedMicroseconds / _nq).toStringAsFixed(1)} µs/q   '
      'self top-1 ${((bhit / _nq) * 100).toStringAsFixed(1)} %',
    );

    // (2) IndexLSH over the float corpus — random-projection
    //     binarization; recall grows with nbits.
    for (final nbits in [64, 128, 256, 512]) {
      final lsh = IndexLSH(d: _d, nbits: nbits)..add(xb);
      sw = Stopwatch()..start();
      final r = lsh.search(xq, _k);
      sw.stop();
      final recall = _recallAtK(r, truth, _k);
      print(
        '  LSH nbits=${nbits.toString().padLeft(3)}      '
        'search ${(sw.elapsedMicroseconds / _nq).toStringAsFixed(1)} µs/q   '
        'recall@10 ${(recall * 100).toStringAsFixed(1)} %   '
        'bytes/vec ${(nbits + 7) ~/ 8}',
      );
    }
  }

  // Pre-transforms — cheap preprocessors applied via IndexPreTransform.
  print('\nPre-transforms:');
  {
    // Rotate then L2-normalize before feeding a Flat inner index.
    final wrapped = IndexPreTransform(
      chain: [
        RandomRotationTransform(d: _d, seed: 7),
        L2NormTransform(_d),
      ],
      inner: IndexFlatL2(_d),
    )..add(xb);
    final wrappedTimed = _timedSearch(wrapped, xq, _k);
    print(
      '  RandomRot+L2Norm→Flat  '
      'search ${(wrappedTimed.wall.inMicroseconds / _nq).toStringAsFixed(1)} µs/q   '
      'ntotal=${wrapped.ntotal}',
    );

    // Demonstrate the L2 ≡ IP-on-sphere identity: L2Norm chained onto
    // IndexFlatIP yields the same top-k ids as a pure IndexFlatL2 on
    // L2-normalized data.
    final l2 = IndexFlatL2(_d)..add(L2NormTransform(_d).apply(xb));
    final ipWrap = IndexPreTransform(
      chain: [L2NormTransform(_d)],
      inner: IndexFlatIP(_d),
    )..add(xb);
    final rA = l2.search(L2NormTransform(_d).apply(xq), _k);
    final rB = ipWrap.search(xq, _k);
    var identical = true;
    for (var qi = 0; qi < _nq && identical; qi++) {
      for (var j = 0; j < _k; j++) {
        if (rA.ids[qi][j] != rB.ids[qi][j]) {
          identical = false;
          break;
        }
      }
    }
    print(
      '  L2 ≡ IP on unit sphere: L2Norm+FlatIP vs FlatL2 top-$_k match: '
      '${identical ? "OK" : "MISMATCH"}',
    );

    // Persistence round-trip.
    final bytes = writeIndex(wrapped);
    final loaded = readIndex(bytes) as IndexPreTransform;
    final origRes = wrapped.search(xq.sublist(0, 5), _k);
    final loadRes = loaded.search(xq.sublist(0, 5), _k);
    var parity = true;
    for (var qi = 0; qi < 5 && parity; qi++) {
      for (var j = 0; j < _k; j++) {
        if (origRes.ids[qi][j] != loadRes.ids[qi][j]) {
          parity = false;
          break;
        }
      }
    }
    print(
      '  Persistence            ${(bytes.length / 1024).toStringAsFixed(1)} KiB   '
      'chain=${loaded.chain.length}   parity: ${parity ? "OK" : "MISMATCH"}',
    );
  }

  print(
    '\nDone. Larger corpora and higher-recall settings tighten the picture.',
  );
}

({Duration wall, RangeSearchResult result}) _timedRange(
  Index idx,
  List<Float32List> queries,
  double radius,
) {
  final sw = Stopwatch()..start();
  final r = idx.rangeSearch(queries, radius);
  sw.stop();
  return (wall: sw.elapsed, result: r);
}
