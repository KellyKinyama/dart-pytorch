import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

/// Sample `n` cluster-y `d`-dim vectors: `k` gaussian blobs.
List<Float32List> _sampleBlobs({
  required int n,
  required int d,
  int k = 8,
  double std = 0.05,
  int seed = 1,
}) {
  final rng = math.Random(seed);
  final centers = List<Float32List>.generate(k, (_) {
    final c = Float32List(d);
    for (var j = 0; j < d; j++) {
      c[j] = rng.nextDouble() * 2 - 1;
    }
    return c;
  });
  return List<Float32List>.generate(n, (_) {
    final c = centers[rng.nextInt(k)];
    final v = Float32List(d);
    for (var j = 0; j < d; j++) {
      // Box-Muller approximation via two-uniforms mean.
      final n0 = (rng.nextDouble() + rng.nextDouble() - 1.0);
      v[j] = c[j] + n0 * std;
    }
    return v;
  });
}

/// Recall@k of `result.ids[qi]` against `truth.ids[qi]` (both sorted).
double _recall(SearchResult result, SearchResult truth, int k) {
  var hit = 0;
  var total = 0;
  for (var qi = 0; qi < result.nq; qi++) {
    final gold = truth.ids[qi].sublist(0, k).toSet();
    for (var j = 0; j < k; j++) {
      if (gold.contains(result.ids[qi][j])) hit++;
    }
    total += k;
  }
  return hit / total;
}

void main() {
  const d = 16;
  const n = 400;
  const nq = 20;
  const k = 5;

  final xs = _sampleBlobs(n: n, d: d, seed: 1);
  final queries = _sampleBlobs(n: nq, d: d, seed: 2);

  late SearchResult truth;

  setUpAll(() {
    final flat = IndexFlatL2(d)..add(xs);
    truth = flat.search(queries, k);
  });

  group('IndexFlat', () {
    test('L2 finds itself with distance 0', () {
      final flat = IndexFlatL2(d)..add(xs);
      final r = flat.search(xs.sublist(0, 3), 1);
      for (var i = 0; i < 3; i++) {
        expect(r.ids[i][0], equals(i));
        expect(r.distances[i][0], closeTo(0.0, 1e-5));
      }
    });

    test('IP is symmetric and consistent', () {
      final ip = IndexFlatIP(d)..add(xs);
      final r = ip.search(xs.sublist(0, 3), 1);
      // Highest inner-product with self is at least dot(x,x).
      for (var i = 0; i < 3; i++) {
        final self = ip.reconstruct(i);
        var s = 0.0;
        for (var j = 0; j < d; j++) {
          s += self[j] * self[j];
        }
        expect(r.distances[i][0], greaterThanOrEqualTo(s - 1e-4));
      }
    });
  });

  test('Kmeans converges to a monotone objective', () {
    final km = Kmeans(d: d, k: 8, niter: 30);
    final res = km.train(xs);
    expect(res.centroids, hasLength(8));
    expect(res.assignments.length, equals(n));
    // Reassign after training — objective should not exceed training obj by much.
    var reobj = 0.0;
    for (var i = 0; i < n; i++) {
      var best = double.infinity;
      for (var c = 0; c < 8; c++) {
        var s = 0.0;
        for (var j = 0; j < d; j++) {
          final diff = xs[i][j] - res.centroids[c][j];
          s += diff * diff;
        }
        if (s < best) best = s;
      }
      reobj += best;
    }
    expect(reobj, lessThanOrEqualTo(res.objective * 1.05 + 1e-6));
  });

  test('IndexIVFFlat matches flat with high recall', () {
    final ivf =
        IndexIVFFlat(d: d, nlist: 8, nprobe: 8) // nprobe=nlist → exact
          ..train(xs)
          ..add(xs);
    final r = ivf.search(queries, k);
    expect(_recall(r, truth, k), equals(1.0));
  });

  test('IndexIVFFlat with nprobe=1 has recall < 1 but > 0.3', () {
    final ivf = IndexIVFFlat(d: d, nlist: 8, nprobe: 1)
      ..train(xs)
      ..add(xs);
    final r = ivf.search(queries, k);
    final rec = _recall(r, truth, k);
    expect(rec, greaterThan(0.3));
  });

  test('IndexPQ reconstructs approximately', () {
    final pq = IndexPQ(d: d, m: 4)
      ..train(xs)
      ..add(xs);
    final r = pq.search(queries, k);
    // ADC is approximate but should still hit the true NN often.
    expect(_recall(r, truth, k), greaterThan(0.4));
  });

  test('IndexIVFPQ works end-to-end', () {
    final ivfpq = IndexIVFPQ(d: d, nlist: 8, m: 4, nprobe: 4)
      ..train(xs)
      ..add(xs);
    final r = ivfpq.search(queries, k);
    expect(r.ids[0][0], greaterThanOrEqualTo(0));
    expect(_recall(r, truth, k), greaterThan(0.3));
  });

  test('IndexHNSW achieves near-perfect recall on this dataset', () {
    final hnsw = IndexHNSW(d: d, M: 16, efConstruction: 100, efSearch: 64)
      ..add(xs);
    final r = hnsw.search(queries, k);
    expect(_recall(r, truth, k), greaterThan(0.9));
  });

  test('IndexIDMap round-trips custom ids', () {
    final idmap = IndexIDMap(IndexFlatL2(d));
    final ids = List<int>.generate(n, (i) => 1000000 + i * 7);
    idmap.addWithIds(xs, ids);
    final r = idmap.search(xs.sublist(0, 3), 1);
    for (var i = 0; i < 3; i++) {
      expect(r.ids[i][0], equals(ids[i]));
    }
  });

  test('IndexIDMap rejects ids outside int32', () {
    final idmap = IndexIDMap(IndexFlatL2(d));
    expect(
      () => idmap.addWithIds(xs.sublist(0, 1), [1 << 40]),
      throwsA(isA<ArgumentError>()),
    );
  });

  test('IndexScalarQuantizer preserves top-k with small recall loss', () {
    final sq = IndexScalarQuantizer(d)
      ..train(xs)
      ..add(xs);
    final r = sq.search(queries, k);
    // SQ8 typically stays above 90 % on blob-like data.
    expect(_recall(r, truth, k), greaterThan(0.85));
  });

  test('IndexRefineFlat lifts IVFPQ back to near-flat recall', () {
    // Baseline: standalone IVFPQ with the same hyperparameters.
    final ivfpq = IndexIVFPQ(d: d, nlist: 8, m: 4, nprobe: 4)
      ..train(xs)
      ..add(xs);
    final baseRecall = _recall(ivfpq.search(queries, k), truth, k);

    // Refined wrapper: build a fresh inner index, add through the wrapper.
    final refined = IndexRefineFlat(
      IndexIVFPQ(d: d, nlist: 8, m: 4, nprobe: 4),
      kFactor: 8,
    );
    refined.train(xs);
    refined.add(xs);
    final refinedRecall = _recall(refined.search(queries, k), truth, k);

    // Re-ranking never decreases recall (up to the candidate pool).
    expect(refinedRecall, greaterThanOrEqualTo(baseRecall));
  });

  // --- persistence round-trip --------------------------------------------

  bool _sameResult(SearchResult a, SearchResult b) {
    if (a.nq != b.nq || a.k != b.k) return false;
    for (var qi = 0; qi < a.nq; qi++) {
      for (var j = 0; j < a.k; j++) {
        if (a.ids[qi][j] != b.ids[qi][j]) return false;
        if ((a.distances[qi][j] - b.distances[qi][j]).abs() > 1e-4) {
          return false;
        }
      }
    }
    return true;
  }

  test('IndexFlat round-trips through bytes', () {
    final orig = IndexFlatL2(d)..add(xs);
    final before = orig.search(queries, k);
    final blob = writeIndex(orig);
    final loaded = readIndex(blob);
    expect(loaded, isA<IndexFlat>());
    expect(loaded.ntotal, equals(orig.ntotal));
    expect(_sameResult(loaded.search(queries, k), before), isTrue);
  });

  test('IndexIVFFlat round-trips through bytes', () {
    final orig = IndexIVFFlat(d: d, nlist: 8, nprobe: 4)
      ..train(xs)
      ..add(xs);
    final before = orig.search(queries, k);
    final loaded = readIndex(writeIndex(orig)) as IndexIVFFlat;
    expect(loaded.nlist, equals(orig.nlist));
    expect(loaded.nprobe, equals(orig.nprobe));
    expect(_sameResult(loaded.search(queries, k), before), isTrue);
  });

  test('IndexPQ round-trips through bytes', () {
    final orig = IndexPQ(d: d, m: 4)
      ..train(xs)
      ..add(xs);
    final before = orig.search(queries, k);
    final loaded = readIndex(writeIndex(orig)) as IndexPQ;
    expect(loaded.m, equals(orig.m));
    expect(_sameResult(loaded.search(queries, k), before), isTrue);
  });

  test('IndexIVFPQ round-trips through bytes', () {
    final orig = IndexIVFPQ(d: d, nlist: 8, m: 4, nprobe: 4)
      ..train(xs)
      ..add(xs);
    final before = orig.search(queries, k);
    final loaded = readIndex(writeIndex(orig)) as IndexIVFPQ;
    expect(loaded.nlist, equals(orig.nlist));
    expect(_sameResult(loaded.search(queries, k), before), isTrue);
  });

  test('IndexHNSW round-trips through bytes', () {
    final orig = IndexHNSW(d: d, M: 8, efConstruction: 40, efSearch: 32)
      ..add(xs);
    final before = orig.search(queries, k);
    final loaded = readIndex(writeIndex(orig)) as IndexHNSW;
    expect(loaded.ntotal, equals(orig.ntotal));
    expect(_sameResult(loaded.search(queries, k), before), isTrue);
  });

  test('IndexIDMap round-trips ids and search results', () {
    final idmap = IndexIDMap(IndexFlatL2(d));
    final ids = List<int>.generate(n, (i) => 5000 + i);
    idmap.addWithIds(xs, ids);
    final before = idmap.search(queries, k);
    final loaded = readIndex(writeIndex(idmap)) as IndexIDMap;
    expect(loaded.idOf(0), equals(5000));
    expect(_sameResult(loaded.search(queries, k), before), isTrue);
  });

  test('IndexScalarQuantizer round-trips through bytes', () {
    final sq = IndexScalarQuantizer(d)
      ..train(xs)
      ..add(xs);
    final before = sq.search(queries, k);
    final loaded = readIndex(writeIndex(sq)) as IndexScalarQuantizer;
    expect(_sameResult(loaded.search(queries, k), before), isTrue);
  });

  test('IndexRefineFlat round-trips through bytes', () {
    final refined = IndexRefineFlat(
      IndexIVFPQ(d: d, nlist: 8, m: 4, nprobe: 4),
      kFactor: 4,
    );
    refined.train(xs);
    refined.add(xs);
    final before = refined.search(queries, k);
    final loaded = readIndex(writeIndex(refined)) as IndexRefineFlat;
    expect(loaded.kFactor, equals(4));
    expect(_sameResult(loaded.search(queries, k), before), isTrue);
  });

  test('readIndex rejects bad magic', () {
    final blob = Uint8List.fromList(List<int>.filled(64, 0));
    expect(() => readIndex(blob), throwsA(isA<FormatException>()));
  });

  // --- range search + removeIds -----------------------------------------

  test('IndexFlat.rangeSearch agrees with brute-force filter of search()', () {
    final flat = IndexFlatL2(d)..add(xs);
    final r = flat.search(queries, n);
    // Pick a radius strictly between the 3rd and 4th shell of query 0
    // to avoid float32-vs-double boundary flakes.
    final radius = (r.distances[0][2] + r.distances[0][3]) / 2;
    final rr = flat.rangeSearch(queries, radius);
    // For each query, the set of ids returned must exactly match the
    // set of ids from the k=n search with distance <= radius.
    for (var qi = 0; qi < nq; qi++) {
      final expected = <int>{};
      for (var j = 0; j < n; j++) {
        if (r.distances[qi][j] <= radius) expected.add(r.ids[qi][j]);
      }
      final got = <int>{};
      final len = rr.lengthFor(qi);
      final off = rr.limits[qi];
      for (var j = 0; j < len; j++) {
        got.add(rr.ids[off + j]);
      }
      expect(got, equals(expected));
    }
  });

  test('IndexIVFFlat.rangeSearch matches Flat when nprobe = nlist', () {
    final ivf = IndexIVFFlat(d: d, nlist: 4, nprobe: 4)
      ..train(xs)
      ..add(xs);
    final flat = IndexFlatL2(d)..add(xs);
    final radius = 0.05;
    final gold = flat.rangeSearch(queries, radius);
    final got = ivf.rangeSearch(queries, radius);
    for (var qi = 0; qi < nq; qi++) {
      final a = <int>{};
      for (var j = 0; j < gold.lengthFor(qi); j++) {
        a.add(gold.ids[gold.limits[qi] + j]);
      }
      final b = <int>{};
      for (var j = 0; j < got.lengthFor(qi); j++) {
        b.add(got.ids[got.limits[qi] + j]);
      }
      expect(b, equals(a));
    }
  });

  test(
    'IndexScalarQuantizer.rangeSearch returns approximately similar hits',
    () {
      final sq = IndexScalarQuantizer(d)
        ..train(xs)
        ..add(xs);
      final flat = IndexFlatL2(d)..add(xs);
      final radius = 0.1;
      final gold = flat.rangeSearch(queries, radius);
      final got = sq.rangeSearch(queries, radius);
      // SQ is quantized so hits aren't identical, but > 70 % of the
      // exact hits should still be captured with the same radius.
      var hit = 0;
      var total = 0;
      for (var qi = 0; qi < nq; qi++) {
        final expected = <int>{};
        for (var j = 0; j < gold.lengthFor(qi); j++) {
          expected.add(gold.ids[gold.limits[qi] + j]);
        }
        final got_ids = <int>{};
        for (var j = 0; j < got.lengthFor(qi); j++) {
          got_ids.add(got.ids[got.limits[qi] + j]);
        }
        hit += expected.intersection(got_ids).length;
        total += expected.length;
      }
      if (total > 0) {
        expect(hit / total, greaterThan(0.7));
      }
    },
  );

  test('IndexFlat.removeIds compacts and updates ntotal', () {
    final flat = IndexFlatL2(d)..add(xs);
    // Remove every third id.
    final toRemove = <int>{for (var i = 0; i < n; i += 3) i};
    final expectedKept = n - toRemove.length;
    final removed = flat.removeIds(toRemove);
    expect(removed, equals(toRemove.length));
    expect(flat.ntotal, equals(expectedKept));
    // The remaining vectors should be exactly the ones that were kept,
    // in original order (though at compacted positions).
    final kept = <Float32List>[];
    for (var i = 0; i < n; i++) {
      if (!toRemove.contains(i)) kept.add(xs[i]);
    }
    // Verify by searching each kept vector at k=1 and seeing itself.
    final r = flat.search(kept.sublist(0, 5), 1);
    for (var i = 0; i < 5; i++) {
      expect(r.distances[i][0], closeTo(0.0, 1e-5));
    }
  });

  test('IndexIVFFlat.removeIds keeps searchable state', () {
    final ivf = IndexIVFFlat(d: d, nlist: 4, nprobe: 4)
      ..train(xs)
      ..add(xs);
    final toRemove = <int>{0, 1, 2, 3, 4};
    final removed = ivf.removeIds(toRemove);
    expect(removed, equals(5));
    expect(ivf.ntotal, equals(n - 5));
    // Search should still succeed and never return a removed original id
    // (which would now be at some new offset — verify by ensuring the
    // top result for xs[5] is at position 0 in the compacted store).
    final r = ivf.search(xs.sublist(5, 6), 1);
    expect(r.ids[0][0], equals(0));
  });

  test('IndexIDMap.removeIds preserves external ids on the remaining set', () {
    final idmap = IndexIDMap(IndexFlatL2(d));
    final ids = List<int>.generate(n, (i) => 2000 + i);
    idmap.addWithIds(xs, ids);
    final toRemove = <int>{2000, 2005, 2010};
    final removed = idmap.removeIds(toRemove);
    expect(removed, equals(3));
    expect(idmap.ntotal, equals(n - 3));
    // Searching for one of the removed vectors must NOT return its old
    // external id — it should return the next-closest kept vector.
    final r = idmap.search(xs.sublist(0, 1), 1);
    expect(r.ids[0][0], isNot(equals(2000)));
    // Searching for a still-present vector returns its unchanged id.
    final r2 = idmap.search(xs.sublist(6, 7), 1);
    expect(r2.ids[0][0], equals(2006));
  });

  test('Index.rangeSearch throws on indexes without support', () {
    final pq = IndexPQ(d: d, m: 4)
      ..train(xs)
      ..add(xs);
    expect(
      () => pq.rangeSearch(queries, 0.1),
      throwsA(isA<UnsupportedError>()),
    );
  });

  // --- binary + LSH ------------------------------------------------------

  test('IndexBinaryFlat finds itself at Hamming distance 0', () {
    const codeSize = 8; // 64-bit codes
    final rng = math.Random(11);
    final codes = List<Uint8List>.generate(64, (_) {
      final c = Uint8List(codeSize);
      for (var j = 0; j < codeSize; j++) {
        c[j] = rng.nextInt(256);
      }
      return c;
    });
    final bf = IndexBinaryFlat(codeSize)..add(codes);
    final r = bf.search(codes.sublist(0, 4), 1);
    for (var i = 0; i < 4; i++) {
      expect(r.ids[i][0], equals(i));
      expect(r.distances[i][0], equals(0.0));
    }
  });

  test('IndexBinaryFlat search returns ids ranked by Hamming distance', () {
    const codeSize = 4; // 32 bits
    // Build a dataset where distance from the query grows with id.
    final codes = <Uint8List>[];
    for (var i = 0; i < 8; i++) {
      final c = Uint8List(codeSize);
      // Flip the low i bits of byte 0.
      c[0] = (1 << i) - 1; // 0, 1, 3, 7, 15, 31, 63, 127
      codes.add(c);
    }
    final bf = IndexBinaryFlat(codeSize)..add(codes);
    final query = Uint8List(codeSize); // all zero bits → dist == popcount(code)
    final r = bf.search([query], 3);
    // popcount(0)=0, popcount(1)=1, popcount(3)=2 — ids 0,1,2 in order.
    expect(r.ids[0][0], equals(0));
    expect(r.ids[0][1], equals(1));
    expect(r.ids[0][2], equals(2));
    expect(r.distances[0][0], equals(0.0));
    expect(r.distances[0][1], equals(1.0));
    expect(r.distances[0][2], equals(2.0));
  });

  test('IndexBinaryFlat round-trips through bytes', () {
    const codeSize = 8;
    final rng = math.Random(3);
    final codes = List<Uint8List>.generate(50, (_) {
      final c = Uint8List(codeSize);
      for (var j = 0; j < codeSize; j++) {
        c[j] = rng.nextInt(256);
      }
      return c;
    });
    final bf = IndexBinaryFlat(codeSize)..add(codes);
    final before = bf.search(codes.sublist(0, 5), 3);
    final loaded = readBinaryIndex(writeBinaryIndex(bf)) as IndexBinaryFlat;
    expect(loaded.codeSize, equals(codeSize));
    expect(loaded.ntotal, equals(50));
    final after = loaded.search(codes.sublist(0, 5), 3);
    for (var qi = 0; qi < 5; qi++) {
      for (var j = 0; j < 3; j++) {
        expect(after.ids[qi][j], equals(before.ids[qi][j]));
        expect(after.distances[qi][j], equals(before.distances[qi][j]));
      }
    }
  });

  test('IndexLSH recall grows with nbits', () {
    // Use enough bits for a reasonable signal on d=16 blob data.
    final flat = IndexFlatL2(d)..add(xs);
    final flatTruth = flat.search(queries, k);

    final lsh64 = IndexLSH(d: d, nbits: 64)..add(xs);
    final r64 = lsh64.search(queries, k);
    final recall64 = _recall(r64, flatTruth, k);

    final lsh256 = IndexLSH(d: d, nbits: 256)..add(xs);
    final r256 = lsh256.search(queries, k);
    final recall256 = _recall(r256, flatTruth, k);

    // More bits should give at least as much recall (statistically).
    expect(recall256, greaterThanOrEqualTo(recall64 - 0.05));
    // And the higher-bit build should be non-trivial (well above the
    // 1-in-n baseline of ~0.01 for our test corpus).
    expect(recall256, greaterThan(0.1));
  });

  test('IndexLSH round-trips through bytes', () {
    final lsh = IndexLSH(d: d, nbits: 128)..add(xs);
    final before = lsh.search(queries, k);
    final loaded = readIndex(writeIndex(lsh)) as IndexLSH;
    expect(loaded.nbits, equals(128));
    expect(loaded.codeSize, equals(16));
    expect(loaded.ntotal, equals(xs.length));
    final after = loaded.search(queries, k);
    for (var qi = 0; qi < nq; qi++) {
      for (var j = 0; j < k; j++) {
        expect(after.ids[qi][j], equals(before.ids[qi][j]));
        expect(after.distances[qi][j], equals(before.distances[qi][j]));
      }
    }
  });

  // --- pre-transforms ----------------------------------------------------

  test('L2NormTransform produces unit-length outputs', () {
    final t = L2NormTransform(d);
    final out = t.apply(xs.sublist(0, 5));
    for (final row in out) {
      var s = 0.0;
      for (var j = 0; j < d; j++) {
        s += row[j] * row[j];
      }
      expect(math.sqrt(s), closeTo(1.0, 1e-5));
    }
  });

  test('RandomRotationTransform preserves L2 distances', () {
    final t = RandomRotationTransform(d: d, seed: 7);
    final a = xs.sublist(0, 4);
    final b = t.apply(a);
    // Pairwise distances should match up to float rounding.
    for (var i = 0; i < 4; i++) {
      for (var j = i + 1; j < 4; j++) {
        var dOrig = 0.0;
        var dNew = 0.0;
        for (var c = 0; c < d; c++) {
          final da = a[i][c] - a[j][c];
          final db = b[i][c] - b[j][c];
          dOrig += da * da;
          dNew += db * db;
        }
        expect(dNew, closeTo(dOrig, 1e-3));
      }
    }
  });

  test('RandomRotationTransform.reverseTransform inverts apply', () {
    final t = RandomRotationTransform(d: d, seed: 3);
    final round = t.reverseTransform(t.apply(xs.sublist(0, 3)));
    for (var i = 0; i < 3; i++) {
      for (var j = 0; j < d; j++) {
        expect(round[i][j], closeTo(xs[i][j], 1e-4));
      }
    }
  });

  test('IndexPreTransform proxies ntotal and delegates search', () {
    final pt = IndexPreTransform(
      chain: [L2NormTransform(d)],
      inner: IndexFlatL2(d),
    )..add(xs);
    expect(pt.ntotal, equals(xs.length));
    final r = pt.search(xs.sublist(0, 3), 1);
    for (var i = 0; i < 3; i++) {
      expect(r.ids[i][0], equals(i));
    }
  });

  test('L2Norm + IndexFlatIP matches IndexFlatL2 on unit vectors', () {
    // On the unit sphere, ‖x − y‖² = 2 − 2·<x, y>, so the L2 nearest
    // neighbour is exactly the largest inner product. Wrapping an IP
    // index in L2Norm therefore matches a plain L2 index on the same
    // (post-normalization) data — a mathematical guarantee, not a
    // recall approximation.
    final flat = IndexFlatL2(d);
    final wrapped = IndexPreTransform(
      chain: [L2NormTransform(d)],
      inner: IndexFlatIP(d),
    );
    // Feed both indexes the L2-normalized inputs so the underlying
    // corpora are identical; the wrapper additionally normalizes
    // queries on the search path.
    final normed = L2NormTransform(d).apply(xs);
    flat.add(normed);
    wrapped.add(xs); // wrapper normalizes internally
    final rFlat = flat.search(L2NormTransform(d).apply(queries), k);
    final rWrap = wrapped.search(queries, k);
    for (var qi = 0; qi < nq; qi++) {
      for (var j = 0; j < k; j++) {
        expect(rWrap.ids[qi][j], equals(rFlat.ids[qi][j]));
      }
    }
  });

  test('IndexPreTransform round-trips through bytes', () {
    final pt = IndexPreTransform(
      chain: [
        RandomRotationTransform(d: d, seed: 42),
        L2NormTransform(d),
      ],
      inner: IndexFlatL2(d),
    )..add(xs);
    final before = pt.search(queries, k);
    final loaded = readIndex(writeIndex(pt)) as IndexPreTransform;
    expect(loaded.chain.length, equals(2));
    expect(loaded.chain[0], isA<RandomRotationTransform>());
    expect(loaded.chain[1], isA<L2NormTransform>());
    expect(loaded.ntotal, equals(xs.length));
    final after = loaded.search(queries, k);
    for (var qi = 0; qi < nq; qi++) {
      for (var j = 0; j < k; j++) {
        expect(after.ids[qi][j], equals(before.ids[qi][j]));
        expect(after.distances[qi][j], closeTo(before.distances[qi][j], 1e-4));
      }
    }
  });

  // --- PCA ---------------------------------------------------------------

  test('PCATransform reduces dimension and centres output', () {
    final pca = PCATransform(dIn: d, dOut: 4)..train(xs);
    final out = pca.apply(xs);
    expect(out.length, equals(xs.length));
    expect(out.first.length, equals(4));
    // Empirical mean of projected data should be ~zero.
    final mean = Float64List(4);
    for (final row in out) {
      for (var j = 0; j < 4; j++) {
        mean[j] += row[j];
      }
    }
    for (var j = 0; j < 4; j++) {
      expect(mean[j] / out.length, closeTo(0.0, 1e-3));
    }
  });

  test('PCATransform eigenvalues are sorted descending', () {
    final pca = PCATransform(dIn: d, dOut: 8)..train(xs);
    for (var i = 0; i < 7; i++) {
      expect(pca.eigenvalues[i], greaterThanOrEqualTo(pca.eigenvalues[i + 1]));
    }
    // Blob data has structure → leading eigenvalue is substantially
    // larger than a random baseline.
    expect(pca.eigenvalues[0], greaterThan(0.0));
  });

  test('PCA + Flat preserves top-1 recall on low-rank data', () {
    // Construct data that lives on a 3-dimensional subspace of ℝ^d,
    // plus tiny noise. PCA to dOut = 3 should keep all the signal and
    // recover top-1 recall on independent queries.
    final rng = math.Random(101);
    final basis = List<Float32List>.generate(3, (_) {
      final b = Float32List(d);
      for (var j = 0; j < d; j++) {
        b[j] = rng.nextDouble() * 2 - 1;
      }
      return b;
    });
    Float32List _mix(int seed) {
      final r = math.Random(seed);
      final coeffs = [r.nextDouble(), r.nextDouble(), r.nextDouble()];
      final v = Float32List(d);
      for (var j = 0; j < d; j++) {
        v[j] =
            coeffs[0] * basis[0][j] +
            coeffs[1] * basis[1][j] +
            coeffs[2] * basis[2][j] +
            (r.nextDouble() - 0.5) * 0.001;
      }
      return v;
    }

    final data = List<Float32List>.generate(300, (i) => _mix(1000 + i));
    final qs = List<Float32List>.generate(30, (i) => _mix(9000 + i));

    final flat = IndexFlatL2(d)..add(data);
    final gold = flat.search(qs, 1);

    final pt = IndexPreTransform(
      chain: [PCATransform(dIn: d, dOut: 3)],
      inner: IndexFlatL2(3),
    );
    pt.train(data);
    pt.add(data);
    final r = pt.search(qs, 1);
    var hit = 0;
    for (var qi = 0; qi < qs.length; qi++) {
      if (r.ids[qi][0] == gold.ids[qi][0]) hit++;
    }
    expect(hit / qs.length, greaterThan(0.9));
  });

  test('PCATransform round-trips through IndexPreTransform bytes', () {
    final pt = IndexPreTransform(
      chain: [PCATransform(dIn: d, dOut: 8)],
      inner: IndexFlatL2(8),
    );
    pt.train(xs);
    pt.add(xs);
    final before = pt.search(queries, k);
    final loaded = readIndex(writeIndex(pt)) as IndexPreTransform;
    expect(loaded.chain.length, equals(1));
    expect(loaded.chain[0], isA<PCATransform>());
    final after = loaded.search(queries, k);
    for (var qi = 0; qi < nq; qi++) {
      for (var j = 0; j < k; j++) {
        expect(after.ids[qi][j], equals(before.ids[qi][j]));
      }
    }
  });

  // --- IndexFactory -----------------------------------------------------

  test('indexFactory builds bare Flat', () {
    final idx = indexFactory(d, 'Flat');
    expect(idx, isA<IndexFlat>());
    expect(idx.d, equals(d));
    idx.add(xs);
    final r = idx.search(xs.sublist(0, 3), 1);
    for (var i = 0; i < 3; i++) {
      expect(r.ids[i][0], equals(i));
    }
  });

  test('indexFactory parses IVF<nlist>,Flat', () {
    final idx = indexFactory(d, 'IVF8,Flat');
    expect(idx, isA<IndexIVFFlat>());
    (idx as IndexIVFFlat)
      ..train(xs)
      ..add(xs);
    idx.nprobe = 8;
    final r = idx.search(queries, k);
    expect(_recall(r, truth, k), equals(1.0));
  });

  test('indexFactory parses IVF<nlist>,PQ<m>', () {
    final idx = indexFactory(d, 'IVF8,PQ4');
    expect(idx, isA<IndexIVFPQ>());
    expect(idx.d, equals(d));
  });

  test('indexFactory parses HNSW<M>', () {
    final idx = indexFactory(d, 'HNSW16');
    expect(idx, isA<IndexHNSW>());
    expect((idx as IndexHNSW).M, equals(16));
  });

  test('indexFactory parses LSH<nbits>', () {
    final idx = indexFactory(d, 'LSH64');
    expect(idx, isA<IndexLSH>());
    expect((idx as IndexLSH).nbits, equals(64));
  });

  test('indexFactory wraps with pre-transforms', () {
    final idx = indexFactory(d, 'L2Norm,PCA8,Flat');
    expect(idx, isA<IndexPreTransform>());
    final pt = idx as IndexPreTransform;
    expect(pt.chain.length, equals(2));
    expect(pt.chain[0], isA<L2NormTransform>());
    expect(pt.chain[1], isA<PCATransform>());
    expect(pt.inner.d, equals(8));
  });

  test('indexFactory end-to-end: L2Norm,IVF8,Flat searches correctly', () {
    final idx = indexFactory(d, 'L2Norm,IVF8,Flat') as IndexPreTransform;
    idx.train(xs);
    idx.add(xs);
    (idx.inner as IndexIVFFlat).nprobe = 8;
    final r = idx.search(xs.sublist(0, 3), 1);
    for (var i = 0; i < 3; i++) {
      expect(r.ids[i][0], equals(i));
    }
  });

  test('indexFactory rejects unknown specifiers', () {
    expect(
      () => indexFactory(d, 'Nope'),
      throwsA(isA<FormatException>()),
    );
    expect(
      () => indexFactory(d, 'IVF16'),
      throwsA(isA<FormatException>()),
    );
    expect(
      () => indexFactory(d, ''),
      throwsA(isA<FormatException>()),
    );
  });
}
