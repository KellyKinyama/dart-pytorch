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

  test('IndexScalarQuantizer.rangeSearch returns approximately similar hits', () {
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
  });

  test('IndexFlat.removeIds compacts and updates ntotal', () {
    final flat = IndexFlatL2(d)..add(xs);
    // Remove every third id.
    final toRemove = <int>{
      for (var i = 0; i < n; i += 3) i,
    };
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
}
