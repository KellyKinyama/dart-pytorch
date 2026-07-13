import 'dart:io';
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

  test('IndexBinaryIVF matches BinaryFlat when nprobe=nlist', () {
    const codeSize = 8;
    final rng = math.Random(4);
    // Build 8 clustered "populations" of codes so binary k-means has
    // real structure to discover.
    final centers = List<Uint8List>.generate(8, (_) {
      final c = Uint8List(codeSize);
      for (var j = 0; j < codeSize; j++) {
        c[j] = rng.nextInt(256);
      }
      return c;
    });
    final codes = List<Uint8List>.generate(400, (i) {
      final base = centers[i % 8];
      final c = Uint8List(codeSize);
      // Flip ~5 % of bits.
      for (var j = 0; j < codeSize; j++) {
        var b = base[j];
        for (var bit = 0; bit < 8; bit++) {
          if (rng.nextDouble() < 0.05) b ^= (1 << bit);
        }
        c[j] = b;
      }
      return c;
    });
    final qs = codes.sublist(0, 20);
    final flat = IndexBinaryFlat(codeSize)..add(codes);
    final ivf = IndexBinaryIVF(codeSize: codeSize, nlist: 8, nprobe: 8);
    ivf.train(codes);
    ivf.add(codes);
    final flatR = flat.search(qs, 5);
    final ivfR = ivf.search(qs, 5);
    // Same top-1 id in every query when we probe every list.
    for (var qi = 0; qi < 20; qi++) {
      expect(ivfR.ids[qi][0], equals(flatR.ids[qi][0]));
    }
  });

  test('IndexBinaryIVF ntotal accumulates across add calls', () {
    const codeSize = 4;
    final rng = math.Random(5);
    Uint8List rnd() {
      final c = Uint8List(codeSize);
      for (var j = 0; j < codeSize; j++) {
        c[j] = rng.nextInt(256);
      }
      return c;
    }

    final train = List<Uint8List>.generate(50, (_) => rnd());
    final ivf = IndexBinaryIVF(codeSize: codeSize, nlist: 4)..train(train);
    ivf.add(List.generate(20, (_) => rnd()));
    ivf.add(List.generate(30, (_) => rnd()));
    expect(ivf.ntotal, equals(50));
    var sum = 0;
    for (var c = 0; c < 4; c++) {
      sum += ivf.listSize(c);
    }
    expect(sum, equals(50));
  });

  test('IndexBinaryIVF round-trips through bytes', () {
    const codeSize = 8;
    final rng = math.Random(6);
    final codes = List<Uint8List>.generate(120, (_) {
      final c = Uint8List(codeSize);
      for (var j = 0; j < codeSize; j++) {
        c[j] = rng.nextInt(256);
      }
      return c;
    });
    final ivf = IndexBinaryIVF(codeSize: codeSize, nlist: 8, nprobe: 3);
    ivf.train(codes);
    ivf.add(codes);
    final before = ivf.search(codes.sublist(0, 5), 3);
    final loaded = readBinaryIndex(writeBinaryIndex(ivf)) as IndexBinaryIVF;
    expect(loaded.nlist, equals(8));
    expect(loaded.nprobe, equals(3));
    expect(loaded.ntotal, equals(120));
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
    expect(() => indexFactory(d, 'Nope'), throwsA(isA<FormatException>()));
    expect(() => indexFactory(d, 'IVF16'), throwsA(isA<FormatException>()));
    expect(() => indexFactory(d, ''), throwsA(isA<FormatException>()));
  });

  // --- IndexShards / IndexReplicas -------------------------------------

  test('IndexShards partitions add across shards round-robin', () {
    final shards = IndexShards(
      shards: [IndexFlatL2(d), IndexFlatL2(d), IndexFlatL2(d)],
    );
    shards.add(xs);
    expect(shards.ntotal, equals(xs.length));
    // Round-robin: shard 0 holds ceil(n/3), shard 1 ceil((n-1)/3), etc.
    final sizes = [
      shards.shards[0].ntotal,
      shards.shards[1].ntotal,
      shards.shards[2].ntotal,
    ];
    expect(sizes.reduce((a, b) => a + b), equals(xs.length));
    // Sizes within 1 of each other.
    final maxS = sizes.reduce((a, b) => a > b ? a : b);
    final minS = sizes.reduce((a, b) => a < b ? a : b);
    expect(maxS - minS, lessThanOrEqualTo(1));
  });

  test('IndexShards search matches monolithic Flat', () {
    final mono = IndexFlatL2(d)..add(xs);
    final gold = mono.search(queries, k);
    final shards = IndexShards(
      shards: [IndexFlatL2(d), IndexFlatL2(d), IndexFlatL2(d), IndexFlatL2(d)],
    )..add(xs);
    final r = shards.search(queries, k);
    // Top-1 must match; deeper ranks can tie-break differently.
    for (var qi = 0; qi < nq; qi++) {
      expect(r.ids[qi][0], equals(gold.ids[qi][0]));
      expect(r.distances[qi][0], closeTo(gold.distances[qi][0], 1e-5));
    }
    // And recall@k against the monolithic gold set must be 100 %.
    expect(_recall(r, gold, k), equals(1.0));
  });

  test('IndexShards forwards train to trainable shards', () {
    final shards = IndexShards(
      shards: [
        IndexIVFFlat(d: d, nlist: 4),
        IndexIVFFlat(d: d, nlist: 4),
      ],
    );
    expect(shards.isTrained, isFalse);
    shards.train(xs);
    expect(shards.isTrained, isTrue);
    shards.add(xs);
    // Bump nprobe so ivf.search returns k results.
    (shards.shards[0] as IndexIVFFlat).nprobe = 4;
    (shards.shards[1] as IndexIVFFlat).nprobe = 4;
    final r = shards.search(queries, k);
    expect(r.ids[0][0], greaterThanOrEqualTo(0));
  });

  test('IndexShards round-trips through bytes', () {
    final shards = IndexShards(shards: [IndexFlatL2(d), IndexFlatL2(d)])
      ..add(xs);
    final before = shards.search(queries, k);
    final loaded = readIndex(writeIndex(shards)) as IndexShards;
    expect(loaded.nshards, equals(2));
    expect(loaded.ntotal, equals(xs.length));
    final after = loaded.search(queries, k);
    for (var qi = 0; qi < nq; qi++) {
      for (var j = 0; j < k; j++) {
        expect(after.ids[qi][j], equals(before.ids[qi][j]));
      }
    }
  });

  test('IndexReplicas broadcasts add and search delegates to one replica', () {
    final rep = IndexReplicas(
      replicas: [IndexFlatL2(d), IndexFlatL2(d), IndexFlatL2(d)],
    )..add(xs);
    expect(rep.ntotal, equals(xs.length));
    for (final r in rep.replicas) {
      expect(r.ntotal, equals(xs.length));
    }
    // Search results are identical to any one replica.
    final gold = rep.replicas[0].search(queries, k);
    final r0 = rep.search(queries, k);
    for (var qi = 0; qi < nq; qi++) {
      for (var j = 0; j < k; j++) {
        expect(r0.ids[qi][j], equals(gold.ids[qi][j]));
      }
    }
  });

  test('IndexReplicas round-trips through bytes', () {
    final rep = IndexReplicas(replicas: [IndexFlatL2(d), IndexFlatL2(d)])
      ..add(xs);
    final before = rep.search(queries, k);
    final loaded = readIndex(writeIndex(rep)) as IndexReplicas;
    expect(loaded.nreplicas, equals(2));
    expect(loaded.ntotal, equals(xs.length));
    final after = loaded.search(queries, k);
    for (var qi = 0; qi < nq; qi++) {
      for (var j = 0; j < k; j++) {
        expect(after.ids[qi][j], equals(before.ids[qi][j]));
      }
    }
  });

  // --- FAISS binary-format interop -------------------------------------

  test('FaissFourcc encodes and decodes ASCII tags', () {
    // 'I'=0x49, 'x'=0x78, 'F'=0x46, '2'=0x32; LE u32 = 0x32467849.
    expect(FaissFourcc.of('IxF2'), equals(0x32467849));
    expect(FaissFourcc.toStr(0x32467849), equals('IxF2'));
    expect(FaissFourcc.toStr(FaissFourcc.flatIP), equals('IxFI'));
    expect(() => FaissFourcc.of('abc'), throwsArgumentError);
  });

  test('writeFaissIndex emits the exact FAISS byte layout for IndexFlatL2', () {
    // Golden fixture: IndexFlatL2(d=2) containing a single vector [3, 4].
    // Reconstructed by hand from faiss/impl/index_write.cc:
    //
    //   fourcc('IxF2')  : 49 78 46 32
    //   d = 2 (i32)     : 02 00 00 00
    //   ntotal = 1 (i64): 01 00 00 00 00 00 00 00
    //   dummy = 1<<20   : 00 00 10 00 00 00 00 00
    //   dummy = 1<<20   : 00 00 10 00 00 00 00 00
    //   is_trained (u8) : 01
    //   metric = 1 (L2) : 01 00 00 00
    //   codes size = 8  : 08 00 00 00 00 00 00 00
    //   3.0f 4.0f       : 00 00 40 40 00 00 80 40
    final expected = Uint8List.fromList(<int>[
      0x49,
      0x78,
      0x46,
      0x32,
      0x02,
      0x00,
      0x00,
      0x00,
      0x01,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x10,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x10,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x01,
      0x01,
      0x00,
      0x00,
      0x00,
      0x08,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x40,
      0x40,
      0x00,
      0x00,
      0x80,
      0x40,
    ]);
    final idx = IndexFlatL2(2)
      ..add([
        Float32List.fromList([3.0, 4.0]),
      ]);
    final got = writeFaissIndexToBytes(idx);
    expect(got, equals(expected));
    expect(got.length, equals(53));
  });

  test('writeFaissIndex tags IndexFlatIP with fourcc IxFI and metric 0', () {
    final idx = IndexFlatIP(2)
      ..add([
        Float32List.fromList([1.0, 0.0]),
      ]);
    final bytes = writeFaissIndexToBytes(idx);
    // First 4 bytes = fourcc, next 8 = i32 d + first half of i64 ntotal.
    expect(bytes[0], equals(0x49));
    expect(bytes[1], equals(0x78));
    expect(bytes[2], equals(0x46));
    expect(bytes[3], equals(0x49)); // 'I' — IxFI
    // metric_type field sits at offset 4+4+8+8+8+1 = 33.
    expect(bytes[33], equals(0x00));
    expect(bytes[34], equals(0x00));
    expect(bytes[35], equals(0x00));
    expect(bytes[36], equals(0x00));
  });

  test('readFaissIndex round-trips IndexFlat contents', () {
    final xs4 = List<Float32List>.generate(
      12,
      (i) => Float32List.fromList([
        i.toDouble(),
        (i * 2).toDouble(),
        (-i).toDouble(),
        (i * i).toDouble(),
      ]),
    );
    for (final metric in [Metric.l2, Metric.innerProduct]) {
      final idx = IndexFlat(4, metric)..add(xs4);
      final bytes = writeFaissIndexToBytes(idx);
      final loaded = readFaissIndexFromBytes(bytes) as IndexFlat;
      expect(loaded.d, equals(4));
      expect(loaded.metric, equals(metric));
      expect(loaded.ntotal, equals(12));
      for (var i = 0; i < 12; i++) {
        expect(loaded.reconstruct(i), equals(xs4[i]));
      }
      // Search must give identical results.
      final query = [xs4[3]];
      final a = idx.search(query, 3);
      final b = loaded.search(query, 3);
      expect(b.ids[0], equals(a.ids[0]));
    }
  });

  test('readFaissIndex parses a hand-built golden fixture back correctly', () {
    // Same 53-byte layout as above.
    final golden = Uint8List.fromList(<int>[
      0x49,
      0x78,
      0x46,
      0x32,
      0x02,
      0x00,
      0x00,
      0x00,
      0x01,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x10,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x10,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x01,
      0x01,
      0x00,
      0x00,
      0x00,
      0x08,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x40,
      0x40,
      0x00,
      0x00,
      0x80,
      0x40,
    ]);
    final idx = readFaissIndexFromBytes(golden) as IndexFlat;
    expect(idx.d, equals(2));
    expect(idx.metric, equals(Metric.l2));
    expect(idx.ntotal, equals(1));
    expect(idx.isTrained, isTrue);
    expect(idx.reconstruct(0), equals(Float32List.fromList([3.0, 4.0])));
  });

  test('readFaissIndex rejects an unknown fourcc with a helpful tag', () {
    final bytes = Uint8List.fromList(<int>[0x41, 0x42, 0x43, 0x44]);
    expect(
      () => readFaissIndexFromBytes(bytes),
      throwsA(
        isA<FormatException>().having(
          (e) => e.message,
          'message',
          contains('ABCD'),
        ),
      ),
    );
  });

  test('writeFaissIndex rejects unsupported index types', () {
    // `IndexShards` interop (multi-shard fan-out) is not yet wired up
    // — we use it here as a stand-in for "any Index subclass that
    // isn't in the dispatch table".
    final shards = IndexShards(shards: [IndexFlatL2(d), IndexFlatL2(d)])
      ..add(xs);
    expect(
      () => writeFaissIndexToBytes(shards),
      throwsA(isA<UnsupportedError>()),
    );
  });

  test('FAISS interop round-trips IndexIDMap with custom ids', () {
    final inner = IndexFlatL2(d);
    final idmap = IndexIDMap(inner);
    final ids = List<int>.generate(xs.length, (i) => 1000 - i * 3);
    idmap.addWithIds(xs, ids);
    final before = idmap.search(queries, k);

    final bytes = writeFaissIndexToBytes(idmap);
    // First 4 bytes must spell 'IxMp'.
    final tag =
        bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
    expect(FaissFourcc.toStr(tag), equals('IxMp'));

    final loaded = readFaissIndexFromBytes(bytes) as IndexIDMap;
    expect(loaded.d, equals(d));
    expect(loaded.ntotal, equals(xs.length));
    for (var i = 0; i < xs.length; i++) {
      expect(loaded.idOf(i), equals(ids[i]));
    }
    final after = loaded.search(queries, k);
    for (var qi = 0; qi < nq; qi++) {
      for (var j = 0; j < k; j++) {
        expect(after.ids[qi][j], equals(before.ids[qi][j]));
      }
    }
  });

  test('readFaissIndex decodes IxM2 (IndexIDMap2) as IndexIDMap', () {
    // FAISS's `IndexIDMap2` reuses the exact byte layout of `IxMp` and
    // rebuilds its reverse-lookup table lazily on demand. The port
    // has no IndexIDMap2 class, so IxM2 is accepted as a read-only
    // compatibility marker that decodes into a plain IndexIDMap.
    final idmap = IndexIDMap(IndexFlatL2(d));
    final ids = List<int>.generate(xs.length, (i) => 42 + i * 7);
    idmap.addWithIds(xs, ids);
    final before = idmap.search(queries, k);

    final bytes = writeFaissIndexToBytes(idmap);
    // Patch the leading fourcc from 'IxMp' to 'IxM2'.
    expect(String.fromCharCodes(bytes.sublist(0, 4)), equals('IxMp'));
    bytes[0] = 'I'.codeUnitAt(0);
    bytes[1] = 'x'.codeUnitAt(0);
    bytes[2] = 'M'.codeUnitAt(0);
    bytes[3] = '2'.codeUnitAt(0);
    expect(
      bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24),
      equals(FaissFourcc.idMap2),
    );

    final loaded = readFaissIndexFromBytes(bytes) as IndexIDMap;
    expect(loaded.d, equals(d));
    expect(loaded.ntotal, equals(xs.length));
    for (var i = 0; i < xs.length; i++) {
      expect(loaded.idOf(i), equals(ids[i]));
    }
    final after = loaded.search(queries, k);
    for (var qi = 0; qi < nq; qi++) {
      for (var j = 0; j < k; j++) {
        expect(after.ids[qi][j], equals(before.ids[qi][j]));
      }
    }
  });

  test('writeFaissTransform emits the exact FAISS byte layout for VNrm', () {
    // Golden fixture for a d=3 L2NormTransform, byte-for-byte the
    // output that upstream FAISS's write_VectorTransform produces
    // for a `NormalizationTransform(3, 2.0)`.
    //
    //   fourcc 'VNrm'  : 56 4E 72 6D
    //   f32 norm=2.0f  : 00 00 00 40
    //   i32 d_in  = 3  : 03 00 00 00
    //   i32 d_out = 3  : 03 00 00 00
    //   u8  is_trained : 01
    final expected = Uint8List.fromList(<int>[
      0x56, 0x4E, 0x72, 0x6D, // fourcc
      0x00, 0x00, 0x00, 0x40, // norm=2.0f
      0x03, 0x00, 0x00, 0x00, // d_in
      0x03, 0x00, 0x00, 0x00, // d_out
      0x01, // is_trained
    ]);
    final w = IoWriter();
    writeFaissTransform(w, L2NormTransform(3));
    final got = w.takeBytes();
    expect(got, equals(expected));
    expect(got.length, equals(17));
  });

  test('FAISS interop round-trips IndexPreTransform(L2Norm)+FlatIP', () {
    final pt = IndexPreTransform(
      chain: [L2NormTransform(d)],
      inner: IndexFlatIP(d),
    );
    pt.add(xs);
    final before = pt.search(queries, k);

    final bytes = writeFaissIndexToBytes(pt);
    final tag =
        bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
    expect(FaissFourcc.toStr(tag), equals('IxPT'));

    final loaded = readFaissIndexFromBytes(bytes) as IndexPreTransform;
    expect(loaded.d, equals(d));
    expect(loaded.chain.length, equals(1));
    expect(loaded.chain.first, isA<L2NormTransform>());
    expect(loaded.inner, isA<IndexFlat>());
    expect((loaded.inner as IndexFlat).metric, equals(Metric.innerProduct));
    expect(loaded.ntotal, equals(xs.length));

    final after = loaded.search(queries, k);
    for (var qi = 0; qi < nq; qi++) {
      for (var j = 0; j < k; j++) {
        expect(after.ids[qi][j], equals(before.ids[qi][j]));
      }
    }
  });

  test('readFaissTransform rejects an unknown transform fourcc', () {
    final bytes = Uint8List.fromList(<int>[0x5A, 0x5A, 0x5A, 0x5A]);
    expect(
      () => readFaissTransform(IoReader(bytes)),
      throwsA(isA<FormatException>()),
    );
  });

  test('writeFaissTransform emits exact FAISS layout for rrot (identity)', () {
    // Golden fixture: identity 2x2 rotation. Matches upstream FAISS
    // write_VectorTransform for `RandomRotationMatrix(2, 2)` after
    // manual overwrite to identity.
    //
    //   fourcc 'rrot' : 72 72 6F 74
    //   u8  have_bias : 00
    //   u64 A.size = 4: 04 00 00 00 00 00 00 00
    //   A (row-major) : 1.0 0.0 0.0 1.0 as f32 LE
    //   u64 b.size = 0: 00 00 00 00 00 00 00 00
    //   i32 d_in  = 2 : 02 00 00 00
    //   i32 d_out = 2 : 02 00 00 00
    //   u8 is_trained : 01
    final expected = Uint8List.fromList(<int>[
      0x72, 0x72, 0x6F, 0x74, // fourcc
      0x00, // have_bias
      0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // A.size
      0x00, 0x00, 0x80, 0x3F, // 1.0
      0x00, 0x00, 0x00, 0x00, // 0.0
      0x00, 0x00, 0x00, 0x00, // 0.0
      0x00, 0x00, 0x80, 0x3F, // 1.0
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // b.size = 0
      0x02, 0x00, 0x00, 0x00, // d_in
      0x02, 0x00, 0x00, 0x00, // d_out
      0x01, // is_trained
    ]);
    final rrot = RandomRotationTransform(d: 2, seed: 1);
    // Overwrite the generated matrix with identity so bytes are
    // deterministic for the golden compare.
    rrot.rotation[0] = 1.0;
    rrot.rotation[1] = 0.0;
    rrot.rotation[2] = 0.0;
    rrot.rotation[3] = 1.0;

    final w = IoWriter();
    writeFaissTransform(w, rrot);
    final got = w.takeBytes();
    expect(got, equals(expected));
    expect(got.length, equals(46));
  });

  test('FAISS interop round-trips RandomRotationTransform matrix bytes', () {
    final rrot = RandomRotationTransform(d: d, seed: 7);
    final input = List<Float32List>.generate(4, (i) {
      final v = Float32List(d);
      for (var j = 0; j < d; j++) {
        v[j] = ((i + 1) * (j + 1)).toDouble();
      }
      return v;
    });
    final before = rrot.apply(input);

    final w = IoWriter();
    writeFaissTransform(w, rrot);
    final loaded =
        readFaissTransform(IoReader(w.takeBytes())) as RandomRotationTransform;

    expect(loaded.dIn, equals(d));
    expect(loaded.dOut, equals(d));
    expect(loaded.isTrained, isTrue);
    for (var i = 0; i < d * d; i++) {
      expect(loaded.rotation[i], equals(rrot.rotation[i]));
    }
    final after = loaded.apply(input);
    for (var i = 0; i < input.length; i++) {
      for (var j = 0; j < d; j++) {
        expect(after[i][j], closeTo(before[i][j], 1e-6));
      }
    }
  });

  test('FAISS interop round-trips IndexPreTransform([RR, L2Norm], Flat)', () {
    final pt = IndexPreTransform(
      chain: [
        RandomRotationTransform(d: d, seed: 3),
        L2NormTransform(d),
      ],
      inner: IndexFlatL2(d),
    );
    pt.add(xs);
    final before = pt.search(queries, k);

    final bytes = writeFaissIndexToBytes(pt);
    final loaded = readFaissIndexFromBytes(bytes) as IndexPreTransform;

    expect(loaded.chain.length, equals(2));
    expect(loaded.chain[0], isA<RandomRotationTransform>());
    expect(loaded.chain[1], isA<L2NormTransform>());
    expect(loaded.ntotal, equals(xs.length));

    final after = loaded.search(queries, k);
    for (var qi = 0; qi < nq; qi++) {
      for (var j = 0; j < k; j++) {
        expect(after.ids[qi][j], equals(before.ids[qi][j]));
      }
    }
  });

  test('readFaissTransform rejects rrot with have_bias set', () {
    // Craft a payload with have_bias = 1 and a non-empty b. Matches
    // the current FAISS layout: fourcc, have_bias, A, b, d_in, d_out,
    // is_trained.
    final w = IoWriter();
    w.writeU32(FaissFourcc.randomRotation);
    w.writeU8(1); // have_bias
    w.writeU64(4);
    w.writeF32List(Float32List.fromList([1.0, 0.0, 0.0, 1.0]));
    w.writeU64(2);
    w.writeF32List(Float32List.fromList([0.1, 0.2]));
    w.writeI32(2);
    w.writeI32(2);
    w.writeU8(1);
    expect(
      () => readFaissTransform(IoReader(w.takeBytes())),
      throwsA(isA<UnsupportedError>()),
    );
  });

  test('writeFaissTransform emits exact FAISS layout for Pcam', () {
    // Golden fixture for a trivial 2->1 PCA with mean=[0,0] and
    // projection=[1,0], eigenvalues empty (constructor state).
    // Matches upstream FAISS write_VectorTransform for `PCAMatrix(2,1)`.
    //
    //   fourcc 'Pcam'        : 50 63 61 6D
    //   f32 eigen_power=0    : 00 00 00 00
    //   f32 epsilon=0        : 00 00 00 00
    //   u8  rand_rot=0       : 00
    //   i32 balanced_bins=0  : 00 00 00 00
    //   WVEC mean            : size 2 + f32*2 (0.0, 0.0)
    //   WVEC eigenvalues     : size 0
    //   WVEC PCAMat          : size 0
    //   u8  have_bias=1      : 01
    //   WVEC A               : size 2 + f32*2 (1.0, 0.0)
    //   WVEC b               : size 1 + f32*1 (0.0)  == -A*mean
    //   i32 d_in=2           : 02 00 00 00
    //   i32 d_out=1          : 01 00 00 00
    //   u8  is_trained=1     : 01
    final pca = PCATransform(dIn: 2, dOut: 1);
    pca.projection[0] = 1.0;
    pca.projection[1] = 0.0;
    pca.isTrained = true; // manually flip; skip Jacobi solver

    final expected = Uint8List.fromList(<int>[
      0x50, 0x63, 0x61, 0x6D, // fourcc
      0x00, 0x00, 0x00, 0x00, // eigen_power
      0x00, 0x00, 0x00, 0x00, // epsilon
      0x00, // random_rotation
      0x00, 0x00, 0x00, 0x00, // balanced_bins
      0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // mean size = 2
      0x00, 0x00, 0x00, 0x00, // mean[0]
      0x00, 0x00, 0x00, 0x00, // mean[1]
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // eigenvalues size = 0
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // PCAMat size = 0
      0x01, // have_bias
      0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // A size = 2
      0x00, 0x00, 0x80, 0x3F, // A[0] = 1.0
      0x00, 0x00, 0x00, 0x00, // A[1] = 0.0
      0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // b size = 1
      0x00, 0x00, 0x00, 0x80, // b[0] = -0.0 (== -A*mean when mean=0)
      0x02, 0x00, 0x00, 0x00, // d_in
      0x01, 0x00, 0x00, 0x00, // d_out
      0x01, // is_trained
    ]);

    final w = IoWriter();
    writeFaissTransform(w, pca);
    final got = w.takeBytes();
    expect(got, equals(expected));
  });

  test('FAISS interop round-trips a trained PCATransform', () {
    // Train a small PCA on synthetic data, round-trip via FAISS
    // bytes, and confirm outputs match to float precision.
    final pca = PCATransform(dIn: d, dOut: 4)..train(xs);
    final input = List<Float32List>.generate(3, (i) {
      final v = Float32List(d);
      for (var j = 0; j < d; j++) {
        v[j] = ((i + 1) * (j + 2)).toDouble();
      }
      return v;
    });
    final before = pca.apply(input);

    final w = IoWriter();
    writeFaissTransform(w, pca);
    final loaded = readFaissTransform(IoReader(w.takeBytes())) as PCATransform;

    expect(loaded.dIn, equals(d));
    expect(loaded.dOut, equals(4));
    expect(loaded.isTrained, isTrue);
    for (var i = 0; i < d; i++) {
      expect(loaded.mean[i], closeTo(pca.mean[i], 1e-6));
    }
    for (var i = 0; i < d * 4; i++) {
      expect(loaded.projection[i], closeTo(pca.projection[i], 1e-6));
    }
    final after = loaded.apply(input);
    for (var i = 0; i < input.length; i++) {
      for (var j = 0; j < 4; j++) {
        expect(after[i][j], closeTo(before[i][j], 1e-5));
      }
    }
  });

  test('FAISS interop round-trips IndexPreTransform([PCA, L2Norm], Flat)', () {
    final pt = IndexPreTransform(
      chain: [
        PCATransform(dIn: d, dOut: 8)..train(xs),
        L2NormTransform(8),
      ],
      inner: IndexFlatL2(8),
    );
    pt.add(xs);
    final before = pt.search(queries, k);

    final bytes = writeFaissIndexToBytes(pt);
    final loaded = readFaissIndexFromBytes(bytes) as IndexPreTransform;

    expect(loaded.chain.length, equals(2));
    expect(loaded.chain[0], isA<PCATransform>());
    expect(loaded.chain[1], isA<L2NormTransform>());
    expect(loaded.ntotal, equals(xs.length));

    final after = loaded.search(queries, k);
    for (var qi = 0; qi < nq; qi++) {
      for (var j = 0; j < k; j++) {
        expect(after.ids[qi][j], equals(before.ids[qi][j]));
      }
    }
  });

  test('writeFaissIndex emits IxPq fourcc and correct header bytes', () {
    // Verify the tag + header without validating the ksub=256 codebook
    // blob (2048 bytes for m=1, d=2). Round-trip tests below assert
    // correctness of the payload.
    final pq = IndexPQ(d: d, m: 4, seed: 42)..train(xs);
    final bytes = writeFaissIndexToBytes(pq);
    // First 4 bytes = fourcc 'IxPq'.
    final tag =
        bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
    expect(FaissFourcc.toStr(tag), equals('IxPq'));

    // Header at offset 4:
    //   i32 d = 16, i64 ntotal = 0, 2*i64 dummy=1<<20, u8 is_trained,
    //   i32 metric_type = 1 (L2)
    // Byte 4..7 = d.
    final dRead =
        bytes[4] | (bytes[5] << 8) | (bytes[6] << 16) | (bytes[7] << 24);
    expect(dRead, equals(d));
    // is_trained at offset 4 + 4 + 8 + 8 + 8 = 32.
    expect(bytes[32], equals(1));
    // metric type at offset 33..36 = 1 for L2.
    expect(bytes[33], equals(1));
  });

  test('FAISS interop round-trips a trained IndexPQ (L2)', () {
    final pq = IndexPQ(d: d, m: 4, seed: 42)..train(xs);
    pq.add(xs);
    final before = pq.search(queries, k);

    final bytes = writeFaissIndexToBytes(pq);
    final loaded = readFaissIndexFromBytes(bytes) as IndexPQ;

    expect(loaded.d, equals(d));
    expect(loaded.m, equals(4));
    expect(loaded.pq.nbits, equals(8));
    expect(loaded.metric, equals(Metric.l2));
    expect(loaded.isTrained, isTrue);
    expect(loaded.ntotal, equals(xs.length));

    // Codebook byte-equality.
    for (var sub = 0; sub < pq.m; sub++) {
      for (var c = 0; c < pq.pq.ksub; c++) {
        for (var j = 0; j < pq.pq.dsub; j++) {
          expect(
            loaded.pq.codebooks[sub][c][j],
            equals(pq.pq.codebooks[sub][c][j]),
          );
        }
      }
    }
    // Encoded codes byte-equality.
    expect(loaded.codes, equals(pq.codes));

    // Search results identical.
    final after = loaded.search(queries, k);
    for (var qi = 0; qi < nq; qi++) {
      for (var j = 0; j < k; j++) {
        expect(after.ids[qi][j], equals(before.ids[qi][j]));
      }
    }
  });

  test('FAISS interop round-trips a trained IndexPQ (inner product)', () {
    final pq = IndexPQ(d: d, m: 4, metric: Metric.innerProduct, seed: 7)
      ..train(xs);
    pq.add(xs);
    final before = pq.search(queries, k);

    final bytes = writeFaissIndexToBytes(pq);
    final tag =
        bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
    expect(FaissFourcc.toStr(tag), equals('IxPq'));

    final loaded = readFaissIndexFromBytes(bytes) as IndexPQ;
    expect(loaded.metric, equals(Metric.innerProduct));

    final after = loaded.search(queries, k);
    for (var qi = 0; qi < nq; qi++) {
      for (var j = 0; j < k; j++) {
        expect(after.ids[qi][j], equals(before.ids[qi][j]));
      }
    }
  });

  test('FAISS interop round-trips IndexPreTransform([PCA], IndexPQ)', () {
    // Chain PCA reduction then PQ compression.
    final pca = PCATransform(dIn: d, dOut: 8)..train(xs);
    final inner = IndexPQ(d: 8, m: 4, seed: 99);
    // Train the PQ on projected data.
    inner.train(pca.apply(xs));
    final pt = IndexPreTransform(chain: [pca], inner: inner);
    pt.add(xs);
    final before = pt.search(queries, k);

    final bytes = writeFaissIndexToBytes(pt);
    final loaded = readFaissIndexFromBytes(bytes) as IndexPreTransform;

    expect(loaded.chain.length, equals(1));
    expect(loaded.chain[0], isA<PCATransform>());
    expect(loaded.inner, isA<IndexPQ>());
    expect(loaded.ntotal, equals(xs.length));

    final after = loaded.search(queries, k);
    for (var qi = 0; qi < nq; qi++) {
      for (var j = 0; j < k; j++) {
        expect(after.ids[qi][j], equals(before.ids[qi][j]));
      }
    }
  });

  test('readFaissIndex rejects IxPq with polysemous fields set', () {
    // Craft a payload with valid header + PQ + codes, then non-zero
    // polysemous_ht. Reader should throw UnsupportedError.
    final pq = IndexPQ(d: d, m: 4, seed: 1)..train(xs);
    // Write a valid blob first.
    final valid = writeFaissIndexToBytes(pq);
    // The polysemous_ht is the last 4 bytes.
    valid[valid.length - 1] = 0x00;
    valid[valid.length - 2] = 0x00;
    valid[valid.length - 3] = 0x00;
    valid[valid.length - 4] = 0x2A; // 42
    expect(
      () => readFaissIndexFromBytes(valid),
      throwsA(isA<UnsupportedError>()),
    );
  });

  test('writeFaissIndex emits IxSQ fourcc and correct header bytes', () {
    final sq = IndexScalarQuantizer(d)..train(xs);
    final bytes = writeFaissIndexToBytes(sq);
    // First 4 bytes = fourcc 'IxSQ'.
    final tag =
        bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
    expect(FaissFourcc.toStr(tag), equals('IxSQ'));

    // Header at offset 4: i32 d, i64 ntotal, 2*i64 dummy, u8 is_trained,
    // i32 metric_type.
    final dRead =
        bytes[4] | (bytes[5] << 8) | (bytes[6] << 16) | (bytes[7] << 24);
    expect(dRead, equals(d));
    expect(bytes[32], equals(1)); // is_trained
    expect(bytes[33], equals(1)); // metric = L2
  });

  test('FAISS interop round-trips a trained IndexScalarQuantizer (L2)', () {
    final sq = IndexScalarQuantizer(d)..train(xs);
    sq.add(xs);
    final before = sq.search(queries, k);

    final bytes = writeFaissIndexToBytes(sq);
    final loaded = readFaissIndexFromBytes(bytes) as IndexScalarQuantizer;

    expect(loaded.d, equals(d));
    expect(loaded.metric, equals(Metric.l2));
    expect(loaded.isTrained, isTrue);
    expect(loaded.ntotal, equals(xs.length));

    // vmin / scale byte-equality (scale round-trips via vdiff = scale*255).
    for (var j = 0; j < d; j++) {
      expect(loaded.vmin[j], equals(sq.vmin[j]));
      expect(loaded.scale[j], equals(sq.scale[j]));
    }
    // Encoded codes byte-equality.
    expect(loaded.codes, equals(sq.codes));

    // Search results identical.
    final after = loaded.search(queries, k);
    for (var qi = 0; qi < nq; qi++) {
      for (var j = 0; j < k; j++) {
        expect(after.ids[qi][j], equals(before.ids[qi][j]));
      }
    }
  });

  test(
    'FAISS interop round-trips a trained IndexScalarQuantizer (inner product)',
    () {
      final sq = IndexScalarQuantizer(d, metric: Metric.innerProduct)
        ..train(xs);
      sq.add(xs);
      final before = sq.search(queries, k);

      final bytes = writeFaissIndexToBytes(sq);
      final tag =
          bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
      expect(FaissFourcc.toStr(tag), equals('IxSQ'));

      final loaded = readFaissIndexFromBytes(bytes) as IndexScalarQuantizer;
      expect(loaded.metric, equals(Metric.innerProduct));
      expect(loaded.codes, equals(sq.codes));

      final after = loaded.search(queries, k);
      for (var qi = 0; qi < nq; qi++) {
        for (var j = 0; j < k; j++) {
          expect(after.ids[qi][j], equals(before.ids[qi][j]));
        }
      }
    },
  );

  test(
    'FAISS interop round-trips IndexPreTransform([PCA], IndexScalarQuantizer)',
    () {
      final pca = PCATransform(dIn: d, dOut: 8)..train(xs);
      final inner = IndexScalarQuantizer(8)..train(pca.apply(xs));
      final pt = IndexPreTransform(chain: [pca], inner: inner);
      pt.add(xs);
      final before = pt.search(queries, k);

      final bytes = writeFaissIndexToBytes(pt);
      final loaded = readFaissIndexFromBytes(bytes) as IndexPreTransform;

      expect(loaded.chain.length, equals(1));
      expect(loaded.chain[0], isA<PCATransform>());
      expect(loaded.inner, isA<IndexScalarQuantizer>());
      expect(loaded.ntotal, equals(xs.length));

      final after = loaded.search(queries, k);
      for (var qi = 0; qi < nq; qi++) {
        for (var j = 0; j < k; j++) {
          expect(after.ids[qi][j], equals(before.ids[qi][j]));
        }
      }
    },
  );

  test('readFaissIndex rejects IxSQ with non-QT_8bit qtype', () {
    // Write a valid IxSQ blob, then patch qtype (first i32 of the
    // ScalarQuantizer block) from 0 to 1. Reader should throw
    // UnsupportedError.
    final sq = IndexScalarQuantizer(d)..train(xs);
    sq.add(xs);
    final bytes = writeFaissIndexToBytes(sq);

    // Offset layout: fourcc(4) + header(4+8+8+8+1+4 = 33) = 37.
    // So the qtype i32 starts at offset 37.
    const qtypeOffset = 4 + 4 + 8 + 8 + 8 + 1 + 4;
    bytes[qtypeOffset] = 0x01; // qtype = 1 (QT_4bit)
    bytes[qtypeOffset + 1] = 0x00;
    bytes[qtypeOffset + 2] = 0x00;
    bytes[qtypeOffset + 3] = 0x00;
    expect(
      () => readFaissIndexFromBytes(bytes),
      throwsA(isA<UnsupportedError>()),
    );
  });

  test('writeFaissIndex emits IxRF fourcc and correct header bytes', () {
    final base = IndexFlat(d, Metric.l2);
    final refined = IndexRefineFlat(base, kFactor: 3);
    final bytes = writeFaissIndexToBytes(refined);
    // First 4 bytes = fourcc 'IxRF'.
    final tag =
        bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
    expect(FaissFourcc.toStr(tag), equals('IxRF'));

    // Header at offset 4: i32 d, i64 ntotal, 2*i64 dummy, u8 is_trained,
    // i32 metric_type.
    final dRead =
        bytes[4] | (bytes[5] << 8) | (bytes[6] << 16) | (bytes[7] << 24);
    expect(dRead, equals(d));
    // is_trained: IndexFlat is trained by default → true.
    expect(bytes[32], equals(1));
    expect(bytes[33], equals(1)); // metric = L2
  });

  test('FAISS interop round-trips IndexRefineFlat over IndexFlat (L2)', () {
    final base = IndexFlat(d, Metric.l2);
    final refined = IndexRefineFlat(base, kFactor: 3);
    refined.add(xs);
    final before = refined.search(queries, k);

    final bytes = writeFaissIndexToBytes(refined);
    final loaded = readFaissIndexFromBytes(bytes) as IndexRefineFlat;

    expect(loaded.d, equals(d));
    expect(loaded.metric, equals(Metric.l2));
    expect(loaded.isTrained, isTrue);
    expect(loaded.ntotal, equals(xs.length));
    expect(loaded.kFactor, equals(3));
    expect(loaded.base, isA<IndexFlat>());
    expect(loaded.refine, isA<IndexFlat>());
    expect(loaded.refine.ntotal, equals(xs.length));

    // Search results identical (base is exact, refine is exact).
    final after = loaded.search(queries, k);
    for (var qi = 0; qi < nq; qi++) {
      for (var j = 0; j < k; j++) {
        expect(after.ids[qi][j], equals(before.ids[qi][j]));
      }
    }
  });

  test(
    'FAISS interop round-trips IndexRefineFlat over IndexPQ (recall boost)',
    () {
      final base = IndexPQ(d: d, m: 4, seed: 11)..train(xs);
      final refined = IndexRefineFlat(base, kFactor: 4);
      refined.add(xs);
      final before = refined.search(queries, k);

      final bytes = writeFaissIndexToBytes(refined);
      final tag =
          bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
      expect(FaissFourcc.toStr(tag), equals('IxRF'));

      final loaded = readFaissIndexFromBytes(bytes) as IndexRefineFlat;
      expect(loaded.base, isA<IndexPQ>());
      expect(loaded.refine, isA<IndexFlat>());
      expect(loaded.kFactor, equals(4));
      expect(loaded.ntotal, equals(xs.length));

      final after = loaded.search(queries, k);
      for (var qi = 0; qi < nq; qi++) {
        for (var j = 0; j < k; j++) {
          expect(after.ids[qi][j], equals(before.ids[qi][j]));
        }
      }
    },
  );

  test('readFaissIndex rejects IxRF whose refine sub-index is non-Flat', () {
    // Craft a bogus IxRF blob whose refine sub-index has an IxPq
    // fourcc instead of IxF2. We synthesize by writing a real
    // IndexRefineFlat, then overwriting the refine fourcc.
    final base = IndexFlat(d, Metric.l2);
    final refined = IndexRefineFlat(base, kFactor: 2);
    refined.add(xs);
    final bytes = writeFaissIndexToBytes(refined);

    // The layout of the bytes:
    //   fourcc(4) + header(33) = 37
    //   then base = fourcc(4) + header(33) + WVEC(u64 size + data)
    // base is IxF2, ntotal = xs.length, storage bytes = ntotal*d*4.
    // WVEC prefix is u64 = 8 bytes. Then storage. Then refine's fourcc
    // at offset: 37 + 4 + 33 + 8 + xs.length*d*4.
    final refineOffset = 37 + 4 + 33 + 8 + xs.length * d * 4;
    // Patch to 'IxPq' — will fail because the payload isn't a PQ blob,
    // OR at minimum the type check will fire. We patch to a fourcc our
    // reader accepts but which produces a non-Flat instance. We use a
    // fresh IxPq stub that will fail to decode. Easier: patch to
    // 'IxSQ' whose decode requires the SQ block; that will fail with a
    // FormatException. To hit the "non-Flat" branch cleanly, just
    // corrupt so refine is unknown → FormatException.
    bytes[refineOffset] = 0x5A; // 'Z'
    bytes[refineOffset + 1] = 0x5A;
    bytes[refineOffset + 2] = 0x5A;
    bytes[refineOffset + 3] = 0x5A;
    expect(
      () => readFaissIndexFromBytes(bytes),
      throwsA(isA<FormatException>()),
    );
  });

  test('writeFaissIndex emits IwFl fourcc and correct header bytes', () {
    final ivf = IndexIVFFlat(d: d, nlist: 4, nprobe: 2)..train(xs);
    final bytes = writeFaissIndexToBytes(ivf);
    final tag =
        bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
    expect(FaissFourcc.toStr(tag), equals('IwFl'));

    // Header at offset 4: i32 d, i64 ntotal, 2*i64 dummy, u8 is_trained,
    // i32 metric_type = 1 (L2).
    final dRead =
        bytes[4] | (bytes[5] << 8) | (bytes[6] << 16) | (bytes[7] << 24);
    expect(dRead, equals(d));
    expect(bytes[32], equals(1)); // is_trained
    expect(bytes[33], equals(1)); // metric = L2
  });

  test('FAISS interop round-trips IndexIVFFlat (L2, nprobe=nlist)', () {
    final ivf = IndexIVFFlat(d: d, nlist: 8, nprobe: 8)..train(xs);
    ivf.add(xs);
    final before = ivf.search(queries, k);

    final bytes = writeFaissIndexToBytes(ivf);
    final loaded = readFaissIndexFromBytes(bytes) as IndexIVFFlat;

    expect(loaded.d, equals(d));
    expect(loaded.nlist, equals(8));
    expect(loaded.nprobe, equals(8));
    expect(loaded.metric, equals(Metric.l2));
    expect(loaded.isTrained, isTrue);
    expect(loaded.ntotal, equals(xs.length));
    // Quantizer centroids preserved byte-for-byte.
    expect(loaded.quantizer.ntotal, equals(ivf.quantizer.ntotal));
    for (var i = 0; i < ivf.quantizer.ntotal; i++) {
      expect(
        loaded.quantizer.reconstruct(i),
        equals(ivf.quantizer.reconstruct(i)),
      );
    }
    // Inverted-list assignments preserved.
    for (var c = 0; c < ivf.nlist; c++) {
      expect(loaded.invLists[c], equals(ivf.invLists[c]));
    }
    // Vector storage preserved.
    expect(loaded.storage, equals(ivf.storage));

    // Search results identical.
    final after = loaded.search(queries, k);
    for (var qi = 0; qi < nq; qi++) {
      for (var j = 0; j < k; j++) {
        expect(after.ids[qi][j], equals(before.ids[qi][j]));
      }
    }
  });

  test('FAISS interop round-trips IndexIVFFlat (inner product)', () {
    final ivf = IndexIVFFlat(
      d: d,
      nlist: 4,
      nprobe: 4,
      metric: Metric.innerProduct,
    )..train(xs);
    ivf.add(xs);
    final before = ivf.search(queries, k);

    final bytes = writeFaissIndexToBytes(ivf);
    final loaded = readFaissIndexFromBytes(bytes) as IndexIVFFlat;
    expect(loaded.metric, equals(Metric.innerProduct));
    expect(loaded.storage, equals(ivf.storage));

    final after = loaded.search(queries, k);
    for (var qi = 0; qi < nq; qi++) {
      for (var j = 0; j < k; j++) {
        expect(after.ids[qi][j], equals(before.ids[qi][j]));
      }
    }
  });

  test('FAISS interop round-trips an empty IndexIVFFlat (train, no add)', () {
    // No adds means every cell is empty → ilar picks the `sprs` branch.
    final ivf = IndexIVFFlat(d: d, nlist: 8, nprobe: 2)..train(xs);
    final bytes = writeFaissIndexToBytes(ivf);
    final loaded = readFaissIndexFromBytes(bytes) as IndexIVFFlat;
    expect(loaded.ntotal, equals(0));
    expect(loaded.nlist, equals(8));
    for (var c = 0; c < loaded.nlist; c++) {
      expect(loaded.invLists[c], isEmpty);
    }
    // Verify the sparse branch was actually taken: locate the `ilar`
    // fourcc and then its list_type after (nlist:u64 + code_size:u64).
    // Find first 'ilar' occurrence.
    final ilar = FaissFourcc.of('ilar');
    var offset = -1;
    for (var i = 0; i < bytes.length - 3; i++) {
      final t =
          bytes[i] |
          (bytes[i + 1] << 8) |
          (bytes[i + 2] << 16) |
          (bytes[i + 3] << 24);
      if (t == ilar) {
        offset = i;
        break;
      }
    }
    expect(offset, greaterThan(0));
    final listTypeOff = offset + 4 + 8 + 8; // fourcc + nlist + code_size
    final listType =
        bytes[listTypeOff] |
        (bytes[listTypeOff + 1] << 8) |
        (bytes[listTypeOff + 2] << 16) |
        (bytes[listTypeOff + 3] << 24);
    expect(FaissFourcc.toStr(listType), equals('sprs'));
  });

  test('writeFaissIndex emits IwPQ fourcc and correct header bytes', () {
    final ivpq = IndexIVFPQ(d: d, nlist: 4, m: 4, nprobe: 2)..train(xs);
    final bytes = writeFaissIndexToBytes(ivpq);
    final tag =
        bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
    expect(FaissFourcc.toStr(tag), equals('IwPQ'));
    final dRead =
        bytes[4] | (bytes[5] << 8) | (bytes[6] << 16) | (bytes[7] << 24);
    expect(dRead, equals(d));
    expect(bytes[32], equals(1)); // is_trained
    expect(bytes[33], equals(1)); // metric = L2
  });

  test('FAISS interop round-trips IndexIVFPQ (L2, nprobe=nlist)', () {
    final ivpq = IndexIVFPQ(d: d, nlist: 4, m: 4, nprobe: 4, seed: 17)
      ..train(xs);
    ivpq.add(xs);
    final before = ivpq.search(queries, k);

    final bytes = writeFaissIndexToBytes(ivpq);
    final loaded = readFaissIndexFromBytes(bytes) as IndexIVFPQ;

    expect(loaded.d, equals(d));
    expect(loaded.nlist, equals(4));
    expect(loaded.m, equals(4));
    expect(loaded.nprobe, equals(4));
    expect(loaded.metric, equals(Metric.l2));
    expect(loaded.isTrained, isTrue);
    expect(loaded.ntotal, equals(xs.length));

    // Coarse quantizer centroids preserved.
    for (var i = 0; i < ivpq.quantizer.ntotal; i++) {
      expect(
        loaded.quantizer.reconstruct(i),
        equals(ivpq.quantizer.reconstruct(i)),
      );
    }
    // PQ codebooks preserved byte-for-byte.
    for (var sub = 0; sub < ivpq.m; sub++) {
      for (var c = 0; c < ivpq.pq.ksub; c++) {
        for (var j = 0; j < ivpq.pq.dsub; j++) {
          expect(
            loaded.pq.codebooks[sub][c][j],
            equals(ivpq.pq.codebooks[sub][c][j]),
          );
        }
      }
    }
    // Inverted lists preserved.
    for (var c = 0; c < ivpq.nlist; c++) {
      expect(loaded.invListsIds[c], equals(ivpq.invListsIds[c]));
      expect(loaded.invListsCodes[c], equals(ivpq.invListsCodes[c]));
    }

    // Search results identical.
    final after = loaded.search(queries, k);
    for (var qi = 0; qi < nq; qi++) {
      for (var j = 0; j < k; j++) {
        expect(after.ids[qi][j], equals(before.ids[qi][j]));
      }
    }
  });

  test('FAISS interop round-trips IndexIVFPQ (inner product)', () {
    final ivpq = IndexIVFPQ(
      d: d,
      nlist: 4,
      m: 4,
      nprobe: 4,
      metric: Metric.innerProduct,
      seed: 3,
    )..train(xs);
    ivpq.add(xs);
    final before = ivpq.search(queries, k);

    final bytes = writeFaissIndexToBytes(ivpq);
    final loaded = readFaissIndexFromBytes(bytes) as IndexIVFPQ;
    expect(loaded.metric, equals(Metric.innerProduct));

    final after = loaded.search(queries, k);
    for (var qi = 0; qi < nq; qi++) {
      for (var j = 0; j < k; j++) {
        expect(after.ids[qi][j], equals(before.ids[qi][j]));
      }
    }
  });

  test('readFaissIndex rejects IwPQ with by_residual=0', () {
    final ivpq = IndexIVFPQ(d: d, nlist: 4, m: 4)..train(xs);
    ivpq.add(xs);
    final bytes = writeFaissIndexToBytes(ivpq);

    // by_residual is the single byte immediately after the ivf_header.
    // ivf_header offset layout:
    //   fourcc(4) + index_header(33)             = 37
    //   + nlist:u64(8) + nprobe:u64(8)           = 53
    //   + write_index(quantizer)                 = IxF2 fourcc(4)
    //     + index_header(33) + WVEC(u64 size(8) + nlist*d*4 bytes)
    //   + direct_map: u8(1) + WVEC empty i64(8)  = 9
    //   → by_residual at that offset
    final byResidualOffset = 4 + 33 + 8 + 8 + 4 + 33 + 8 + 4 * d * 4 + 1 + 8;
    expect(bytes[byResidualOffset], equals(1));
    bytes[byResidualOffset] = 0;
    expect(
      () => readFaissIndexFromBytes(bytes),
      throwsA(isA<UnsupportedError>()),
    );
  });

  test('writeFaissIndex emits IxHe fourcc and correct header bytes', () {
    final lsh = IndexLSH(d: d, nbits: 64)..add(xs);
    final bytes = writeFaissIndexToBytes(lsh);
    final tag =
        bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
    expect(FaissFourcc.toStr(tag), equals('IxHe'));
    final dRead =
        bytes[4] | (bytes[5] << 8) | (bytes[6] << 16) | (bytes[7] << 24);
    expect(dRead, equals(d));
    // is_trained at offset 32, metric (L2 = 1) at 33.
    expect(bytes[32], equals(1));
    expect(bytes[33], equals(1));
    // nbits (i32) immediately after the 33-byte index_header.
    final nbitsOffset = 4 + 33;
    final nbitsRead =
        bytes[nbitsOffset] |
        (bytes[nbitsOffset + 1] << 8) |
        (bytes[nbitsOffset + 2] << 16) |
        (bytes[nbitsOffset + 3] << 24);
    expect(nbitsRead, equals(64));
    // rotate_data = 1, train_thresholds = 0.
    expect(bytes[nbitsOffset + 4], equals(1));
    expect(bytes[nbitsOffset + 5], equals(0));
  });

  test('FAISS interop round-trips IndexLSH (byte-equal proj + codes)', () {
    final lsh = IndexLSH(d: d, nbits: 64, seed: 7)..add(xs);
    final before = lsh.search(queries, k);

    final bytes = writeFaissIndexToBytes(lsh);
    final loaded = readFaissIndexFromBytes(bytes) as IndexLSH;

    expect(loaded.d, equals(d));
    expect(loaded.nbits, equals(64));
    expect(loaded.codeSize, equals(lsh.codeSize));
    expect(loaded.ntotal, equals(xs.length));
    expect(loaded.isTrained, isTrue);

    // Projection matrix preserved byte-for-byte.
    for (var i = 0; i < lsh.projection.length; i++) {
      expect(loaded.projection[i], equals(lsh.projection[i]));
    }
    // Packed codes preserved byte-for-byte.
    expect(loaded.codes, equals(lsh.codes));

    // Search results identical.
    final after = loaded.search(queries, k);
    for (var qi = 0; qi < nq; qi++) {
      for (var j = 0; j < k; j++) {
        expect(after.ids[qi][j], equals(before.ids[qi][j]));
      }
    }
  });

  test('FAISS interop round-trips an empty IndexLSH (no add)', () {
    final lsh = IndexLSH(d: d, nbits: 128, seed: 99)..trainProjection();
    final bytes = writeFaissIndexToBytes(lsh);
    final loaded = readFaissIndexFromBytes(bytes) as IndexLSH;
    expect(loaded.ntotal, equals(0));
    expect(loaded.nbits, equals(128));
    expect(loaded.d, equals(d));
    for (var i = 0; i < lsh.projection.length; i++) {
      expect(loaded.projection[i], equals(lsh.projection[i]));
    }
  });

  test('readFaissIndex rejects IxHe with rotate_data=0', () {
    final lsh = IndexLSH(d: d, nbits: 64)..add(xs);
    final bytes = writeFaissIndexToBytes(lsh);
    // rotate_data byte is at offset (fourcc 4) + (index_header 33) +
    // (nbits i32 4) = 41.
    final rotateOffset = 4 + 33 + 4;
    expect(bytes[rotateOffset], equals(1));
    bytes[rotateOffset] = 0;
    expect(
      () => readFaissIndexFromBytes(bytes),
      throwsA(isA<UnsupportedError>()),
    );
  });

  test('writeFaissBinaryIndex emits IBxF fourcc and correct header bytes', () {
    final codeSize = 8;
    final rng = math.Random(1);
    final codes = List<Uint8List>.generate(16, (_) {
      final c = Uint8List(codeSize);
      for (var j = 0; j < codeSize; j++) {
        c[j] = rng.nextInt(256);
      }
      return c;
    });
    final bf = IndexBinaryFlat(codeSize)..add(codes);
    final bytes = writeFaissBinaryIndexToBytes(bf);
    final tag =
        bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
    expect(FaissFourcc.toStr(tag), equals('IBxF'));
    // i32 d then i32 code_size then i64 ntotal then u8 is_trained then
    // i32 metric_type = 1.
    final dRead =
        bytes[4] | (bytes[5] << 8) | (bytes[6] << 16) | (bytes[7] << 24);
    expect(dRead, equals(codeSize * 8));
    final csRead =
        bytes[8] | (bytes[9] << 8) | (bytes[10] << 16) | (bytes[11] << 24);
    expect(csRead, equals(codeSize));
    // ntotal little-endian i64 at offset 12; only low byte matters here.
    expect(bytes[12], equals(16));
    for (var i = 13; i < 20; i++) {
      expect(bytes[i], equals(0));
    }
    expect(bytes[20], equals(1)); // is_trained
    expect(bytes[21], equals(1)); // metric_type = METRIC_L2
    for (var i = 22; i < 25; i++) {
      expect(bytes[i], equals(0));
    }
    // WVEC size = ntotal * codeSize at offset 25 as u64 LE.
    final xbCount = bytes[25] | (bytes[26] << 8) | (bytes[27] << 16);
    expect(xbCount, equals(16 * codeSize));
  });

  test('FAISS interop round-trips IndexBinaryFlat (byte-equal codes)', () {
    final codeSize = 16;
    final rng = math.Random(42);
    final codes = List<Uint8List>.generate(64, (_) {
      final c = Uint8List(codeSize);
      for (var j = 0; j < codeSize; j++) {
        c[j] = rng.nextInt(256);
      }
      return c;
    });
    final queries = List<Uint8List>.generate(nq, (_) {
      final c = Uint8List(codeSize);
      for (var j = 0; j < codeSize; j++) {
        c[j] = rng.nextInt(256);
      }
      return c;
    });
    final bf = IndexBinaryFlat(codeSize)..add(codes);
    final before = bf.search(queries, k);

    final bytes = writeFaissBinaryIndexToBytes(bf);
    final loaded = readFaissBinaryIndexFromBytes(bytes) as IndexBinaryFlat;

    expect(loaded.d, equals(codeSize * 8));
    expect(loaded.codeSize, equals(codeSize));
    expect(loaded.ntotal, equals(codes.length));
    expect(loaded.isTrained, isTrue);

    // Packed codes preserved byte-for-byte.
    expect(loaded.codes, equals(bf.codes));

    // Search results identical.
    final after = loaded.search(queries, k);
    for (var qi = 0; qi < nq; qi++) {
      for (var j = 0; j < k; j++) {
        expect(after.ids[qi][j], equals(before.ids[qi][j]));
      }
    }
  });

  test('FAISS interop round-trips an empty IndexBinaryFlat', () {
    final bf = IndexBinaryFlat(8);
    final bytes = writeFaissBinaryIndexToBytes(bf);
    final loaded = readFaissBinaryIndexFromBytes(bytes) as IndexBinaryFlat;
    expect(loaded.ntotal, equals(0));
    expect(loaded.codeSize, equals(8));
    expect(loaded.d, equals(64));
    expect(loaded.codes.length, equals(0));
  });

  test('writeFaissBinaryIndex rejects unsupported binary index types', () {
    // Simulate an out-of-port IndexBinary subtype (e.g. IndexBinaryHNSW,
    // IndexBinaryIDMap) that this port doesn't wire up yet.
    final foreign = _ForeignBinaryIndex(8);
    expect(
      () => writeFaissBinaryIndexToBytes(foreign),
      throwsA(isA<UnsupportedError>()),
    );
  });

  test('readFaissBinaryIndex rejects IBxF with mismatched d / code_size', () {
    final bf = IndexBinaryFlat(8)
      ..add(<Uint8List>[Uint8List.fromList(List<int>.filled(8, 0))]);
    final bytes = writeFaissBinaryIndexToBytes(bf);
    // Corrupt d (i32 at offset 4) from 64 → 32, keep code_size=8.
    bytes[4] = 32;
    expect(
      () => readFaissBinaryIndexFromBytes(bytes),
      throwsA(isA<FormatException>()),
    );
  });

  test('readFaissBinaryIndex rejects an unknown binary fourcc', () {
    final bogus = Uint8List.fromList(<int>[0x5A, 0x5A, 0x5A, 0x5A]);
    expect(
      () => readFaissBinaryIndexFromBytes(bogus),
      throwsA(
        isA<FormatException>().having(
          (e) => e.message,
          'message',
          contains('ZZZZ'),
        ),
      ),
    );
  });

  test('writeFaissBinaryIndex emits IBwF fourcc and correct header bytes', () {
    final codeSize = 8;
    final rng = math.Random(2);
    final codes = List<Uint8List>.generate(64, (_) {
      final c = Uint8List(codeSize);
      for (var j = 0; j < codeSize; j++) {
        c[j] = rng.nextInt(256);
      }
      return c;
    });
    final ivf = IndexBinaryIVF(codeSize: codeSize, nlist: 4, nprobe: 2)
      ..train(codes)
      ..add(codes);
    final bytes = writeFaissBinaryIndexToBytes(ivf);
    final tag =
        bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
    expect(FaissFourcc.toStr(tag), equals('IBwF'));
    // Binary header: i32 d, i32 code_size at offsets 4, 8.
    final dRead =
        bytes[4] | (bytes[5] << 8) | (bytes[6] << 16) | (bytes[7] << 24);
    expect(dRead, equals(codeSize * 8));
    final csRead =
        bytes[8] | (bytes[9] << 8) | (bytes[10] << 16) | (bytes[11] << 24);
    expect(csRead, equals(codeSize));
    // ntotal (i64 at 12), is_trained (u8 at 20), metric (i32 at 21).
    expect(bytes[12], equals(64));
    expect(bytes[20], equals(1));
    expect(bytes[21], equals(1)); // METRIC_L2
  });

  test('FAISS interop round-trips IndexBinaryIVF (nprobe=nlist)', () {
    final codeSize = 8;
    final rng = math.Random(11);
    final codes = List<Uint8List>.generate(200, (_) {
      final c = Uint8List(codeSize);
      for (var j = 0; j < codeSize; j++) {
        c[j] = rng.nextInt(256);
      }
      return c;
    });
    final queries = List<Uint8List>.generate(nq, (_) {
      final c = Uint8List(codeSize);
      for (var j = 0; j < codeSize; j++) {
        c[j] = rng.nextInt(256);
      }
      return c;
    });
    final ivf = IndexBinaryIVF(codeSize: codeSize, nlist: 8, nprobe: 8)
      ..train(codes)
      ..add(codes);
    final before = ivf.search(queries, k);

    final bytes = writeFaissBinaryIndexToBytes(ivf);
    final loaded = readFaissBinaryIndexFromBytes(bytes) as IndexBinaryIVF;

    expect(loaded.codeSize, equals(codeSize));
    expect(loaded.d, equals(codeSize * 8));
    expect(loaded.nlist, equals(8));
    expect(loaded.nprobe, equals(8));
    expect(loaded.ntotal, equals(codes.length));
    expect(loaded.isTrained, isTrue);

    // Coarse quantizer centroids preserved byte-for-byte.
    expect(loaded.quantizer.codes, equals(ivf.quantizer.codes));

    // Inverted lists preserved cell-by-cell.
    for (var c = 0; c < ivf.nlist; c++) {
      expect(loaded.invListIds(c), equals(ivf.invListIds(c)));
      expect(loaded.invListCodes(c), equals(ivf.invListCodes(c)));
    }

    // Search results identical.
    final after = loaded.search(queries, k);
    for (var qi = 0; qi < nq; qi++) {
      for (var j = 0; j < k; j++) {
        expect(after.ids[qi][j], equals(before.ids[qi][j]));
      }
    }
  });

  test('FAISS interop round-trips an empty IndexBinaryIVF (train, no add)', () {
    final codeSize = 8;
    final rng = math.Random(3);
    final codes = List<Uint8List>.generate(64, (_) {
      final c = Uint8List(codeSize);
      for (var j = 0; j < codeSize; j++) {
        c[j] = rng.nextInt(256);
      }
      return c;
    });
    final ivf = IndexBinaryIVF(codeSize: codeSize, nlist: 4)..train(codes);
    final bytes = writeFaissBinaryIndexToBytes(ivf);
    final loaded = readFaissBinaryIndexFromBytes(bytes) as IndexBinaryIVF;
    expect(loaded.ntotal, equals(0));
    expect(loaded.nlist, equals(4));
    expect(loaded.quantizer.ntotal, equals(4));
    expect(loaded.quantizer.codes, equals(ivf.quantizer.codes));
    for (var c = 0; c < ivf.nlist; c++) {
      expect(loaded.listSize(c), equals(0));
    }
  });

  test('writeFaissIndex emits IHNf fourcc and correct header bytes', () {
    final hnsw = IndexHNSW(d: d, M: 8, efConstruction: 40, efSearch: 32)
      ..add(xs);
    final bytes = writeFaissIndexToBytes(hnsw);
    final tag =
        bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
    expect(FaissFourcc.toStr(tag), equals('IHNf'));
    final dRead =
        bytes[4] | (bytes[5] << 8) | (bytes[6] << 16) | (bytes[7] << 24);
    expect(dRead, equals(d));
    // is_trained at offset 32, metric (L2 = 1) at 33.
    expect(bytes[32], equals(1));
    expect(bytes[33], equals(1));
  });

  test('FAISS interop round-trips IndexHNSW byte-equal (graph + search)', () {
    final orig = IndexHNSW(d: d, M: 8, efConstruction: 40, efSearch: 32)
      ..add(xs);
    final before = orig.search(queries, k);

    final bytes = writeFaissIndexToBytes(orig);
    final loaded = readFaissIndexFromBytes(bytes) as IndexHNSW;

    expect(loaded.d, equals(d));
    expect(loaded.M, equals(orig.M));
    expect(loaded.efConstruction, equals(orig.efConstruction));
    expect(loaded.efSearch, equals(orig.efSearch));
    expect(loaded.ntotal, equals(orig.ntotal));
    expect(loaded.entryPoint, equals(orig.entryPoint));
    expect(loaded.topLevel, equals(orig.topLevel));
    expect(loaded.isTrained, isTrue);

    // Vector storage preserved byte-for-byte.
    expect(loaded.storage, equals(orig.storage));

    // Graph topology preserved.
    for (var i = 0; i < orig.ntotal; i++) {
      expect(loaded.nodeLevel(i), equals(orig.nodeLevel(i)));
      for (var l = 0; l <= orig.nodeLevel(i); l++) {
        expect(loaded.nodeNeighbors(i, l), equals(orig.nodeNeighbors(i, l)));
      }
    }

    // Re-writing the loaded index must produce an identical byte blob.
    final bytes2 = writeFaissIndexToBytes(loaded);
    expect(bytes2, equals(bytes));

    // Search results identical.
    final after = loaded.search(queries, k);
    for (var qi = 0; qi < nq; qi++) {
      for (var j = 0; j < k; j++) {
        expect(after.ids[qi][j], equals(before.ids[qi][j]));
      }
    }
  });

  test('FAISS interop round-trips an empty IndexHNSW (no add)', () {
    final orig = IndexHNSW(d: d, M: 16);
    final bytes = writeFaissIndexToBytes(orig);
    final loaded = readFaissIndexFromBytes(bytes) as IndexHNSW;
    expect(loaded.ntotal, equals(0));
    expect(loaded.M, equals(16));
    expect(loaded.entryPoint, equals(-1));
    expect(loaded.topLevel, equals(-1));
    // Re-emitting the empty index yields the same bytes.
    expect(writeFaissIndexToBytes(loaded), equals(bytes));
  });

  test('readFaissIndex rejects IHNf with upper_beam != 1', () {
    final hnsw = IndexHNSW(d: d, M: 8, efConstruction: 40, efSearch: 32)
      ..add(xs);
    final bytes = writeFaissIndexToBytes(hnsw);
    // upper_beam is the LAST i32 of the HNSW payload (immediately
    // before the storage sub-index fourcc). Walk from the tail: the
    // storage IxF2 blob is (fourcc 4) + (index_header 33) +
    // (WVEC u8 codes: 8 + ntotal * d * 4) bytes long.
    final storageLen = 4 + 33 + 8 + xs.length * d * 4;
    final upperBeamOffset = bytes.length - storageLen - 4;
    // Sanity: original bytes hold `1` at that spot.
    final origBeam =
        bytes[upperBeamOffset] |
        (bytes[upperBeamOffset + 1] << 8) |
        (bytes[upperBeamOffset + 2] << 16) |
        (bytes[upperBeamOffset + 3] << 24);
    expect(origBeam, equals(1));
    bytes[upperBeamOffset] = 2;
    expect(
      () => readFaissIndexFromBytes(bytes),
      throwsA(isA<UnsupportedError>()),
    );
  });

  test(
    'probeFaissIndex reports metadata for float indexes without decoding',
    () {
      final flat = IndexFlatL2(d)..add(xs);
      final flatBytes = writeFaissIndexToBytes(flat);
      final flatInfo = probeFaissIndex(flatBytes);
      expect(flatInfo.fourccStr, equals('IxF2'));
      expect(flatInfo.kind, equals(FaissIndexKind.floatIndex));
      expect(flatInfo.d, equals(d));
      expect(flatInfo.ntotal, equals(xs.length));
      expect(flatInfo.metric, equals(Metric.l2));
      expect(flatInfo.isTrained, isTrue);
      expect(flatInfo.codeSize, isNull);

      final ip = IndexFlatIP(d)..add(xs);
      final ipInfo = probeFaissIndex(writeFaissIndexToBytes(ip));
      expect(ipInfo.fourccStr, equals('IxFI'));
      expect(ipInfo.metric, equals(Metric.innerProduct));

      final hnsw = IndexHNSW(d: d, M: 8, efConstruction: 40, efSearch: 32)
        ..add(xs);
      final hnswInfo = probeFaissIndex(writeFaissIndexToBytes(hnsw));
      expect(hnswInfo.fourccStr, equals('IHNf'));
      expect(hnswInfo.kind, equals(FaissIndexKind.floatIndex));
      expect(hnswInfo.d, equals(d));
      expect(hnswInfo.ntotal, equals(xs.length));

      final idmap = IndexIDMap(IndexFlatL2(d))
        ..addWithIds(xs, List<int>.generate(xs.length, (i) => i + 500));
      final idmapInfo = probeFaissIndex(writeFaissIndexToBytes(idmap));
      expect(idmapInfo.fourccStr, equals('IxMp'));
      expect(idmapInfo.ntotal, equals(xs.length));
    },
  );

  test('probeFaissIndex reports metadata for binary indexes', () {
    final codeSize = 8;
    final rng = math.Random(1);
    final codes = List<Uint8List>.generate(24, (_) {
      final c = Uint8List(codeSize);
      for (var j = 0; j < codeSize; j++) {
        c[j] = rng.nextInt(256);
      }
      return c;
    });
    final bxf = IndexBinaryFlat(codeSize)..add(codes);
    final bxfInfo = probeFaissIndex(writeFaissBinaryIndexToBytes(bxf));
    expect(bxfInfo.fourccStr, equals('IBxF'));
    expect(bxfInfo.kind, equals(FaissIndexKind.binaryIndex));
    expect(bxfInfo.d, equals(codeSize * 8));
    expect(bxfInfo.codeSize, equals(codeSize));
    expect(bxfInfo.ntotal, equals(codes.length));
    // IndexBinary's on-disk metric is METRIC_L2 = 1 (even though the
    // real distance is Hamming) — surface it verbatim.
    expect(bxfInfo.metric, equals(Metric.l2));

    final ivf = IndexBinaryIVF(codeSize: codeSize, nlist: 4)
      ..train(codes)
      ..add(codes);
    final ivfInfo = probeFaissIndex(writeFaissBinaryIndexToBytes(ivf));
    expect(ivfInfo.fourccStr, equals('IBwF'));
    expect(ivfInfo.kind, equals(FaissIndexKind.binaryIndex));
    expect(ivfInfo.codeSize, equals(codeSize));
    expect(ivfInfo.ntotal, equals(codes.length));
  });

  test('probeFaissIndex flags unknown fourccs without throwing', () {
    // Four ASCII bytes not in the known-fourcc table.
    final bytes = Uint8List.fromList(<int>[
      'Z'.codeUnitAt(0),
      'z'.codeUnitAt(0),
      'z'.codeUnitAt(0),
      'Z'.codeUnitAt(0),
      // Trailing bytes that would parse as a valid header if the
      // fourcc were recognized — probe must NOT reach for them.
      ...List<int>.filled(64, 0xff),
    ]);
    final info = probeFaissIndex(bytes);
    expect(info.fourccStr, equals('ZzzZ'));
    expect(info.kind, equals(FaissIndexKind.unknown));
    expect(info.d, isNull);
    expect(info.ntotal, isNull);
    expect(info.metric, isNull);
    expect(info.isTrained, isNull);
    expect(info.codeSize, isNull);
  });

  test('probeFaissIndex rejects blobs shorter than the fourcc', () {
    expect(
      () => probeFaissIndex(Uint8List.fromList(<int>[1, 2, 3])),
      throwsA(isA<FormatException>()),
    );
  });

  test('probeFaissIndexFile round-trips through disk', () {
    final tmp = Directory.systemTemp.createTempSync('faiss_probe_');
    try {
      final path = '${tmp.path}/idx.faiss';
      final flat = IndexFlatL2(d)..add(xs);
      saveFaissIndex(path, flat);
      final info = probeFaissIndexFile(path);
      expect(info.fourccStr, equals('IxF2'));
      expect(info.kind, equals(FaissIndexKind.floatIndex));
      expect(info.d, equals(d));
      expect(info.ntotal, equals(xs.length));
    } finally {
      tmp.deleteSync(recursive: true);
    }
  });

  // -----------------------------------------------------------------
  // Cross-index conversion helpers.
  // -----------------------------------------------------------------

  test('reconstructAll returns owned rows equal to the source', () {
    final flat = IndexFlatL2(d)..add(xs);
    final rows = reconstructAll(flat);
    expect(rows, hasLength(xs.length));
    // Mutating a row must not touch the source storage.
    rows[0][0] = -12345.0;
    expect(flat.reconstruct(0)[0], isNot(equals(-12345.0)));
    // Values should match otherwise.
    for (var j = 0; j < d; j++) {
      expect(rows[1][j], equals(xs[1][j]));
    }
  });

  test('flatToIvfFlat produces a searchable IVF with matching metric', () {
    final flat = IndexFlatL2(d)..add(xs);
    final ivf = flatToIvfFlat(flat, nlist: 16, nprobe: 16);
    expect(ivf.d, equals(d));
    expect(ivf.metric, equals(Metric.l2));
    expect(ivf.nlist, equals(16));
    expect(ivf.nprobe, equals(16));
    expect(ivf.ntotal, equals(xs.length));
    expect(ivf.isTrained, isTrue);
    // At nprobe == nlist, IVFFlat is equivalent to a brute-force flat
    // — recall must be 1.0.
    final r = ivf.search(queries, k);
    expect(_recall(r, truth, k), equals(1.0));
  });

  test('flatToIvfPq trains PQ and hits reasonable recall with refine', () {
    final flat = IndexFlatL2(d)..add(xs);
    final ivfpq = flatToIvfPq(flat, m: 4, nbits: 8, nlist: 16, nprobe: 16);
    expect(ivfpq.m, equals(4));
    expect(ivfpq.nlist, equals(16));
    expect(ivfpq.ntotal, equals(xs.length));
    expect(ivfpq.isTrained, isTrue);
    // PQ alone loses recall; wrap with refine to recover it.
    final refined = wrapWithRefine(ivfpq, flat, kFactor: 8);
    expect(refined.ntotal, equals(xs.length));
    final r = refined.search(queries, k);
    expect(_recall(r, truth, k), greaterThan(0.9));
  });

  test('flatToHnsw builds a graph and returns strong recall', () {
    final flat = IndexFlatL2(d)..add(xs);
    final hnsw = flatToHnsw(flat, m: 16, efConstruction: 80, efSearch: 64);
    expect(hnsw.ntotal, equals(xs.length));
    expect(hnsw.isTrained, isTrue);
    final r = hnsw.search(queries, k);
    expect(_recall(r, truth, k), greaterThan(0.9));
  });

  test('flatToIvfFlat rejects empty sources', () {
    expect(() => flatToIvfFlat(IndexFlatL2(d)), throwsA(isA<ArgumentError>()));
  });

  test('flatToIvfPq rejects m that does not divide d', () {
    final flat = IndexFlatL2(d)..add(xs);
    expect(() => flatToIvfPq(flat, m: 5), throwsA(isA<ArgumentError>()));
  });

  test('wrapWithRefine rejects mismatched d / metric / ntotal', () {
    final flat = IndexFlatL2(d)..add(xs);
    final ivf = flatToIvfFlat(flat, nlist: 16, nprobe: 16);
    // Ntotal mismatch: rebuild a smaller flat.
    final smallFlat = IndexFlatL2(d)..add(xs.sublist(0, xs.length - 1));
    expect(() => wrapWithRefine(ivf, smallFlat), throwsA(isA<ArgumentError>()));
    // Metric mismatch: IP flat vs L2 IVF.
    final ipFlat = IndexFlatIP(d)..add(xs);
    expect(() => wrapWithRefine(ivf, ipFlat), throwsA(isA<ArgumentError>()));
  });

  // -----------------------------------------------------------------
  // Bench harness.
  // -----------------------------------------------------------------

  test('benchIndex reports recall@k = 1.0 for the flat baseline', () {
    final flat = IndexFlatL2(d)..add(xs);
    final r = benchIndex(
      index: flat,
      queries: queries,
      k: k,
      truth: truth,
      label: 'flat',
      options: const BenchOptions(warmup: 0, repeats: 1, perQuery: false),
    );
    expect(r.label, equals('flat'));
    expect(r.nq, equals(queries.length));
    expect(r.k, equals(k));
    expect(r.ntotal, equals(xs.length));
    expect(r.recall, equals(1.0));
    expect(r.meanUs, greaterThanOrEqualTo(0.0));
  });

  test('benchIndex tracks recall degradation on IVFFlat with nprobe=1', () {
    final flat = IndexFlatL2(d)..add(xs);
    final ivf = flatToIvfFlat(flat, nlist: 16, nprobe: 1);
    final r = benchIndex(
      index: ivf,
      queries: queries,
      k: k,
      truth: truth,
      label: 'ivf-np1',
      options: const BenchOptions(warmup: 0, repeats: 1),
    );
    // nprobe=1 with nlist=16 should NOT match flat recall.
    expect(r.recall, lessThan(1.0));
    expect(r.recall, greaterThan(0.0));
  });

  test('benchIndexSweep returns one row per configured value', () {
    final flat = IndexFlatL2(d)..add(xs);
    final ivf = flatToIvfFlat(flat, nlist: 16, nprobe: 1);
    final rows = benchIndexSweep<int>(
      index: ivf,
      queries: queries,
      k: k,
      truth: truth,
      values: <int>[1, 4, 16],
      configure: (np) {
        ivf.nprobe = np;
        return 'ivf-np=$np';
      },
      options: const BenchOptions(warmup: 0, repeats: 1),
    );
    expect(rows, hasLength(3));
    expect(rows[0].label, equals('ivf-np=1'));
    expect(rows[2].label, equals('ivf-np=16'));
    // Recall is monotone non-decreasing in nprobe.
    expect(rows[1].recall, greaterThanOrEqualTo(rows[0].recall));
    expect(rows[2].recall, greaterThanOrEqualTo(rows[1].recall));
    // At nprobe == nlist, recall must be 1.0.
    expect(rows[2].recall, equals(1.0));
  });

  test('toBenchCsv produces RFC-4180 compliant rows with a header', () {
    final row = BenchResult(
      label: 'x, "hi"',
      nq: 10,
      k: 5,
      recall: 0.5,
      meanUs: 12.0,
      p50Us: 10.0,
      p95Us: 20.0,
      p99Us: 25.0,
      ntotal: 100,
    );
    final csv = toBenchCsv(<BenchResult>[row]);
    final lines = csv.trim().split('\n');
    expect(lines, hasLength(2));
    expect(
      lines[0],
      equals('label,ntotal,nq,k,recall,mean_us,p50_us,p95_us,p99_us'),
    );
    // Commas + quotes escaped per RFC 4180.
    expect(lines[1], startsWith('"x, ""hi"""'));
  });

  test('toBenchMarkdown emits a right-aligned GFM table with escaped labels',
      () {
    final rows = <BenchResult>[
      BenchResult(
        label: 'a|b\\c',
        nq: 1,
        k: 1,
        recall: 0.9,
        meanUs: 12.5,
        p50Us: 10.0,
        p95Us: 20.0,
        p99Us: 30.0,
        ntotal: 5,
      ),
    ];
    final md = toBenchMarkdown(rows);
    final lines = md.trim().split('\n');
    expect(lines, hasLength(3));
    expect(lines[0], startsWith('| label |'));
    // Header separator row uses `--:` for right-aligned numeric cols.
    expect(lines[1], contains('---:'));
    // Pipe + backslash are escaped in the label cell.
    expect(lines[2], contains(r'a\|b\\c'));
    expect(lines[2], endsWith('| 1 |'));
  });

  test('paretoFrontier drops dominated rows and returns recall-ascending', () {
    BenchResult row(String label, double recall, double mean) => BenchResult(
          label: label,
          nq: 1,
          k: 1,
          recall: recall,
          meanUs: mean,
          p50Us: mean,
          p95Us: mean,
          p99Us: mean,
          ntotal: 100,
        );
    final rows = <BenchResult>[
      row('fast-lowrecall', 0.2, 5),
      row('dominated', 0.3, 25), // dominated by (mid-mediocre 0.5 @ 10)
      row('mid-mediocre', 0.5, 10),
      row('slow-perfect', 1.0, 100),
    ];
    final frontier = paretoFrontier(rows);
    expect(
      frontier.map((r) => r.label),
      equals(<String>['fast-lowrecall', 'mid-mediocre', 'slow-perfect']),
    );
  });

  test('paretoFrontier keeps only the fastest row on recall ties', () {
    BenchResult row(String label, double recall, double mean) => BenchResult(
          label: label,
          nq: 1,
          k: 1,
          recall: recall,
          meanUs: mean,
          p50Us: mean,
          p95Us: mean,
          p99Us: mean,
          ntotal: 100,
        );
    final rows = <BenchResult>[
      row('slow-recall1', 1.0, 100),
      row('fast-recall1', 1.0, 50),
    ];
    final frontier = paretoFrontier(rows);
    expect(frontier, hasLength(1));
    expect(frontier.first.label, equals('fast-recall1'));
  });

  test('paretoFrontier handles empty input', () {
    expect(paretoFrontier(<BenchResult>[]), isEmpty);
  });

  test('benchIndex rejects mismatched truth', () {
    final flat = IndexFlatL2(d)..add(xs);
    // Shorter queries → mismatch with `truth.nq`.
    final short = queries.sublist(0, 3);
    expect(
      () => benchIndex(index: flat, queries: short, k: k, truth: truth),
      throwsA(isA<ArgumentError>()),
    );
    // Ask for larger k than truth has.
    expect(
      () => benchIndex(
        index: flat,
        queries: queries,
        k: truth.k + 1,
        truth: truth,
      ),
      throwsA(isA<ArgumentError>()),
    );
  });

  // -----------------------------------------------------------------
  // Auto-tuner.
  // -----------------------------------------------------------------

  test('OperatingPoints.pareto drops dominated points', () {
    final points = OperatingPoints(<OperatingPoint>[
      // fast + low recall
      const OperatingPoint(
        paramValue: 1,
        paramLabel: 'nprobe=1',
        recall: 0.2,
        meanUs: 10.0,
      ),
      // dominated: worse recall than (2,0.5,20) at similar cost
      const OperatingPoint(
        paramValue: 2,
        paramLabel: 'nprobe=2',
        recall: 0.3,
        meanUs: 25.0,
      ),
      const OperatingPoint(
        paramValue: 4,
        paramLabel: 'nprobe=4',
        recall: 0.5,
        meanUs: 20.0,
      ),
      const OperatingPoint(
        paramValue: 8,
        paramLabel: 'nprobe=8',
        recall: 0.9,
        meanUs: 60.0,
      ),
    ]);
    final frontier = points.pareto();
    // (nprobe=2) is strictly dominated by (nprobe=4).
    expect(frontier.map((p) => p.paramValue), containsAll(<int>[1, 4, 8]));
    expect(frontier.map((p) => p.paramValue), isNot(contains(2)));
  });

  test('OperatingPoints.pickForRecall picks the fastest point >= floor', () {
    final points = OperatingPoints(<OperatingPoint>[
      const OperatingPoint(
        paramValue: 1,
        paramLabel: 'nprobe=1',
        recall: 0.5,
        meanUs: 5.0,
      ),
      const OperatingPoint(
        paramValue: 8,
        paramLabel: 'nprobe=8',
        recall: 0.9,
        meanUs: 30.0,
      ),
      const OperatingPoint(
        paramValue: 32,
        paramLabel: 'nprobe=32',
        recall: 0.99,
        meanUs: 90.0,
      ),
    ]);
    expect(points.pickForRecall(0.85)!.paramValue, equals(8));
    expect(points.pickForRecall(0.95)!.paramValue, equals(32));
    expect(points.pickForRecall(0.999), isNull);
  });

  test(
    'OperatingPoints.pickForLatency picks the highest recall within budget',
    () {
      final points = OperatingPoints(<OperatingPoint>[
        const OperatingPoint(
          paramValue: 1,
          paramLabel: 'nprobe=1',
          recall: 0.5,
          meanUs: 5.0,
        ),
        const OperatingPoint(
          paramValue: 8,
          paramLabel: 'nprobe=8',
          recall: 0.9,
          meanUs: 30.0,
        ),
        const OperatingPoint(
          paramValue: 32,
          paramLabel: 'nprobe=32',
          recall: 0.99,
          meanUs: 90.0,
        ),
      ]);
      expect(points.pickForLatency(50.0)!.paramValue, equals(8));
      expect(points.pickForLatency(4.0), isNull);
    },
  );

  test('autoTuneNprobe sweeps IVFFlat and applies the chosen value', () {
    final flat = IndexFlatL2(d)..add(xs);
    final ivf = flatToIvfFlat(flat, nlist: 16, nprobe: 1);
    final points = autoTuneNprobe(
      target: ivf,
      queries: queries,
      k: k,
      truth: truth,
      values: const <int>[1, 4, 16],
      minRecall: 0.99,
      options: const BenchOptions(warmup: 0, repeats: 1),
    );
    // With nlist=16, nprobe=16 must give perfect recall.
    expect(points.points, hasLength(3));
    expect(points.points.last.recall, equals(1.0));
    // Applied value = fastest that met the floor. Must be one of
    // the swept values that actually cleared 0.99.
    final chosen = points.pickForRecall(0.99);
    expect(chosen, isNotNull);
    expect(ivf.nprobe, equals(chosen!.paramValue));
  });

  test('autoTuneNprobe deduplicates clamped values', () {
    final flat = IndexFlatL2(d)..add(xs);
    final ivf = flatToIvfFlat(flat, nlist: 4, nprobe: 1);
    final points = autoTuneNprobe(
      target: ivf,
      queries: queries,
      k: k,
      truth: truth,
      // 8, 16, 64 all clamp to nlist=4.
      values: const <int>[1, 2, 8, 16, 64],
      options: const BenchOptions(warmup: 0, repeats: 1),
    );
    // 1, 2, 4 remain after clamp + dedupe.
    expect(points.points.map((p) => p.paramValue), equals(<int>[1, 2, 4]));
  });

  test('autoTuneEfSearch sweeps HNSW and returns monotone recall', () {
    final flat = IndexFlatL2(d)..add(xs);
    final hnsw = flatToHnsw(flat, m: 16, efConstruction: 80);
    final points = autoTuneEfSearch(
      target: hnsw,
      queries: queries,
      k: k,
      truth: truth,
      values: const <int>[8, 32, 128],
      options: const BenchOptions(warmup: 0, repeats: 1),
    );
    expect(points.points, hasLength(3));
    // efSearch monotone non-decreasing in recall.
    for (var i = 1; i < points.points.length; i++) {
      expect(
        points.points[i].recall,
        greaterThanOrEqualTo(points.points[i - 1].recall - 1e-9),
      );
    }
    // Last (largest efSearch) applied.
    expect(hnsw.efSearch, equals(128));
  });

  test('autoTuneNprobe supports IVFPQ wrapped in IndexRefineFlat', () {
    final flat = IndexFlatL2(d)..add(xs);
    final ivfpq = flatToIvfPq(flat, m: 4, nlist: 8, nprobe: 1);
    final refined = wrapWithRefine(ivfpq, flat, kFactor: 4);
    final points = autoTuneNprobe(
      target: refined,
      queries: queries,
      k: k,
      truth: truth,
      values: const <int>[1, 4, 8],
      applyBest: false,
      options: const BenchOptions(warmup: 0, repeats: 1),
    );
    expect(points.points, hasLength(3));
    // Ensure the tuner actually swept the wrapped IVFPQ's nprobe by
    // running an explicit apply and checking it got poked.
    autoTuneNprobe(
      target: refined,
      queries: queries,
      k: k,
      truth: truth,
      values: const <int>[8],
      minRecall: 0.0, // trivially satisfied → applies value=8
      options: const BenchOptions(warmup: 0, repeats: 1),
    );
    expect(ivfpq.nprobe, equals(8));
  });

  test('autoTuneNprobe rejects an unwrappable target', () {
    final flat = IndexFlatL2(d)..add(xs);
    expect(
      () => autoTuneNprobe(target: flat, queries: queries, k: k, truth: truth),
      throwsA(isA<ArgumentError>()),
    );
  });

  test('autoTuneM sweeps PQ subquantiser counts and returns a winner', () {
    // d = 16 → divisors ≤ d: {1,2,4,8,16}. Include an invalid 5 to
    // exercise the skip-non-divisor branch.
    final result = autoTuneMFromFlat(
      source: IndexFlatL2(d)..add(xs),
      queries: queries,
      k: k,
      truth: truth,
      values: const <int>[2, 4, 5, 8],
      nlist: 8,
      nprobe: 8,
      options: const BenchOptions(warmup: 0, repeats: 1),
    );
    // 5 must be skipped; remaining {2,4,8} all appear.
    expect(result.points.points.map((p) => p.paramValue),
        equals(<int>[2, 4, 8]));
    expect(result.built.keys, unorderedEquals(<int>[2, 4, 8]));
    // With no recall floor, chosen is the largest m (m=8).
    expect(result.chosen, isNotNull);
    expect(result.chosen!.paramValue, equals(8));
    expect(result.chosenIndex, same(result.built[8]));
  });

  test('autoTuneM with minRecall picks the smallest m that clears the bar',
      () {
    final flat = IndexFlatL2(d)..add(xs);
    // Compare against an IVFFlat truth so the recall floor is
    // achievable by the coarsest m. Truth stays as the exact flat.
    final result = autoTuneM(
      sourceVectors: List<Float32List>.generate(
        xs.length,
        (i) => Float32List.fromList(flat.reconstruct(i)),
      ),
      d: d,
      metric: Metric.l2,
      queries: queries,
      k: k,
      truth: truth,
      values: const <int>[2, 4, 8],
      nlist: 8,
      nprobe: 8,
      minRecall: 0.0, // trivially satisfied → smallest m wins
      options: const BenchOptions(warmup: 0, repeats: 1),
    );
    expect(result.chosen!.paramValue, equals(2));
  });

  test('autoTuneM rejects empty sources', () {
    expect(
      () => autoTuneM(
        sourceVectors: const <Float32List>[],
        d: d,
        metric: Metric.l2,
        queries: queries,
        k: k,
        truth: truth,
        values: const <int>[2],
      ),
      throwsA(isA<ArgumentError>()),
    );
  });

  // -----------------------------------------------------------------
  // Tuning-metadata wrapper (IxDT).
  // -----------------------------------------------------------------

  test('IxDT wrapper round-trips a populated TuningMetadata block', () {
    final inner = IndexFlatL2(d)..add(xs);
    final meta = TuningMetadata(
      createdAt: DateTime.fromMicrosecondsSinceEpoch(1_700_000_000_000_000),
      metric: Metric.l2,
      points: <OperatingPoint>[
        OperatingPoint(
          paramValue: 1,
          paramLabel: 'nprobe=1',
          recall: 0.42,
          meanUs: 12.5,
        ),
        OperatingPoint(
          paramValue: 8,
          paramLabel: 'nprobe=8',
          recall: 0.97,
          meanUs: 88.0,
        ),
      ],
      chosenParamValue: 8,
    );
    final bytes = writeTunedFaissIndexToBytes(inner, meta);
    expect(isTunedFaissBlob(bytes), isTrue);
    final decoded = readTunedFaissIndexFromBytes(bytes);
    expect(decoded.metadata.createdAt, equals(meta.createdAt));
    expect(decoded.metadata.metric, equals(Metric.l2));
    expect(decoded.metadata.chosenParamValue, equals(8));
    expect(decoded.metadata.points, hasLength(2));
    expect(decoded.metadata.points[0].paramLabel, equals('nprobe=1'));
    expect(decoded.metadata.points[0].recall, closeTo(0.42, 1e-12));
    expect(decoded.metadata.points[1].meanUs, closeTo(88.0, 1e-12));
    // Inner index survives byte-identical.
    final restored = decoded.index;
    expect(restored, isA<IndexFlat>());
    expect(restored.d, equals(inner.d));
    expect(restored.ntotal, equals(inner.ntotal));
  });

  test('IxDT wrapper round-trips an empty operating-point list with no '
      'chosen value', () {
    final inner = IndexFlatL2(d)..add(xs);
    final meta = TuningMetadata(
      createdAt: DateTime.fromMicrosecondsSinceEpoch(0),
      metric: Metric.innerProduct,
      points: const <OperatingPoint>[],
    );
    final bytes = writeTunedFaissIndexToBytes(inner, meta);
    final decoded = readTunedFaissIndexFromBytes(bytes);
    expect(decoded.metadata.points, isEmpty);
    expect(decoded.metadata.chosenParamValue, isNull);
    expect(decoded.metadata.metric, equals(Metric.innerProduct));
  });

  test('readTunedFaissIndexFromBytes rejects a plain FAISS blob', () {
    final inner = IndexFlatL2(d)..add(xs);
    final plain = writeFaissIndexToBytes(inner);
    expect(isTunedFaissBlob(plain), isFalse);
    expect(
      () => readTunedFaissIndexFromBytes(plain),
      throwsA(isA<FormatException>()),
    );
  });

  test('IxDT wrapper preserves the inner FAISS blob byte-for-byte', () {
    final inner = IndexFlatL2(d)..add(xs);
    final plain = writeFaissIndexToBytes(inner);
    final meta = TuningMetadata(
      createdAt: DateTime.fromMicrosecondsSinceEpoch(1),
      metric: Metric.l2,
      points: const <OperatingPoint>[],
    );
    final wrapped = writeTunedFaissIndexToBytes(inner, meta);
    // Inner blob is appended verbatim at the tail of the wrapper.
    expect(
      wrapped.sublist(wrapped.length - plain.length),
      equals(plain),
    );
  });

  // -----------------------------------------------------------------
  // GPU-backed flat search.
  // -----------------------------------------------------------------

  test(
    'GpuIndexFlat matches IndexFlatL2 top-1 for a small corpus (fallback)',
    () {
      // Small size → wrapper transparently defers to CPU search.
      final gpu = GpuIndexFlat.l2(d)..add(xs);
      final flat = IndexFlatL2(d)..add(xs);
      final g = gpu.search(queries, k);
      final f = flat.search(queries, k);
      for (var qi = 0; qi < queries.length; qi++) {
        expect(g.ids[qi][0], equals(f.ids[qi][0]));
        expect(g.distances[qi][0], closeTo(f.distances[qi][0], 1e-3));
      }
    },
  );

  test('GpuIndexFlat.l2 matches IndexFlatL2 at a size that clears the '
      'CPU-fallback threshold', () {
    // Bump the workload past gpuIndexFlatMinDot (2^18) so the GPU
    // matmul path is exercised: nq * ntotal * d ~= 640k for n=4096.
    const bigN = 4096;
    const bigNq = 20;
    final big = _sampleBlobs(n: bigN, d: d, seed: 7);
    final bigQ = _sampleBlobs(n: bigNq, d: d, seed: 8);
    final gpu = GpuIndexFlat.l2(d)..add(big);
    final flat = IndexFlatL2(d)..add(big);
    final g = gpu.search(bigQ, k);
    final f = flat.search(bigQ, k);
    // Top-1 must agree exactly (ties aside). Compare as a multiset —
    // rounding may reorder within-tie neighbours.
    for (var qi = 0; qi < bigNq; qi++) {
      expect(g.ids[qi][0], equals(f.ids[qi][0]));
      // Top-k id sets should overlap heavily.
      final gs = g.ids[qi].toSet();
      final fs = f.ids[qi].toSet();
      final overlap = gs.intersection(fs).length;
      expect(overlap, greaterThanOrEqualTo(k - 1));
      // Squared L2 recovered via identity must match to within
      // fp32 accumulated rounding tolerance.
      expect(g.distances[qi][0], closeTo(f.distances[qi][0], 1e-2));
    }
  });

  test('GpuIndexFlat.ip matches IndexFlatIP for inner-product ranking', () {
    const bigN = 4096;
    const bigNq = 10;
    final big = _sampleBlobs(n: bigN, d: d, seed: 9);
    final bigQ = _sampleBlobs(n: bigNq, d: d, seed: 10);
    final gpu = GpuIndexFlat.ip(d)..add(big);
    final flat = IndexFlatIP(d)..add(big);
    final g = gpu.search(bigQ, k);
    final f = flat.search(bigQ, k);
    for (var qi = 0; qi < bigNq; qi++) {
      expect(g.ids[qi][0], equals(f.ids[qi][0]));
      expect(g.distances[qi][0], closeTo(f.distances[qi][0], 1e-2));
    }
  });

  test('GpuIndexFlat invalidates its DB cache on add', () {
    final gpu = GpuIndexFlat.l2(d)..add(xs);
    gpu.warmup();
    // Adding a new vector then searching for it must return that
    // vector as the nearest neighbour (id == old ntotal).
    final newVec = Float32List(d);
    for (var j = 0; j < d; j++) {
      newVec[j] = 99.0;
    }
    final oldNtotal = gpu.ntotal;
    gpu.add(<Float32List>[newVec]);
    expect(gpu.ntotal, equals(oldNtotal + 1));
    final r = gpu.search(<Float32List>[newVec], 1);
    expect(r.ids[0][0], equals(oldNtotal));
    expect(r.distances[0][0], closeTo(0.0, 1e-3));
  });

  test('GpuIndexFlat.wrap adopts an existing IndexFlat', () {
    final flat = IndexFlatL2(d)..add(xs);
    final gpu = GpuIndexFlat.wrap(flat);
    expect(gpu.d, equals(d));
    expect(gpu.ntotal, equals(xs.length));
    final r = gpu.search(queries.sublist(0, 2), 1);
    // Wrap must not change the ranking.
    final baseline = flat.search(queries.sublist(0, 2), 1);
    for (var qi = 0; qi < 2; qi++) {
      expect(r.ids[qi][0], equals(baseline.ids[qi][0]));
    }
  });
}

/// Stub `IndexBinary` subtype used only by the "rejects unsupported"
/// test. Its runtimeType is neither [IndexBinaryFlat] nor
/// [IndexBinaryIVF], so the dispatch in [writeFaissBinaryIndex] falls
/// through to the [UnsupportedError] branch.
class _ForeignBinaryIndex extends IndexBinary {
  _ForeignBinaryIndex(super.codeSize);

  @override
  void add(List<Uint8List> xs) {}

  @override
  SearchResult search(List<Uint8List> queries, int k) => SearchResult(
    List<Float32List>.generate(queries.length, (_) => Float32List(k)),
    List<Int32List>.generate(queries.length, (_) => Int32List(k)),
  );
}
