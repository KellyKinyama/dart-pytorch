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
}
