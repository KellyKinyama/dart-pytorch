# **3. INDEXFLAT: THE GROUND TRUTH**

`IndexFlat` is brute-force k-NN. It stores every vector uncompressed
and scans all of them at query time. Nothing to train, no cells, no
graph, no compression. It is the recall-100% row in the demo output
and the reference every other index is measured against.

Source: [../../lib/core/vector_store/index_flat.dart](../../lib/core/vector_store/index_flat.dart).

Two subclasses fix the metric for you:

```dart
final l2 = IndexFlatL2(d);   // metric = Metric.l2
final ip = IndexFlatIP(d);   // metric = Metric.innerProduct
```

Both are just thin constructors around `IndexFlat(d, metric)`.

## **3.1. Storage**

<sup>from [../../lib/core/vector_store/index_flat.dart](../../lib/core/vector_store/index_flat.dart)</sup>

```dart
class IndexFlat extends Index {
  IndexFlat(super.d, super.metric);

  Float32List _storage = Float32List(0);
  int _capacity = 0; // in vectors
  ...
}
```

The whole database is one flat `Float32List`. `_capacity` counts
vectors (not floats) and doubles on demand. Growth is amortised O(1):

<sup>from [../../lib/core/vector_store/index_flat.dart](../../lib/core/vector_store/index_flat.dart)</sup>

```dart
if (ntotal + n > _capacity) {
  var newCap = _capacity == 0 ? 1024 : _capacity;
  while (newCap < ntotal + n) {
    newCap *= 2;
  }
  final grown = Float32List(newCap * d);
  for (var i = 0; i < ntotal * d; i++) {
    grown[i] = _storage[i];
  }
  _storage = grown;
  _capacity = newCap;
}
```

After the `add` loop, `ntotal += n`. That is the entire write path.

## **3.2. `reconstruct`**

Because storage is uncompressed, a flat index can hand any vector
back verbatim. This is used by the composed indexes (chapter 7): an
`IndexRefineFlat` wraps an approximate index but keeps an inner
`IndexFlat` on the side purely so it can re-rank the top candidates
by exact distance.

<sup>from [../../lib/core/vector_store/index_flat.dart](../../lib/core/vector_store/index_flat.dart)</sup>

```dart
/// Direct access to the encoded vector at position `id`.
/// Returns a view; do not mutate.
Float32List reconstruct(int id) {
  if (id < 0 || id >= ntotal) {
    throw RangeError('id $id out of range [0, $ntotal)');
  }
  return Float32List.sublistView(_storage, id * d, (id + 1) * d);
}
```

`Float32List.sublistView` is zero-copy, so this is O(1) and safe to
call in a tight inner loop.

## **3.3. The search hot loop**

<sup>from [../../lib/core/vector_store/index_flat.dart](../../lib/core/vector_store/index_flat.dart)</sup>

```dart
@override
SearchResult search(List<Float32List> queries, int k) {
  final nq = queries.length;
  final distances = List<Float32List>.generate(nq, (_) => Float32List(k));
  final ids = List<Int32List>.generate(nq, (_) => Int32List(k));

  for (var qi = 0; qi < nq; qi++) {
    final q = queries[qi];
    if (q.length != d) {
      throw ArgumentError('query $qi length ${q.length} != $d');
    }
    final heap = TopK(k, maxIsWorst: metric == Metric.l2);

    // Hot loop: iterate every stored vector.
    for (var vi = 0; vi < ntotal; vi++) {
      final base = vi * d;
      var s = 0.0;
      if (metric == Metric.l2) {
        for (var j = 0; j < d; j++) {
          final diff = q[j] - _storage[base + j];
          s += diff * diff;
        }
      } else {
        for (var j = 0; j < d; j++) {
          s += q[j] * _storage[base + j];
        }
      }
      heap.push(s, vi);
    }
    heap.drainSorted(distances[qi], ids[qi]);
  }
  return SearchResult(distances, ids);
}
```

Two things to notice:

1. **The metric branch is outside the inner loop.** For a scan of
   `ntotal * d` floats you want the JIT to specialise one tight body,
   not check `metric` on every iteration. FAISS' C++ takes the same
   shape.
2. **`maxIsWorst` inverts the heap orientation.** For L2 (smaller is
   better) the heap keeps the *largest* seen so far at the top so we
   can evict it. For IP (larger is better) the invariant flips.

## **3.4. When to actually use it**

Never in production for large `nb`. Always for:

* small `nb` (say `< 10k` for `d = 64`) where the constants win;
* correctness testing — every other index in this repo has unit tests
  that build a matching `IndexFlat`, run both, and check that
  approximate recall stays above a threshold;
* refine wrappers — `IndexRefineFlat` re-ranks approximate hits with
  exact distances (chapter 7);
* GPU offload — [`GpuIndexFlat`](../../lib/core/vector_store/gpu_index_flat.dart)
  is a drop-in replacement that batches the query-vs-database matmul
  onto CUDA when the workload is large enough to pay off.

## **3.5. Why the demo starts here**

Every recall number you will see later — 61.4%, 97.8%, 99.7% — is
`|approx_topk intersect flat_topk| / k`. The flat result *is* the
denominator. If flat is broken, every number in the report is a lie.
So the demo builds `IndexFlatL2` first, stores its `SearchResult` as
`truth`, and passes that into `_recallAtK` for every approximate row.

The next chapter takes the first step away from brute force:
inverted-file cell-probe search.

---

<div align="right">

[&lt;&nbsp;&nbsp;Previous: VECTORS AND THE INDEX CONTRACT](./02-VECTORS-AND-THE-INDEX-CONTRACT.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next: INDEXIVFFLAT: CELL-PROBE SEARCH&nbsp;&nbsp;&gt;](./04-INDEXIVFFLAT-CELL-PROBE-SEARCH.md)

</div>
