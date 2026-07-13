# **4. INDEXIVFFLAT: CELL-PROBE SEARCH**

`IndexIVFFlat` is the first real approximate index. The trick is old
and simple: partition the feature space into `nlist` k-means cells,
assign every database vector to its closest centroid, and at query
time only look inside the `nprobe` cells nearest the query. If
`nprobe << nlist` you visit a small fraction of the database and pay
a small fraction of the flat cost.

Source: [../../lib/core/vector_store/index_ivf_flat.dart](../../lib/core/vector_store/index_ivf_flat.dart).

## **4.1. Anatomy**

<sup>from [../../lib/core/vector_store/index_ivf_flat.dart](../../lib/core/vector_store/index_ivf_flat.dart)</sup>

```dart
class IndexIVFFlat extends Index {
  IndexIVFFlat({
    required int d,
    required this.nlist,
    Metric metric = Metric.l2,
    this.nprobe = 1,
    this.kmeansIters = 20,
    this.seed = 1234,
  }) : quantizer = IndexFlat(d, metric),
       super(d, metric) {
    isTrained = false;
    _invLists = List<List<int>>.generate(nlist, (_) => <int>[]);
  }

  /// Coarse quantizer holding the `nlist` k-means centroids.
  final IndexFlat quantizer;

  /// Number of cells (k-means clusters).
  final int nlist;

  /// Number of cells to probe at search time.
  int nprobe;
  ...
}
```

Three pieces of state:

* `quantizer` — an `IndexFlat` that owns the k-means centroids. It is
  itself a full index (of the centroids), which is exactly what we
  need for "find the `nprobe` nearest cells to `q`".
* `_storage` — the same growing `Float32List` as `IndexFlat`, holding
  the raw database vectors. IVFFlat does not compress; it just
  reorders the search work.
* `_invLists[cell]` — for each cell, the ordered ids of vectors
  assigned to it. Insertion order.

## **4.2. Train**

`train(xs)` runs k-means over `xs` to produce `nlist` centroids and
loads them into `quantizer`. After that `isTrained = true` and you
may `add`.

Key rules of thumb (same as upstream FAISS):

* `nlist` is typically `~sqrt(nb)`. The
  [../../lib/core/vector_store/index_convert.dart](../../lib/core/vector_store/index_convert.dart)
  helper `flatToIvfFlat` picks exactly that default.
* Training set can be a subset of the corpus (100k vectors is more
  than enough for millions).

## **4.3. Add**

`add(xs)` does three things per vector:

1. Ask `quantizer.search(x, 1)` for the id of the nearest centroid
   (this is a brute-force scan of `nlist` centroids — cheap).
2. Append the raw vector to `_storage` at position `ntotal + i`.
3. Push the new id onto `_invLists[centroid]`.

After a full `add`, `_invLists` partitions `[0, ntotal)` and every
vector is exactly once in exactly one cell.

## **4.4. Search: the tradeoff knob**

Search runs in two stages:

1. **Coarse.** `quantizer.search(q, nprobe)` — the `nprobe` nearest
   cells to `q`.
2. **Fine.** For each probed cell, iterate `_invLists[cell]`, fetch
   the raw vector from `_storage`, compute the exact distance, and
   push into a `TopK(k)` heap.

The whole design collapses to one dial: **`nprobe`**.

| `nprobe` | Behaviour |
| ---: | --- |
| `1` | Fastest. Visits ~`ntotal / nlist` vectors per query. Recall depends heavily on whether the true nearest cell contains all the top-k. |
| `~sqrt(nlist)` | Common default. Recall ~90-98% at a fraction of flat cost. |
| `nlist` | Equivalent to brute force plus one extra pass through the coarse quantizer. Wasteful; use `IndexFlat` instead. |

In the sample demo output:

```
IVFFlat nprobe=8    build  62 ms   search  195 us/q   recall@10  97.8%
IVFFlat nprobe=1    build  59 ms   search   30 us/q   recall@10  61.4%
```

Same index built twice, only `nprobe` changed. 8x more probed cells,
~6.5x more work, +36 percentage points of recall.

## **4.5. Why recall can drop for a cell-probe index**

A query near a cell boundary might have its true nearest neighbour
in the *neighbouring* cell. If `nprobe = 1`, you never look there.
That is why:

* You always want `nprobe > 1` in practice.
* HNSW (chapter 6) does better on this case because its graph edges
  cross cell boundaries by construction.
* Auto-tuning nprobe (chapter 9) sweeps a small ladder of values and
  picks the one that hits your recall target — exactly because the
  right value is workload-dependent.

## **4.6. Range search**

IVFFlat supports `rangeSearch(queries, radius)` too. Same two stages;
the fine loop just keeps every hit that satisfies the radius
predicate instead of pushing into a top-k heap, and packs the results
into the CSR `RangeSearchResult` layout described in chapter 2.

## **4.7. Room to grow**

IVFFlat still stores the raw vectors uncompressed. For a database of
billions of high-dimensional vectors this dominates memory. The next
chapter replaces per-vector storage with a Product Quantizer:
`IndexIVFPQ`.

---

<div align="right">

[&lt;&nbsp;&nbsp;Previous: INDEXFLAT: THE GROUND TRUTH](./03-INDEXFLAT-THE-GROUND-TRUTH.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next: INDEXIVFPQ: PRODUCT QUANTIZATION&nbsp;&nbsp;&gt;](./05-INDEXIVFPQ-PRODUCT-QUANTIZATION.md)

</div>
