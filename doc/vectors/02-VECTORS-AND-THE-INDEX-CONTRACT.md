# **2. VECTORS AND THE INDEX CONTRACT**

Before we dive into any particular index type we need to agree on the
shape of the data and the API that every index implements. Both live
in [../../lib/core/vector_store/index.dart](../../lib/core/vector_store/index.dart).

## **2.1. What is a "vector" here?**

A vector is a `Float32List` of length `d`. A database is a
`List<Float32List>` where every entry has exactly length `d` — the
port does not accept ragged input and will throw `ArgumentError` at
`add`-time if you try.

The stored representation, on the other hand, is one big contiguous
`Float32List` of length `ntotal * d`. Every concrete index owns one
(or a few) of these buffers and grows them by doubling. Vector `i`
lives at `[i*d, (i+1)*d)`. The reason for this layout is exactly the
reason FAISS uses it: the search hot loop dereferences one vector per
comparison, and a contiguous slice fits nicely in L1.

## **2.2. Metrics**

<sup>from [../../lib/core/vector_store/index.dart](../../lib/core/vector_store/index.dart)</sup>

```dart
/// Distance metric used by an [Index].
///
/// * [Metric.l2] - squared Euclidean distance (the FAISS default).
/// * [Metric.innerProduct] - dot product. Cosine similarity is obtained
///   by pre-normalising both database and query vectors.
enum Metric { l2, innerProduct }
```

Two things to note:

* `Metric.l2` is *squared* Euclidean distance. FAISS omits the `sqrt`
  because k-NN ordering is invariant under monotonic transforms and
  the square root would be pure waste.
* There is no first-class cosine metric; you compose it. Either wrap
  your index in [L2NormTransform](../../lib/core/vector_store/l2_norm_transform.dart)
  via `IndexPreTransform` (chapter 7), or pre-normalise your vectors
  yourself before `add` / `search`. Then set the metric to
  `Metric.innerProduct` and every dot product is a cosine similarity
  in `[-1, 1]`.

## **2.3. The `Index` contract**

Every concrete index derives from a small abstract base:

* `int d` — vector dimension, fixed at construction.
* `Metric metric` — fixed at construction.
* `int ntotal` — how many vectors are currently in the index.
* `bool isTrained` — `true` for [IndexFlat](../../lib/core/vector_store/index_flat.dart),
  `false` until `train()` is called for the IVF and PQ families.
* `void add(List<Float32List> xs)` — append vectors, assigning ids
  `[ntotal, ntotal + xs.length)`.
* `SearchResult search(List<Float32List> queries, int k)` — return the
  top-`k` neighbours per query.
* `void train(List<Float32List> xs)` — optional; a no-op for `IndexFlat`.

Anything else (range search, id remapping, GPU offload) is layered on
top and covered in the type-specific chapters.

## **2.4. `SearchResult`**

Every search returns the same struct:

<sup>from [../../lib/core/vector_store/index.dart](../../lib/core/vector_store/index.dart)</sup>

```dart
class SearchResult {
  SearchResult(this.distances, this.ids);
  final List<Float32List> distances;
  final List<Int32List> ids;

  int get nq => distances.length;
  int get k => nq == 0 ? 0 : distances[0].length;
}
```

So `distances[qi][j]` is the distance from query `qi` to its `j`th
neighbour and `ids[qi][j]` is that neighbour's integer id. Two edge
cases to remember:

* If an index has fewer than `k` vectors for a query, unused slots
  are filled with `double.infinity` for L2 (or `-double.infinity`
  for IP) and `-1` for ids.
* Ids default to sequential `[0, ntotal)` at `add` time. If you want
  arbitrary application ids, wrap in
  [IndexIDMap](../../lib/core/vector_store/index_id_map.dart) — that
  index stores an `int64` translation table on top of any inner
  index.

## **2.5. Range search (`RangeSearchResult`)**

Only the flat and IVF families support radius search today. The
return type mirrors FAISS' CSR layout — one flat `distances` /
`ids` buffer plus a `limits` array so `perQuery(qi) = [limits[qi],
limits[qi+1])`:

<sup>from [../../lib/core/vector_store/index.dart](../../lib/core/vector_store/index.dart)</sup>

```dart
class RangeSearchResult {
  final Int32List limits;
  final Float32List distances;
  final Int32List ids;

  int get nq => limits.length - 1;
  int get totalMatches => limits.isEmpty ? 0 : limits[limits.length - 1];
  int lengthFor(int qi) => limits[qi + 1] - limits[qi];
  ...
}
```

The semantics of "radius" flip with the metric: for `Metric.l2` a
match satisfies `dist <= radius`, but for `Metric.innerProduct` it
satisfies `dot >= radius` (i.e. the radius is a **similarity**
threshold, not a distance one). This too matches FAISS.

## **2.6. The shared top-k heap**

Every non-graph index uses the same `TopK` structure from
[../../lib/core/vector_store/index.dart](../../lib/core/vector_store/index.dart)
to accumulate its best-k results as it scans a candidate stream.
It is a fixed-capacity binary heap that pushes candidates in and pops
sorted at the end. The reason we mention it here is that when you
read the search loops in later chapters, you will see the same
`TopK(k, maxIsWorst: metric == Metric.l2)` pattern everywhere.

With the data shape and API pinned down, we can now walk through the
concrete indexes. The next chapter starts with the simplest one:
brute force.

---

<div align="right">

[&lt;&nbsp;&nbsp;Previous: RUNNING THE FAISS DEMO](./01-RUNNING-THE-FAISS-DEMO.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next: INDEXFLAT: THE GROUND TRUTH&nbsp;&nbsp;&gt;](./03-INDEXFLAT-THE-GROUND-TRUTH.md)

</div>
