# **5. INDEXIVFPQ: PRODUCT QUANTIZATION**

`IndexIVFPQ`, also known in the literature as **IVFADC**, is FAISS'
workhorse for billion-scale search. It keeps the cell-probe skeleton
from chapter 4 but replaces per-cell raw storage with a Product
Quantizer. Memory drops from `d * 4` bytes per vector to `m` bytes
per vector; `m` is usually 8, 16, or 32.

Sources:
* [../../lib/core/vector_store/index_ivf_pq.dart](../../lib/core/vector_store/index_ivf_pq.dart) — the composed index.
* [../../lib/core/vector_store/index_pq.dart](../../lib/core/vector_store/index_pq.dart) — the underlying `ProductQuantizer`.

## **5.1. What is a Product Quantizer?**

Split each `d`-dim vector into `m` equal-size sub-vectors. Independently
k-means each of the `m` "columns" of the training set to produce `2^nbits`
centroids per column (usually `nbits = 8`, so 256 centroids). To encode
a new vector, replace each sub-vector by the id (a byte) of its nearest
column-centroid. To reconstruct, concatenate the corresponding
centroids.

Every vector shrinks from `d * 4` bytes to `m` bytes. For `d = 128` and
`m = 16`, that is a **32x compression ratio**.

## **5.2. Pipeline recap**

<sup>from [../../lib/core/vector_store/index_ivf_pq.dart](../../lib/core/vector_store/index_ivf_pq.dart)</sup>

```dart
/// Pipeline:
///   1. k-means over the raw vectors -> `nlist` coarse centroids.
///   2. Each database vector's residual (`x - centroid`) is encoded
///      with a Product Quantizer of `m` subquantizers.
///   3. At query time, probe `nprobe` cells; for each probed cell,
///      compute the query's residual w.r.t. that cell's centroid,
///      build a PQ distance LUT, and scan the inverted list via ADC.
```

The two additions over `IndexIVFFlat`:

* **Residuals.** We never PQ-encode the raw vector, only `x - c_cell`.
  The residual has much lower variance, so the PQ can afford tighter
  cells and preserve more precision. This is the "ADC" (Asymmetric
  Distance Computation) part.
* **LUT-based scan.** At query time we compute, once per probed cell,
  a `m x 256` lookup table of squared distances from the query's
  residual to each PQ centroid. The inner loop then decodes one byte
  at a time and accumulates the LUT entries — no float multiplication
  per hit.

## **5.3. Storage layout**

<sup>from [../../lib/core/vector_store/index_ivf_pq.dart](../../lib/core/vector_store/index_ivf_pq.dart)</sup>

```dart
/// `_invListsIds[cell][slot]` = database id of that stored vector.
late final List<List<int>> _invListsIds;

/// `_invListsCodes[cell]` = flat list of PQ code bytes for the cell,
/// length `m * len(_invListsIds[cell])`.
late final List<List<int>> _invListsCodes;
```

Two lists per cell:

* `_invListsIds[cell]` — same as IVFFlat, insertion-ordered ids.
* `_invListsCodes[cell]` — a byte-packed run of PQ codes. Slot `j` in
  the cell owns bytes `[j*m, (j+1)*m)`. This is the compressed store.

Notice what's *missing*: no `_storage` of raw floats. The full
database is only the PQ codes plus the two centroid tables (coarse
`nlist * d` floats, PQ `m * 256 * (d/m)` floats).

## **5.4. Build parameters**

<sup>from [../../lib/core/vector_store/index_ivf_pq.dart](../../lib/core/vector_store/index_ivf_pq.dart)</sup>

```dart
IndexIVFPQ({
  required int d,
  required this.nlist,
  required this.m,
  int nbits = 8,
  this.nprobe = 1,
  this.kmeansIters = 20,
  this.pqKmeansIters = 25,
  this.seed = 1234,
  Metric metric = Metric.l2,
})
```

The three dials you actually touch:

| Dial | Effect |
| --- | --- |
| `nlist` | As in IVFFlat. `~sqrt(nb)` default. |
| `nprobe` | As in IVFFlat. Query-time speed vs recall. |
| `m` | Bytes per encoded vector. Must divide `d`. Bigger = more accurate reconstruction, more memory, slower scan. |

`nbits` is almost always left at 8 (256 centroids per sub-quantizer).
Reducing it doubles compression but loses recall fast.

## **5.5. Why recall dips even at large nprobe**

Unlike `IndexIVFFlat` where `nprobe = nlist` is exact, `IndexIVFPQ`
loses accuracy at the **PQ encoding step**, before search even
starts. Every vector was replaced by its `m`-byte code — there is
information you cannot recover. So `IVFPQ`'s recall ceiling is
strictly below flat, regardless of nprobe. To get back to near-flat
recall you wrap it in `IndexRefineFlat` (chapter 7), which re-scores
the top `k_factor * k` PQ candidates using an inner exact
`IndexFlat`.

## **5.6. `autoTuneM`: the expensive dial**

`nprobe` is a query-time knob — a single `IVFPQ` can be probed at
1, 2, 4, 8, 16 in one auto-tune sweep. `m`, though, is baked into
the index at build time. Changing `m` requires re-training the PQ
codebooks and re-encoding every vector, so `autoTuneM` in
[../../lib/core/vector_store/auto_tune.dart](../../lib/core/vector_store/auto_tune.dart)
sweeps candidate values by rebuilding the whole index each time.
Expect it to be slow.

The returned `TuneMResult` records the built indexes so you can pick
one without rebuilding again. Chapter 9 walks through the API.

## **5.7. Rule of thumb**

| Corpus size | Reasonable starting point |
| --- | --- |
| `< 100k` | `IndexFlat` or `IndexIVFFlat` |
| `100k - 10M` | `IndexIVFFlat` or `IndexHNSW` |
| `10M - 1B` | `IndexIVFPQ` + optional `IndexRefineFlat` |
| `> 1B` | `IndexIVFPQ` with tuned `m`, sharded via `IndexShards` |

The next chapter moves off the k-means grid entirely and searches a
graph instead.

---

<div align="right">

[&lt;&nbsp;&nbsp;Previous: INDEXIVFFLAT: CELL-PROBE SEARCH](./04-INDEXIVFFLAT-CELL-PROBE-SEARCH.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next: INDEXHNSW: GRAPH-BASED SEARCH&nbsp;&nbsp;&gt;](./06-INDEXHNSW-GRAPH-BASED-SEARCH.md)

</div>
