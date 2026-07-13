# **0. INFRASTRUCTURE**

Before we can trace a single vector through the port, it helps to know
what the plumbing looks like. Everything vector-search-related lives in
two places:

* [../../lib/core/vector_store](../../lib/core/vector_store) — the port
  itself. Pure Dart, `Float32List`-based, CPU-only for now, no FFI.
* [../../bin](../../bin) — command-line demos and tools that consume
  the library from the outside (`demo_faiss.dart`, `bench_faiss.dart`,
  `faiss_describe.dart`, `faiss_strip.dart`).

## **0.1. Package layout**

The top-level barrel [../../lib/dart_pytorch.dart](../../lib/dart_pytorch.dart)
re-exports every public vector-store type. Any consumer only needs:

```dart
import 'package:dart_pytorch/dart_pytorch.dart';
```

Inside `lib/core/vector_store/` the files split by responsibility:

| File | What lives here |
| --- | --- |
| [index.dart](../../lib/core/vector_store/index.dart) | Abstract `Index`, `Metric`, `SearchResult`, `RangeSearchResult`, the shared `TopK` heap |
| [index_flat.dart](../../lib/core/vector_store/index_flat.dart) | `IndexFlat`, `IndexFlatL2`, `IndexFlatIP` — the brute-force ground truth |
| [index_ivf_flat.dart](../../lib/core/vector_store/index_ivf_flat.dart) | `IndexIVFFlat` — k-means cells + flat per-cell storage |
| [index_ivf_pq.dart](../../lib/core/vector_store/index_ivf_pq.dart) | `IndexIVFPQ` — IVF with product-quantized residuals |
| [index_pq.dart](../../lib/core/vector_store/index_pq.dart) | `ProductQuantizer`, `IndexPQ` |
| [index_hnsw.dart](../../lib/core/vector_store/index_hnsw.dart) | `IndexHNSW` — Malkov-Yashunin hierarchical graph |
| [index_refine_flat.dart](../../lib/core/vector_store/index_refine_flat.dart) | `IndexRefineFlat` — re-rank an approximate index with exact fp32 |
| [index_id_map.dart](../../lib/core/vector_store/index_id_map.dart) | `IndexIDMap` — attach arbitrary i64 ids to any inner index |
| [vector_transform.dart](../../lib/core/vector_store/vector_transform.dart) + `l2_norm_transform.dart` / `pca_transform.dart` / `random_rotation_transform.dart` | Pre-transforms wired via `IndexPreTransform` |
| [index_factory.dart](../../lib/core/vector_store/index_factory.dart) | `indexFactory(d, "PCA32,IVF64,PQ16")` string-DSL builder |
| [kmeans.dart](../../lib/core/vector_store/kmeans.dart) | Shared k-means used by IVF and PQ training |
| [bench.dart](../../lib/core/vector_store/bench.dart) | `BenchResult`, `runBench`, `toBenchMarkdown`, `paretoFrontier` |
| [auto_tune.dart](../../lib/core/vector_store/auto_tune.dart) | `autoTuneNprobe`, `autoTuneEfSearch`, `autoTuneM`, `OperatingPoint(s)` |
| [faiss_io.dart](../../lib/core/vector_store/faiss_io.dart) | On-disk FAISS binary format + `IxDT` tuning wrapper |
| [index_io.dart](../../lib/core/vector_store/index_io.dart) | Low-level `IoReader` / `IoWriter` + the port's native `FAISDART` format |
| [gpu_index_flat.dart](../../lib/core/vector_store/gpu_index_flat.dart) | GPU-backed flat search (fallback-safe) |

## **0.2. Numerical conventions**

The port sticks to FAISS' choices so blobs stay interoperable:

* **Dtype.** All database vectors and queries are `Float32List`. Every
  distance is a `double` internally but rounded to `float32` in
  `SearchResult.distances`.
* **Layout.** Storage is a single flat `Float32List` of length
  `ntotal * d`. Vector `i` lives at byte offsets `[i*d, (i+1)*d)`.
  This keeps the hot inner loop cache-friendly.
* **Metrics.** `Metric.l2` is *squared* Euclidean distance (the FAISS
  default — no `sqrt`). `Metric.innerProduct` is a raw dot product;
  cosine similarity is L2-normalise on both sides plus IP search.
* **Ids.** Sequential `int64` from `0` at `add` time. To attach your
  own ids, wrap the index in `IndexIDMap`.
* **Endianness.** The on-disk FAISS format is little-endian only and
  assumes 64-bit `idx_t` (see chapter 8).

## **0.3. What is NOT covered here**

* GPU IVF / GPU PQ — only `GpuIndexFlat` is wired today.
* Compressed on-disk formats (OPQ, IVFSpectralHash, HNSW-with-PQ).
* Distributed / sharded query routing beyond `IndexShards` and
  `IndexReplicas` primitives.

Everything else is fair game and gets traced end-to-end in the next
nine chapters.

---

<div align="right">

[&lt;&nbsp;&nbsp;Previous: Home](./README.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next: RUNNING THE FAISS DEMO&nbsp;&nbsp;&gt;](./01-RUNNING-THE-FAISS-DEMO.md)

</div>
