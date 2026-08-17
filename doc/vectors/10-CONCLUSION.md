# **10. CONCLUSION**

The adventure of a single vector, in one page:

1. Enter as a `Float32List` of dimension `d`, matching the metric the
   index was built for. Live in one contiguous
   `Float32List` of length `ntotal * d` in the concrete index of
   [../../lib/core/vector_store](../../lib/core/vector_store)
   (chapter 2).
2. If the index is `IndexFlat`, get scanned brute-force against every
   query. This is the recall-100% ground truth (chapter 3).
3. If the index is `IndexIVFFlat`, get routed into one of `nlist`
   k-means cells and only scanned when a query probes that cell
   (chapter 4).
4. If the index is `IndexIVFPQ`, get first residualized against a
   cell centroid, then replaced by an `m`-byte PQ code that
   dominates the memory story for billion-scale (chapter 5).
5. If the index is `IndexHNSW`, get inserted into a multi-layer
   navigable-small-world graph, reached at query time by greedy
   descent + a beam search of width `efSearch` (chapter 6).
6. Optionally get seen through pre-transforms (`L2Norm`, `PCA`,
   `RR`), have an application id attached via `IndexIDMap`, or be
   the target of a re-ranking `IndexRefineFlat` — all composable
   under the `indexFactory` DSL (chapter 7).
7. Get serialized into a byte-compatible FAISS blob, optionally
   wrapped in an `IxDT` tuning envelope that documents which
   `nprobe` / `efSearch` won the last sweep (chapter 8).
8. Get sweeped, benchmarked, and CLI-tooled into a curated
   deployment artefact (chapter 9).

## **10.1. Where to go next in the codebase**

If you want to extend the port, the highest-leverage places are:

* **New index types.** Every index has the same shape: state,
  optional `train`, `add`, `search`, plus an `_ioReadPayload` /
  `_ioWritePayload` pair for FAISS-format compatibility. Look at
  [../../lib/core/vector_store/index_lsh.dart](../../lib/core/vector_store/index_lsh.dart)
  for the smallest complete example.
* **New tuning targets.** `autoTuneNprobe`, `autoTuneEfSearch`,
  and `autoTuneM` all share the same `OperatingPoints` output
  shape, so adding e.g. `autoTuneKFactor` for `IndexRefineFlat` is
  a copy-paste-and-swap-the-inner-loop job in
  [../../lib/core/vector_store/auto_tune.dart](../../lib/core/vector_store/auto_tune.dart).
* **New CLI tools.** The three existing ones (`bench_faiss`,
  `faiss_describe`, `faiss_strip`) are all `< 350` lines and
  share no plumbing; copy the closest one.

## **10.2. Where to go next outside the codebase**

* [FAISS getting-started tutorial](https://github.com/facebookresearch/faiss/wiki/Getting-started).
  The demo in chapter 1 is a direct port.
* [FAISS index composition guide](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index).
  The rule-of-thumb tables in chapters 5 and 7 are simplified from this.
* [Malkov & Yashunin, "Efficient and robust approximate nearest neighbor
  search using Hierarchical Navigable Small World graphs" (arXiv:1603.09320)](https://arxiv.org/abs/1603.09320).
  The paper HNSW is ported from; the code in `index_hnsw.dart` cites
  it inline in the pruning routine.
* [Product Quantization for Nearest Neighbor Search — Jegou et al. 2011](https://ieeexplore.ieee.org/document/5432202).
  The canonical PQ / IVFADC reference.

## **10.3. Reading suggestion**

The most efficient way to learn this codebase is to run it. In order:

```
dart run bin/demo_faiss.dart
dart run bin/bench_faiss.dart --nb 20000 --nq 200 --d 64 --k 10 --pareto
dart run bin/faiss_describe.dart <some/file.faiss>
```

Each of those maps cleanly onto a chapter above.

---

<div align="right">

[&lt;&nbsp;&nbsp;Previous: AUTO-TUNING AND CLI TOOLS](./09-AUTO-TUNING-AND-CLI-TOOLS.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Home: Tutorial README](./README.md)

</div>
