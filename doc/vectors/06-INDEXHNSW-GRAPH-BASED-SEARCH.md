# **6. INDEXHNSW: GRAPH-BASED SEARCH**

The IVF family works by partitioning space; `IndexHNSW` works by
building a **graph** whose short paths approximate nearest-neighbour
walks. It is a direct port of Malkov and Yashunin's Hierarchical
Navigable Small World algorithm ([arXiv:1603.09320](https://arxiv.org/abs/1603.09320))
at the same level of detail that `hnswlib` and FAISS's
`IndexHNSWFlat` use.

Source: [../../lib/core/vector_store/index_hnsw.dart](../../lib/core/vector_store/index_hnsw.dart).

## **6.1. The idea in one paragraph**

Build a multi-layer graph. Every node exists in layer 0; roughly
`1/M` of them also exist in layer 1; `1/M^2` in layer 2; etc.
Neighbours are only within the same layer. To search, start from an
entry point at the top layer, greedy-descend to the closest node
there, drop down a layer, greedy-descend again, and so on until
layer 0. At layer 0 do a beam search of width `efSearch` and return
the top `k`. Because the top layers are sparse the greedy descent
skips huge distances in one hop; the beam at layer 0 does the
fine-grained work.

## **6.2. Anatomy**

<sup>from [../../lib/core/vector_store/index_hnsw.dart](../../lib/core/vector_store/index_hnsw.dart)</sup>

```dart
class IndexHNSW extends Index {
  IndexHNSW({
    required int d,
    Metric metric = Metric.l2,
    this.M = 16,
    this.efConstruction = 100,
    this.efSearch = 32,
    this.seed = 1234,
  }) : Mmax = M,
       Mmax0 = 2 * M,
       _mL = 1.0 / math.log(M.toDouble()),
       super(d, metric);

  /// Max connections per node in layers > 0.
  final int M;
  /// Same as `M` (Malkov's default).
  final int Mmax;
  /// Larger budget for layer 0 (Malkov's default: `2 * M`).
  final int Mmax0;
  /// Beam size at insertion time (higher = better graph, slower build).
  final int efConstruction;
  /// Beam size at search time (higher = better recall, slower search).
  int efSearch;
  ...
}
```

Three tunable numbers, each with a distinct role.

## **6.3. The three dials**

| Dial | When it matters | Effect |
| --- | --- | --- |
| `M` | Build time. Baked into the graph. | Out-degree budget per layer. Bigger = more edges, more memory, better recall ceiling. Typical: 8-32. |
| `efConstruction` | Build time. Baked into the graph. | Beam width at insertion. Larger = graph quality up, build slower. Typical: 100-400. Setting it below `M` is pointless. |
| `efSearch` | Query time. Cheap to change. | Beam width at layer 0 during search. Larger = higher recall, slower per query. Auto-tunable. |

`efSearch` is the one you sweep in `autoTuneEfSearch` (chapter 9);
`M` and `efConstruction` you decide once at build time and live with.

## **6.4. Insertion (`add`)**

For each new vector:

1. Sample a random level `l ~ floor(-log(U) * mL)`, where `mL = 1/ln(M)`.
   Nodes are exponentially rarer at higher levels.
2. Store the raw vector in `_storage` (uncompressed float32 — this is
   `IndexHNSWFlat`, not a quantised variant).
3. Starting from the current entry point at the top layer, greedy-descend
   through layers `topLevel .. l+1` looking for the single best neighbour
   at each layer.
4. For layers `l .. 0`, run a beam search of width `efConstruction` to
   find candidate neighbours, then apply Malkov's `select-M-heuristic`
   (§4.3) to prune to at most `Mmax` (or `Mmax0` for layer 0) edges.
5. Insert bidirectional edges. If a neighbour now exceeds its degree
   budget, prune that neighbour too with the same heuristic.
6. If `l > topLevel`, this node becomes the new entry point.

## **6.5. Search**

Two phases:

1. **Descent.** From the entry point, greedy-search layer by layer down
   to layer 1. At each layer, from your current best node, examine its
   neighbours, jump to whichever is closer to the query, and repeat
   until no neighbour improves.
2. **Layer 0 beam.** At layer 0 run a beam of width `max(efSearch, k)`.
   Keep two heaps (candidates to expand, results seen so far), pull the
   closest unexpanded candidate, add its unvisited neighbours to both,
   and stop when the top-of-candidates is worse than the worst result.
   Return the top `k`.

Cost per query is roughly `efSearch * M * log(ntotal)`, empirically flat
in `ntotal` for constant recall.

## **6.6. Why the demo often shows HNSW as the fastest good option**

From the sample output in chapter 1:

```
HNSW32              build   485 ms   search      55.0 us/q   recall@10  99.7%
```

Compare against `IVFFlat nprobe=8` at 195 us/q, 97.8% recall. HNSW is
~3-4x faster and slightly more accurate — at the cost of a 6-8x
longer build. That trade is worth it for read-mostly corpora; it is
not worth it for corpora you rebuild every hour.

## **6.7. Gotchas**

* **Cosine similarity.** HNSW does not offer a first-class cosine
  metric. Normalise both database and query and use
  `Metric.innerProduct`. Wrapping in `L2NormTransform` via
  `IndexPreTransform` (chapter 7) makes this automatic.
* **Deletions.** Not supported. HNSW is append-only in this port
  (and in FAISS). If you need to churn, use IVFFlat.
* **Determinism.** Given a fixed `seed`, insertion order matters. If
  you change insertion order you will get a different graph — recall
  will be the same on average, results may differ.

Next chapter zooms out to composed indexes: pre-transforms, id maps,
refine wrappers, and the `indexFactory` string DSL that sits on top
of all of them.

---

<div align="right">

[&lt;&nbsp;&nbsp;Previous: INDEXIVFPQ: PRODUCT QUANTIZATION](./05-INDEXIVFPQ-PRODUCT-QUANTIZATION.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next: TRANSFORMS AND COMPOSED INDEXES&nbsp;&nbsp;&gt;](./07-TRANSFORMS-AND-COMPOSED-INDEXES.md)

</div>
