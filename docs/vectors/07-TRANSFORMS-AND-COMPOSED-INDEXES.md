# **7. TRANSFORMS AND COMPOSED INDEXES**

The indexes in chapters 3-6 are the "leaves" of the search tree.
Everything interesting in a production pipeline stacks additional
layers around them: pre-transforms that reshape the vector space,
id-maps that swap in application ids, and refine wrappers that
recover recall lost to compression. This chapter walks through the
composed types and finishes with `indexFactory` — the string DSL
that assembles them.

## **7.1. Pre-transforms**

A pre-transform is a bijective (or at least well-defined-forward)
`d_in -> d_out` map applied to every vector on the way into `add` and
`search`. The port ships four:

| File | Class | Purpose |
| --- | --- | --- |
| [l2_norm_transform.dart](../../lib/core/vector_store/l2_norm_transform.dart) | `L2NormTransform` | Divide by `||x||_2`. Turns IP into cosine. `d_out = d_in`. |
| [random_rotation_transform.dart](../../lib/core/vector_store/random_rotation_transform.dart) | `RandomRotationTransform` | Orthonormal `d x d` rotation. Useful before PQ to spread information across sub-quantizers. |
| [pca_transform.dart](../../lib/core/vector_store/pca_transform.dart) | `PCATransform` | Learned `d_out x d_in` projection + mean-centring. Optional whitening. |
| [vector_transform.dart](../../lib/core/vector_store/vector_transform.dart) | Base `VectorTransform` | Shared interface (`dIn`, `dOut`, `apply`, `applyMany`, `isTrained`, `train`). |

You wire one or more transforms in front of an inner index using
[../../lib/core/vector_store/index_pre_transform.dart](../../lib/core/vector_store/index_pre_transform.dart):

```dart
final index = IndexPreTransform(
  chain: [L2NormTransform(dIn), PCATransform(dIn: dIn, dOut: 32)],
  inner: IndexIVFFlat(d: 32, nlist: 64, nprobe: 8),
);
```

Order matters: transforms are applied left-to-right, and the inner
index sees vectors of dimension equal to the **last** transform's
`dOut`.

Training semantics: calling `train(xs)` on the pre-transform will
train every un-trained transform in the chain **in order**, threading
the reduced-dimension output through, and then train the inner index
on the final vectors. `add` and `search` do the same forward pass
without training.

## **7.2. `IndexIDMap`**

Source: [../../lib/core/vector_store/index_id_map.dart](../../lib/core/vector_store/index_id_map.dart).

Every index in chapters 3-6 gives you sequential `[0, ntotal)` ids
whether you like it or not. `IndexIDMap` fixes that: it wraps any
inner index and lets `add_with_ids(xs, ids)` attach an arbitrary
`int64` per vector. Searches return your ids, not the inner ids.

```dart
final index = IndexIDMap(IndexFlatL2(d));
index.addWithIds(xs, myIds); // myIds is Int64List
final r = index.search(qs, 10);
// r.ids[qi][j] is now in myIds space, not [0, ntotal).
```

Internally it stores an `Int64List` translation table, indexed by
the inner id.

## **7.3. `IndexRefineFlat`**

Source: [../../lib/core/vector_store/index_refine_flat.dart](../../lib/core/vector_store/index_refine_flat.dart).

`IndexRefineFlat` is the answer to "IVFPQ lost too much recall".
It wraps an approximate `base` index plus a full-precision
`IndexFlat` that mirrors the same vectors. At search time it asks
`base` for the top `k_factor * k` candidates, re-ranks those with
exact fp32 distances via the inner flat, and returns the true top
`k`.

Cost: memory doubles (raw floats live in both places). Benefit:
recall recovers to something very close to `IndexFlat` while the
candidate-generation step stays fast.

Building one via the helper in
[../../lib/core/vector_store/index_convert.dart](../../lib/core/vector_store/index_convert.dart):

```dart
final flat = IndexFlatL2(d)..add(xs);
final ivfpq = flatToIvfPq(flat, nlist: 64, m: 16, nprobe: 8);
final refined = wrapWithRefine(ivfpq, flat, kFactor: 4);
```

Note that `wrapWithRefine` requires `base.d == src.d`, matching
metrics, matching `ntotal`, and `base.isTrained`.

## **7.4. `IndexShards` and `IndexReplicas`**

Two small composers for scale-out patterns:

* [index_shards.dart](../../lib/core/vector_store/index_shards.dart) —
  fan out `add` and `search` across N independent inner indexes, each
  holding a slice of the corpus, and merge the top-k results in the
  parent. Ids stay unique because each shard gets a disjoint id range.
* [index_replicas.dart](../../lib/core/vector_store/index_replicas.dart) —
  same corpus in every replica; round-robin queries across them for
  throughput.

Both are the pure-Dart shape of the eventual multi-process or
multi-GPU story. They're not doing IPC, just batching.

## **7.5. `indexFactory`**

Source: [../../lib/core/vector_store/index_factory.dart](../../lib/core/vector_store/index_factory.dart).

The composed types in this chapter are exactly what makes FAISS
production configs unreadable. `indexFactory` lets you spell a whole
chain as a short string:

<sup>from [../../lib/core/vector_store/index_factory.dart](../../lib/core/vector_store/index_factory.dart)</sup>

```dart
/// A description is a comma-separated chain of tokens:
///   * pre-transforms (left side, in order): `L2Norm`, `RR`,
///     `PCA<dOut>`, `PCAW<dOut>` (whitened).
///   * an index specifier (right side, exactly one): `Flat`,
///     `IVF<nlist>,Flat`, `IVF<nlist>,PQ<m>[x<nbits>]`, `PQ<m>[x<nbits>]`,
///     `HNSW<M>`, `SQ8`, `RFlat<inner>` (refine wrapper), `LSH<nbits>`.
///
/// Examples:
///   * `"Flat"`
///   * `"PCA32,IVF64,Flat"`
///   * `"L2Norm,IVF128,PQ16"`
///   * `"HNSW32"`
///   * `"L2Norm,LSH128"`
```

Metric is a parameter, not part of the string, matching FAISS'
`index_factory(d, description, metric)` API:

```dart
final idx = indexFactory(128, 'L2Norm,IVF128,PQ16',
    metric: Metric.innerProduct);
idx.train(xs);
idx.add(xs);
```

Anything you can build with `indexFactory` can also be built by hand;
the factory is just a parser for the string DSL over the same public
constructors.

## **7.6. Rule of thumb for composition**

| Goal | Composition |
| --- | --- |
| Cosine similarity | `L2Norm,<inner>` with `Metric.innerProduct` |
| Reduce dimensionality before quantization | `PCA<dOut>,<inner>` |
| Spread info across PQ subquantizers | `RR,PQ<m>` or `RR,IVF<nlist>,PQ<m>` |
| Recover recall lost to PQ | `RFlat<inner>` via `wrapWithRefine` |
| Custom application ids | `IndexIDMap(inner)` |

Now that we have indexes and composed indexes, the last thing we need
is a way to save and restore them. The next chapter covers the FAISS
on-disk format and this port's `IxDT` tuning wrapper.

---

<div align="right">

[&lt;&nbsp;&nbsp;Previous: INDEXHNSW: GRAPH-BASED SEARCH](./06-INDEXHNSW-GRAPH-BASED-SEARCH.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next: PERSISTENCE: FAISS FORMAT AND IXDT&nbsp;&nbsp;&gt;](./08-PERSISTENCE-FAISS-FORMAT-AND-IXDT.md)

</div>
