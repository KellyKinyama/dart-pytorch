# **1. RUNNING THE FAISS DEMO**

The fastest way to see the whole toolkit at once is to run the port of
the upstream FAISS getting-started tutorial:
[../../bin/demo_faiss.dart](../../bin/demo_faiss.dart).

```
dart run bin/demo_faiss.dart
```

Sample output (numbers depend on your CPU):

```
=== Dart FAISS demo - d=64  nb=5000  nq=100  k=10 ===

Ground truth: IndexFlatL2 brute force
  IndexFlatL2         build     3 ms   search    1620.0 us/q   recall@10 100.0%

  sanity: query 0 -> id 0, dist 0.0000
          query 1 -> id 1, dist 0.0000

Approximate indexes:
  IVFFlat nprobe=8    build    62 ms   search     195.4 us/q   recall@10  97.8%
  IVFFlat nprobe=1    build    59 ms   search      29.6 us/q   recall@10  61.4%
  IVFPQ m=8           build   110 ms   search     240.1 us/q   recall@10  74.5%
  HNSW32              build   485 ms   search      55.0 us/q   recall@10  99.7%
  ...
```

That single output already encodes most of what this tutorial covers:

* the ground-truth line at 100% recall,
* every subsequent row trading a slice of recall for a big drop in
  search latency,
* and a build-time column that shows why some indexes are expensive to
  set up but cheap to query.

## **1.1. What the demo does**

Reading [../../bin/demo_faiss.dart](../../bin/demo_faiss.dart) top to
bottom:

<sup>from [../../bin/demo_faiss.dart](../../bin/demo_faiss.dart)</sup>

```dart
const int _d = 64;
const int _nb = 5000;
const int _nq = 100;
const int _k = 10;

List<Float32List> _makeData(int n, {required int seed}) {
  final rng = math.Random(seed);
  return List<Float32List>.generate(n, (i) {
    final v = Float32List(_d);
    for (var j = 0; j < _d; j++) {
      v[j] = rng.nextDouble();
    }
    v[0] += i / 1000.0; // first-axis smear (mirrors the FAISS tutorial)
    return v;
  });
}
```

`_d = 64` mirrors the upstream tutorial's dimensionality. The
`v[0] += i / 1000.0` smear guarantees that nearby ids are actually
near each other in space, so recall can be measured meaningfully.

Then the demo:

1. Builds an [IndexFlatL2](../../lib/core/vector_store/index_flat.dart)
   and measures brute-force search — this is the recall@10 = 100%
   reference line.
2. Sanity-checks the FAISS getting-started property: `query[i]` from
   the database returns id `i` at distance `0`.
3. Builds every approximate index in the port
   ([IndexIVFFlat](../../lib/core/vector_store/index_ivf_flat.dart),
   [IndexIVFPQ](../../lib/core/vector_store/index_ivf_pq.dart),
   [IndexHNSW](../../lib/core/vector_store/index_hnsw.dart),
   [IndexPQ](../../lib/core/vector_store/index_pq.dart),
   [IndexScalarQuantizer](../../lib/core/vector_store/index_scalar_quantizer.dart),
   [IndexLSH](../../lib/core/vector_store/index_lsh.dart),
   [IndexRefineFlat](../../lib/core/vector_store/index_refine_flat.dart)),
   times each one, and computes recall@10 against the flat truth.

## **1.2. The two questions the demo lets you ask**

The report is designed to answer two questions at a glance:

* **How much recall am I losing?** The `recall@10` column, always
  compared to the same brute-force ground truth. FAISS calls this
  "recall@k" — the fraction of the true top-k that survived the
  approximation.
* **How much am I saving?** The `search us/q` column vs. the flat
  baseline. On the sample output above, `IVFFlat nprobe=1` is ~55x
  faster than flat but only 61% recall; `HNSW32` is 30x faster and
  loses almost nothing.

Every chapter from here on picks one of the rows in the demo output
and shows how it works internally, why its trade-off looks the way it
does, and what knobs you have to move along that curve.

## **1.3. Where to run interactively**

Two more entrypoints are useful while reading this series:

* [../../bin/bench_faiss.dart](../../bin/bench_faiss.dart) — a
  reusable benchmark harness with `--nb`, `--nq`, `--d`, `--k`,
  `--csv PATH`, `--md PATH`, `--pareto` flags. Covered in chapter 9.
* [../../bin/vector_store_demo.dart](../../bin/vector_store_demo.dart) —
  a text-vector demo that uses the port's transformer encoders to
  produce embeddings and then does cosine search over them. Useful
  when you want vectors that come from a real model rather than a
  uniform PRNG.

---

<div align="right">

[&lt;&nbsp;&nbsp;Previous: INFRASTRUCTURE](./00-INFRASTRUCTURE.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next: VECTORS AND THE INDEX CONTRACT&nbsp;&nbsp;&gt;](./02-VECTORS-AND-THE-INDEX-CONTRACT.md)

</div>
