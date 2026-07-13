# **9. AUTO-TUNING, BENCHMARKING, AND THE CLI TOOLS**

Every approximate index in this port has at least one dial you have
to pick: `nprobe`, `efSearch`, `m`. Picking them well is not
guesswork — you sweep and measure. This chapter walks through the
three subsystems that make that pleasant:

* [bench.dart](../../lib/core/vector_store/bench.dart) — a reusable
  measurement harness and Pareto-frontier extractor.
* [auto_tune.dart](../../lib/core/vector_store/auto_tune.dart) —
  targeted sweeps for `nprobe`, `efSearch`, and `m`.
* The four CLIs in [../../bin](../../bin) that wrap them.

## **9.1. `bench.dart` — measuring anything against ground truth**

`runBench(index, queries, k, {truth, ...})` runs a search, times it,
and (given the flat ground truth) computes recall@k. Every result is
a `BenchResult` record with `label`, `recall`, `meanUs`, `buildMs`.

Two helpers turn a bag of results into something publishable:

* `String toBenchMarkdown(Iterable<BenchResult>)` — emits a GitHub-flavoured
  Markdown table with right-aligned numeric columns and escaped labels.
* `List<BenchResult> paretoFrontier(Iterable<BenchResult>)` — walks the
  results sorted by `(recall desc, meanUs asc)` and returns only the
  ones that Pareto-dominate on `(recall, meanUs)`. Everything else is
  strictly worse than something on the frontier.

These are the API that `bench_faiss.dart` uses under the hood.

## **9.2. `auto_tune.dart` — three targeted sweeps**

### `autoTuneNprobe(index, queries, {targets, k, truth})`

Given a trained IVF-family index (`IndexIVFFlat`, `IndexIVFPQ`, or
either wrapped in `IndexRefineFlat`), sweep a candidate ladder of
`nprobe` values, measure recall@k and latency at each, and return an
`OperatingPoints` record. The underlying index's `.nprobe` is
temporarily mutated during the sweep and restored on the way out.

### `autoTuneEfSearch(index, queries, {targets, k, truth})`

Same idea for `IndexHNSW.efSearch`. Cheap — every point is a fresh
search, no rebuild.

### `autoTuneM(...)` and `autoTuneMFromFlat(...)`

Expensive. `m` is baked into the PQ codebooks; sweeping it means
rebuilding the whole IVFPQ (train + add) for every candidate value.
The returned `TuneMResult` carries the built indexes so you can adopt
the winner without rebuilding a fourth time:

```dart
class TuneMResult {
  final List<OperatingPoint> points;   // one per m value
  final List<IndexIVFPQ> built;        // parallel to points
  final IndexIVFPQ? chosen;            // best under the target
  final int? chosenIndex;
}
```

### `OperatingPoint` and `OperatingPoints`

Every sweep records a list of these:

```dart
class OperatingPoint {
  final int paramValue;
  final String paramLabel;     // e.g. "nprobe=8"
  final double recall;
  final double meanUs;
}

class OperatingPoints {
  final List<OperatingPoint> points;
  OperatingPoints pareto();                        // frontier only
  OperatingPoint? pickForRecall(double target);    // cheapest above recall
  OperatingPoint? pickForLatency(double budgetUs); // best recall under budget
}
```

The output feeds directly into `TuningMetadata.fromOperatingPoints` or
`TuningMetadata.fromTuneMResult` (chapter 8) so the tuning history
persists inside the `IxDT` blob.

## **9.3. `bin/bench_faiss.dart`**

A reusable harness for the demo output shape:

```
dart run bin/bench_faiss.dart --nb 50000 --nq 500 --d 128 --k 10 \
  --csv /tmp/bench.csv --md /tmp/bench.md --pareto
```

Flags:

| Flag | Effect |
| --- | --- |
| `--nb N` | Database size |
| `--nq N` | Query count |
| `--d D` | Vector dimension |
| `--k K` | Top-k |
| `--seed S` | PRNG seed |
| `--csv PATH` | Write raw results as CSV |
| `--md PATH` | Write Markdown table (via `toBenchMarkdown`) |
| `--pareto` | Filter CSV + Markdown to the Pareto frontier and print a second stdout table |

## **9.4. `bin/faiss_describe.dart`**

Probes any FAISS blob (or `IxDT`-wrapped blob) without loading it:

```
$ dart run bin/faiss_describe.dart /tmp/mine.faiss
/tmp/mine.faiss
  kind:      float index
  fourcc:    IwFl  (0x6C46_7749)
  d:         128
  ntotal:    50000
  metric:    l2
  trained:   true
```

For tuned blobs it recurses into the inner and reports both:

```
/tmp/mine.tuned.faiss
  kind:         tuned wrapper
  fourcc:       IxDT
  tuning:       6 points, chosen nprobe=8, recall=0.94, latency=180.2us
  inner:
    kind:       float index
    fourcc:     IwFl
    d:          128
    ntotal:     50000
    ...
```

Pass `--json` for a jq-friendly nested record instead of the text
report; it emits `JsonEncoder.withIndent('  ')` output and encodes
per-file errors as `{"error": "..."}` entries so a bad blob doesn't
break batch mode.

## **9.5. `bin/faiss_strip.dart`**

Peels the `IxDT` wrapper off a tuned blob, leaving the raw inner
FAISS bytes (byte-identical to what `saveFaissIndex` would have
written for the same inner index):

```
# Single input: write output to a specific path.
dart run bin/faiss_strip.dart -o /tmp/plain.faiss /tmp/tuned.faiss

# Multi input: writes sibling foo.stripped.faiss for each.
dart run bin/faiss_strip.dart a.faiss b.faiss c.faiss
```

Exit codes: 0 all good, 1 usage error, 2 one-or-more input failed.

Use case: you tuned a blob with the port, want to hand it to
upstream FAISS, which does not know the `IxDT` fourcc. Strip
first, ship second.

## **9.6. `bin/demo_faiss.dart`**

Already discussed in chapter 1. Serves double duty: it's the first
program a new user runs, and it's a working reference for how to
wire ground truth + recall@k + latency reporting yourself if
`bench_faiss.dart`'s CLI shape doesn't fit.

## **9.7. Putting it together**

The typical workflow for a production corpus:

1. Build an index of the type you want. `indexFactory("PCA64,IVF256,PQ32", ...)`.
2. `train(subset); add(corpus)`.
3. `autoTuneNprobe` or `autoTuneEfSearch` against a held-out query
   set to pick the best knob for your recall target.
4. `saveTunedFaissIndex(path, index, TuningMetadata.fromOperatingPoints(...))`.
5. In production: `loadTunedFaissIndex(path, applyTuning: true)`.
   Same code path can also `stripTuningWrapperFile` before shipping
   to a FAISS-C++ consumer.

You now have the end-to-end story. The conclusion pulls it together.

---

<div align="right">

[&lt;&nbsp;&nbsp;Previous: PERSISTENCE: FAISS FORMAT AND IXDT](./08-PERSISTENCE-FAISS-FORMAT-AND-IXDT.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next: CONCLUSION&nbsp;&nbsp;&gt;](./10-CONCLUSION.md)

</div>
