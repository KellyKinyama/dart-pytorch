# **8. PERSISTENCE: FAISS FORMAT AND THE IXDT WRAPPER**

An index is not useful unless you can save it and load it again.
This port supports two on-disk formats:

* **FAISS binary format** ([faiss_io.dart](../../lib/core/vector_store/faiss_io.dart))
  — byte-compatible with the upstream C++/Python library. Blobs
  written here can be read by real FAISS; blobs written by real
  FAISS can be read here.
* **Port-native FAISDART format** ([index_io.dart](../../lib/core/vector_store/index_io.dart))
  — a simpler format the port owns end-to-end. Not interchangeable
  with FAISS. Useful for tests and prototypes.

Prefer FAISS format unless you have a reason not to.

## **8.1. The FAISS binary format**

<sup>from [../../lib/core/vector_store/faiss_io.dart](../../lib/core/vector_store/faiss_io.dart)</sup>

```
FAISS's binary format is little-endian only and assumes a 64-bit
host: `idx_t` is `int64_t`, `size_t` is 8 bytes. `WRITEVECTOR`
prefixes a raw byte blob with the element count as a `size_t`.
Every top-level index begins with a 4-byte ASCII fourcc tag
read as a little-endian u32.
```

The common float-index header is:

```
i32   d              // dimension
i64   ntotal
i64   dummy = 1<<20  // legacy ntotal_prev
i64   dummy = 1<<20  // legacy
u8    is_trained
i32   metric_type    // 0 = IP, 1 = L2
```

That is 33 bytes. Binary indexes use a different 21-byte header. Each
concrete index adds its own body after the header.

## **8.2. Supported fourccs**

Every supported index has a four-character tag:

| Fourcc | Class |
| --- | --- |
| `IxF2` / `IxFI` | `IndexFlatL2` / `IndexFlatIP` |
| `IxMp` / `IxM2` | `IndexIDMap` (both variants decode to the same class) |
| `IxPT` | `IndexPreTransform` (chain + inner) |
| `IxPq` | `IndexPQ` |
| `IxSQ` | `IndexScalarQuantizer` (8-bit min-max) |
| `IxRF` | `IndexRefineFlat` |
| `IwFl` | `IndexIVFFlat` |
| `IwPQ` | `IndexIVFPQ` |
| `IxHe` | `IndexLSH` |
| `IBxF` | `IndexBinaryFlat` |
| `IBwF` | `IndexBinaryIVF` |
| `IHNf` | `IndexHNSWFlat` |
| `VNrm` | `L2NormTransform` |
| `rrot` | `RandomRotationTransform` |
| `Pcam` | `PCATransform` |

Anything else raises a `FormatException` on read or `UnsupportedError`
on write, with the offending fourcc reported verbatim.

## **8.3. Save / load**

Float indexes:

```dart
saveFaissIndex('/tmp/index.faiss', myIndex);
final same = loadFaissIndex('/tmp/index.faiss');
// Or purely in memory:
final bytes = writeFaissIndexToBytes(myIndex);
final rebuilt = readFaissIndexFromBytes(bytes);
```

Binary indexes have a parallel API because `IndexBinary` is a
separate class hierarchy:

```dart
saveFaissBinaryIndex(path, myBinIndex);
final same = loadFaissBinaryIndex(path);
```

## **8.4. Probing without loading**

If you have a blob and just want to know what's inside without
paying the deserialization cost, use the probe API:

```dart
final info = probeFaissIndexFile('/tmp/index.faiss');
print('${info.kind} d=${info.d} ntotal=${info.ntotal} metric=${info.metric}');
```

`FaissIndexInfo` carries `fourcc`, `fourccStr`, `kind`
(`floatIndex` / `binaryIndex` / `tunedWrapper` / `unknown`), `d`,
`ntotal`, `metric`, `isTrained`, `codeSize`, plus optional
`tuning` and recursive `inner` when the blob is an `IxDT` wrapper
(section 8.5).

The `bin/faiss_describe.dart` CLI (chapter 9) is a thin wrapper
over this probe API.

## **8.5. The `IxDT` tuning wrapper**

FAISS' format has no place for "hey, the last time we auto-tuned
this index we picked `nprobe = 8` and it scored `recall=0.94, latency=180us`".
This port adds a byte-compatible outer wrapper for that metadata.
Fourcc: `IxDT`. Layout:

```
u32  IxDT
u32  version = 1
u64  tuningLen
[tuning payload, tuningLen bytes]
[inner FAISS blob, verbatim]
```

The tuning payload records the sweep that produced the current
best config, so the file *documents its own history*:

```
i64  createdAtMicros
i32  innerMetric              // 0 = IP, 1 = L2
u32  numPoints
per point:
  i32   paramValue            // e.g. nprobe=8 -> 8
  u32   labelLen + utf-8 label
  f64   recall
  f64   meanUs
u8   chosenPresent
if chosenPresent:
  i32  chosenParamValue
```

API surface in [faiss_io.dart](../../lib/core/vector_store/faiss_io.dart):

```dart
// Wrap an inner index with a tuning record.
saveTunedFaissIndex(path, inner, meta);
final rec = loadTunedFaissIndex(path);           // (index, metadata)

// Optional: apply the recorded winning parameter on load.
final warm = loadTunedFaissIndex(path, applyTuning: true);
// warm.index already has nprobe / efSearch set from meta.chosenParamValue.

// Detect + strip.
final tagged = isTunedFaissBlob(bytes);
final rawFaiss = stripTuningWrapper(bytes);      // returns the inner FAISS blob
stripTuningWrapperFile(inPath, outPath);         // file variant
```

The idea is: tune once, save with `saveTunedFaissIndex`; every future
consumer can either honour the recorded params (`applyTuning: true`)
or ignore them (`stripTuningWrapper`) — the inner blob is always
byte-identical to what `saveFaissIndex` would have written.

Constructing metadata:

```dart
final meta = TuningMetadata.fromOperatingPoints(
  points: opPoints,           // from OperatingPoints.pareto()
  metric: Metric.l2,
  chosenParamValue: 8,        // the winning nprobe
);
// Or straight from an autoTuneM sweep:
final meta = TuningMetadata.fromTuneMResult(result: tuneMResult,
    metric: Metric.l2);
```

## **8.6. `applyTuningToIndex`**

The heavy lifter behind `applyTuning: true` is a small public helper:

```dart
bool applyTuningToIndex(Index inner, TuningMetadata meta);
```

It inspects `meta.chosenParamValue` and the matching
`OperatingPoint.paramLabel` prefix:

* `nprobe=...` -> sets `.nprobe` on `IndexIVFFlat` / `IndexIVFPQ`
  (also unwraps `IndexRefineFlat`).
* `efSearch=...` -> sets `.efSearch` on `IndexHNSW`.
* Anything else (e.g. an `autoTuneM` sweep whose winner is a
  particular `m`) -> returns `false`; those parameters are baked
  into the trained index and cannot be applied post-facto.

Returns `true` iff a parameter was actually set. Throws
`ArgumentError` if the recognised label targets an index type it
doesn't fit (for example, `nprobe=8` on an `IndexFlat`).

## **8.7. When to skip FAISS format**

Two cases:

* You need `AddTime` or `SearchTime` tensor callbacks that FAISS'
  format has no place for. Use the port-native
  [index_io.dart](../../lib/core/vector_store/index_io.dart) format;
  it's simpler to extend.
* You're testing round-trips of a type FAISS doesn't have. Same
  answer.

Otherwise, always FAISS format — it keeps you interoperable and
lets you use `faiss_describe` / `faiss_strip` on the artefacts.

The last piece is picking the right knobs. That's the next chapter.

---

<div align="right">

[&lt;&nbsp;&nbsp;Previous: TRANSFORMS AND COMPOSED INDEXES](./07-TRANSFORMS-AND-COMPOSED-INDEXES.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next: AUTO-TUNING AND CLI TOOLS&nbsp;&nbsp;&gt;](./09-AUTO-TUNING-AND-CLI-TOOLS.md)

</div>
