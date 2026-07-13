/// FAISS binary-format interop.
///
/// Reads and writes indexes in the on-disk format produced by the
/// upstream C++ / Python FAISS library (see `faiss/impl/index_read.cc`
/// and `faiss/impl/index_write.cc`). Complementary to the native
/// `FAISDART` format in `index_io.dart`; the two formats are not
/// interchangeable, and each index has its own read/write pair.
///
/// ## Supported types
///
/// * `IndexFlat`         — fourcc `IxF2` (L2) / `IxFI` (inner product).
/// * `IndexIDMap`        — fourcc `IxMp`, wraps a sub-index + `i64` id table.
///   `IxM2` (FAISS's `IndexIDMap2`) is accepted on read with the same
///   payload layout — the extra reverse-lookup table is regenerated
///   lazily by the port, so the two fourccs decode to the same
///   [IndexIDMap] object.
/// * `IndexPreTransform` — fourcc `IxPT`, chain of transforms + inner.
/// * `IndexPQ`           — fourcc `IxPq`, product-quantised flat index.
/// * `IndexScalarQuantizer` — fourcc `IxSQ`, 8-bit per-dim scalar
///   quantization (`QT_8bit` + `RS_minmax`).
/// * `IndexRefineFlat`   — fourcc `IxRF`, base approximate index +
///   exact fp32 refine store + `k_factor` candidate multiplier.
/// * `IndexIVFFlat`      — fourcc `IwFl`, cell-probe IVF with a flat
///   (uncompressed) code store per cell. Uses the `ilar`
///   ArrayInvertedLists container in either `full` or `sprs` layout.
/// * `IndexIVFPQ`        — fourcc `IwPQ`, cell-probe IVF with a PQ
///   residual code store per cell.
/// * `IndexLSH`          — fourcc `IxHe`, random-projection LSH with
///   an embedded rectangular `rrot` transform of shape
///   `[nbits, d]` (no thresholds, `rotate_data = true`).
/// * `IndexBinaryFlat`   — fourcc `IBxF`, brute-force Hamming-distance
///   index over binary vectors. Serialized/parsed via the parallel
///   [writeFaissBinaryIndex] / [readFaissBinaryIndex] entry points
///   because [IndexBinary] is a separate class hierarchy from [Index].
/// * `IndexBinaryIVF`    — fourcc `IBwF`, cell-probe binary IVF with an
///   `IndexBinaryFlat` coarse quantizer and per-cell packed code
///   storage (same `ilar` container as the float IVF families, with
///   `code_size` in bytes rather than fp32 dims).
/// * `IndexHNSW`         — fourcc `IHNf` (IndexHNSWFlat), Malkov-style
///   hierarchical graph with an inline `IndexFlat` storage sub-index.
///   The port's per-node adjacency lists are converted to/from FAISS's
///   flat `offsets` / `neighbors` CSR layout on write / read.
/// * `L2NormTransform`   — fourcc `VNrm`, on-sphere normalization
///   (FAISS's `NormalizationTransform` with `norm = 2.0`).
/// * `RandomRotationTransform` — fourcc `rrot`, a `LinearTransform`
///   with an orthonormal `d×d` matrix and no bias.
/// * `PCATransform`      — fourcc `Pcam`, a `LinearTransform` with a
///   trained projection + mean-centring bias.
///
/// Other index families will be added in later batches. Anything
/// unsupported raises a [FormatException] on read or an
/// [UnsupportedError] on write, with the offending fourcc / type
/// reported verbatim.
///
/// ## Format notes
///
/// FAISS's binary format is little-endian only and assumes a 64-bit
/// host: `idx_t` is `int64_t`, `size_t` is 8 bytes. `WRITEVECTOR`
/// prefixes a raw byte blob with the element count as a `size_t`.
/// Every top-level index begins with a 4-byte ASCII **fourcc** tag
/// read as a little-endian u32.
///
/// The common `write_index_header` layout for float indexes is:
///
/// ```
///   i32   d                 (dimension)
///   i64   ntotal            (number of vectors, idx_t)
///   i64   dummy = 1<<20     (legacy `ntotal_prev`)
///   i64   dummy = 1<<20     (legacy)
///   u8    is_trained
///   i32   metric_type       (0 = INNER_PRODUCT, 1 = L2)
///   [f32  metric_arg]       (only if metric_type > 1)
/// ```
///
/// The `write_VectorTransform` layout mirrors upstream exactly:
///
/// ```
///   u32   fourcc                 (subtype tag)
///   ...   subtype-specific fields
///   [if LinearTransform sub-type:
///     u8       have_bias
///     WVEC f32 A                 (d_out * d_in floats)
///     WVEC f32 b                 (d_out floats when have_bias, else 0)
///   ]
///   i32   d_in
///   i32   d_out
///   u8    is_trained             (VectorTransform base field)
/// ```
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'auto_tune.dart';
import 'index.dart';
import 'index_binary.dart';
import 'index_binary_flat.dart';
import 'index_binary_ivf.dart';
import 'index_flat.dart';
import 'index_hnsw.dart';
import 'index_id_map.dart';
import 'index_io.dart';
import 'index_ivf_flat.dart';
import 'index_ivf_pq.dart';
import 'index_lsh.dart';
import 'index_pq.dart';
import 'index_pre_transform.dart';
import 'index_refine_flat.dart';
import 'index_scalar_quantizer.dart';
import 'l2_norm_transform.dart';
import 'pca_transform.dart';
import 'random_rotation_transform.dart';
import 'vector_transform.dart';

/// FAISS 4-character type tags encoded as little-endian u32.
///
/// `FaissFourcc.of('IxF2')` returns the same u32 that FAISS's
/// `fourcc("IxF2")` macro emits on a little-endian host.
class FaissFourcc {
  /// Encode a 4-character ASCII string as a little-endian u32.
  static int of(String s) {
    if (s.length != 4) {
      throw ArgumentError('Fourcc must be 4 chars, got "$s"');
    }
    return s.codeUnitAt(0) |
        (s.codeUnitAt(1) << 8) |
        (s.codeUnitAt(2) << 16) |
        (s.codeUnitAt(3) << 24);
  }

  /// Decode a fourcc u32 back into its 4-character ASCII form. Useful
  /// for error messages when an unknown tag is encountered.
  static String toStr(int v) {
    return String.fromCharCodes(<int>[
      v & 0xff,
      (v >> 8) & 0xff,
      (v >> 16) & 0xff,
      (v >> 24) & 0xff,
    ]);
  }

  /// `IxF2` — `IndexFlat` with `METRIC_L2`.
  static final int flatL2 = of('IxF2');

  /// `IxFI` — `IndexFlat` with `METRIC_INNER_PRODUCT`.
  static final int flatIP = of('IxFI');

  /// `IxFl` — generic `IndexFlat`, used for metric codes other than L2 / IP.
  /// The trailing character is a lowercase letter `l`.
  static final int flat = of('IxFl');

  /// `IxMp` — `IndexIDMap` wrapping a sub-index with a custom i64 id table.
  static final int idMap = of('IxMp');

  /// `IxM2` — FAISS's `IndexIDMap2` (adds a reverse id lookup on top
  /// of `IndexIDMap`). On disk the layout is byte-identical to
  /// [idMap]; the reverse map is rebuilt lazily on demand and never
  /// serialized. The port has no dedicated `IndexIDMap2` class, so
  /// this fourcc is a read-only compatibility marker —
  /// [readFaissIndex] decodes `IxM2` blobs into a plain [IndexIDMap].
  static final int idMap2 = of('IxM2');

  /// `IxPT` — `IndexPreTransform`, a chain of vector transforms + inner index.
  static final int preTransform = of('IxPT');

  /// `L2nT` — alias for `VNrm` retained for backwards compatibility
  /// with earlier drafts. **Do not use.** Left as a compile-time
  /// constant so any stale caller keeps compiling; if you actually
  /// need to match FAISS bytes, use [normalizationTransform].
  @Deprecated(
    'This was a mistake in batch 12; real FAISS uses fourcc "VNrm". '
    'Use FaissFourcc.normalizationTransform instead.',
  )
  static final int l2NormTransform = of('L2nT');

  /// `VNrm` — `NormalizationTransform`. Layout:
  ///
  /// ```
  ///   u32 fourcc('VNrm')
  ///   f32 norm            (always 2.0f for L2 normalization)
  ///   i32 d_in            (common tail)
  ///   i32 d_out
  ///   u8  is_trained
  /// ```
  static final int normalizationTransform = of('VNrm');

  /// `rrot` — `RandomRotationMatrix`, a `LinearTransform` sub-class
  /// with an orthonormal `d×d` matrix and no bias. Shares the
  /// `have_bias`, `A`, `b` LinearTransform body plus the common
  /// `d_in`, `d_out`, `is_trained` tail with other LinearTransform
  /// sub-types (Pcam, Viqm, LTra).
  static final int randomRotation = of('rrot');

  /// `Pcam` — `PCAMatrix`, a `LinearTransform` sub-class holding a
  /// trained projection. Subtype-specific fields (eigen_power,
  /// epsilon, random_rotation flag, balanced_bins, mean, eigenvalues,
  /// PCAMat) precede the LinearTransform body.
  static final int pca = of('Pcam');

  /// `IxPq` — `IndexPQ`. Layout (after fourcc):
  ///
  /// ```
  ///   [write_index_header]
  ///   [write_ProductQuantizer]  u64 d, u64 M, u64 nbits,
  ///                             WVEC f32 centroids  (M*ksub*dsub)
  ///   WVEC u8 codes             (ntotal * M bytes)
  ///   i32 search_type           (0 = ST_PQ)
  ///   u8  encode_signs          (bool, 0 outside polysemous mode)
  ///   i32 polysemous_ht         (0 outside polysemous mode)
  /// ```
  static final int indexPq = of('IxPq');

  /// `IxSQ` — `IndexScalarQuantizer`. Layout (after fourcc):
  ///
  /// ```
  ///   [write_index_header]
  ///   [write_ScalarQuantizer]:
  ///     i32 qtype        (0 = QT_8bit)
  ///     i32 rangestat    (0 = RS_minmax)
  ///     f32 rangestat_arg
  ///     u64 d
  ///     u64 code_size    (= d for QT_8bit)
  ///     WVEC f32 trained (2*d floats: [vmin..., vdiff...])
  ///   WVEC u8 codes      (ntotal * d bytes)
  /// ```
  static final int indexSq = of('IxSQ');

  /// `IxRF` — `IndexRefine` / `IndexRefineFlat`. Layout (after fourcc):
  ///
  /// ```
  ///   [write_index_header]
  ///   write_index(base_index)    (recursive)
  ///   write_index(refine_index)  (recursive; upstream always IndexFlat)
  ///   f32 k_factor               (candidate multiplier)
  /// ```
  ///
  /// FAISS stores `k_factor` as a float; our port keeps it as an `int`,
  /// so on write we cast to `f32` and on read we round to the nearest
  /// integer.
  static final int indexRefine = of('IxRF');

  /// `IwFl` — `IndexIVFFlat`. Layout (after fourcc):
  ///
  /// ```
  ///   [write_ivf_header]:
  ///     [write_index_header]
  ///     u64 nlist
  ///     u64 nprobe
  ///     write_index(quantizer)   (recursive; upstream IndexFlat)
  ///     [write_direct_map]:
  ///       u8  type (0 = NoMap)
  ///       WVEC i64 array         (empty for NoMap)
  ///   [write_InvertedLists]:
  ///     u32 fourcc('ilar')
  ///     u64 nlist
  ///     u64 code_size            (= d * 4 for IndexIVFFlat)
  ///     u32 list_type fourcc ('full' or 'sprs')
  ///       'full': WVEC u64 sizes[nlist]
  ///       'sprs': WVEC u64 pairs (flat [list_idx, size, ...] for non-empty)
  ///     per non-empty cell:
  ///       (n * code_size) raw code bytes
  ///       (n * 8)         raw idx_t (int64) ids
  /// ```
  static final int indexIvfFlat = of('IwFl');

  /// `ilar` — `ArrayInvertedLists` container (per-cell storage). See
  /// [indexIvfFlat] for the full layout.
  static final int invListsArray = of('ilar');

  /// `full` — dense sizes vector for `ilar` (one size per cell).
  static final int invListsFull = of('full');

  /// `sprs` — sparse sizes vector for `ilar` (pairs of `[cell_idx, size]`
  /// for non-empty cells only).
  static final int invListsSparse = of('sprs');

  /// `IwPQ` — `IndexIVFPQ`. Layout (after fourcc):
  ///
  /// ```
  ///   [write_ivf_header]
  ///   u8  by_residual         (this port always emits/expects 1)
  ///   u64 code_size           (= pq.M for QT_8bit)
  ///   [write_ProductQuantizer]
  ///   [write_InvertedLists]   (ilar with code_size = M)
  /// ```
  static final int indexIvfPq = of('IwPQ');

  /// `IxHe` — `IndexLSH`. Layout (after fourcc):
  ///
  /// ```
  ///   [write_index_header]
  ///   i32 nbits
  ///   u8  rotate_data          (this port always emits/expects 1)
  ///   u8  train_thresholds     (this port always emits/expects 0)
  ///   WVEC f32 thresholds      (empty when train_thresholds = 0)
  ///   i32 code_size            (= (nbits + 7) / 8)
  ///   [inline rrot LinearTransform, non-square `[nbits, d]`]:
  ///     u32 fourcc('rrot')
  ///     u8  have_bias = 0
  ///     WVEC f32 A             (nbits * d floats, row-major)
  ///     WVEC f32 b             (empty)
  ///     i32 d_in  = d
  ///     i32 d_out = nbits
  ///     u8  is_trained
  ///   WVEC u8 codes            (ntotal * code_size bytes)
  /// ```
  ///
  /// FAISS's `IndexLSH` embeds a `RandomRotationMatrix` sub-transform
  /// that carries the actual projection. Because that transform is
  /// rectangular (`d_in != d_out` in general), we can't route it
  /// through [FaissFourcc.randomRotation]'s square-only reader; the
  /// projection is instead serialized inline directly by the IxHe
  /// dispatch.
  static final int indexLsh = of('IxHe');

  /// `IBxF` — `IndexBinaryFlat`. Layout (after fourcc):
  ///
  /// ```
  ///   [write_index_binary_header]:
  ///     i32 d                (bit dimension = code_size * 8)
  ///     i32 code_size        (bytes per vector)
  ///     i64 ntotal
  ///     u8  is_trained
  ///     i32 metric_type      (upstream default = METRIC_L2 = 1)
  ///   WVEC u8 xb             (ntotal * code_size bytes)
  /// ```
  ///
  /// Note: FAISS's binary indexes share the [MetricType] enum with
  /// their float cousins and the `IndexBinary` base default is
  /// `METRIC_L2 = 1`; `IndexBinaryFlat` does not override it, even
  /// though the actual scoring is always Hamming. This port therefore
  /// always emits `metric_type = 1` and accepts any of the port's
  /// supported float metric codes on read.
  static final int indexBinaryFlat = of('IBxF');

  /// `IBwF` — `IndexBinaryIVF`. Layout (after fourcc):
  ///
  /// ```
  ///   [write_binary_ivf_header]:
  ///     [write_index_binary_header]
  ///     u64 nlist
  ///     u64 nprobe
  ///     write_index_binary(quantizer)   (recursive; IBxF for this port)
  ///     [write_direct_map]:
  ///       u8   type (0 = NoMap)
  ///       WVEC i64 array                (empty for NoMap)
  ///   [write_InvertedLists]:
  ///     u32 fourcc('ilar')
  ///     u64 nlist
  ///     u64 code_size                   (bytes per binary vector)
  ///     u32 list_type fourcc ('full' or 'sprs')
  ///       'full': WVEC u64 sizes[nlist]
  ///       'sprs': WVEC u64 pairs (flat [cell_idx, size, ...] for non-empty)
  ///     per non-empty cell:
  ///       (n * code_size) raw code bytes
  ///       (n * 8)         raw idx_t (int64) ids
  /// ```
  static final int indexBinaryIvf = of('IBwF');

  /// `IHNf` — `IndexHNSWFlat`. Layout (after fourcc):
  ///
  /// ```
  ///   [write_index_header]
  ///   [write_HNSW]:
  ///     WVEC f64 assign_probas             (per-level sampling weights,
  ///                                         length = max_expected_level + 1)
  ///     WVEC i32 cum_nneighbor_per_level   (prefix sums of per-layer slot
  ///                                         budgets; [0, Mmax0, Mmax0 + Mmax,
  ///                                         Mmax0 + 2*Mmax, ...])
  ///     WVEC i32 levels                    (per-node top_layer + 1)
  ///     WVEC u64 offsets                   (start of node's slot region in
  ///                                         `neighbors`, length = ntotal + 1)
  ///     WVEC i32 neighbors                 (flat neighbor array; unused
  ///                                         slots are -1)
  ///     i32 entry_point                    (storage id or -1 when empty)
  ///     i32 max_level                      (== port's _topLevel)
  ///     i32 efConstruction
  ///     i32 efSearch
  ///     i32 upper_beam                     (deprecated; always 1)
  ///   write_index(storage)                 (recursive; IxF2/IxFI IndexFlat)
  /// ```
  ///
  /// The port stores per-layer adjacency lists directly on each node,
  /// while FAISS packs them into a CSR-ish `offsets` + `neighbors`
  /// pair with per-layer slot budgets from `cum_nneighbor_per_level`.
  /// This dispatch does the conversion in both directions: on write
  /// each node's lists are padded with `-1` to the slot budget; on
  /// read the `-1` sentinels are filtered back out.
  static final int indexHnswFlat = of('IHNf');
}

/// FAISS `MetricType` enum values.
///
/// ```
///   METRIC_INNER_PRODUCT = 0
///   METRIC_L2            = 1
/// ```
///
/// Our port only exposes the first two; other FAISS metrics
/// (`METRIC_L1`, `METRIC_Linf`, `METRIC_Canberra`, ...) throw.
int _metricToFaiss(Metric m) => m == Metric.l2 ? 1 : 0;

Metric _metricFromFaiss(int v) {
  switch (v) {
    case 0:
      return Metric.innerProduct;
    case 1:
      return Metric.l2;
    default:
      throw FormatException('Unsupported FAISS metric code $v');
  }
}

/// Writes the common `Index` header, matching FAISS's
/// `write_index_header` in `impl/index_write.cc` byte-for-byte.
void _writeHeader(IoWriter w, Index x) {
  w.writeI32(x.d);
  w.writeI64(x.ntotal);
  w.writeI64(1 << 20); // legacy dummy `ntotal_prev`
  w.writeI64(1 << 20); // legacy dummy
  w.writeU8(x.isTrained ? 1 : 0);
  w.writeI32(_metricToFaiss(x.metric));
  // metric_arg only written when metric_type > 1; L2/IP never emit it.
}

/// Header fields extracted from a FAISS-format stream.
typedef _Header = ({int d, int ntotal, bool isTrained, Metric metric});

_Header _readHeader(IoReader r) {
  final d = r.readI32();
  final ntotal = r.readI64();
  r.readI64(); // dummy `ntotal_prev`
  r.readI64(); // dummy
  final isTrained = r.readU8() != 0;
  final metric = _metricFromFaiss(r.readI32());
  return (d: d, ntotal: ntotal, isTrained: isTrained, metric: metric);
}

/// FAISS `WRITEVECTOR(vec)` for a `std::vector<uint8_t>`:
///
/// ```
///   size_t size = vec.size();
///   WRITE1(size);
///   writer(&vec[0], sizeof(uint8_t), size);
/// ```
///
/// The size field is 8 bytes on all 64-bit hosts.
void _writeVectorU8(IoWriter w, Uint8List xs) {
  w.writeU64(xs.length);
  w.writeBytes(xs);
}

Uint8List _readVectorU8(IoReader r) {
  final n = r.readU64();
  return r.readBytes(n);
}

/// FAISS `WRITEVECTOR` for `std::vector<int64_t>`:
///   size_t element_count (little-endian u64)
///   element_count * 8 raw bytes, each an int64 LE.
void _writeVectorI64(IoWriter w, List<int> xs) {
  w.writeU64(xs.length);
  for (final v in xs) {
    w.writeI64(v);
  }
}

List<int> _readVectorI64(IoReader r) {
  final n = r.readU64();
  final out = List<int>.filled(n, 0);
  for (var i = 0; i < n; i++) {
    out[i] = r.readI64();
  }
  return out;
}

/// FAISS `WRITEVECTOR` for `std::vector<float>`:
///   size_t element_count (u64 LE)
///   element_count * 4 raw bytes, each an f32 LE.
void _writeVectorF32(IoWriter w, Float32List xs) {
  w.writeU64(xs.length);
  if (xs.isNotEmpty) {
    w.writeF32List(xs);
  }
}

Float32List _readVectorF32(IoReader r) {
  final n = r.readU64();
  return n == 0 ? Float32List(0) : r.readF32List(n);
}

/// FAISS `WRITEVECTOR` for `std::vector<double>`:
///   size_t element_count (u64 LE)
///   element_count * 8 raw bytes, each an f64 LE.
///
/// Used by HNSW's `assign_probas`.
void _writeVectorF64(IoWriter w, List<double> xs) {
  w.writeU64(xs.length);
  for (final v in xs) {
    w.writeF64(v);
  }
}

List<double> _readVectorF64(IoReader r) {
  final n = r.readU64();
  final out = List<double>.filled(n, 0.0);
  for (var i = 0; i < n; i++) {
    out[i] = r.readF64();
  }
  return out;
}

/// FAISS `WRITEVECTOR` for `std::vector<int>` (32-bit).
///
/// Used by HNSW's `cum_nneighbor_per_level`, `levels`, and
/// `neighbors` arrays (with a leading `u64` element count).
void _writeVectorI32(IoWriter w, List<int> xs) {
  w.writeU64(xs.length);
  for (final v in xs) {
    w.writeI32(v);
  }
}

List<int> _readVectorI32(IoReader r) {
  final n = r.readU64();
  final out = List<int>.filled(n, 0);
  for (var i = 0; i < n; i++) {
    out[i] = r.readI32();
  }
  return out;
}

/// Writes the `LinearTransform` body block that follows the subtype
/// fields for every LinearTransform sub-class:
///
/// ```
///   u8       have_bias
///   WVEC f32 A          (d_out * d_in floats, row-major)
///   WVEC f32 b          (d_out floats when have_bias else 0)
/// ```
void _writeLTBody(
  IoWriter w, {
  required bool haveBias,
  required Float32List a,
  required Float32List b,
}) {
  w.writeU8(haveBias ? 1 : 0);
  _writeVectorF32(w, a);
  _writeVectorF32(w, b);
}

typedef _LTBody = ({bool haveBias, Float32List a, Float32List b});

_LTBody _readLTBody(IoReader r) {
  final haveBias = r.readU8() != 0;
  final a = _readVectorF32(r);
  final b = _readVectorF32(r);
  return (haveBias: haveBias, a: a, b: b);
}

/// Writes the common `VectorTransform` tail that terminates every
/// `write_VectorTransform` call in upstream FAISS:
///
/// ```
///   i32 d_in
///   i32 d_out
///   u8  is_trained
/// ```
void _writeVTCommon(IoWriter w, VectorTransform vt) {
  w.writeI32(vt.dIn);
  w.writeI32(vt.dOut);
  w.writeU8(vt.isTrained ? 1 : 0);
}

typedef _VTCommon = ({int dIn, int dOut, bool isTrained});

_VTCommon _readVTCommon(IoReader r) {
  final dIn = r.readI32();
  final dOut = r.readI32();
  final isTrained = r.readU8() != 0;
  return (dIn: dIn, dOut: dOut, isTrained: isTrained);
}

/// Serializes a [ProductQuantizer] using FAISS's `write_ProductQuantizer`
/// layout:
///
/// ```
///   u64 d      (size_t)
///   u64 M      (size_t)
///   u64 nbits  (size_t)
///   WVEC f32 centroids  (M * ksub * dsub floats, row-major
///                        [sub][c][j])
/// ```
///
/// The order in the centroid blob is `sub` outermost, then `c` (code),
/// then `j` (sub-dim). Length = `M * (1 << nbits) * (d / M)`.
void _writeProductQuantizer(IoWriter w, ProductQuantizer pq) {
  w.writeU64(pq.d);
  w.writeU64(pq.m);
  w.writeU64(pq.nbits);
  final ksub = pq.ksub;
  final dsub = pq.dsub;
  final flat = Float32List(pq.m * ksub * dsub);
  final books = pq.codebooks;
  for (var sub = 0; sub < pq.m; sub++) {
    final base = sub * ksub * dsub;
    for (var c = 0; c < ksub; c++) {
      final cen = books[sub][c];
      final off = base + c * dsub;
      for (var j = 0; j < dsub; j++) {
        flat[off + j] = cen[j];
      }
    }
  }
  _writeVectorF32(w, flat);
}

/// Parses the FAISS `write_ProductQuantizer` payload from [r], returning
/// a fully-populated [ProductQuantizer]. Only `nbits == 8` is supported
/// by this port; other values throw [UnsupportedError].
ProductQuantizer _readProductQuantizer(IoReader r) {
  final d = r.readU64();
  final m = r.readU64();
  final nbits = r.readU64();
  if (nbits != 8) {
    throw UnsupportedError(
      'IxPq: only nbits == 8 is supported by this port, got $nbits.',
    );
  }
  final pq = ProductQuantizer(d: d, m: m);
  final ksub = pq.ksub;
  final dsub = pq.dsub;
  final expected = m * ksub * dsub;
  final flat = _readVectorF32(r);
  if (flat.length != expected) {
    throw FormatException(
      'IxPq: centroid blob has ${flat.length} floats, expected $expected '
      'for d=$d, M=$m, nbits=$nbits.',
    );
  }
  final books = pq.codebooks;
  for (var sub = 0; sub < m; sub++) {
    final base = sub * ksub * dsub;
    for (var c = 0; c < ksub; c++) {
      final cen = books[sub][c];
      final off = base + c * dsub;
      for (var j = 0; j < dsub; j++) {
        cen[j] = flat[off + j];
      }
    }
  }
  pq.isTrained = true;
  return pq;
}

/// Serializes an [IndexScalarQuantizer] as FAISS's `write_ScalarQuantizer`
/// block (the trained parameters, sans wrapper fourcc). Layout:
///
/// ```
///   i32 qtype        (0 = QT_8bit)
///   i32 rangestat    (0 = RS_minmax)
///   f32 rangestat_arg
///   u64 d
///   u64 code_size    (= d for QT_8bit)
///   WVEC f32 trained (2 * d floats: vmin[0..d-1] then vdiff[0..d-1])
/// ```
///
/// Our port only supports the `(QT_8bit, RS_minmax)` combination — the
/// only mode where our internal `(_vmin, _scale)` representation maps
/// losslessly to FAISS's `(vmin, vdiff)`. `vdiff = _scale * 255`.
void _writeScalarQuantizer(IoWriter w, IndexScalarQuantizer sq) {
  w.writeI32(0); // qtype = QT_8bit
  w.writeI32(0); // rangestat = RS_minmax
  w.writeF32(0.0); // rangestat_arg (unused by RS_minmax in our port)
  w.writeU64(sq.d);
  w.writeU64(sq.d); // code_size == d for QT_8bit
  final trained = Float32List(2 * sq.d);
  for (var j = 0; j < sq.d; j++) {
    trained[j] = sq.vmin[j];
    trained[sq.d + j] = sq.scale[j] * 255.0;
  }
  _writeVectorF32(w, trained);
}

/// Parses FAISS's `write_ScalarQuantizer` payload and returns a
/// populated [IndexScalarQuantizer] with `_vmin` and `_scale` set (but
/// no codes / ntotal yet). The caller is responsible for reading the
/// codes blob that follows.
IndexScalarQuantizer _readScalarQuantizer(IoReader r, Metric metric) {
  final qtype = r.readI32();
  final rangestat = r.readI32();
  r.readF32(); // rangestat_arg (unused by RS_minmax in our port)
  final d = r.readU64();
  final codeSize = r.readU64();
  final trained = _readVectorF32(r);
  if (qtype != 0) {
    throw UnsupportedError(
      'IxSQ: only QT_8bit (qtype=0) is supported, got qtype=$qtype.',
    );
  }
  if (rangestat != 0) {
    throw UnsupportedError(
      'IxSQ: only RS_minmax (rangestat=0) is supported, '
      'got rangestat=$rangestat.',
    );
  }
  if (codeSize != d) {
    throw FormatException('IxSQ: code_size ($codeSize) != d ($d) for QT_8bit.');
  }
  if (trained.length != 2 * d) {
    throw FormatException(
      'IxSQ: trained has ${trained.length} floats, expected ${2 * d} '
      'for QT_8bit + RS_minmax.',
    );
  }
  final sq = IndexScalarQuantizer(d, metric: metric);
  for (var j = 0; j < d; j++) {
    sq.vmin[j] = trained[j];
    final vdiff = trained[d + j];
    sq.scale[j] = vdiff == 0 ? 1.0 : vdiff / 255.0;
  }
  sq.isTrained = true;
  return sq;
}

/// Serializes FAISS's `write_direct_map` block for an IVF-family index
/// that has no direct map maintained (`DirectMap::NoMap`, the only mode
/// this port supports). Layout:
///
/// ```
///   u8       type            (0 = NoMap, 1 = Array, 2 = Hashtable)
///   WVEC i64 array           (empty for NoMap)
///   [WVEC pair<i64,i64>]     (only when type == Hashtable)
/// ```
void _writeDirectMapNone(IoWriter w) {
  w.writeU8(0); // DirectMap::NoMap
  _writeVectorI64(w, const <int>[]);
}

/// Parses FAISS's `write_direct_map` block. Our port only supports
/// `NoMap` (type 0); any other value raises [UnsupportedError].
void _readDirectMap(IoReader r) {
  final type = r.readU8();
  if (type != 0) {
    throw UnsupportedError(
      'DirectMap: only NoMap (type=0) supported, got type=$type',
    );
  }
  final array = _readVectorI64(r);
  if (array.isNotEmpty) {
    throw FormatException(
      'DirectMap: NoMap should carry an empty array, got ${array.length} entries',
    );
  }
}

/// Serializes an [IndexIVFFlat]'s IVF header block, mirroring FAISS's
/// `write_ivf_header`. Note: FAISS's function does *not* emit
/// `by_residual`; subclasses that need it write it themselves. `IwFl`
/// is not one of them.
void _writeIvfHeader(IoWriter w, IndexIVFFlat ivf) {
  _writeIvfHeaderGeneric(
    w,
    header: ivf,
    nlist: ivf.nlist,
    nprobe: ivf.nprobe,
    quantizer: ivf.quantizer,
  );
}

/// Generic IVF header writer used by both `IwFl` and `IwPQ`. Mirrors
/// FAISS's `write_ivf_header`: common index header + nlist + nprobe +
/// recursive quantizer + direct_map (NoMap only). The `header`
/// parameter supplies d / ntotal / is_trained / metric.
void _writeIvfHeaderGeneric(
  IoWriter w, {
  required Index header,
  required int nlist,
  required int nprobe,
  required IndexFlat quantizer,
}) {
  _writeHeader(w, header);
  w.writeU64(nlist);
  w.writeU64(nprobe);
  writeFaissIndex(w, quantizer);
  _writeDirectMapNone(w);
}

/// Parses FAISS's `write_ivf_header` block. Returns the common header
/// plus the extracted `nlist`, `nprobe`, and coarse `quantizer` (which
/// must decode to [IndexFlat] for our port).
({
  ({int d, int ntotal, bool isTrained, Metric metric}) h,
  int nlist,
  int nprobe,
  IndexFlat quantizer,
})
_readIvfHeader(IoReader r) {
  final h = _readHeader(r);
  final nlist = r.readU64();
  final nprobe = r.readU64();
  final quantizer = readFaissIndex(r);
  if (quantizer is! IndexFlat) {
    throw FormatException(
      'IVF header: quantizer sub-index is ${quantizer.runtimeType}, '
      'expected IndexFlat',
    );
  }
  if (quantizer.d != h.d || quantizer.metric != h.metric) {
    throw FormatException(
      'IVF header: quantizer (d=${quantizer.d}, metric=${quantizer.metric}) '
      'disagrees with header (d=${h.d}, metric=${h.metric})',
    );
  }
  if (quantizer.ntotal != nlist) {
    throw FormatException(
      'IVF header: quantizer has ${quantizer.ntotal} centroids, '
      'header nlist=$nlist',
    );
  }
  _readDirectMap(r);
  return (h: h, nlist: nlist, nprobe: nprobe, quantizer: quantizer);
}

/// Serializes an [IndexIVFFlat]'s inverted-list payload as FAISS's
/// `ArrayInvertedLists` (`ilar`) container. Picks between `full` and
/// `sprs` size-vector encodings using FAISS's own heuristic
/// (`sprs` when > half of the cells are empty).
void _writeArrayInvertedLists(IoWriter w, IndexIVFFlat ivf) {
  final codeSize = ivf.d * 4;
  w.writeU32(FaissFourcc.invListsArray);
  w.writeU64(ivf.nlist);
  w.writeU64(codeSize);

  var nNon0 = 0;
  for (var i = 0; i < ivf.nlist; i++) {
    if (ivf.invLists[i].isNotEmpty) nNon0++;
  }
  if (nNon0 > ivf.nlist ~/ 2) {
    // `full`: one size per cell.
    w.writeU32(FaissFourcc.invListsFull);
    final sizes = List<int>.generate(ivf.nlist, (i) => ivf.invLists[i].length);
    _writeVectorI64(w, sizes);
  } else {
    // `sprs`: (cell_idx, size) pairs for non-empty cells only.
    w.writeU32(FaissFourcc.invListsSparse);
    final pairs = <int>[];
    for (var i = 0; i < ivf.nlist; i++) {
      final n = ivf.invLists[i].length;
      if (n > 0) {
        pairs.add(i);
        pairs.add(n);
      }
    }
    _writeVectorI64(w, pairs);
  }

  // Per-cell payload: codes (n * code_size bytes) then ids (n * 8 bytes).
  final storage = ivf.storage;
  for (var i = 0; i < ivf.nlist; i++) {
    final ids = ivf.invLists[i];
    final n = ids.length;
    if (n == 0) continue;
    final codesFlat = Float32List(n * ivf.d);
    for (var j = 0; j < n; j++) {
      final id = ids[j];
      final srcBase = id * ivf.d;
      final dstBase = j * ivf.d;
      for (var k = 0; k < ivf.d; k++) {
        codesFlat[dstBase + k] = storage[srcBase + k];
      }
    }
    w.writeF32List(codesFlat);
    for (final id in ids) {
      w.writeI64(id);
    }
  }
}

/// Parses FAISS's `ArrayInvertedLists` (`ilar`) container into a
/// port-shaped state tuple: contiguous fp32 storage indexed by id, and
/// a per-cell list of ids. The port only supports contiguous ids in
/// `[0, ntotal)`.
({Float32List storage, List<List<int>> invLists}) _readArrayInvertedLists(
  IoReader r,
  int d,
  int nlist,
  int ntotal,
) {
  final tag = r.readU32();
  if (tag != FaissFourcc.invListsArray) {
    throw UnsupportedError(
      'IwFl: only ilar InvertedLists supported, got fourcc '
      '"${FaissFourcc.toStr(tag)}"',
    );
  }
  final storedNlist = r.readU64();
  final codeSize = r.readU64();
  if (storedNlist != nlist) {
    throw FormatException(
      'ilar: nlist mismatch (payload=$storedNlist, header=$nlist)',
    );
  }
  if (codeSize != d * 4) {
    throw FormatException('ilar: code_size $codeSize != d*4 = ${d * 4}');
  }

  final sizes = List<int>.filled(nlist, 0);
  final listType = r.readU32();
  if (listType == FaissFourcc.invListsFull) {
    final xs = _readVectorI64(r);
    if (xs.length != nlist) {
      throw FormatException(
        'ilar full: sizes length ${xs.length} != nlist $nlist',
      );
    }
    for (var i = 0; i < nlist; i++) {
      sizes[i] = xs[i];
    }
  } else if (listType == FaissFourcc.invListsSparse) {
    final pairs = _readVectorI64(r);
    if (pairs.length.isOdd) {
      throw FormatException(
        'ilar sprs: pairs length ${pairs.length} is not even',
      );
    }
    for (var j = 0; j < pairs.length; j += 2) {
      final idx = pairs[j];
      if (idx < 0 || idx >= nlist) {
        throw FormatException('ilar sprs: cell idx $idx out of [0, $nlist)');
      }
      sizes[idx] = pairs[j + 1];
    }
  } else {
    throw UnsupportedError(
      'ilar: list_type "${FaissFourcc.toStr(listType)}" not supported',
    );
  }

  var sum = 0;
  for (final s in sizes) {
    sum += s;
  }
  if (sum != ntotal) {
    throw FormatException(
      'ilar: sum of sizes ($sum) != header ntotal ($ntotal)',
    );
  }

  final storage = Float32List(ntotal * d);
  final invLists = List<List<int>>.generate(nlist, (_) => <int>[]);
  for (var i = 0; i < nlist; i++) {
    final n = sizes[i];
    if (n == 0) continue;
    final codesFlat = r.readF32List(n * d);
    for (var j = 0; j < n; j++) {
      final id = r.readI64();
      if (id < 0 || id >= ntotal) {
        throw UnsupportedError(
          'IwFl: ids must be contiguous in [0, ntotal); got id=$id '
          '(ntotal=$ntotal). Non-contiguous FAISS blobs are not yet '
          'supported.',
        );
      }
      final dstBase = id * d;
      final srcBase = j * d;
      for (var k = 0; k < d; k++) {
        storage[dstBase + k] = codesFlat[srcBase + k];
      }
      invLists[i].add(id);
    }
  }
  return (storage: storage, invLists: invLists);
}

/// IVFPQ variant of [_writeArrayInvertedLists]. The `ilar` container
/// wire format is identical, but the code source is the port's
/// per-cell byte lists (one code = `m` bytes) rather than a flat fp32
/// storage.
void _writeArrayInvertedListsIVFPQ(IoWriter w, IndexIVFPQ ivpq) {
  final codeSize = ivpq.m;
  w.writeU32(FaissFourcc.invListsArray);
  w.writeU64(ivpq.nlist);
  w.writeU64(codeSize);

  var nNon0 = 0;
  for (var i = 0; i < ivpq.nlist; i++) {
    if (ivpq.invListsIds[i].isNotEmpty) nNon0++;
  }
  if (nNon0 > ivpq.nlist ~/ 2) {
    w.writeU32(FaissFourcc.invListsFull);
    final sizes = List<int>.generate(
      ivpq.nlist,
      (i) => ivpq.invListsIds[i].length,
    );
    _writeVectorI64(w, sizes);
  } else {
    w.writeU32(FaissFourcc.invListsSparse);
    final pairs = <int>[];
    for (var i = 0; i < ivpq.nlist; i++) {
      final n = ivpq.invListsIds[i].length;
      if (n > 0) {
        pairs.add(i);
        pairs.add(n);
      }
    }
    _writeVectorI64(w, pairs);
  }

  for (var i = 0; i < ivpq.nlist; i++) {
    final ids = ivpq.invListsIds[i];
    final n = ids.length;
    if (n == 0) continue;
    final codes = ivpq.invListsCodes[i];
    if (codes.length != n * codeSize) {
      throw StateError(
        'IwPQ: cell $i has ${codes.length} code bytes, expected '
        '${n * codeSize} (n=$n, m=$codeSize)',
      );
    }
    // codes are Dart ints in [0, 255]; write as raw bytes.
    final buf = Uint8List(codes.length);
    for (var j = 0; j < codes.length; j++) {
      buf[j] = codes[j] & 0xff;
    }
    w.writeBytes(buf);
    for (final id in ids) {
      w.writeI64(id);
    }
  }
}

/// IVFPQ variant of [_readArrayInvertedLists]. Returns per-cell ids
/// and per-cell code-byte lists, matching [IndexIVFPQ]'s internal
/// representation. Asserts contiguous ids in `[0, ntotal)`.
({List<List<int>> ids, List<List<int>> codes}) _readArrayInvertedListsIVFPQ(
  IoReader r,
  int m,
  int nlist,
  int ntotal,
) {
  final tag = r.readU32();
  if (tag != FaissFourcc.invListsArray) {
    throw UnsupportedError(
      'IwPQ: only ilar InvertedLists supported, got fourcc '
      '"${FaissFourcc.toStr(tag)}"',
    );
  }
  final storedNlist = r.readU64();
  final codeSize = r.readU64();
  if (storedNlist != nlist) {
    throw FormatException(
      'ilar: nlist mismatch (payload=$storedNlist, header=$nlist)',
    );
  }
  if (codeSize != m) {
    throw FormatException('ilar: code_size $codeSize != m $m for IwPQ');
  }

  final sizes = List<int>.filled(nlist, 0);
  final listType = r.readU32();
  if (listType == FaissFourcc.invListsFull) {
    final xs = _readVectorI64(r);
    if (xs.length != nlist) {
      throw FormatException(
        'ilar full: sizes length ${xs.length} != nlist $nlist',
      );
    }
    for (var i = 0; i < nlist; i++) {
      sizes[i] = xs[i];
    }
  } else if (listType == FaissFourcc.invListsSparse) {
    final pairs = _readVectorI64(r);
    if (pairs.length.isOdd) {
      throw FormatException(
        'ilar sprs: pairs length ${pairs.length} is not even',
      );
    }
    for (var j = 0; j < pairs.length; j += 2) {
      final idx = pairs[j];
      if (idx < 0 || idx >= nlist) {
        throw FormatException('ilar sprs: cell idx $idx out of [0, $nlist)');
      }
      sizes[idx] = pairs[j + 1];
    }
  } else {
    throw UnsupportedError(
      'ilar: list_type "${FaissFourcc.toStr(listType)}" not supported',
    );
  }

  var sum = 0;
  for (final s in sizes) {
    sum += s;
  }
  if (sum != ntotal) {
    throw FormatException(
      'ilar: sum of sizes ($sum) != header ntotal ($ntotal)',
    );
  }

  final ids = List<List<int>>.generate(nlist, (_) => <int>[]);
  final codes = List<List<int>>.generate(nlist, (_) => <int>[]);
  for (var i = 0; i < nlist; i++) {
    final n = sizes[i];
    if (n == 0) continue;
    final rawCodes = r.readBytes(n * m);
    codes[i].addAll(rawCodes);
    for (var j = 0; j < n; j++) {
      final id = r.readI64();
      if (id < 0 || id >= ntotal) {
        throw UnsupportedError(
          'IwPQ: ids must be contiguous in [0, ntotal); got id=$id '
          '(ntotal=$ntotal). Non-contiguous FAISS blobs are not yet '
          'supported.',
        );
      }
      ids[i].add(id);
    }
  }
  return (ids: ids, codes: codes);
}

/// Serializes a [VectorTransform] using FAISS's `write_VectorTransform`
/// dispatch, keyed on the concrete Dart type. The exact layout mirrors
/// `faiss/impl/index_write.cpp` verbatim: subtype fourcc + subtype
/// fields, then (for LinearTransform) the LT body, then the common
/// `d_in / d_out / is_trained` tail.
void writeFaissTransform(IoWriter w, VectorTransform vt) {
  if (vt is L2NormTransform) {
    // VNrm layout:
    //   u32 fourcc('VNrm')
    //   f32 norm                (2.0f for L2)
    //   [common VT tail]
    w.writeU32(FaissFourcc.normalizationTransform);
    w.writeF32(2.0);
    _writeVTCommon(w, vt);
    return;
  }
  if (vt is RandomRotationTransform) {
    // rrot layout:
    //   u32 fourcc('rrot')      [no subtype-specific fields]
    //   [LinearTransform body]  have_bias=0, A=rotation, b=[]
    //   [common VT tail]
    w.writeU32(FaissFourcc.randomRotation);
    _writeLTBody(w, haveBias: false, a: vt.rotation, b: Float32List(0));
    _writeVTCommon(w, vt);
    return;
  }
  if (vt is PCATransform) {
    // Pcam layout (matches upstream write_VectorTransform):
    //   u32 fourcc('Pcam')
    //   f32 eigen_power
    //   f32 epsilon
    //   u8  random_rotation
    //   i32 balanced_bins
    //   WVEC f32 mean           (d_in floats)
    //   WVEC f32 eigenvalues    (upstream writes d_in; we write d_out
    //                            because that's what we retain)
    //   WVEC f32 PCAMat         (upstream writes d_in*d_in; we write
    //                            empty since we don't store the full
    //                            eigenbasis, only the truncated
    //                            projection A)
    //   [LinearTransform body]  have_bias=1, A=projection, b=-A*mean
    //   [common VT tail]
    w.writeU32(FaissFourcc.pca);
    w.writeF32(vt.eigenPower);
    w.writeF32(0.0); // epsilon (unused in this port)
    w.writeU8(0); // random_rotation
    w.writeI32(0); // balanced_bins
    _writeVectorF32(w, vt.mean);
    _writeVectorF32(w, vt.eigenvalues);
    _writeVectorF32(w, Float32List(0)); // PCAMat (not retained)
    // Compute b = -A * mean so a FAISS-side reader can apply the
    // transform directly without recomputing the bias.
    final b = Float32List(vt.dOut);
    for (var r = 0; r < vt.dOut; r++) {
      var s = 0.0;
      final base = r * vt.dIn;
      for (var c = 0; c < vt.dIn; c++) {
        s += vt.projection[base + c] * vt.mean[c];
      }
      b[r] = -s;
    }
    _writeLTBody(w, haveBias: true, a: vt.projection, b: b);
    _writeVTCommon(w, vt);
    return;
  }
  throw UnsupportedError(
    'writeFaissTransform: ${vt.runtimeType} not yet supported.',
  );
}

/// Parses one [VectorTransform] from the reader, dispatching on the
/// fourcc tag that appears at the current offset.
VectorTransform readFaissTransform(IoReader r) {
  final tag = r.readU32();
  if (tag == FaissFourcc.normalizationTransform) {
    final norm = r.readF32();
    final c = _readVTCommon(r);
    if (c.dIn != c.dOut) {
      throw FormatException(
        'VNrm: d_in (${c.dIn}) != d_out (${c.dOut}); '
        'NormalizationTransform must preserve dimension.',
      );
    }
    if ((norm - 2.0).abs() > 1e-6) {
      throw FormatException(
        'VNrm: only norm == 2.0 is supported by this port, got $norm.',
      );
    }
    return L2NormTransform(c.dIn);
  }
  if (tag == FaissFourcc.randomRotation) {
    final body = _readLTBody(r);
    final c = _readVTCommon(r);
    if (c.dIn != c.dOut) {
      throw FormatException(
        'rrot: d_in (${c.dIn}) != d_out (${c.dOut}); this port only '
        'supports square (rotation) matrices.',
      );
    }
    if (body.a.length != c.dIn * c.dOut) {
      throw FormatException(
        'rrot: A has ${body.a.length} floats, expected ${c.dIn * c.dOut}.',
      );
    }
    if (body.haveBias || body.b.isNotEmpty) {
      throw UnsupportedError(
        'rrot: bias is not supported (RandomRotationMatrix has none).',
      );
    }
    final rrot = RandomRotationTransform(d: c.dIn, seed: 0);
    for (var i = 0; i < body.a.length; i++) {
      rrot.rotation[i] = body.a[i];
    }
    rrot.isTrained = c.isTrained;
    return rrot;
  }
  if (tag == FaissFourcc.pca) {
    final eigenPower = r.readF32();
    r.readF32(); // epsilon
    r.readU8(); // random_rotation flag (ignored)
    r.readI32(); // balanced_bins (ignored)
    final mean = _readVectorF32(r);
    final eigenvalues = _readVectorF32(r);
    _readVectorF32(r); // PCAMat — not retained by this port
    final body = _readLTBody(r);
    final c = _readVTCommon(r);
    if (body.a.length != c.dIn * c.dOut) {
      throw FormatException(
        'Pcam: A has ${body.a.length} floats, expected '
        '${c.dIn * c.dOut} for a $c projection.',
      );
    }
    if (mean.length != c.dIn) {
      throw FormatException(
        'Pcam: mean has ${mean.length} floats, expected ${c.dIn}.',
      );
    }
    final pca = PCATransform(
      dIn: c.dIn,
      dOut: c.dOut,
      eigenPower: eigenPower.toDouble(),
    );
    for (var i = 0; i < c.dIn; i++) {
      pca.mean[i] = mean[i];
    }
    for (var i = 0; i < body.a.length; i++) {
      pca.projection[i] = body.a[i];
    }
    // eigenvalues are informational only: this port's apply() uses
    // projection + mean directly. We discard them but the read is
    // still consistent because we consumed them from the stream above.
    // (Explicit ignore to silence "unused local" warnings.)
    // ignore: unused_local_variable
    final _ = eigenvalues;
    pca.isTrained = c.isTrained;
    return pca;
  }
  throw FormatException(
    'Unsupported FAISS transform fourcc "${FaissFourcc.toStr(tag)}" '
    '(0x${tag.toRadixString(16).padLeft(8, '0')})',
  );
}

// --- HNSW helpers -----------------------------------------------------------

/// Recomputes FAISS's `assign_probas` table from `M` and `mL`.
///
/// Mirrors `HNSW::set_default_probas`: keep adding levels while the
/// per-level probability exceeds `1e-9`, at which point the loop
/// breaks. Also returns the parallel `cum_nneighbor_per_level` table
/// so the port can size the neighbor slot buckets consistently.
///
/// Layer 0 gets `2 * M` slots and each upper layer gets `M` slots, so
/// the cumulative table is `[0, 2M, 2M + M, 2M + 2M, ...]`. The
/// returned tables always have `assign_probas.length + 1 ==
/// cum_nneighbor_per_level.length`, matching FAISS's invariant.
({List<double> assignProbas, List<int> cumNneighborPerLevel})
_hnswDefaultProbas(int M, double mL) {
  final assignProbas = <double>[];
  final cumNneighborPerLevel = <int>[0];
  var nn = 0;
  for (var level = 0; ; level++) {
    final proba = math.exp(-level / mL) * (1 - math.exp(-1 / mL));
    if (proba < 1e-9) break;
    assignProbas.add(proba);
    nn += level == 0 ? M * 2 : M;
    cumNneighborPerLevel.add(nn);
  }
  return (
    assignProbas: assignProbas,
    cumNneighborPerLevel: cumNneighborPerLevel,
  );
}

/// Converts the port's per-node adjacency lists into FAISS's flat
/// CSR-ish HNSW payload and writes it to `w`. Does NOT emit the
/// storage sub-index — the caller is responsible for that.
void _writeHnsw(IoWriter w, IndexHNSW idx) {
  final n = idx.ntotal;
  final topLevel = idx.topLevel;
  // Build cum_nneighbor_per_level long enough to cover every observed
  // node level. `_hnswDefaultProbas` already gives us the FAISS-shaped
  // table; extend it if this particular graph happens to be taller
  // than the default probability tail.
  final defaults = _hnswDefaultProbas(idx.M, 1.0 / math.log(idx.M));
  final cum = <int>[...defaults.cumNneighborPerLevel];
  final needed = math.max(topLevel + 1, 0);
  while (cum.length < needed + 1) {
    // Each additional level gets `M` slots (layer 0 already covered
    // by the seed table).
    cum.add(cum.last + idx.M);
  }
  // Levels array: FAISS stores `top_layer + 1`.
  final levels = List<int>.filled(n, 0);
  final offsets = List<int>.filled(n + 1, 0);
  for (var i = 0; i < n; i++) {
    levels[i] = idx.nodeLevel(i) + 1;
    offsets[i + 1] = offsets[i] + cum[levels[i]];
  }
  // Neighbors: -1-filled slot buffer, then overwrite from adjacency
  // lists.
  final neighbors = List<int>.filled(offsets[n], -1);
  for (var i = 0; i < n; i++) {
    final base = offsets[i];
    for (var l = 0; l <= idx.nodeLevel(i); l++) {
      final layerStart = base + cum[l];
      final layerEnd = base + cum[l + 1];
      final adj = idx.nodeNeighbors(i, l);
      if (adj.length > layerEnd - layerStart) {
        throw StateError(
          'HNSW node $i layer $l has ${adj.length} neighbors, '
          'but slot budget is ${layerEnd - layerStart}',
        );
      }
      for (var j = 0; j < adj.length; j++) {
        neighbors[layerStart + j] = adj[j];
      }
    }
  }
  _writeVectorF64(w, defaults.assignProbas);
  _writeVectorI32(w, cum);
  _writeVectorI32(w, levels);
  // `offsets` is a std::vector<size_t> in FAISS, which is u64 on 64-bit
  // little-endian hosts (the only supported build).
  w.writeU64(offsets.length);
  for (final o in offsets) {
    w.writeU64(o);
  }
  _writeVectorI32(w, neighbors);
  w.writeI32(idx.entryPoint);
  w.writeI32(topLevel);
  w.writeI32(idx.efConstruction);
  w.writeI32(idx.efSearch);
  w.writeI32(1); // upper_beam (deprecated; always 1 in modern FAISS)
}

/// Decoded HNSW payload used to hydrate a fresh [IndexHNSW].
class _HnswPayload {
  _HnswPayload({
    required this.M,
    required this.efConstruction,
    required this.efSearch,
    required this.entryPoint,
    required this.topLevel,
    required this.nodeLevels,
    required this.perNodePerLayerNeighbors,
  });
  final int M;
  final int efConstruction;
  final int efSearch;
  final int entryPoint;
  final int topLevel;
  final List<int> nodeLevels;
  final List<List<List<int>>> perNodePerLayerNeighbors;
}

/// Parses the FAISS HNSW payload from `r` and converts it back to the
/// port's per-node adjacency representation.
_HnswPayload _readHnsw(IoReader r, int ntotal) {
  final assignProbas = _readVectorF64(r);
  final cum = _readVectorI32(r);
  if (cum.isEmpty || cum.first != 0) {
    throw FormatException(
      'IHNf: cum_nneighbor_per_level must start with 0 '
      '(got ${cum.isEmpty ? "empty" : cum.first})',
    );
  }
  if (assignProbas.length + 1 != cum.length) {
    throw FormatException(
      'IHNf: assign_probas length ${assignProbas.length} inconsistent '
      'with cum_nneighbor_per_level length ${cum.length}',
    );
  }
  // Recover `M` from the slot budgets. Layer 0 gets `2 * M`; if there's
  // a layer 1, its per-layer budget is exactly `M`.
  final int m0 = cum.length >= 2 ? cum[1] : 0;
  final int mUpper = cum.length >= 3 ? cum[2] - cum[1] : (m0 ~/ 2);
  final int M = mUpper > 0 ? mUpper : (m0 ~/ 2).clamp(1, 1 << 30);
  final levels = _readVectorI32(r);
  if (levels.length != ntotal) {
    throw FormatException(
      'IHNf: levels length ${levels.length} != ntotal $ntotal',
    );
  }
  final offsetsCount = r.readU64();
  if (offsetsCount != ntotal + 1) {
    throw FormatException(
      'IHNf: offsets length $offsetsCount != ntotal + 1 = ${ntotal + 1}',
    );
  }
  final offsets = List<int>.generate(offsetsCount, (_) => r.readU64());
  final neighbors = _readVectorI32(r);
  if (offsets.isNotEmpty && offsets.last != neighbors.length) {
    throw FormatException(
      'IHNf: offsets tail ${offsets.last} != neighbors length '
      '${neighbors.length}',
    );
  }
  final entryPoint = r.readI32();
  final maxLevel = r.readI32();
  final efConstruction = r.readI32();
  final efSearch = r.readI32();
  final upperBeam = r.readI32();
  if (upperBeam != 1) {
    throw UnsupportedError(
      'IHNf: only upper_beam=1 is supported, got $upperBeam',
    );
  }
  // Rebuild per-node adjacency lists, filtering out `-1` sentinels.
  final nodeLevels = List<int>.filled(ntotal, 0);
  final perNodePerLayerNeighbors = List<List<List<int>>>.generate(
    ntotal,
    (_) => <List<int>>[],
  );
  for (var i = 0; i < ntotal; i++) {
    final level = levels[i] - 1;
    if (level < 0) {
      throw FormatException('IHNf: levels[$i] = ${levels[i]} < 1');
    }
    nodeLevels[i] = level;
    final layers = perNodePerLayerNeighbors[i];
    final base = offsets[i];
    for (var l = 0; l <= level; l++) {
      if (l + 1 >= cum.length) {
        throw FormatException(
          'IHNf: node $i requests layer $l but cum table only has '
          '${cum.length - 1} layers',
        );
      }
      final layerStart = base + cum[l];
      final layerEnd = base + cum[l + 1];
      final adj = <int>[];
      for (var j = layerStart; j < layerEnd; j++) {
        final v = neighbors[j];
        if (v < 0) break;
        adj.add(v);
      }
      layers.add(adj);
    }
  }
  return _HnswPayload(
    M: M,
    efConstruction: efConstruction,
    efSearch: efSearch,
    entryPoint: entryPoint,
    topLevel: maxLevel,
    nodeLevels: nodeLevels,
    perNodePerLayerNeighbors: perNodePerLayerNeighbors,
  );
}

/// Serializes [x] into FAISS binary format.
///
/// Currently supports `IndexFlat`. Composite / approximate index
/// types will follow in later batches; each raises [UnsupportedError]
/// today so callers get an actionable message instead of a corrupt
/// file.
void writeFaissIndex(IoWriter w, Index x) {
  if (x is IndexFlat) {
    final tag = x.metric == Metric.l2 ? FaissFourcc.flatL2 : FaissFourcc.flatIP;
    w.writeU32(tag);
    _writeHeader(w, x);
    // IndexFlat stores its vectors as std::vector<uint8_t> whose byte
    // content is the row-major float32 matrix. Reconstruct that blob
    // from our own Float32List backing store.
    final n = x.ntotal;
    final codes = Uint8List(n * x.d * 4);
    if (n > 0) {
      final asFloats = codes.buffer.asFloat32List(0, n * x.d);
      for (var i = 0; i < n; i++) {
        final row = x.reconstruct(i);
        for (var j = 0; j < x.d; j++) {
          asFloats[i * x.d + j] = row[j];
        }
      }
    }
    _writeVectorU8(w, codes);
    return;
  }
  if (x is IndexIDMap) {
    // IxMp layout:
    //   fourcc('IxMp')
    //   write_index_header(this)
    //   write_index(sub-index)     [recursive]
    //   WRITEVECTOR(id_map)        [vector<int64_t>]
    w.writeU32(FaissFourcc.idMap);
    _writeHeader(w, x);
    writeFaissIndex(w, x.inner);
    // Our IndexIDMap keeps the id table in insertion order; the length
    // is guaranteed to equal ntotal.
    final ids = List<int>.generate(x.ntotal, x.idOf);
    _writeVectorI64(w, ids);
    return;
  }
  if (x is IndexPreTransform) {
    // IxPT layout:
    //   fourcc('IxPT')
    //   write_index_header(this)
    //   u32 chain_length
    //   for each transform: write_VectorTransform
    //   write_index(inner)         [recursive]
    w.writeU32(FaissFourcc.preTransform);
    _writeHeader(w, x);
    w.writeU32(x.chain.length);
    for (final t in x.chain) {
      writeFaissTransform(w, t);
    }
    writeFaissIndex(w, x.inner);
    return;
  }
  if (x is IndexPQ) {
    // IxPq layout:
    //   fourcc('IxPq')
    //   write_index_header(this)
    //   write_ProductQuantizer(pq)
    //   WVEC u8 codes             (ntotal * M bytes)
    //   i32 search_type = 0       (ST_PQ; polysemous variants not
    //                              supported)
    //   u8  encode_signs = 0
    //   i32 polysemous_ht = 0
    w.writeU32(FaissFourcc.indexPq);
    _writeHeader(w, x);
    _writeProductQuantizer(w, x.pq);
    _writeVectorU8(w, x.codes);
    w.writeI32(0);
    w.writeU8(0);
    w.writeI32(0);
    return;
  }
  if (x is IndexScalarQuantizer) {
    // IxSQ layout:
    //   fourcc('IxSQ')
    //   write_index_header(this)
    //   write_ScalarQuantizer(sq)
    //   WVEC u8 codes             (ntotal * d bytes)
    w.writeU32(FaissFourcc.indexSq);
    _writeHeader(w, x);
    _writeScalarQuantizer(w, x);
    _writeVectorU8(w, x.codes);
    return;
  }
  if (x is IndexRefineFlat) {
    // IxRF layout:
    //   fourcc('IxRF')
    //   write_index_header(this)
    //   write_index(base_index)     [recursive]
    //   write_index(refine_index)   [recursive; must be IndexFlat]
    //   f32 k_factor
    w.writeU32(FaissFourcc.indexRefine);
    _writeHeader(w, x);
    writeFaissIndex(w, x.base);
    writeFaissIndex(w, x.refine);
    w.writeF32(x.kFactor.toDouble());
    return;
  }
  if (x is IndexIVFFlat) {
    // IwFl layout:
    //   fourcc('IwFl')
    //   write_ivf_header (header + nlist + nprobe + quantizer + direct_map)
    //   write_InvertedLists (ilar container)
    w.writeU32(FaissFourcc.indexIvfFlat);
    _writeIvfHeader(w, x);
    _writeArrayInvertedLists(w, x);
    return;
  }
  if (x is IndexIVFPQ) {
    // IwPQ layout:
    //   fourcc('IwPQ')
    //   write_ivf_header
    //   u8  by_residual         (always 1 for this port)
    //   u64 code_size           (= m for QT_8bit PQ)
    //   write_ProductQuantizer
    //   write_InvertedLists     (ilar, code_size = m)
    w.writeU32(FaissFourcc.indexIvfPq);
    _writeIvfHeaderGeneric(
      w,
      header: x,
      nlist: x.nlist,
      nprobe: x.nprobe,
      quantizer: x.quantizer,
    );
    w.writeU8(1); // by_residual = true
    w.writeU64(x.m); // code_size = m
    _writeProductQuantizer(w, x.pq);
    _writeArrayInvertedListsIVFPQ(w, x);
    return;
  }
  if (x is IndexLSH) {
    // IxHe layout: see [FaissFourcc.indexLsh] for the full spec.
    // The port's IndexLSH stores the raw random projection directly;
    // FAISS wraps the same matrix in a `RandomRotationMatrix`
    // sub-transform whose `A` field IS the projection, so on the wire
    // we emit the LinearTransform block inline (rectangular
    // `[nbits, d]`).
    w.writeU32(FaissFourcc.indexLsh);
    _writeHeader(w, x);
    w.writeI32(x.nbits);
    w.writeU8(1); // rotate_data
    w.writeU8(0); // train_thresholds
    _writeVectorF32(w, Float32List(0)); // thresholds (empty)
    w.writeI32(x.codeSize);
    // Inline `rrot` LinearTransform block (rectangular).
    w.writeU32(FaissFourcc.randomRotation);
    _writeLTBody(w, haveBias: false, a: x.projection, b: Float32List(0));
    w.writeI32(x.d); // d_in
    w.writeI32(x.nbits); // d_out
    w.writeU8(x.isTrained ? 1 : 0);
    // Trailing packed codes.
    final n = x.ntotal;
    final codes = Uint8List(n * x.codeSize);
    if (n > 0) {
      final src = x.codes;
      for (var i = 0; i < codes.length; i++) {
        codes[i] = src[i];
      }
    }
    _writeVectorU8(w, codes);
    return;
  }
  if (x is IndexHNSW) {
    // IHNf layout: see [FaissFourcc.indexHnswFlat] for the full spec.
    w.writeU32(FaissFourcc.indexHnswFlat);
    _writeHeader(w, x);
    _writeHnsw(w, x);
    // Storage sub-index: rebuild an IndexFlat holding the raw vectors
    // so we can reuse the existing IxF2/IxFI encoder path.
    final storageIdx = IndexFlat(x.d, x.metric);
    if (x.ntotal > 0) {
      final rows = List<Float32List>.generate(x.ntotal, (i) {
        return Float32List.sublistView(x.storage, i * x.d, (i + 1) * x.d);
      });
      storageIdx.add(rows);
    }
    storageIdx.isTrained = x.isTrained;
    writeFaissIndex(w, storageIdx);
    return;
  }
  throw UnsupportedError(
    'writeFaissIndex: ${x.runtimeType} not yet supported.',
  );
}

/// Parses a FAISS-format blob starting at the current reader position.
///
/// Reads the fourcc tag and dispatches to the correct concrete type.
Index readFaissIndex(IoReader r) {
  final tag = r.readU32();
  if (tag == FaissFourcc.flatL2 ||
      tag == FaissFourcc.flatIP ||
      tag == FaissFourcc.flat) {
    final h = _readHeader(r);
    final codes = _readVectorU8(r);
    final expectedBytes = h.ntotal * h.d * 4;
    if (codes.length != expectedBytes) {
      throw FormatException(
        'IndexFlat payload size ${codes.length} != ntotal*d*4 = $expectedBytes',
      );
    }
    final idx = IndexFlat(h.d, h.metric);
    if (h.ntotal > 0) {
      final asFloats = codes.buffer.asFloat32List(
        codes.offsetInBytes,
        h.ntotal * h.d,
      );
      final rows = List<Float32List>.generate(
        h.ntotal,
        (i) => Float32List.sublistView(asFloats, i * h.d, (i + 1) * h.d),
      );
      idx.add(rows);
    }
    idx.isTrained = h.isTrained;
    return idx;
  }
  if (tag == FaissFourcc.idMap || tag == FaissFourcc.idMap2) {
    // IxMp and IxM2 share the same on-disk layout — IndexIDMap2 only
    // adds an in-memory reverse-lookup table that FAISS rebuilds on
    // demand, so nothing extra is serialized. The port has no
    // IndexIDMap2 class; both fourccs decode to a plain IndexIDMap.
    final tagStr = FaissFourcc.toStr(tag);
    // Header is that of the wrapper (mirrors the sub-index's d/metric).
    final h = _readHeader(r);
    final inner = readFaissIndex(r);
    final ids = _readVectorI64(r);
    if (ids.length != h.ntotal) {
      throw FormatException(
        '$tagStr: id_map length ${ids.length} != header ntotal ${h.ntotal}',
      );
    }
    if (inner.ntotal != h.ntotal) {
      throw FormatException(
        '$tagStr: inner ntotal ${inner.ntotal} != wrapper ntotal ${h.ntotal}',
      );
    }
    final idmap = IndexIDMap(inner);
    // The sub-index already contains the vectors; we only need to
    // restore the id table without re-adding to inner. Reconstruct
    // each row from inner and re-add via addWithIds so IndexIDMap
    // internal bookkeeping stays consistent. This *does* double the
    // storage momentarily, but keeps encapsulation clean.
    if (h.ntotal > 0) {
      if (inner is! IndexFlat) {
        throw UnsupportedError(
          '$tagStr: reading with a non-Flat inner (${inner.runtimeType}) '
          'is not yet supported.',
        );
      }
      // Rebuild the IDMap on a fresh copy of the inner so that after
      // addWithIds the two ntotals match exactly.
      final freshInner = IndexFlat(inner.d, inner.metric);
      final rebuilt = IndexIDMap(freshInner);
      final rows = List<Float32List>.generate(
        h.ntotal,
        (i) => Float32List.fromList(inner.reconstruct(i)),
      );
      rebuilt.addWithIds(rows, ids);
      rebuilt.isTrained = h.isTrained;
      return rebuilt;
    }
    idmap.isTrained = h.isTrained;
    return idmap;
  }
  if (tag == FaissFourcc.preTransform) {
    final h = _readHeader(r);
    final nt = r.readU32();
    final chain = List<VectorTransform>.generate(
      nt,
      (_) => readFaissTransform(r),
    );
    final inner = readFaissIndex(r);
    if (inner.ntotal != h.ntotal) {
      throw FormatException(
        'IxPT: inner ntotal ${inner.ntotal} != wrapper ntotal ${h.ntotal}',
      );
    }
    final pt = IndexPreTransform(chain: chain, inner: inner);
    pt.ntotal = inner.ntotal;
    return pt;
  }
  if (tag == FaissFourcc.indexPq) {
    // IxPq: header, PQ, codes, search_type, encode_signs, polysemous_ht.
    final h = _readHeader(r);
    final pq = _readProductQuantizer(r);
    if (pq.d != h.d) {
      throw FormatException('IxPq: PQ d (${pq.d}) != header d (${h.d})');
    }
    final codes = _readVectorU8(r);
    final expected = h.ntotal * pq.m;
    if (codes.length != expected) {
      throw FormatException(
        'IxPq: codes length ${codes.length} != ntotal*M = $expected',
      );
    }
    final searchType = r.readI32();
    final encodeSigns = r.readU8();
    final polysemousHt = r.readI32();
    if (searchType != 0 || encodeSigns != 0 || polysemousHt != 0) {
      throw UnsupportedError(
        'IxPq: polysemous variants not supported '
        '(search_type=$searchType, encode_signs=$encodeSigns, '
        'polysemous_ht=$polysemousHt).',
      );
    }
    final idx = IndexPQ(d: h.d, m: pq.m, metric: h.metric);
    // Splice the trained PQ centroids into the new IndexPQ's internal
    // ProductQuantizer. We can't just assign `idx.pq = pq` because
    // it's a final field; instead, copy the codebook floats over.
    final books = idx.pq.codebooks;
    final srcBooks = pq.codebooks;
    for (var sub = 0; sub < pq.m; sub++) {
      for (var c = 0; c < pq.ksub; c++) {
        final src = srcBooks[sub][c];
        final dst = books[sub][c];
        for (var j = 0; j < pq.dsub; j++) {
          dst[j] = src[j];
        }
      }
    }
    idx.pq.isTrained = true;
    idx.isTrained = h.isTrained;
    idx.ioSetCodes(codes, h.ntotal);
    return idx;
  }
  if (tag == FaissFourcc.indexSq) {
    // IxSQ: header, ScalarQuantizer, codes.
    final h = _readHeader(r);
    final sq = _readScalarQuantizer(r, h.metric);
    if (sq.d != h.d) {
      throw FormatException('IxSQ: SQ d (${sq.d}) != header d (${h.d})');
    }
    final codes = _readVectorU8(r);
    final expected = h.ntotal * h.d;
    if (codes.length != expected) {
      throw FormatException(
        'IxSQ: codes length ${codes.length} != ntotal*d = $expected',
      );
    }
    sq.isTrained = h.isTrained;
    sq.ioSetCodes(codes, h.ntotal);
    return sq;
  }
  if (tag == FaissFourcc.indexRefine) {
    // IxRF: header, base index, refine index (must be IndexFlat), k_factor.
    final h = _readHeader(r);
    final base = readFaissIndex(r);
    final refine = readFaissIndex(r);
    if (refine is! IndexFlat) {
      throw FormatException(
        'IxRF: refine sub-index is ${refine.runtimeType}, expected IndexFlat',
      );
    }
    if (base.d != h.d || base.metric != h.metric) {
      throw FormatException(
        'IxRF: header (d=${h.d}, metric=${h.metric}) disagrees with '
        'base (d=${base.d}, metric=${base.metric})',
      );
    }
    final kFactor = r.readF32();
    return IndexRefineFlat.ioLoad(
      base: base,
      refine: refine,
      kFactor: kFactor.round(),
      ntotal: h.ntotal,
      isTrained: h.isTrained,
    );
  }
  if (tag == FaissFourcc.indexIvfFlat) {
    // IwFl: ivf_header + ilar InvertedLists.
    final ivfHead = _readIvfHeader(r);
    final h = ivfHead.h;
    final ivf = IndexIVFFlat(
      d: h.d,
      nlist: ivfHead.nlist,
      metric: h.metric,
      nprobe: ivfHead.nprobe,
    );
    // Splice loaded centroids into the freshly-created coarse quantizer.
    // IndexIVFFlat's constructor allocates an empty IndexFlat; we need
    // it populated so search() / add() can assign to cells.
    final centroids = List<Float32List>.generate(
      ivfHead.quantizer.ntotal,
      (i) => Float32List.fromList(ivfHead.quantizer.reconstruct(i)),
    );
    ivf.quantizer.add(centroids);
    ivf.isTrained = h.isTrained;
    final pay = _readArrayInvertedLists(r, h.d, ivfHead.nlist, h.ntotal);
    ivf.ioSetStorageAndInvLists(pay.storage, pay.invLists, h.ntotal);
    return ivf;
  }
  if (tag == FaissFourcc.indexIvfPq) {
    // IwPQ: ivf_header + by_residual + code_size + PQ + ilar codes.
    final ivfHead = _readIvfHeader(r);
    final h = ivfHead.h;
    final byResidual = r.readU8();
    if (byResidual != 1) {
      throw UnsupportedError(
        'IwPQ: only by_residual=1 is supported, got $byResidual',
      );
    }
    final codeSize = r.readU64();
    final pq = _readProductQuantizer(r);
    if (pq.d != h.d) {
      throw FormatException('IwPQ: PQ d (${pq.d}) != header d (${h.d})');
    }
    if (codeSize != pq.m) {
      throw FormatException(
        'IwPQ: code_size ($codeSize) != pq.M (${pq.m}) for QT_8bit PQ',
      );
    }
    if (pq.nbits != 8) {
      throw UnsupportedError(
        'IwPQ: only nbits=8 PQ is supported, got nbits=${pq.nbits}',
      );
    }
    final ivpq = IndexIVFPQ(
      d: h.d,
      nlist: ivfHead.nlist,
      m: pq.m,
      nprobe: ivfHead.nprobe,
      metric: h.metric,
    );
    // Populate coarse quantizer centroids.
    final centroids = List<Float32List>.generate(
      ivfHead.quantizer.ntotal,
      (i) => Float32List.fromList(ivfHead.quantizer.reconstruct(i)),
    );
    ivpq.quantizer.add(centroids);
    // Splice PQ codebook centroids into the fresh ProductQuantizer.
    final books = ivpq.pq.codebooks;
    final srcBooks = pq.codebooks;
    for (var sub = 0; sub < pq.m; sub++) {
      for (var c = 0; c < pq.ksub; c++) {
        final src = srcBooks[sub][c];
        final dst = books[sub][c];
        for (var j = 0; j < pq.dsub; j++) {
          dst[j] = src[j];
        }
      }
    }
    ivpq.pq.isTrained = true;
    ivpq.isTrained = h.isTrained;
    final pay = _readArrayInvertedListsIVFPQ(r, pq.m, ivfHead.nlist, h.ntotal);
    ivpq.ioSetInvLists(pay.ids, pay.codes, h.ntotal);
    return ivpq;
  }
  if (tag == FaissFourcc.indexLsh) {
    // IxHe: header + nbits + rotate_data + train_thresholds +
    //       thresholds + code_size + inline rrot + codes.
    final h = _readHeader(r);
    if (h.metric != Metric.l2) {
      throw FormatException(
        'IxHe: only METRIC_L2 is supported, got ${h.metric}',
      );
    }
    final nbits = r.readI32();
    final rotateData = r.readU8();
    if (rotateData != 1) {
      throw UnsupportedError(
        'IxHe: only rotate_data=1 is supported, got $rotateData',
      );
    }
    final trainThresholds = r.readU8();
    if (trainThresholds != 0) {
      throw UnsupportedError(
        'IxHe: only train_thresholds=0 is supported, got $trainThresholds',
      );
    }
    final thresholds = _readVectorF32(r);
    if (thresholds.isNotEmpty) {
      throw FormatException(
        'IxHe: thresholds must be empty when train_thresholds=0, '
        'got ${thresholds.length} floats',
      );
    }
    final codeSize = r.readI32();
    final expectedCodeSize = (nbits + 7) ~/ 8;
    if (codeSize != expectedCodeSize) {
      throw FormatException(
        'IxHe: code_size ($codeSize) != (nbits + 7) / 8 = $expectedCodeSize',
      );
    }
    // Inline `rrot` LinearTransform (rectangular `[nbits, d]`).
    final rrotTag = r.readU32();
    if (rrotTag != FaissFourcc.randomRotation) {
      throw FormatException(
        'IxHe: expected inline rrot fourcc, got '
        '"${FaissFourcc.toStr(rrotTag)}"',
      );
    }
    final body = _readLTBody(r);
    final c = _readVTCommon(r);
    if (c.dIn != h.d || c.dOut != nbits) {
      throw FormatException(
        'IxHe: rrot d_in=${c.dIn}, d_out=${c.dOut} but header d=${h.d}, '
        'nbits=$nbits',
      );
    }
    if (body.haveBias || body.b.isNotEmpty) {
      throw UnsupportedError(
        'IxHe: rrot with bias is not supported (RandomRotationMatrix has none)',
      );
    }
    if (body.a.length != nbits * h.d) {
      throw FormatException(
        'IxHe: rrot A has ${body.a.length} floats, expected '
        '${nbits * h.d}',
      );
    }
    final codes = _readVectorU8(r);
    if (codes.length != h.ntotal * codeSize) {
      throw FormatException(
        'IxHe: codes length ${codes.length} != ntotal * code_size = '
        '${h.ntotal * codeSize}',
      );
    }
    final idx = IndexLSH(d: h.d, nbits: nbits);
    idx.ioSetProjectionAndCodes(body.a, codes, h.ntotal);
    idx.isTrained = h.isTrained;
    return idx;
  }
  if (tag == FaissFourcc.indexHnswFlat) {
    // IHNf: header, HNSW graph payload, storage sub-index (IndexFlat).
    final h = _readHeader(r);
    final payload = _readHnsw(r, h.ntotal);
    final storage = readFaissIndex(r);
    if (storage is! IndexFlat) {
      throw FormatException(
        'IHNf: storage sub-index is ${storage.runtimeType}, '
        'expected IndexFlat',
      );
    }
    if (storage.d != h.d || storage.metric != h.metric) {
      throw FormatException(
        'IHNf: storage (d=${storage.d}, metric=${storage.metric}) '
        'disagrees with header (d=${h.d}, metric=${h.metric})',
      );
    }
    if (storage.ntotal != h.ntotal) {
      throw FormatException(
        'IHNf: storage ntotal ${storage.ntotal} != header ntotal '
        '${h.ntotal}',
      );
    }
    final idx = IndexHNSW(
      d: h.d,
      metric: h.metric,
      M: payload.M,
      efConstruction: payload.efConstruction,
      efSearch: payload.efSearch,
    );
    final flatStorage = Float32List(h.ntotal * h.d);
    for (var i = 0; i < h.ntotal; i++) {
      final row = storage.reconstruct(i);
      for (var j = 0; j < h.d; j++) {
        flatStorage[i * h.d + j] = row[j];
      }
    }
    idx.ioSetGraph(
      newStorage: flatStorage,
      nodeLevels: payload.nodeLevels,
      perNodePerLayerNeighbors: payload.perNodePerLayerNeighbors,
      newEntryPoint: payload.entryPoint,
      newTopLevel: payload.topLevel,
      newNtotal: h.ntotal,
    );
    idx.isTrained = h.isTrained;
    return idx;
  }
  throw FormatException(
    'Unsupported FAISS fourcc "${FaissFourcc.toStr(tag)}" '
    '(0x${tag.toRadixString(16).padLeft(8, '0')})',
  );
}

/// Convenience: serialize [x] to a fresh byte buffer.
Uint8List writeFaissIndexToBytes(Index x) {
  final w = IoWriter();
  writeFaissIndex(w, x);
  return w.takeBytes();
}

/// Convenience: parse a byte buffer written by [writeFaissIndexToBytes]
/// or by upstream FAISS.
Index readFaissIndexFromBytes(Uint8List bytes) {
  return readFaissIndex(IoReader(bytes));
}

/// Save [x] to [path] in FAISS binary format.
///
/// The resulting file has the same magic layout that
/// `faiss::write_index(idx, path)` produces on 64-bit little-endian
/// hosts (the only configuration FAISS supports).
void saveFaissIndex(String path, Index x) {
  File(path).writeAsBytesSync(writeFaissIndexToBytes(x));
}

/// Load a FAISS-format file from [path].
Index loadFaissIndex(String path) {
  return readFaissIndexFromBytes(File(path).readAsBytesSync());
}

// -----------------------------------------------------------------------------
// Cheap metadata inspection (probe)
// -----------------------------------------------------------------------------

/// Category of a FAISS-format blob, distinguished by the leading
/// fourcc tag. Determines which header layout follows and which
/// `readFaissIndex*` entry point can decode the payload.
enum FaissIndexKind {
  /// Recognized float-index fourcc (`Ix*`, `Iw*`, `IHNf`, ...). The
  /// blob starts with the 33-byte float `index_header` immediately
  /// after the fourcc and can be decoded by [readFaissIndex].
  floatIndex,

  /// Recognized binary-index fourcc (`IB*`). The blob starts with the
  /// 21-byte binary `index_header` and can be decoded by
  /// [readFaissBinaryIndex].
  binaryIndex,

  /// Port-specific `IxDT` wrapper (see [writeTunedFaissIndexToBytes])
  /// that carries an [OperatingPoints] sweep alongside an inner
  /// FAISS blob. When [probeFaissIndex] returns this kind, the
  /// returned [FaissIndexInfo] populates both [FaissIndexInfo.tuning]
  /// and [FaissIndexInfo.inner]; the top-level `d`/`ntotal`/`metric`/
  /// `isTrained` are copied from the inner probe as a convenience.
  tunedWrapper,

  /// Fourcc is not in the port's known-index table. Only the raw
  /// fourcc bytes are populated on the returned [FaissIndexInfo] —
  /// the header layout is unknown, so `d` / `ntotal` / etc. are left
  /// `null`.
  unknown,
}

/// Metadata extracted from the leading bytes of a FAISS-format blob
/// by [probeFaissIndex]. The header fields are only populated when
/// the fourcc is recognized ([kind] != [FaissIndexKind.unknown]);
/// unknown blobs expose just [fourcc] / [fourccStr] so tools can
/// surface an actionable "unsupported FAISS type X" message without
/// paying the full decode cost.
class FaissIndexInfo {
  FaissIndexInfo._({
    required this.fourcc,
    required this.fourccStr,
    required this.kind,
    this.d,
    this.ntotal,
    this.metric,
    this.isTrained,
    this.codeSize,
    this.tuning,
    this.inner,
  });

  /// Raw little-endian u32 fourcc read from the first 4 bytes.
  final int fourcc;

  /// Human-readable 4-char ASCII rendering of [fourcc] (e.g. `IxF2`).
  final String fourccStr;

  /// Whether the fourcc names a float, binary, or unknown index type.
  final FaissIndexKind kind;

  /// Vector dimensionality (elements for float, bits for binary). Only
  /// populated when [kind] is not [FaissIndexKind.unknown].
  final int? d;

  /// Number of vectors currently stored. Populated for both float and
  /// binary indexes when [kind] is not [FaissIndexKind.unknown].
  final int? ntotal;

  /// Distance metric. Always populated for float indexes with a
  /// recognized metric code; for binary indexes this is `Metric.l2`
  /// because FAISS's `IndexBinary` stores `METRIC_L2 = 1` in its
  /// header even though the real distance is Hamming.
  final Metric? metric;

  /// `true` when the index self-reports as trained. Populated for
  /// recognized fourccs only.
  final bool? isTrained;

  /// Packed code size in **bytes** per vector. Populated only for
  /// binary indexes; `null` for float indexes.
  final int? codeSize;

  /// Tuning metadata decoded from the `IxDT` wrapper. Populated only
  /// when [kind] is [FaissIndexKind.tunedWrapper]; `null` for all
  /// other kinds.
  final TuningMetadata? tuning;

  /// One-level probe of the inner FAISS blob carried by the `IxDT`
  /// wrapper. Populated only when [kind] is
  /// [FaissIndexKind.tunedWrapper]; `null` for all other kinds.
  final FaissIndexInfo? inner;

  @override
  String toString() =>
      'FaissIndexInfo(fourcc=$fourccStr, kind=$kind, '
      'd=$d, ntotal=$ntotal, metric=$metric, isTrained=$isTrained'
      '${codeSize != null ? ", codeSize=$codeSize" : ""}'
      '${tuning != null ? ", tuning=$tuning" : ""})';
}

bool _isKnownFloatFourcc(int tag) {
  return tag == FaissFourcc.flatL2 ||
      tag == FaissFourcc.flatIP ||
      tag == FaissFourcc.flat ||
      tag == FaissFourcc.idMap ||
      tag == FaissFourcc.idMap2 ||
      tag == FaissFourcc.preTransform ||
      tag == FaissFourcc.indexPq ||
      tag == FaissFourcc.indexSq ||
      tag == FaissFourcc.indexRefine ||
      tag == FaissFourcc.indexIvfFlat ||
      tag == FaissFourcc.indexIvfPq ||
      tag == FaissFourcc.indexLsh ||
      tag == FaissFourcc.indexHnswFlat;
}

bool _isKnownBinaryFourcc(int tag) {
  return tag == FaissFourcc.indexBinaryFlat ||
      tag == FaissFourcc.indexBinaryIvf;
}

/// Attempts to decode a [Metric] from the FAISS raw metric code
/// without throwing. Returns `null` when the code is not one of the
/// two metrics the port supports.
Metric? _tryMetricFromFaiss(int v) {
  switch (v) {
    case 0:
      return Metric.innerProduct;
    case 1:
      return Metric.l2;
    default:
      return null;
  }
}

/// Peeks the fourcc + immediate index header from a FAISS-format blob
/// without decoding the full payload.
///
/// This is a cheap O(1) inspection helper — the reader advances at
/// most 4 + 33 bytes (float indexes) or 4 + 21 bytes (binary indexes)
/// and never allocates the underlying storage or graph structures.
/// It is intended for CLI inspection tools, format-sniffing routers,
/// and pre-flight validation before committing to a full
/// [readFaissIndex] / [readFaissBinaryIndex] call.
///
/// For unrecognized fourccs the returned [FaissIndexInfo] has
/// [FaissIndexInfo.kind] set to [FaissIndexKind.unknown] and only the
/// fourcc fields populated — callers can then surface a targeted
/// error message rather than a truncated-read exception.
FaissIndexInfo probeFaissIndex(Uint8List bytes) {
  if (bytes.length < 4) {
    throw FormatException(
      'probeFaissIndex: blob is ${bytes.length} bytes, need at least 4 '
      'for the fourcc',
    );
  }
  final r = IoReader(bytes);
  final tag = r.readU32();
  final tagStr = FaissFourcc.toStr(tag);
  if (tag == _fourccIxDT) {
    // Unwrap the port-specific tuning wrapper and probe the inner blob
    // one level down. The wrapper itself doesn't have a FAISS-style
    // header, so top-level d/ntotal/etc. are propagated from the
    // inner probe.
    return _probeTunedWrapper(bytes, r);
  }
  if (_isKnownFloatFourcc(tag)) {
    // Float header is 33 bytes: i32 d, i64 ntotal, 2 * i64 dummy,
    // u8 is_trained, i32 metric.
    if (bytes.length < 4 + 33) {
      throw FormatException(
        'probeFaissIndex: blob is ${bytes.length} bytes but the float '
        '"$tagStr" index_header needs 4 + 33 = 37',
      );
    }
    final h = _readHeader(r);
    return FaissIndexInfo._(
      fourcc: tag,
      fourccStr: tagStr,
      kind: FaissIndexKind.floatIndex,
      d: h.d,
      ntotal: h.ntotal,
      metric: h.metric,
      isTrained: h.isTrained,
    );
  }
  if (_isKnownBinaryFourcc(tag)) {
    // Binary header is 21 bytes: i32 d, i32 code_size, i64 ntotal,
    // u8 is_trained, i32 metric.
    if (bytes.length < 4 + 21) {
      throw FormatException(
        'probeFaissIndex: blob is ${bytes.length} bytes but the binary '
        '"$tagStr" index_header needs 4 + 21 = 25',
      );
    }
    final h = _readBinaryHeader(r);
    return FaissIndexInfo._(
      fourcc: tag,
      fourccStr: tagStr,
      kind: FaissIndexKind.binaryIndex,
      d: h.d,
      ntotal: h.ntotal,
      metric: _tryMetricFromFaiss(h.metric),
      isTrained: h.isTrained,
      codeSize: h.codeSize,
    );
  }
  return FaissIndexInfo._(
    fourcc: tag,
    fourccStr: tagStr,
    kind: FaissIndexKind.unknown,
  );
}

/// File-based variant of [probeFaissIndex]. Reads at most the first
/// 37 bytes of [path] (enough to cover both float and binary headers)
/// — significantly cheaper than [loadFaissIndex] for inspection use
/// cases that only need the top-level metadata.
///
/// For `IxDT` tuning-wrapper files the reader falls back to loading
/// the full file, since the tuning payload length — and therefore the
/// offset of the inner blob — is not known until the wrapper header
/// is decoded. Wrapped files are typically small, so this stays cheap
/// in practice.
FaissIndexInfo probeFaissIndexFile(String path) {
  final f = File(path);
  final raf = f.openSync();
  try {
    final len = raf.lengthSync();
    if (len >= 4) {
      final head = Uint8List(4);
      raf.readIntoSync(head);
      final tag = head[0] |
          (head[1] << 8) |
          (head[2] << 16) |
          (head[3] << 24);
      if (tag == _fourccIxDT) {
        raf.setPositionSync(0);
        final full = Uint8List(len);
        raf.readIntoSync(full);
        return probeFaissIndex(full);
      }
      raf.setPositionSync(0);
    }
    // 4-byte fourcc + up to 33-byte float header = 37 bytes.
    final want = len < 37 ? len : 37;
    final buf = Uint8List(want);
    raf.readIntoSync(buf);
    return probeFaissIndex(buf);
  } finally {
    raf.closeSync();
  }
}

/// Decodes the `IxDT` wrapper header at the current reader position
/// (immediately after the 4-byte fourcc) and returns a
/// [FaissIndexInfo] with the tuning metadata + inner probe attached.
FaissIndexInfo _probeTunedWrapper(Uint8List bytes, IoReader r) {
  final tagStr = FaissFourcc.toStr(_fourccIxDT);
  if (r.remaining < 4 + 8) {
    throw FormatException(
      'probeFaissIndex: IxDT wrapper header truncated — need at least '
      '4 (version) + 8 (tuningLen) bytes after the fourcc',
    );
  }
  final version = r.readU32();
  if (version != _ixDTVersion) {
    throw FormatException(
      'probeFaissIndex: unsupported IxDT wrapper version $version '
      '(this build handles version $_ixDTVersion)',
    );
  }
  final tuningLen = r.readU64();
  if (r.remaining < tuningLen) {
    throw FormatException(
      'probeFaissIndex: IxDT tuning block declares $tuningLen bytes '
      'but only ${r.remaining} bytes remain in the blob',
    );
  }
  final tuningBytes = r.readBytes(tuningLen);
  final meta = _readTuningPayload(tuningBytes);
  final innerStart = r.position;
  final innerBytes = Uint8List.sublistView(bytes, innerStart);
  final inner = probeFaissIndex(innerBytes);
  return FaissIndexInfo._(
    fourcc: _fourccIxDT,
    fourccStr: tagStr,
    kind: FaissIndexKind.tunedWrapper,
    d: inner.d,
    ntotal: inner.ntotal,
    metric: inner.metric,
    isTrained: inner.isTrained,
    codeSize: inner.codeSize,
    tuning: meta,
    inner: inner,
  );
}

// -----------------------------------------------------------------------------
// Binary indexes
// -----------------------------------------------------------------------------

/// Writes the common `IndexBinary` header, matching FAISS's
/// `write_index_binary_header` in `impl/index_write.cc` byte-for-byte.
///
/// ```
///   i32 d           (bit dimension = code_size * 8)
///   i32 code_size
///   i64 ntotal
///   u8  is_trained
///   i32 metric_type (always 1 = METRIC_L2 on write; upstream default)
/// ```
void _writeBinaryHeader(IoWriter w, IndexBinary x) {
  w.writeI32(x.d);
  w.writeI32(x.codeSize);
  w.writeI64(x.ntotal);
  w.writeU8(x.isTrained ? 1 : 0);
  // IndexBinary's default metric_type is METRIC_L2 = 1 (even though the
  // real distance is Hamming); IndexBinaryFlat does not override it.
  w.writeI32(1);
}

/// Header fields extracted from a FAISS binary-index stream. `metric`
/// is kept as the raw i32 so exotic values (e.g. non-L2) can be
/// surfaced verbatim in error messages instead of throwing during
/// header parsing.
typedef _BinaryHeader = ({
  int d,
  int codeSize,
  int ntotal,
  bool isTrained,
  int metric,
});

_BinaryHeader _readBinaryHeader(IoReader r) {
  final d = r.readI32();
  final codeSize = r.readI32();
  final ntotal = r.readI64();
  final isTrained = r.readU8() != 0;
  final metric = r.readI32();
  return (
    d: d,
    codeSize: codeSize,
    ntotal: ntotal,
    isTrained: isTrained,
    metric: metric,
  );
}

/// Serializes an [IndexBinaryIVF]'s IVF header block, mirroring
/// FAISS's `write_binary_ivf_header`: the binary-index common header,
/// then `nlist`, `nprobe`, the recursive quantizer, then a `NoMap`
/// direct-map block.
void _writeBinaryIvfHeader(IoWriter w, IndexBinaryIVF ivf) {
  _writeBinaryHeader(w, ivf);
  w.writeU64(ivf.nlist);
  w.writeU64(ivf.nprobe);
  writeFaissBinaryIndex(w, ivf.quantizer);
  _writeDirectMapNone(w);
}

/// Parses FAISS's `write_binary_ivf_header` block. Returns the common
/// header plus the extracted `nlist`, `nprobe`, and coarse
/// `quantizer` (which must decode to [IndexBinaryFlat] for our port).
({
  ({int d, int codeSize, int ntotal, bool isTrained, int metric}) h,
  int nlist,
  int nprobe,
  IndexBinaryFlat quantizer,
})
_readBinaryIvfHeader(IoReader r) {
  final h = _readBinaryHeader(r);
  final nlist = r.readU64();
  final nprobe = r.readU64();
  final q = readFaissBinaryIndex(r);
  if (q is! IndexBinaryFlat) {
    throw FormatException(
      'Binary IVF header: quantizer is ${q.runtimeType}, '
      'expected IndexBinaryFlat',
    );
  }
  if (q.codeSize != h.codeSize) {
    throw FormatException(
      'Binary IVF header: quantizer codeSize=${q.codeSize} disagrees '
      'with header codeSize=${h.codeSize}',
    );
  }
  if (q.ntotal != nlist) {
    throw FormatException(
      'Binary IVF header: quantizer has ${q.ntotal} centroids, '
      'header nlist=$nlist',
    );
  }
  _readDirectMap(r);
  return (h: h, nlist: nlist, nprobe: nprobe, quantizer: q);
}

/// Binary IVF variant of [_writeArrayInvertedLists]. Same `ilar`
/// wire format as the float families, but the code source is the
/// port's per-cell `Uint8List` bytes (`n * codeSize`) instead of a
/// flat fp32 storage.
void _writeArrayInvertedListsBinary(IoWriter w, IndexBinaryIVF ivf) {
  final codeSize = ivf.codeSize;
  w.writeU32(FaissFourcc.invListsArray);
  w.writeU64(ivf.nlist);
  w.writeU64(codeSize);

  var nNon0 = 0;
  for (var i = 0; i < ivf.nlist; i++) {
    if (ivf.listSize(i) > 0) nNon0++;
  }
  if (nNon0 > ivf.nlist ~/ 2) {
    w.writeU32(FaissFourcc.invListsFull);
    final sizes = List<int>.generate(ivf.nlist, ivf.listSize);
    _writeVectorI64(w, sizes);
  } else {
    w.writeU32(FaissFourcc.invListsSparse);
    final pairs = <int>[];
    for (var i = 0; i < ivf.nlist; i++) {
      final n = ivf.listSize(i);
      if (n > 0) {
        pairs.add(i);
        pairs.add(n);
      }
    }
    _writeVectorI64(w, pairs);
  }

  for (var i = 0; i < ivf.nlist; i++) {
    final n = ivf.listSize(i);
    if (n == 0) continue;
    w.writeBytes(ivf.invListCodes(i));
    for (final id in ivf.invListIds(i)) {
      w.writeI64(id);
    }
  }
}

/// Binary IVF variant of [_readArrayInvertedLists]. Returns per-cell
/// ids and per-cell packed code buffers, matching
/// [IndexBinaryIVF.ioSetInvertedLists]'s expectations. Asserts
/// contiguous ids in `[0, ntotal)`.
({List<List<int>> ids, List<Uint8List> codes}) _readArrayInvertedListsBinary(
  IoReader r,
  int codeSize,
  int nlist,
  int ntotal,
) {
  final tag = r.readU32();
  if (tag != FaissFourcc.invListsArray) {
    throw UnsupportedError(
      'IBwF: only ilar InvertedLists supported, got fourcc '
      '"${FaissFourcc.toStr(tag)}"',
    );
  }
  final storedNlist = r.readU64();
  final storedCodeSize = r.readU64();
  if (storedNlist != nlist) {
    throw FormatException(
      'ilar: nlist mismatch (payload=$storedNlist, header=$nlist)',
    );
  }
  if (storedCodeSize != codeSize) {
    throw FormatException(
      'ilar: code_size $storedCodeSize != $codeSize for IBwF',
    );
  }

  final sizes = List<int>.filled(nlist, 0);
  final listType = r.readU32();
  if (listType == FaissFourcc.invListsFull) {
    final xs = _readVectorI64(r);
    if (xs.length != nlist) {
      throw FormatException(
        'ilar full: sizes length ${xs.length} != nlist $nlist',
      );
    }
    for (var i = 0; i < nlist; i++) {
      sizes[i] = xs[i];
    }
  } else if (listType == FaissFourcc.invListsSparse) {
    final pairs = _readVectorI64(r);
    if (pairs.length.isOdd) {
      throw FormatException(
        'ilar sprs: pairs length ${pairs.length} is not even',
      );
    }
    for (var j = 0; j < pairs.length; j += 2) {
      final idx = pairs[j];
      if (idx < 0 || idx >= nlist) {
        throw FormatException('ilar sprs: cell idx $idx out of [0, $nlist)');
      }
      sizes[idx] = pairs[j + 1];
    }
  } else {
    throw UnsupportedError(
      'ilar: list_type "${FaissFourcc.toStr(listType)}" not supported',
    );
  }

  var sum = 0;
  for (final s in sizes) {
    sum += s;
  }
  if (sum != ntotal) {
    throw FormatException(
      'ilar: sum of sizes ($sum) != header ntotal ($ntotal)',
    );
  }

  final ids = List<List<int>>.generate(nlist, (_) => <int>[]);
  final codes = List<Uint8List>.generate(nlist, (_) => Uint8List(0));
  for (var i = 0; i < nlist; i++) {
    final n = sizes[i];
    if (n == 0) continue;
    codes[i] = r.readBytes(n * codeSize);
    final cellIds = <int>[];
    for (var j = 0; j < n; j++) {
      final id = r.readI64();
      if (id < 0 || id >= ntotal) {
        throw UnsupportedError(
          'IBwF: ids must be contiguous in [0, ntotal); got id=$id '
          '(ntotal=$ntotal). Non-contiguous FAISS blobs are not yet '
          'supported.',
        );
      }
      cellIds.add(id);
    }
    ids[i] = cellIds;
  }
  return (ids: ids, codes: codes);
}

/// Serializes a binary index [x] in FAISS's on-disk format.
///
/// Parallel to [writeFaissIndex] but for the `IndexBinary` hierarchy,
/// which FAISS keeps in a separate top-level dispatch
/// (`write_index_binary` in `impl/index_write.cc`).
void writeFaissBinaryIndex(IoWriter w, IndexBinary x) {
  if (x is IndexBinaryFlat) {
    // IBxF layout:
    //   fourcc('IBxF')
    //   write_index_binary_header(this)
    //   WVEC u8 xb           (ntotal * code_size bytes)
    w.writeU32(FaissFourcc.indexBinaryFlat);
    _writeBinaryHeader(w, x);
    _writeVectorU8(w, x.codes);
    return;
  }
  if (x is IndexBinaryIVF) {
    // IBwF layout: see [FaissFourcc.indexBinaryIvf] for the full spec.
    w.writeU32(FaissFourcc.indexBinaryIvf);
    _writeBinaryIvfHeader(w, x);
    _writeArrayInvertedListsBinary(w, x);
    return;
  }
  throw UnsupportedError(
    'writeFaissBinaryIndex: ${x.runtimeType} not yet supported.',
  );
}

/// Parses a FAISS binary-index blob starting at the current reader
/// position.
IndexBinary readFaissBinaryIndex(IoReader r) {
  final tag = r.readU32();
  if (tag == FaissFourcc.indexBinaryFlat) {
    final h = _readBinaryHeader(r);
    if (h.codeSize * 8 != h.d) {
      throw FormatException(
        'IBxF: header d=${h.d} but code_size=${h.codeSize} '
        '(expected d = code_size * 8)',
      );
    }
    final codes = _readVectorU8(r);
    if (codes.length != h.ntotal * h.codeSize) {
      throw FormatException(
        'IBxF: xb payload size ${codes.length} != ntotal * code_size = '
        '${h.ntotal * h.codeSize}',
      );
    }
    final idx = IndexBinaryFlat(h.codeSize);
    idx.ioSetCodes(codes, h.ntotal);
    idx.isTrained = h.isTrained;
    return idx;
  }
  if (tag == FaissFourcc.indexBinaryIvf) {
    final head = _readBinaryIvfHeader(r);
    if (head.h.codeSize * 8 != head.h.d) {
      throw FormatException(
        'IBwF: header d=${head.h.d} but code_size=${head.h.codeSize} '
        '(expected d = code_size * 8)',
      );
    }
    final pay = _readArrayInvertedListsBinary(
      r,
      head.h.codeSize,
      head.nlist,
      head.h.ntotal,
    );
    final idx = IndexBinaryIVF(
      codeSize: head.h.codeSize,
      nlist: head.nlist,
      nprobe: head.nprobe,
    );
    // Rehydrate the coarse quantizer directly from the payload we just
    // parsed rather than the freshly-constructed placeholder.
    idx.quantizer.ioSetCodes(head.quantizer.codes, head.quantizer.ntotal);
    idx.quantizer.isTrained = head.quantizer.isTrained;
    idx.isTrained = head.h.isTrained;
    idx.ioSetInvertedLists(pay.ids, pay.codes, head.h.ntotal);
    return idx;
  }
  throw FormatException(
    'Unsupported FAISS binary fourcc "${FaissFourcc.toStr(tag)}" '
    '(0x${tag.toRadixString(16).padLeft(8, '0')})',
  );
}

/// Convenience: serialize a binary index [x] to a fresh byte buffer.
Uint8List writeFaissBinaryIndexToBytes(IndexBinary x) {
  final w = IoWriter();
  writeFaissBinaryIndex(w, x);
  return w.takeBytes();
}

/// Convenience: parse a byte buffer written by
/// [writeFaissBinaryIndexToBytes] or by upstream FAISS
/// (`write_index_binary`).
IndexBinary readFaissBinaryIndexFromBytes(Uint8List bytes) {
  return readFaissBinaryIndex(IoReader(bytes));
}

/// Save a binary index [x] to [path] in FAISS binary-index format.
///
/// The resulting file has the same magic layout that
/// `faiss::write_index_binary(idx, path)` produces on 64-bit
/// little-endian hosts.
void saveFaissBinaryIndex(String path, IndexBinary x) {
  File(path).writeAsBytesSync(writeFaissBinaryIndexToBytes(x));
}

/// Load a FAISS binary-index file from [path].
IndexBinary loadFaissBinaryIndex(String path) {
  return readFaissBinaryIndexFromBytes(File(path).readAsBytesSync());
}

// =============================================================================
// Tuning-metadata wrapper (port-specific fourcc `IxDT`)
// =============================================================================
//
// Persists an auto-tuner's operating-point sweep alongside the inner
// FAISS blob so a later loader can pick a working `nprobe` / `efSearch`
// without redoing the recall benchmark. The wrapper is a Dart-only
// extension — upstream FAISS does not know about `IxDT` and will
// reject a wrapped file. Plain `saveFaissIndex` / `writeFaissIndex...`
// stay 100% byte-compatible with upstream.
//
// Wire format (all little-endian):
//
//   u32 fourcc      = 'I'|'x'<<8|'D'<<16|'T'<<24  (= "IxDT")
//   u32 version     = 1
//   -- tuning block (WVEC-style) --
//   u64 tuningLen
//   [tuningLen bytes of tuning payload]  (see below)
//   -- inner FAISS blob --
//   [remaining bytes = raw upstream-compatible FAISS blob]
//
// Tuning payload:
//   i64 createdAtMicros  (Unix epoch, DateTime.microsecondsSinceEpoch)
//   i32 innerMetric      (0 = IP, 1 = L2 — sanity marker; NOT
//                         authoritative for the inner index which owns
//                         its own metric in its header)
//   u32 numPoints
//   for each point:
//     i32 paramValue
//     u32 labelLen
//     [labelLen bytes utf-8 label]
//     f64 recall
//     f64 meanUs
//   u8  chosenPresent    (0 or 1)
//   if chosenPresent == 1:
//     i32 chosenParamValue

/// Port-specific fourcc marking a tuning-metadata wrapper (see
/// [writeTunedFaissIndexToBytes]). Upstream FAISS does not recognize
/// this tag; keep it out of files intended for cross-tool interop.
final int _fourccIxDT = FaissFourcc.of('IxDT');

/// Current on-disk version of the `IxDT` wrapper. Bumped on any
/// backwards-incompatible payload change.
const int _ixDTVersion = 1;

/// Auto-tuner metadata persisted alongside a FAISS index by the
/// `IxDT` wrapper. Captures the operating-point sweep + the chosen
/// parameter so a downstream loader can restore the tuning decision
/// without re-running the benchmark.
class TuningMetadata {
  TuningMetadata({
    required this.createdAt,
    required this.metric,
    required this.points,
    this.chosenParamValue,
  });

  /// Convenience: snapshot the [OperatingPoints] returned by
  /// [autoTuneNprobe] or [autoTuneEfSearch] into a persistable block.
  /// [chosenParamValue] usually comes from
  /// `OperatingPoints.pickForRecall(...)?.paramValue` or
  /// `OperatingPoints.pickForLatency(...)?.paramValue`.
  factory TuningMetadata.fromOperatingPoints({
    required OperatingPoints points,
    required Metric metric,
    int? chosenParamValue,
    DateTime? createdAt,
  }) {
    return TuningMetadata(
      createdAt: createdAt ?? DateTime.now(),
      metric: metric,
      points: List<OperatingPoint>.from(points.points),
      chosenParamValue: chosenParamValue,
    );
  }

  /// Convenience: snapshot the outcome of [autoTuneM]. The chosen
  /// `paramValue` here is the winning subquantiser count `m` when
  /// [TuneMResult.chosen] is non-null.
  factory TuningMetadata.fromTuneMResult({
    required TuneMResult result,
    required Metric metric,
    DateTime? createdAt,
  }) {
    return TuningMetadata(
      createdAt: createdAt ?? DateTime.now(),
      metric: metric,
      points: List<OperatingPoint>.from(result.points.points),
      chosenParamValue: result.chosen?.paramValue,
    );
  }

  /// When the tuning sweep was recorded.
  final DateTime createdAt;

  /// Sanity marker for the inner index's metric. Not authoritative —
  /// the wrapped FAISS blob owns its own metric in its header.
  final Metric metric;

  /// Operating points from the sweep, in the order they were emitted
  /// by the auto-tuner.
  final List<OperatingPoint> points;

  /// The parameter value chosen by the auto-tuner (e.g. the selected
  /// `nprobe` or `efSearch`), or `null` if the sweep did not pick a
  /// winner.
  final int? chosenParamValue;

  @override
  String toString() =>
      'TuningMetadata(createdAt=$createdAt, metric=$metric, '
      'points=${points.length}, chosen=$chosenParamValue)';
}

/// Serialize [inner] with an attached [TuningMetadata] block using
/// the port-specific `IxDT` wrapper fourcc.
///
/// The resulting blob is NOT loadable by upstream FAISS. Use
/// [saveFaissIndex] / [writeFaissIndexToBytes] when byte compatibility
/// with upstream is required.
Uint8List writeTunedFaissIndexToBytes(Index inner, TuningMetadata meta) {
  final w = IoWriter();
  w.writeU32(_fourccIxDT);
  w.writeU32(_ixDTVersion);
  final tuning = _writeTuningPayload(meta);
  w.writeU64(tuning.length);
  w.writeBytes(tuning);
  w.writeBytes(writeFaissIndexToBytes(inner));
  return w.takeBytes();
}

/// Parse a byte buffer previously written by
/// [writeTunedFaissIndexToBytes]. Throws [FormatException] if the
/// leading fourcc is not `IxDT` or the wrapper version is unknown.
///
/// When [applyTuning] is `true`, the chosen operating-point value
/// from the tuning block is applied to the decoded index via
/// [applyTuningToIndex] before returning — useful for one-call
/// "load and warm up" workflows.
({Index index, TuningMetadata metadata}) readTunedFaissIndexFromBytes(
  Uint8List bytes, {
  bool applyTuning = false,
}) {
  if (bytes.length < 4) {
    throw FormatException(
      'readTunedFaissIndex: blob is ${bytes.length} bytes, need at '
      'least 4 for the fourcc',
    );
  }
  final r = IoReader(bytes);
  final tag = r.readU32();
  if (tag != _fourccIxDT) {
    throw FormatException(
      'readTunedFaissIndex: expected fourcc "IxDT" but got '
      '"${FaissFourcc.toStr(tag)}" — this blob is not a tuned wrapper',
    );
  }
  final version = r.readU32();
  if (version != _ixDTVersion) {
    throw FormatException(
      'readTunedFaissIndex: unsupported IxDT wrapper version $version '
      '(this build handles version $_ixDTVersion)',
    );
  }
  final tuningLen = r.readU64();
  if (r.remaining < tuningLen) {
    throw FormatException(
      'readTunedFaissIndex: tuning block declares $tuningLen bytes but '
      'only ${r.remaining} bytes remain in the blob',
    );
  }
  final tuningBytes = r.readBytes(tuningLen);
  final meta = _readTuningPayload(tuningBytes);
  final innerBytes = r.readBytes(r.remaining);
  final inner = readFaissIndexFromBytes(innerBytes);
  if (applyTuning) applyTuningToIndex(inner, meta);
  return (index: inner, metadata: meta);
}

/// File variant of [writeTunedFaissIndexToBytes].
void saveTunedFaissIndex(String path, Index inner, TuningMetadata meta) {
  File(path).writeAsBytesSync(writeTunedFaissIndexToBytes(inner, meta));
}

/// File variant of [readTunedFaissIndexFromBytes]. Forwards
/// [applyTuning] so callers can load-and-warm-up in a single call.
({Index index, TuningMetadata metadata}) loadTunedFaissIndex(
  String path, {
  bool applyTuning = false,
}) {
  return readTunedFaissIndexFromBytes(
    File(path).readAsBytesSync(),
    applyTuning: applyTuning,
  );
}

/// Apply the chosen operating-point value from [meta] to [index].
///
/// Inspects [TuningMetadata.chosenParamValue] together with the
/// matching point's `paramLabel` to decide which knob to set:
///
///   * `nprobe` prefix   -> `IndexIVFFlat` / `IndexIVFPQ` (also when
///                          wrapped in an `IndexRefineFlat`) `.nprobe`
///   * `efSearch` prefix -> `IndexHNSW.efSearch`
///
/// Returns `true` when a setter was found and applied; `false` when
/// [TuningMetadata.chosenParamValue] is `null` or the sweep produced
/// a parameter class this helper doesn't know how to re-apply
/// post-facto (e.g. PQ subquantiser count `m`, which is baked into
/// the trained index).
///
/// Throws [ArgumentError] when the chosen paramValue is not in
/// [TuningMetadata.points], or when a recognised label does not
/// match the index type (e.g. `nprobe=8` targeting an `IndexFlat`).
bool applyTuningToIndex(Index index, TuningMetadata meta) {
  final chosen = meta.chosenParamValue;
  if (chosen == null) return false;
  final point = meta.points.firstWhere(
    (p) => p.paramValue == chosen,
    orElse: () => throw ArgumentError(
      'applyTuningToIndex: chosen paramValue $chosen is not present '
      'in TuningMetadata.points',
    ),
  );
  final label = point.paramLabel.toLowerCase();
  if (label.startsWith('nprobe')) {
    _applyNprobe(index, chosen);
    return true;
  }
  if (label.startsWith('efsearch')) {
    _applyEfSearch(index, chosen);
    return true;
  }
  return false;
}

void _applyNprobe(Index index, int value) {
  if (index is IndexIVFFlat) {
    index.nprobe = value;
    return;
  }
  if (index is IndexIVFPQ) {
    index.nprobe = value;
    return;
  }
  if (index is IndexRefineFlat) {
    final base = index.base;
    if (base is IndexIVFFlat) {
      base.nprobe = value;
      return;
    }
    if (base is IndexIVFPQ) {
      base.nprobe = value;
      return;
    }
  }
  throw ArgumentError(
    'applyTuningToIndex: no nprobe setter for ${index.runtimeType}',
  );
}

void _applyEfSearch(Index index, int value) {
  if (index is IndexHNSW) {
    index.efSearch = value;
    return;
  }
  throw ArgumentError(
    'applyTuningToIndex: no efSearch setter for ${index.runtimeType}',
  );
}

/// Returns `true` when [bytes] begins with the `IxDT` wrapper fourcc.
/// Useful for tools that want to route a blob between
/// [readFaissIndexFromBytes] and [readTunedFaissIndexFromBytes]
/// without catching a `FormatException`.
bool isTunedFaissBlob(Uint8List bytes) {
  if (bytes.length < 4) return false;
  final tag = bytes[0] |
      (bytes[1] << 8) |
      (bytes[2] << 16) |
      (bytes[3] << 24);
  return tag == _fourccIxDT;
}

/// Return the raw inner FAISS blob from an `IxDT`-wrapped byte
/// buffer, discarding the tuning-metadata block. The result is
/// byte-identical to what upstream FAISS would produce for the inner
/// index; feed it back to [readFaissIndexFromBytes] or to
/// `faiss::read_index` in the reference implementation.
///
/// Throws a [FormatException] when the leading fourcc is not `IxDT`
/// or the wrapper header is truncated.
Uint8List stripTuningWrapper(Uint8List bytes) {
  if (bytes.length < 4) {
    throw FormatException(
      'stripTuningWrapper: blob is ${bytes.length} bytes, need at '
      'least 4 for the fourcc',
    );
  }
  final r = IoReader(bytes);
  final tag = r.readU32();
  if (tag != _fourccIxDT) {
    throw FormatException(
      'stripTuningWrapper: expected fourcc "IxDT" but got '
      '"${FaissFourcc.toStr(tag)}" — this blob is not a tuned wrapper',
    );
  }
  final version = r.readU32();
  if (version != _ixDTVersion) {
    throw FormatException(
      'stripTuningWrapper: unsupported IxDT wrapper version $version '
      '(this build handles version $_ixDTVersion)',
    );
  }
  final tuningLen = r.readU64();
  if (r.remaining < tuningLen) {
    throw FormatException(
      'stripTuningWrapper: IxDT tuning block declares $tuningLen '
      'bytes but only ${r.remaining} bytes remain in the blob',
    );
  }
  final innerStart = r.position + tuningLen;
  return Uint8List.fromList(Uint8List.sublistView(bytes, innerStart));
}

/// File variant of [stripTuningWrapper]: read the `IxDT` file at
/// [inPath], write the inner FAISS blob to [outPath]. The written
/// file is loadable by both [loadFaissIndex] and upstream
/// `faiss::read_index`.
void stripTuningWrapperFile(String inPath, String outPath) {
  final bytes = File(inPath).readAsBytesSync();
  File(outPath).writeAsBytesSync(stripTuningWrapper(bytes));
}

Uint8List _writeTuningPayload(TuningMetadata meta) {
  final w = IoWriter();
  w.writeI64(meta.createdAt.microsecondsSinceEpoch);
  w.writeI32(meta.metric == Metric.innerProduct ? 0 : 1);
  w.writeU32(meta.points.length);
  for (final p in meta.points) {
    w.writeI32(p.paramValue);
    final label = utf8Encode(p.paramLabel);
    w.writeU32(label.length);
    w.writeBytes(label);
    w.writeF64(p.recall);
    w.writeF64(p.meanUs);
  }
  final chosen = meta.chosenParamValue;
  if (chosen == null) {
    w.writeU8(0);
  } else {
    w.writeU8(1);
    w.writeI32(chosen);
  }
  return w.takeBytes();
}

TuningMetadata _readTuningPayload(Uint8List bytes) {
  final r = IoReader(bytes);
  final createdAt = DateTime.fromMicrosecondsSinceEpoch(r.readI64());
  final metricCode = r.readI32();
  final metric = metricCode == 0 ? Metric.innerProduct : Metric.l2;
  final n = r.readU32();
  final points = <OperatingPoint>[];
  for (var i = 0; i < n; i++) {
    final paramValue = r.readI32();
    final labelLen = r.readU32();
    final labelBytes = r.readBytes(labelLen);
    final label = utf8Decode(labelBytes);
    final recall = r.readF64();
    final meanUs = r.readF64();
    points.add(OperatingPoint(
      paramValue: paramValue,
      paramLabel: label,
      recall: recall,
      meanUs: meanUs,
    ));
  }
  final chosenPresent = r.readU8();
  int? chosen;
  if (chosenPresent == 1) {
    chosen = r.readI32();
  }
  return TuningMetadata(
    createdAt: createdAt,
    metric: metric,
    points: points,
    chosenParamValue: chosen,
  );
}

/// Minimal UTF-8 encoder / decoder used by the tuning-payload codec.
/// Dart's `dart:convert` is intentionally avoided here to keep this
/// module free of extra imports; labels emitted by the auto-tuner are
/// short ASCII strings ("nprobe=8"), so the fast path dominates.
Uint8List utf8Encode(String s) {
  final out = <int>[];
  for (final code in s.runes) {
    if (code < 0x80) {
      out.add(code);
    } else if (code < 0x800) {
      out.add(0xc0 | (code >> 6));
      out.add(0x80 | (code & 0x3f));
    } else if (code < 0x10000) {
      out.add(0xe0 | (code >> 12));
      out.add(0x80 | ((code >> 6) & 0x3f));
      out.add(0x80 | (code & 0x3f));
    } else {
      out.add(0xf0 | (code >> 18));
      out.add(0x80 | ((code >> 12) & 0x3f));
      out.add(0x80 | ((code >> 6) & 0x3f));
      out.add(0x80 | (code & 0x3f));
    }
  }
  return Uint8List.fromList(out);
}

String utf8Decode(Uint8List bytes) {
  final codes = <int>[];
  var i = 0;
  while (i < bytes.length) {
    final b = bytes[i++];
    if (b < 0x80) {
      codes.add(b);
    } else if ((b & 0xe0) == 0xc0) {
      codes.add(((b & 0x1f) << 6) | (bytes[i++] & 0x3f));
    } else if ((b & 0xf0) == 0xe0) {
      final b1 = bytes[i++] & 0x3f;
      final b2 = bytes[i++] & 0x3f;
      codes.add(((b & 0x0f) << 12) | (b1 << 6) | b2);
    } else {
      final b1 = bytes[i++] & 0x3f;
      final b2 = bytes[i++] & 0x3f;
      final b3 = bytes[i++] & 0x3f;
      codes.add(((b & 0x07) << 18) | (b1 << 12) | (b2 << 6) | b3);
    }
  }
  return String.fromCharCodes(codes);
}
