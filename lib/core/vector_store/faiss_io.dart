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
/// * `IndexPreTransform` — fourcc `IxPT`, chain of transforms + inner.
/// * `IndexPQ`           — fourcc `IxPq`, product-quantised flat index.
/// * `IndexScalarQuantizer` — fourcc `IxSQ`, 8-bit per-dim scalar
///   quantization (`QT_8bit` + `RS_minmax`).
/// * `IndexRefineFlat`   — fourcc `IxRF`, base approximate index +
///   exact fp32 refine store + `k_factor` candidate multiplier.
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
import 'dart:typed_data';

import 'index.dart';
import 'index_flat.dart';
import 'index_id_map.dart';
import 'index_io.dart';
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
  if (tag == FaissFourcc.idMap) {
    // Header is that of the wrapper (mirrors the sub-index's d/metric).
    final h = _readHeader(r);
    final inner = readFaissIndex(r);
    final ids = _readVectorI64(r);
    if (ids.length != h.ntotal) {
      throw FormatException(
        'IxMp: id_map length ${ids.length} != header ntotal ${h.ntotal}',
      );
    }
    if (inner.ntotal != h.ntotal) {
      throw FormatException(
        'IxMp: inner ntotal ${inner.ntotal} != wrapper ntotal ${h.ntotal}',
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
          'IxMp: reading with a non-Flat inner (${inner.runtimeType}) '
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
