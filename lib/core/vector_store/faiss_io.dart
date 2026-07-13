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
/// * `L2NormTransform`   — fourcc `L2nT`, on-sphere normalization.
/// * `RandomRotationTransform` — fourcc `rrot`, a `LinearTransform`
///   with an orthonormal `d×d` matrix and no bias.
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
library;

import 'dart:io';
import 'dart:typed_data';

import 'index.dart';
import 'index_flat.dart';
import 'index_id_map.dart';
import 'index_io.dart';
import 'index_pre_transform.dart';
import 'l2_norm_transform.dart';
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

  /// `L2nT` — `NormalizationTransform` with `norm = 2.0` (unit-sphere
  /// projection). Used as one element of an `IxPT` chain.
  static final int l2NormTransform = of('L2nT');

  /// `rrot` — `RandomRotationMatrix`, a `LinearTransform` sub-class
  /// with an orthonormal `d×d` matrix and no bias. Shares the common
  /// `write_LinearTransform` tail (A, b, have_bias, is_orthonormal)
  /// with `PCAm`, `Viqm`, `LTra` etc. — future batches will piggyback
  /// on the same helper.
  static final int randomRotation = of('rrot');
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

/// Emits the tail shared by every `LinearTransform` sub-type: the base
/// `write_LinearTransform` block in `faiss/impl/index_write.cc`.
///
/// ```
///   i32 d_in
///   i32 d_out
///   u8  is_trained
///   WRITEVECTOR A        (d_out * d_in floats, row-major)
///   WRITEVECTOR b        (d_out floats when have_bias, else 0)
///   u8  have_bias
///   u8  is_orthonormal
/// ```
void _writeLinearTransformTail(
  IoWriter w, {
  required int dIn,
  required int dOut,
  required bool isTrained,
  required Float32List a,
  required Float32List b,
  required bool haveBias,
  required bool isOrthonormal,
}) {
  w.writeI32(dIn);
  w.writeI32(dOut);
  w.writeU8(isTrained ? 1 : 0);
  _writeVectorF32(w, a);
  _writeVectorF32(w, b);
  w.writeU8(haveBias ? 1 : 0);
  w.writeU8(isOrthonormal ? 1 : 0);
}

/// Companion of [_writeLinearTransformTail]. The subtype fourcc has
/// already been consumed by the caller; any subtype-specific fields
/// (PCA's eigenvalues etc.) must also be handled by the caller before
/// this runs.
typedef _LinearTail = ({
  int dIn,
  int dOut,
  bool isTrained,
  Float32List a,
  Float32List b,
  bool haveBias,
  bool isOrthonormal,
});

_LinearTail _readLinearTransformTail(IoReader r) {
  final dIn = r.readI32();
  final dOut = r.readI32();
  final isTrained = r.readU8() != 0;
  final a = _readVectorF32(r);
  final b = _readVectorF32(r);
  final haveBias = r.readU8() != 0;
  final isOrthonormal = r.readU8() != 0;
  return (
    dIn: dIn,
    dOut: dOut,
    isTrained: isTrained,
    a: a,
    b: b,
    haveBias: haveBias,
    isOrthonormal: isOrthonormal,
  );
}

/// Serializes a [VectorTransform] using FAISS's `write_VectorTransform`
/// dispatch, keyed on the concrete Dart type.
void writeFaissTransform(IoWriter w, VectorTransform vt) {
  if (vt is L2NormTransform) {
    // NormalizationTransform layout:
    //   fourcc('L2nT')
    //   i32 d_in
    //   i32 d_out
    //   f32 norm      (always 2.0f for L2 normalization)
    w.writeU32(FaissFourcc.l2NormTransform);
    w.writeI32(vt.dIn);
    w.writeI32(vt.dOut);
    w.writeF32(2.0);
    return;
  }
  if (vt is RandomRotationTransform) {
    // rrot layout (per faiss/impl/index_write.cc):
    //   fourcc('rrot')             [no subtype-specific fields]
    //   write_LinearTransform tail: dIn, dOut, isTrained, A, b,
    //                               have_bias, is_orthonormal
    // RandomRotationMatrix has no bias and is always orthonormal.
    w.writeU32(FaissFourcc.randomRotation);
    _writeLinearTransformTail(
      w,
      dIn: vt.dIn,
      dOut: vt.dOut,
      isTrained: vt.isTrained,
      a: vt.rotation,
      b: Float32List(0),
      haveBias: false,
      isOrthonormal: true,
    );
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
  if (tag == FaissFourcc.l2NormTransform) {
    final dIn = r.readI32();
    final dOut = r.readI32();
    final norm = r.readF32();
    if (dIn != dOut) {
      throw FormatException(
        'L2nT: d_in ($dIn) != d_out ($dOut); NormalizationTransform '
        'must preserve dimension.',
      );
    }
    if ((norm - 2.0).abs() > 1e-6) {
      throw FormatException(
        'L2nT: only norm == 2.0 is supported by this port, got $norm.',
      );
    }
    return L2NormTransform(dIn);
  }
  if (tag == FaissFourcc.randomRotation) {
    final t = _readLinearTransformTail(r);
    if (t.dIn != t.dOut) {
      throw FormatException(
        'rrot: d_in (${t.dIn}) != d_out (${t.dOut}); this port only '
        'supports square (rotation) matrices.',
      );
    }
    if (t.a.length != t.dIn * t.dOut) {
      throw FormatException(
        'rrot: A has ${t.a.length} floats, expected ${t.dIn * t.dOut}.',
      );
    }
    if (t.haveBias || t.b.isNotEmpty) {
      throw UnsupportedError(
        'rrot: bias is not supported (RandomRotationMatrix has none).',
      );
    }
    if (!t.isOrthonormal) {
      throw FormatException(
        'rrot: is_orthonormal flag must be true for RandomRotationMatrix.',
      );
    }
    // Construct then overwrite the internal matrix. `seed = 0` is
    // harmless — the generated matrix is immediately replaced.
    final rrot = RandomRotationTransform(d: t.dIn, seed: 0);
    for (var i = 0; i < t.a.length; i++) {
      rrot.rotation[i] = t.a[i];
    }
    rrot.isTrained = t.isTrained;
    return rrot;
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
