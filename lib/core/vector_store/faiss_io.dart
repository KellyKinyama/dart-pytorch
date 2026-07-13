/// FAISS binary-format interop.
///
/// Reads and writes indexes in the on-disk format produced by the
/// upstream C++ / Python FAISS library (see `faiss/impl/index_read.cc`
/// and `faiss/impl/index_write.cc`). Complementary to the native
/// `FAISDART` format in `index_io.dart`; the two formats are not
/// interchangeable, and each index has its own read/write pair.
///
/// ## Supported types (batch 11)
///
/// * `IndexFlat`  — fourcc `IxF2` (L2) / `IxFI` (inner product).
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
import 'index_io.dart';

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

/// Serializes [x] into FAISS binary format.
///
/// Currently supports `IndexFlat`. Composite / approximate index
/// types will follow in later batches; each raises [UnsupportedError]
/// today so callers get an actionable message instead of a corrupt
/// file.
void writeFaissIndex(IoWriter w, Index x) {
  if (x is IndexFlat) {
    final tag = x.metric == Metric.l2
        ? FaissFourcc.flatL2
        : FaissFourcc.flatIP;
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
  throw UnsupportedError(
    'writeFaissIndex: ${x.runtimeType} not yet supported. '
    'Batch 11 covers IndexFlat only.',
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
