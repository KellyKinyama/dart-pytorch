/// Persistence layer for the FAISS-in-Dart index toolkit.
///
/// Every index type implements `void writeTo(IoWriter w)` and a
/// matching static `readFrom(IoReader r)`. The top-level entry points
/// [writeIndex] / [readIndex] handle magic-byte dispatch so any saved
/// blob decodes back to the correct concrete class.
///
/// Wire format (little-endian throughout):
///
///   file header:
///     magic       = 'FAISDART'   (8 bytes, ASCII)
///     version     = u32          (currently 1)
///     kind        = u32          (see [IndexKind])
///
///   common index header (written first by every `writeTo`):
///     d           = u32
///     metric      = u32          (0 = L2, 1 = IP)
///     ntotal      = u32
///     isTrained   = u8           (0 or 1)
///
///   payload: index-specific.
///
/// Nested indexes (IDMap, RefineFlat, IVFFlat, IVFPQ) embed a child by
/// calling `writeChild(w, child)` — which writes magic + version + kind
/// + payload just like a top-level blob. This makes every sub-slice a
/// standalone valid file at ~16 bytes of overhead.
library;

import 'dart:io';
import 'dart:typed_data';

import 'index.dart';
import 'index_flat.dart';
import 'index_hnsw.dart';
import 'index_id_map.dart';
import 'index_ivf_flat.dart';
import 'index_ivf_pq.dart';
import 'index_pq.dart';
import 'index_refine_flat.dart';
import 'index_scalar_quantizer.dart';
import 'index_binary.dart';
import 'index_binary_flat.dart';
import 'index_binary_ivf.dart';
import 'index_lsh.dart';
import 'index_pre_transform.dart';
import 'index_shards.dart';
import 'index_replicas.dart';

/// Discriminator values distinguishing each supported index type on
/// disk. Never change existing values — they are the compatibility
/// contract with previously written files.
class IndexKind {
  static const int flat = 0x01;
  static const int ivfFlat = 0x02;
  static const int pq = 0x03;
  static const int ivfPq = 0x04;
  static const int hnsw = 0x05;
  static const int idMap = 0x06;
  static const int scalarQuantizer = 0x07;
  static const int refineFlat = 0x08;
  static const int lsh = 0x09;
  static const int binaryFlat = 0x0A;
  static const int preTransform = 0x0B;
  static const int binaryIVF = 0x0C;
  static const int shards = 0x0D;
  static const int replicas = 0x0E;
}

/// File-format magic and version.
const List<int> _magicBytes = <int>[
  0x46,
  0x41,
  0x49,
  0x53,
  0x44,
  0x41,
  0x52,
  0x54,
]; // 'FAISDART'
const int _version = 1;

/// Little-endian writer over a growing `BytesBuilder`.
class IoWriter {
  final BytesBuilder _b = BytesBuilder();
  final ByteData _scratch = ByteData(8);

  int get length => _b.length;
  Uint8List takeBytes() => _b.takeBytes();

  void writeU8(int v) => _b.addByte(v & 0xff);

  void writeU32(int v) {
    _scratch.setUint32(0, v, Endian.little);
    _b.add(_scratch.buffer.asUint8List(0, 4));
  }

  void writeI32(int v) {
    _scratch.setInt32(0, v, Endian.little);
    _b.add(_scratch.buffer.asUint8List(0, 4));
  }

  void writeF32(double v) {
    _scratch.setFloat32(0, v, Endian.little);
    _b.add(_scratch.buffer.asUint8List(0, 4));
  }

  /// Little-endian int64. Used by FAISS-format interop where the C++
  /// side declares fields as `idx_t` (= `int64_t`) and `size_t`.
  void writeI64(int v) {
    _scratch.setInt64(0, v, Endian.little);
    _b.add(_scratch.buffer.asUint8List(0, 8));
  }

  /// Little-endian uint64. Emits the same bit pattern as [writeI64] for
  /// non-negative values.
  void writeU64(int v) => writeI64(v);

  void writeBytes(Uint8List xs) => _b.add(xs);

  /// Bulk-copies a Float32List assuming little-endian host (all
  /// mainstream Dart platforms).
  void writeF32List(Float32List xs) {
    _b.add(xs.buffer.asUint8List(xs.offsetInBytes, xs.lengthInBytes));
  }

  void writeI32List(Int32List xs) {
    _b.add(xs.buffer.asUint8List(xs.offsetInBytes, xs.lengthInBytes));
  }

  void writeU8List(Uint8List xs) => _b.add(xs);
}

/// Little-endian reader over an existing Uint8List.
class IoReader {
  IoReader(Uint8List bytes)
    : _bytes = bytes,
      _view = ByteData.view(
        bytes.buffer,
        bytes.offsetInBytes,
        bytes.lengthInBytes,
      );

  final Uint8List _bytes;
  final ByteData _view;
  int _pos = 0;

  int get position => _pos;
  int get remaining => _bytes.length - _pos;

  int readU8() {
    final v = _view.getUint8(_pos);
    _pos += 1;
    return v;
  }

  int readU32() {
    final v = _view.getUint32(_pos, Endian.little);
    _pos += 4;
    return v;
  }

  int readI32() {
    final v = _view.getInt32(_pos, Endian.little);
    _pos += 4;
    return v;
  }

  double readF32() {
    final v = _view.getFloat32(_pos, Endian.little);
    _pos += 4;
    return v;
  }

  /// Little-endian int64. Companion to [IoWriter.writeI64].
  int readI64() {
    final v = _view.getInt64(_pos, Endian.little);
    _pos += 8;
    return v;
  }

  /// Little-endian uint64. Dart has no unsigned 64-bit type; values
  /// above 2^63 - 1 wrap into negative territory. FAISS files never
  /// encode sizes that large in practice.
  int readU64() => readI64();

  Uint8List readBytes(int n) {
    final out = Uint8List.fromList(
      Uint8List.sublistView(_bytes, _pos, _pos + n),
    );
    _pos += n;
    return out;
  }

  Float32List readF32List(int n) {
    final out = Float32List(n);
    final dst = out.buffer.asUint8List(0, n * 4);
    for (var i = 0; i < n * 4; i++) {
      dst[i] = _bytes[_pos + i];
    }
    _pos += n * 4;
    return out;
  }

  Int32List readI32List(int n) {
    final out = Int32List(n);
    final dst = out.buffer.asUint8List(0, n * 4);
    for (var i = 0; i < n * 4; i++) {
      dst[i] = _bytes[_pos + i];
    }
    _pos += n * 4;
    return out;
  }

  Uint8List readU8List(int n) => readBytes(n);
}

/// Metric ↔ u32 codec.
int metricToU32(Metric m) => m == Metric.l2 ? 0 : 1;
Metric metricFromU32(int v) {
  switch (v) {
    case 0:
      return Metric.l2;
    case 1:
      return Metric.innerProduct;
    default:
      throw FormatException('Unknown metric code $v');
  }
}

/// Writes the file-level header + `kind` + payload for [x] into [w].
/// Used both at top level and for embedding child indexes inside
/// wrappers like IDMap / RefineFlat / IVF*.
void writeChild(IoWriter w, Index x) {
  for (final b in _magicBytes) {
    w.writeU8(b);
  }
  w.writeU32(_version);
  if (x is IndexFlat) {
    w.writeU32(IndexKind.flat);
    x.writeTo(w);
  } else if (x is IndexIVFFlat) {
    w.writeU32(IndexKind.ivfFlat);
    x.writeTo(w);
  } else if (x is IndexPQ) {
    w.writeU32(IndexKind.pq);
    x.writeTo(w);
  } else if (x is IndexIVFPQ) {
    w.writeU32(IndexKind.ivfPq);
    x.writeTo(w);
  } else if (x is IndexHNSW) {
    w.writeU32(IndexKind.hnsw);
    x.writeTo(w);
  } else if (x is IndexIDMap) {
    w.writeU32(IndexKind.idMap);
    x.writeTo(w);
  } else if (x is IndexScalarQuantizer) {
    w.writeU32(IndexKind.scalarQuantizer);
    x.writeTo(w);
  } else if (x is IndexRefineFlat) {
    w.writeU32(IndexKind.refineFlat);
    x.writeTo(w);
  } else if (x is IndexLSH) {
    w.writeU32(IndexKind.lsh);
    x.writeTo(w);
  } else if (x is IndexPreTransform) {
    w.writeU32(IndexKind.preTransform);
    x.writeTo(w);
  } else if (x is IndexShards) {
    w.writeU32(IndexKind.shards);
    x.writeTo(w);
  } else if (x is IndexReplicas) {
    w.writeU32(IndexKind.replicas);
    x.writeTo(w);
  } else {
    throw ArgumentError('writeIndex: unsupported index type ${x.runtimeType}');
  }
}

/// Reads a `writeChild`-encoded blob from [r] and returns the
/// reconstructed index.
Index readChild(IoReader r) {
  for (var i = 0; i < _magicBytes.length; i++) {
    final b = r.readU8();
    if (b != _magicBytes[i]) {
      throw FormatException(
        'readIndex: bad magic byte at offset $i (got 0x${b.toRadixString(16)}, '
        'expected 0x${_magicBytes[i].toRadixString(16)})',
      );
    }
  }
  final version = r.readU32();
  if (version != _version) {
    throw FormatException(
      'readIndex: unsupported version $version (this build understands '
      'version $_version)',
    );
  }
  final kind = r.readU32();
  switch (kind) {
    case IndexKind.flat:
      return IndexFlat.readFrom(r);
    case IndexKind.ivfFlat:
      return IndexIVFFlat.readFrom(r);
    case IndexKind.pq:
      return IndexPQ.readFrom(r);
    case IndexKind.ivfPq:
      return IndexIVFPQ.readFrom(r);
    case IndexKind.hnsw:
      return IndexHNSW.readFrom(r);
    case IndexKind.idMap:
      return IndexIDMap.readFrom(r);
    case IndexKind.scalarQuantizer:
      return IndexScalarQuantizer.readFrom(r);
    case IndexKind.refineFlat:
      return IndexRefineFlat.readFrom(r);
    case IndexKind.lsh:
      return IndexLSH.readFrom(r);
    case IndexKind.preTransform:
      return IndexPreTransform.readFrom(r);
    case IndexKind.shards:
      return IndexShards.readFrom(r);
    case IndexKind.replicas:
      return IndexReplicas.readFrom(r);
    default:
      throw FormatException(
        'readIndex: unknown kind 0x${kind.toRadixString(16)}',
      );
  }
}

/// Serialize [x] to a self-contained byte blob.
Uint8List writeIndex(Index x) {
  final w = IoWriter();
  writeChild(w, x);
  return w.takeBytes();
}

/// Reconstruct an index from a blob written by [writeIndex].
Index readIndex(Uint8List bytes) => readChild(IoReader(bytes));

/// Save [x] to a file on disk.
Future<void> saveIndex(Index x, String path) async {
  final bytes = writeIndex(x);
  await File(path).writeAsBytes(bytes, flush: true);
}

/// Load an index blob written with [saveIndex].
Future<Index> loadIndex(String path) async {
  final bytes = await File(path).readAsBytes();
  return readIndex(bytes);
}

// -----------------------------------------------------------------------------
// Binary-index dispatchers.
//
// `IndexBinary` doesn't extend `Index` (different vector type), so it
// gets its own top-level codec pair. The wire format is identical
// (magic + version + kind + payload); only the `kind` values differ,
// so a caller loading a file always knows which family it belongs to.

/// Serialize an [IndexBinary] to a self-contained byte blob.
Uint8List writeBinaryIndex(IndexBinary x) {
  final w = IoWriter();
  for (final b in _magicBytes) {
    w.writeU8(b);
  }
  w.writeU32(_version);
  if (x is IndexBinaryFlat) {
    w.writeU32(IndexKind.binaryFlat);
    x.writeTo(w);
  } else if (x is IndexBinaryIVF) {
    w.writeU32(IndexKind.binaryIVF);
    x.writeTo(w);
  } else {
    throw ArgumentError('writeBinaryIndex: unsupported type ${x.runtimeType}');
  }
  return w.takeBytes();
}

/// Reconstruct an [IndexBinary] from a blob written by [writeBinaryIndex].
IndexBinary readBinaryIndex(Uint8List bytes) {
  final r = IoReader(bytes);
  for (var i = 0; i < _magicBytes.length; i++) {
    final b = r.readU8();
    if (b != _magicBytes[i]) {
      throw FormatException('readBinaryIndex: bad magic byte at offset $i');
    }
  }
  final version = r.readU32();
  if (version != _version) {
    throw FormatException('readBinaryIndex: unsupported version $version');
  }
  final kind = r.readU32();
  switch (kind) {
    case IndexKind.binaryFlat:
      return IndexBinaryFlat.readFrom(r);
    case IndexKind.binaryIVF:
      return IndexBinaryIVF.readFrom(r);
    default:
      throw FormatException(
        'readBinaryIndex: kind 0x${kind.toRadixString(16)} is not a binary index',
      );
  }
}
