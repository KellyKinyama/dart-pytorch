/// Random-access safetensors reader — parse the header once, keep
/// the file handle open, and pull one tensor at a time via
/// `pread`-style seeks.
///
/// This is the plumbing that makes AirLLM-style layer-by-layer
/// weight streaming possible: only the tensors for the currently
/// resident transformer layer need to sit in RAM, everything else
/// stays on disk.
///
///   final reader = ShardedSafeTensorsReader.singleFile('model.safetensors');
///   // ... or, for a HF sharded checkpoint:
///   final reader = ShardedSafeTensorsReader.fromIndex(
///       'model.safetensors.index.json');
///
///   final w = reader.readTensor('model.layers.0.self_attn.q_proj.weight',
///                               keepFp16: true);
///   // ... use w, drop it, read the next one ...
///   reader.close();
///
/// Only single-file and HF sharded indexes are supported (same set
/// [SafeTensors.loadSharded] handles). Decoding of individual tensors
/// is delegated to [SafeTensors.decodeBlob] so all the dtype/fp16
/// behaviour matches the all-at-once loader.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import '../tensor/tensor.dart';
import 'safetensors.dart';

/// Random-access reader for one safetensors file.
class SafeTensorsReader {
  final String path;
  final RandomAccessFile _raf;
  final int _dataOffset;
  final Map<String, SafeTensorEntry> _entries;
  bool _closed = false;

  SafeTensorsReader._(this.path, this._raf, this._dataOffset, this._entries);

  /// Open [path] and parse just the header — the tensor bodies stay
  /// on disk until [readTensor] is called.
  static SafeTensorsReader open(String path) {
    final raf = File(path).openSync();
    final len = raf.lengthSync();
    if (len < 8) {
      raf.closeSync();
      throw ArgumentError('safetensors: "$path" too small ($len bytes)');
    }
    final header8 = Uint8List(8);
    raf.setPositionSync(0);
    raf.readIntoSync(header8);
    final headerLen = ByteData.sublistView(header8).getUint64(0, Endian.little);
    if (headerLen < 0 || 8 + headerLen > len) {
      raf.closeSync();
      throw ArgumentError('safetensors: "$path" invalid headerLen=$headerLen');
    }
    final headerBytes = Uint8List(8 + headerLen);
    raf.setPositionSync(0);
    raf.readIntoSync(headerBytes);
    final parsed = SafeTensors.parseHeaderOnly(headerBytes);
    final map = <String, SafeTensorEntry>{};
    for (final e in parsed.entries) {
      map[e.name] = e;
    }
    return SafeTensorsReader._(path, raf, parsed.dataOffset, map);
  }

  Iterable<String> get names => _entries.keys;
  bool contains(String name) => _entries.containsKey(name);
  SafeTensorEntry? entry(String name) => _entries[name];

  /// Read and decode a single tensor by name. Seeks straight to the
  /// entry's byte range and reads only that range.
  Tensor readTensor(String name, {bool keepFp16 = false}) {
    if (_closed) {
      throw StateError('safetensors reader for "$path" is closed');
    }
    final e = _entries[name];
    if (e == null) {
      throw ArgumentError('safetensors "$path": no such tensor "$name"');
    }
    final byteLen = e.dataEnd - e.dataStart;
    if (byteLen < 0) {
      throw ArgumentError(
        'safetensors "$path": entry "$name" has negative byte length',
      );
    }
    final blob = Uint8List(byteLen);
    _raf.setPositionSync(_dataOffset + e.dataStart);
    if (byteLen > 0) {
      var got = 0;
      while (got < byteLen) {
        final n = _raf.readIntoSync(blob, got, byteLen);
        if (n <= 0) {
          throw StateError(
            'safetensors "$path": short read for "$name" '
            '($got/$byteLen bytes)',
          );
        }
        got += n;
      }
    }
    return SafeTensors.decodeBlob(e, blob, keepFp16: keepFp16);
  }

  void close() {
    if (_closed) return;
    _closed = true;
    _raf.closeSync();
  }
}

/// A random-access reader that spans one or more safetensors shards.
///
/// Construct either from a single `.safetensors` file
/// ([ShardedSafeTensorsReader.singleFile]) or from a HF
/// `model.safetensors.index.json` file
/// ([ShardedSafeTensorsReader.fromIndex]). The API is identical in
/// both cases.
class ShardedSafeTensorsReader {
  final Map<String, SafeTensorsReader> _readers;
  final Map<String, String> _paramToShard;

  ShardedSafeTensorsReader._(this._readers, this._paramToShard);

  static ShardedSafeTensorsReader singleFile(String path) {
    final reader = SafeTensorsReader.open(path);
    final paramToShard = <String, String>{};
    for (final name in reader.names) {
      paramToShard[name] = path;
    }
    return ShardedSafeTensorsReader._({path: reader}, paramToShard);
  }

  /// Open a sharded checkpoint by its `model.safetensors.index.json`.
  ///
  /// Shard files are opened **lazily** — the first `readTensor` for
  /// a key opens (and keeps open) that key's shard. This lets you
  /// work with partial downloads: as long as every key you actually
  /// touch lives in a shard that's on disk, missing sibling shards
  /// are harmless.
  static ShardedSafeTensorsReader fromIndex(String indexPath) {
    final index = SafeTensors.readShardIndex(indexPath);
    final baseDir = File(indexPath).parent.path;
    final paramToShard = <String, String>{};
    for (final e in index.weightMap.entries) {
      paramToShard[e.key] = '$baseDir${Platform.pathSeparator}${e.value}';
    }
    return ShardedSafeTensorsReader._(<String, SafeTensorsReader>{}, paramToShard);
  }

  /// Auto-detect: pick `fromIndex` if the file ends in `.index.json`,
  /// otherwise assume a single `.safetensors` file.
  static ShardedSafeTensorsReader open(String path) {
    if (path.endsWith('.index.json')) return fromIndex(path);
    return singleFile(path);
  }

  Iterable<String> get names => _paramToShard.keys;
  bool contains(String name) => _paramToShard.containsKey(name);

  /// Returns true iff [name]'s shard is on disk (readable). Useful
  /// for probing partial downloads before touching a key.
  bool shardOnDisk(String name) {
    final shard = _paramToShard[name];
    if (shard == null) return false;
    return File(shard).existsSync();
  }

  SafeTensorsReader _openFor(String name) {
    final shard = _paramToShard[name];
    if (shard == null) {
      throw ArgumentError('no such tensor: "$name"');
    }
    return _readers.putIfAbsent(shard, () => SafeTensorsReader.open(shard));
  }

  Tensor readTensor(String name, {bool keepFp16 = false}) =>
      _openFor(name).readTensor(name, keepFp16: keepFp16);

  /// Read only the entry (dtype/shape/offsets) without decoding the
  /// tensor. Useful for size / layout planning.
  SafeTensorEntry? entry(String name) {
    if (!_paramToShard.containsKey(name)) return null;
    return _openFor(name).entry(name);
  }

  /// Approximate on-disk bytes of a tensor by name (from the header).
  int? tensorBytes(String name) {
    final e = entry(name);
    if (e == null) return null;
    return e.dataEnd - e.dataStart;
  }

  /// Pretty-print the shard layout as a header/manifest string.
  String describe() {
    final buf = StringBuffer();
    buf.writeln(
      'ShardedSafeTensorsReader (${_readers.length} shard(s), '
      '${_paramToShard.length} tensors)',
    );
    final grouped = <String, List<String>>{};
    for (final e in _paramToShard.entries) {
      grouped.putIfAbsent(e.value, () => <String>[]).add(e.key);
    }
    for (final e in grouped.entries) {
      buf.writeln('  ${e.key}  (${e.value.length} tensors)');
    }
    return buf.toString();
  }

  /// Basic JSON header dump for the first shard (used by CLI tools).
  static String dumpHeader(String path) {
    final r = SafeTensorsReader.open(path);
    try {
      final entries = r.names.toList()..sort();
      return const JsonEncoder.withIndent('  ').convert({
        'path': path,
        'tensors': [
          for (final n in entries)
            {
              'name': n,
              'dtype': r.entry(n)!.dtype,
              'shape': r.entry(n)!.shape,
              'bytes': r.entry(n)!.dataEnd - r.entry(n)!.dataStart,
            },
        ],
      });
    } finally {
      r.close();
    }
  }

  void close() {
    for (final r in _readers.values) {
      r.close();
    }
  }
}
