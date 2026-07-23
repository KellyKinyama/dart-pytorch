/// Minimal reader for the [safetensors](https://github.com/huggingface/safetensors)
/// on-disk format.
///
/// Wire format:
///
/// ```text
///   headerLen: uint64 LE                          (8 bytes)
///   header:    UTF-8 JSON, length = headerLen
///              {
///                "__metadata__": { ... },        // optional
///                "<name>": {
///                  "dtype":        "F32" | "F16" | "BF16" | "F64" | ...,
///                  "shape":        [<int>, ...],
///                  "data_offsets": [<start>, <end>]
///                },
///                ...
///              }
///   data:      raw little-endian tensor blobs, indexed by
///              `data_offsets` relative to the start of the data
///              section (i.e. immediately after the header).
/// ```
///
/// Only floating-point tensors are supported (all tensors in this
/// package are float32). `F16`, `BF16`, and `F64` are up- or
/// down-converted to `float32` on load.
///
/// The reader returns a `Map<String, Tensor>` of CPU tensors — the
/// caller is free to `.to(Device.GPU)` them or hand them to a model
/// loader that will `assign` them into module parameters.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import '../tensor/tensor.dart';

/// A single tensor entry parsed from a safetensors header, before the
/// raw bytes have been decoded into a [Tensor].
class SafeTensorEntry {
  final String name;
  final String dtype;
  final List<int> shape;
  final int dataStart;
  final int dataEnd;

  const SafeTensorEntry({
    required this.name,
    required this.dtype,
    required this.shape,
    required this.dataStart,
    required this.dataEnd,
  });

  int get numElements => shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
}

/// Static helpers for reading safetensors files.
class SafeTensors {
  /// The special key HuggingFace uses to store arbitrary string
  /// metadata (e.g. `{"format": "pt"}`). Skipped when enumerating
  /// tensor entries.
  static const String metadataKey = '__metadata__';

  /// Read a safetensors file into a map of tensor name → CPU [Tensor].
  ///
  /// The file is fully loaded into memory. For the models we can
  /// realistically run in-process this is fine (GPT-2 small is ~500 MB
  /// of fp32); for larger models you probably want to stream.
  static Map<String, Tensor> loadFile(String path) {
    final bytes = File(path).readAsBytesSync();
    return loadBytes(bytes);
  }

  /// Parse safetensors from an in-memory byte buffer.
  static Map<String, Tensor> loadBytes(Uint8List bytes) {
    final entries = _parseHeader(bytes);
    final dataOffset = _dataOffset(bytes);
    final result = <String, Tensor>{};
    for (final e in entries) {
      result[e.name] = _readTensor(bytes, dataOffset, e);
    }
    return result;
  }

  /// Just enumerate the header without materializing tensors. Useful
  /// for inspection and for building weight-map summaries.
  static List<SafeTensorEntry> listEntries(String path) {
    final bytes = File(path).readAsBytesSync();
    return _parseHeader(bytes);
  }

  /// Read the free-form `__metadata__` block, if any.
  static Map<String, String> readMetadata(Uint8List bytes) {
    final headerJson = _readHeaderJson(bytes);
    final meta = headerJson[metadataKey];
    if (meta is! Map) return const {};
    return meta.map((k, v) => MapEntry(k.toString(), v.toString()));
  }

  // ---------------- internals ----------------

  static int _dataOffset(Uint8List bytes) {
    if (bytes.length < 8) {
      throw ArgumentError(
        'safetensors: buffer too small (${bytes.length} bytes)',
      );
    }
    final headerLen = ByteData.sublistView(
      bytes,
      0,
      8,
    ).getUint64(0, Endian.little);
    if (headerLen < 0 || 8 + headerLen > bytes.length) {
      throw ArgumentError(
        'safetensors: invalid headerLen=$headerLen for buffer '
        '${bytes.length}',
      );
    }
    return 8 + headerLen;
  }

  static Map<String, dynamic> _readHeaderJson(Uint8List bytes) {
    final dataOff = _dataOffset(bytes);
    final headerBytes = bytes.sublist(8, dataOff);
    final decoded = jsonDecode(utf8.decode(headerBytes));
    if (decoded is! Map) {
      throw ArgumentError(
        'safetensors: header is not a JSON object (got ${decoded.runtimeType})',
      );
    }
    return decoded.cast<String, dynamic>();
  }

  static List<SafeTensorEntry> _parseHeader(Uint8List bytes) {
    final header = _readHeaderJson(bytes);
    final entries = <SafeTensorEntry>[];
    for (final entry in header.entries) {
      if (entry.key == metadataKey) continue;
      final v = entry.value;
      if (v is! Map) {
        throw ArgumentError(
          'safetensors: entry "${entry.key}" is not an object',
        );
      }
      final dtype =
          v['dtype']?.toString() ??
          (throw ArgumentError(
            'safetensors: entry "${entry.key}" missing "dtype"',
          ));
      final rawShape = v['shape'];
      if (rawShape is! List) {
        throw ArgumentError(
          'safetensors: entry "${entry.key}" missing/invalid "shape"',
        );
      }
      final shape = rawShape.map((e) => (e as num).toInt()).toList();
      final rawOffsets = v['data_offsets'];
      if (rawOffsets is! List || rawOffsets.length != 2) {
        throw ArgumentError(
          'safetensors: entry "${entry.key}" invalid "data_offsets"',
        );
      }
      final start = (rawOffsets[0] as num).toInt();
      final end = (rawOffsets[1] as num).toInt();
      entries.add(
        SafeTensorEntry(
          name: entry.key,
          dtype: dtype,
          shape: shape,
          dataStart: start,
          dataEnd: end,
        ),
      );
    }
    return entries;
  }

  static Tensor _readTensor(
    Uint8List bytes,
    int dataOffset,
    SafeTensorEntry e,
  ) {
    final byteLen = e.dataEnd - e.dataStart;
    if (byteLen < 0 || dataOffset + e.dataEnd > bytes.length) {
      throw ArgumentError(
        'safetensors: entry "${e.name}" offsets out of range',
      );
    }
    final view = ByteData.sublistView(
      bytes,
      dataOffset + e.dataStart,
      dataOffset + e.dataEnd,
    );
    final n = e.numElements;
    final data = Float32List(n);
    switch (e.dtype) {
      case 'F32':
        if (byteLen != n * 4) {
          throw ArgumentError(
            'safetensors: "${e.name}" F32 expected ${n * 4} bytes, got '
            '$byteLen',
          );
        }
        for (int i = 0; i < n; i++) {
          data[i] = view.getFloat32(i * 4, Endian.little);
        }
        break;
      case 'F64':
        if (byteLen != n * 8) {
          throw ArgumentError(
            'safetensors: "${e.name}" F64 expected ${n * 8} bytes, got '
            '$byteLen',
          );
        }
        for (int i = 0; i < n; i++) {
          data[i] = view.getFloat64(i * 8, Endian.little);
        }
        break;
      case 'F16':
        if (byteLen != n * 2) {
          throw ArgumentError(
            'safetensors: "${e.name}" F16 expected ${n * 2} bytes, got '
            '$byteLen',
          );
        }
        for (int i = 0; i < n; i++) {
          data[i] = _decodeF16(view.getUint16(i * 2, Endian.little));
        }
        break;
      case 'BF16':
        if (byteLen != n * 2) {
          throw ArgumentError(
            'safetensors: "${e.name}" BF16 expected ${n * 2} bytes, got '
            '$byteLen',
          );
        }
        for (int i = 0; i < n; i++) {
          data[i] = _decodeBF16(view.getUint16(i * 2, Endian.little));
        }
        break;
      case 'U8':
      case 'BOOL':
      case 'I8':
        // Byte-wide integer / boolean tensors. HF PyTorch stores
        // causal-mask buffers this way (`attention.bias`). Decode
        // to float so we can round-trip the entry, but note our
        // loaders never consume these — they build masks on the
        // fly and leave the entry in `unusedKeys`.
        if (byteLen != n) {
          throw ArgumentError(
            'safetensors: "${e.name}" ${e.dtype} expected $n bytes, got '
            '$byteLen',
          );
        }
        for (int i = 0; i < n; i++) {
          final b = view.getUint8(i);
          data[i] = e.dtype == 'I8' && b >= 0x80
              ? (b - 0x100).toDouble()
              : b.toDouble();
        }
        break;
      default:
        throw ArgumentError(
          'safetensors: unsupported dtype "${e.dtype}" for entry '
          '"${e.name}" (supported: F32, F64, F16, BF16, U8, BOOL, I8)',
        );
    }
    return Tensor.fromList(
      e.shape.isEmpty ? const [1] : e.shape,
      data,
      device: Device.CPU,
    );
  }

  /// IEEE-754 half-precision → single-precision decode.
  static double _decodeF16(int bits) {
    final sign = (bits >> 15) & 0x1;
    final exp = (bits >> 10) & 0x1f;
    final mant = bits & 0x3ff;
    int sign32 = sign << 31;
    int exp32;
    int mant32;
    if (exp == 0) {
      if (mant == 0) {
        // ±0
        return _fromU32(sign32);
      }
      // Subnormal — normalize.
      var m = mant;
      var e = 1;
      while ((m & 0x400) == 0) {
        m <<= 1;
        e -= 1;
      }
      m &= 0x3ff;
      exp32 = (127 - 15 + e) << 23;
      mant32 = m << 13;
    } else if (exp == 0x1f) {
      // Inf / NaN.
      exp32 = 0xff << 23;
      mant32 = mant << 13;
    } else {
      exp32 = (exp - 15 + 127) << 23;
      mant32 = mant << 13;
    }
    return _fromU32(sign32 | exp32 | mant32);
  }

  /// bfloat16 → single-precision decode (bf16 is just fp32 with the
  /// low 16 bits zeroed, so we simply place the 16 bits in the high
  /// half).
  static double _decodeBF16(int bits) {
    return _fromU32(bits << 16);
  }

  static final ByteData _f32Scratch = ByteData(4);
  static double _fromU32(int bits) {
    _f32Scratch.setUint32(0, bits & 0xffffffff, Endian.little);
    return _f32Scratch.getFloat32(0, Endian.little);
  }
}
