/// Simple parameter checkpointing for [Module]s.
///
/// The checkpoint format is stable and self-describing:
///
/// ```text
///   magic:       "DPTC"                            (4 bytes)
///   version:     uint32 LE  (currently 1)
///   headerLen:   uint32 LE
///   header:      UTF-8 JSON, length = headerLen
///                {
///                  "version": 1,
///                  "params":  [{"shape": [<int>, ...]}, ...],
///                  "totalScalars": <int>
///                }
///   data:        totalScalars * 4 bytes of Float32 LE
/// ```
///
/// Parameters are serialized in the exact order returned by
/// `module.parameters()` — no names, no reflection. Loading is done
/// against an **already-constructed** module of the same architecture
/// (same order and shapes); each tensor is `assign`ed in place, so
/// optimizer state buffers keyed by parameter identity remain valid.
///
/// GPT-style tied weights (e.g. `GPT` with `tieWeights: true`) work
/// transparently: the tied tensor is exposed once in `parameters()`
/// so it is saved once and loaded once.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import '../tensor/dtype.dart';
import '../tensor/tensor.dart';
import 'module.dart';

class Checkpoint {
  /// Magic bytes at the start of every checkpoint: `D`, `P`, `T`, `C`.
  static const List<int> magic = [0x44, 0x50, 0x54, 0x43];

  /// Serialization format version. Bump when the on-disk layout
  /// changes in a backwards-incompatible way.
  static const int version = 1;

  /// Serialize a module's parameters to a fresh byte buffer.
  ///
  /// If [fp16] is true, every parameter's data is quantized to IEEE-754
  /// half precision on save (halving file size). The header records
  /// `"dtype": "F16"` per compressed parameter so [loadIntoBytes] can
  /// decode back to fp32 automatically. Old checkpoints without any
  /// `dtype` field continue to load as fp32.
  static Uint8List saveBytes(Module module, {bool fp16 = false}) {
    final params = module.parameters();
    final header = jsonEncode({
      'version': version,
      'params': [
        for (final p in params)
          {
            'shape': p.shape,
            if (fp16) 'dtype': 'F16',
          },
      ],
      'totalScalars': params.fold<int>(0, (a, p) => a + p.length),
    });
    final headerBytes = utf8.encode(header);

    final bytesPerScalar = fp16 ? 2 : 4;
    final totalScalars = params.fold<int>(0, (a, p) => a + p.length);
    final dataBytes = bytesPerScalar * totalScalars;
    final preamble = 4 + 4 + 4; // magic + version + headerLen
    final out = ByteData(preamble + headerBytes.length + dataBytes);

    // Preamble.
    var off = 0;
    for (int i = 0; i < 4; i++) {
      out.setUint8(off++, magic[i]);
    }
    out.setUint32(off, version, Endian.little);
    off += 4;
    out.setUint32(off, headerBytes.length, Endian.little);
    off += 4;

    // Header JSON.
    for (int i = 0; i < headerBytes.length; i++) {
      out.setUint8(off++, headerBytes[i]);
    }

    // Data blob. Downloading from GPU when needed happens via toList().
    for (final p in params) {
      final vals = p.toList();
      if (fp16) {
        for (int i = 0; i < vals.length; i++) {
          out.setUint16(off, fp32ToFp16Bits(vals[i]), Endian.little);
          off += 2;
        }
      } else {
        for (int i = 0; i < vals.length; i++) {
          out.setFloat32(off, vals[i], Endian.little);
          off += 4;
        }
      }
    }
    return out.buffer.asUint8List();
  }

  /// Write a checkpoint to disk. Creates parent directories as needed.
  static void saveFile(Module module, String path, {bool fp16 = false}) {
    final f = File(path);
    f.parent.createSync(recursive: true);
    f.writeAsBytesSync(saveBytes(module, fp16: fp16));
  }

  /// Load a checkpoint from a byte buffer into an existing module.
  /// The module must have the same number of parameters with matching
  /// shapes in the same order.
  static void loadIntoBytes(Module module, Uint8List bytes) {
    if (bytes.length < 12) {
      throw ArgumentError(
        'Checkpoint: buffer too small (${bytes.length} bytes)',
      );
    }
    for (int i = 0; i < 4; i++) {
      if (bytes[i] != magic[i]) {
        throw ArgumentError(
          'Checkpoint: bad magic bytes; not a DPTC checkpoint',
        );
      }
    }
    final view = ByteData.sublistView(bytes);
    final ver = view.getUint32(4, Endian.little);
    if (ver != version) {
      throw ArgumentError(
        'Checkpoint: unsupported version $ver (expected $version)',
      );
    }
    final headerLen = view.getUint32(8, Endian.little);
    final headerEnd = 12 + headerLen;
    if (bytes.length < headerEnd) {
      throw ArgumentError(
        'Checkpoint: truncated header (need $headerEnd bytes, have '
        '${bytes.length})',
      );
    }
    final header =
        jsonDecode(utf8.decode(bytes.sublist(12, headerEnd)))
            as Map<String, dynamic>;
    final paramSpecs = (header['params'] as List).cast<Map<String, dynamic>>();
    final params = module.parameters();
    if (paramSpecs.length != params.length) {
      throw ArgumentError(
        'Checkpoint: parameter count mismatch — checkpoint has '
        '${paramSpecs.length}, module has ${params.length}',
      );
    }

    // Verify all shapes match before writing anything (avoid partial
    // loads).
    final paramDtypes = <DType>[];
    for (int i = 0; i < params.length; i++) {
      final want = (paramSpecs[i]['shape'] as List).cast<int>();
      final have = params[i].shape;
      if (want.length != have.length ||
          !List<bool>.generate(
            want.length,
            (k) => want[k] == have[k],
          ).every((b) => b)) {
        throw ArgumentError(
          'Checkpoint: shape mismatch for parameter #$i — checkpoint '
          '$want vs module $have',
        );
      }
      final dtypeStr = paramSpecs[i]['dtype'] as String?;
      switch (dtypeStr) {
        case null:
        case 'F32':
          paramDtypes.add(DType.fp32);
        case 'F16':
          paramDtypes.add(DType.fp16);
        default:
          throw ArgumentError(
            'Checkpoint: unsupported dtype "$dtypeStr" for parameter '
            '#$i (expected F32 or F16)',
          );
      }
    }
    final expectedDataLen = () {
      var n = 0;
      for (int i = 0; i < params.length; i++) {
        n += params[i].length * paramDtypes[i].itemBytes;
      }
      return n;
    }();
    if (bytes.length - headerEnd != expectedDataLen) {
      throw ArgumentError(
        'Checkpoint: data blob length ${bytes.length - headerEnd} '
        'does not match expected $expectedDataLen bytes',
      );
    }

    // Copy in.
    var off = headerEnd;
    for (int i = 0; i < params.length; i++) {
      final p = params[i];
      final n = p.length;
      final f32 = Float32List(n);
      if (paramDtypes[i] == DType.fp16) {
        for (int j = 0; j < n; j++) {
          f32[j] = fp16BitsToFp32(view.getUint16(off, Endian.little));
          off += 2;
        }
      } else {
        for (int j = 0; j < n; j++) {
          f32[j] = view.getFloat32(off, Endian.little);
          off += 4;
        }
      }
      final source = Tensor.fromList(p.shape, f32, device: p.device);
      p.assign(source);
    }
  }

  /// Read a checkpoint from disk into an existing module.
  static void loadIntoFile(Module module, String path) {
    final bytes = File(path).readAsBytesSync();
    loadIntoBytes(module, bytes);
  }
}
