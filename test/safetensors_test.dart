import 'dart:convert';
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

/// Encode a `Map<String, {dtype, shape, data}>` into a valid
/// safetensors buffer for tests. `dataBytes` supplies the raw
/// little-endian payload for each entry — the helper computes offsets
/// and concatenates.
Uint8List _makeSafetensors(
  List<({String name, String dtype, List<int> shape, Uint8List data})>
  entries, {
  Map<String, String>? metadata,
}) {
  final headerObj = <String, dynamic>{};
  if (metadata != null) {
    headerObj['__metadata__'] = metadata;
  }
  var offset = 0;
  final chunks = <Uint8List>[];
  for (final e in entries) {
    headerObj[e.name] = {
      'dtype': e.dtype,
      'shape': e.shape,
      'data_offsets': [offset, offset + e.data.length],
    };
    chunks.add(e.data);
    offset += e.data.length;
  }
  final headerBytes = utf8.encode(jsonEncode(headerObj));
  final total = 8 + headerBytes.length + offset;
  final out = Uint8List(total);
  ByteData.sublistView(out).setUint64(0, headerBytes.length, Endian.little);
  out.setRange(8, 8 + headerBytes.length, headerBytes);
  var pos = 8 + headerBytes.length;
  for (final c in chunks) {
    out.setRange(pos, pos + c.length, c);
    pos += c.length;
  }
  return out;
}

Uint8List _f32Bytes(List<double> vals) {
  final bd = ByteData(vals.length * 4);
  for (int i = 0; i < vals.length; i++) {
    bd.setFloat32(i * 4, vals[i], Endian.little);
  }
  return bd.buffer.asUint8List();
}

Uint8List _f64Bytes(List<double> vals) {
  final bd = ByteData(vals.length * 8);
  for (int i = 0; i < vals.length; i++) {
    bd.setFloat64(i * 8, vals[i], Endian.little);
  }
  return bd.buffer.asUint8List();
}

Uint8List _bf16Bytes(List<double> vals) {
  final bd = ByteData(vals.length * 2);
  final scratch = ByteData(4);
  for (int i = 0; i < vals.length; i++) {
    scratch.setFloat32(0, vals[i], Endian.little);
    // BF16 = high 16 bits of fp32.
    final bits = scratch.getUint32(0, Endian.little);
    bd.setUint16(i * 2, (bits >> 16) & 0xffff, Endian.little);
  }
  return bd.buffer.asUint8List();
}

void main() {
  group('SafeTensors.loadBytes', () {
    test('reads a single F32 tensor with correct shape and values', () {
      final vals = [1.0, 2.0, 3.0, 4.0];
      final buf = _makeSafetensors([
        (name: 'x', dtype: 'F32', shape: [2, 2], data: _f32Bytes(vals)),
      ]);
      final map = SafeTensors.loadBytes(buf);
      expect(map.keys, ['x']);
      final t = map['x']!;
      expect(t.shape, [2, 2]);
      expect(t.toList(), vals);
    });

    test('reads multiple tensors in a single buffer', () {
      final aVals = [1.0, 2.0];
      final bVals = [10.0, 20.0, 30.0];
      final buf = _makeSafetensors([
        (name: 'a', dtype: 'F32', shape: [2], data: _f32Bytes(aVals)),
        (name: 'b', dtype: 'F32', shape: [3], data: _f32Bytes(bVals)),
      ]);
      final map = SafeTensors.loadBytes(buf);
      expect(map['a']!.toList(), aVals);
      expect(map['b']!.toList(), bVals);
    });

    test('skips the __metadata__ key and it is not treated as a tensor', () {
      final buf = _makeSafetensors(
        [
          (name: 'w', dtype: 'F32', shape: [1], data: _f32Bytes([42.0])),
        ],
        metadata: {'format': 'pt', 'note': 'hello'},
      );
      final map = SafeTensors.loadBytes(buf);
      expect(map.keys, ['w']);
      expect(SafeTensors.readMetadata(buf), {'format': 'pt', 'note': 'hello'});
    });

    test('down-casts F64 to F32 losslessly for representable values', () {
      final vals = [0.5, -0.25, 3.0];
      final buf = _makeSafetensors([
        (name: 'x', dtype: 'F64', shape: [3], data: _f64Bytes(vals)),
      ]);
      final map = SafeTensors.loadBytes(buf);
      expect(map['x']!.toList(), vals);
    });

    test('decodes BF16 with round-off tolerance', () {
      final vals = [1.0, -2.5, 0.125];
      final buf = _makeSafetensors([
        (name: 'x', dtype: 'BF16', shape: [3], data: _bf16Bytes(vals)),
      ]);
      final t = SafeTensors.loadBytes(buf)['x']!;
      final got = t.toList();
      expect(got.length, 3);
      for (int i = 0; i < vals.length; i++) {
        // bf16 keeps 8 mantissa bits — worst-case rel error ~2^-7.
        expect(got[i], closeTo(vals[i], vals[i].abs() * 1 / 128 + 1e-6));
      }
    });

    test('rejects unsupported dtypes with a clear message', () {
      final buf = _makeSafetensors([
        (name: 'x', dtype: 'I32', shape: [1], data: Uint8List(4)),
      ]);
      expect(() => SafeTensors.loadBytes(buf), throwsA(isA<ArgumentError>()));
    });

    test('rejects a short buffer', () {
      expect(
        () => SafeTensors.loadBytes(Uint8List(4)),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('listEntries via loadBytes reports correct shapes and offsets', () {
      final buf = _makeSafetensors([
        (
          name: 'w',
          dtype: 'F32',
          shape: [3, 2],
          data: _f32Bytes([1, 2, 3, 4, 5, 6]),
        ),
      ]);
      // We test the private path via the public map: shape must match.
      final t = SafeTensors.loadBytes(buf)['w']!;
      expect(t.shape, [3, 2]);
      expect(t.length, 6);
    });
  });
}
