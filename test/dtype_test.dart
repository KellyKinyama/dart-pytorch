import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:dart_pytorch/core/tensor/dtype.dart';
import 'package:test/test.dart';

void main() {
  group('fp16 codec (dtype.dart)', () {
    test('decodes canonical bit patterns', () {
      // 0x0000 → +0
      expect(fp16BitsToFp32(0x0000), 0.0);
      // 0x8000 → -0
      expect(fp16BitsToFp32(0x8000).isNegative, isTrue);
      expect(fp16BitsToFp32(0x8000), 0.0);
      // 0x3c00 → 1
      expect(fp16BitsToFp32(0x3c00), 1.0);
      // 0xbc00 → -1
      expect(fp16BitsToFp32(0xbc00), -1.0);
      // 0x7c00 → +Inf
      expect(fp16BitsToFp32(0x7c00).isInfinite, isTrue);
      // 0xfc00 → -Inf
      expect(fp16BitsToFp32(0xfc00).isInfinite, isTrue);
      expect(fp16BitsToFp32(0xfc00).isNegative, isTrue);
      // 0x7e00 → NaN
      expect(fp16BitsToFp32(0x7e00).isNaN, isTrue);
    });

    test('encodes canonical values', () {
      expect(fp32ToFp16Bits(0.0), 0x0000);
      expect(fp32ToFp16Bits(1.0), 0x3c00);
      expect(fp32ToFp16Bits(-1.0), 0xbc00);
      expect(fp32ToFp16Bits(double.infinity), 0x7c00);
      expect(fp32ToFp16Bits(double.negativeInfinity), 0xfc00);
      // Overflow saturates to Inf.
      expect(fp32ToFp16Bits(1e30), 0x7c00);
      // Underflow → 0.
      expect(fp32ToFp16Bits(1e-30), 0x0000);
    });

    test('round-trips values within fp16 dynamic range', () {
      final samples = <double>[
        0.0,
        1.0,
        -1.0,
        0.5,
        -0.25,
        3.14,
        -0.001,
        100.0,
        -1024.5,
      ];
      for (final v in samples) {
        final bits = fp32ToFp16Bits(v);
        final back = fp16BitsToFp32(bits);
        // fp16 has ~3 decimal digits of precision.
        expect(back, closeTo(v, v.abs() * 1e-3 + 1e-5));
      }
    });

    test('bulk decode/encode round-trip', () {
      final vals = Float32List.fromList([0.0, 1.0, -2.5, 3.75, 0.125]);
      final bits = encodeFp16Bulk(vals);
      final back = decodeFp16Bulk(bits);
      for (int i = 0; i < vals.length; i++) {
        expect(back[i], closeTo(vals[i], vals[i].abs() * 1e-3 + 1e-5));
      }
    });
  });

  group('Tensor fp16 storage', () {
    test('fromFp16Bits builds an fp16 CPU tensor', () {
      final bits = encodeFp16Bulk(Float32List.fromList([1.0, 2.0, 3.0, 4.0]));
      final t = Tensor.fromFp16Bits([2, 2], bits);
      expect(t.dtype, DType.fp16);
      expect(t.device, Device.CPU);
      expect(t.shape, [2, 2]);
      expect(t.length, 4);
      final host = t.toList();
      expect(host[0], closeTo(1.0, 1e-3));
      expect(host[3], closeTo(4.0, 1e-3));
    });

    test('fromFp16Bits rejects shape/length mismatch', () {
      final bits = Uint16List(3);
      expect(
        () => Tensor.fromFp16Bits([2, 2], bits),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('toFp32 → toFp16 preserves shape and values', () {
      final t = Tensor.fromList([2, 2], [1.0, 2.0, -0.5, 0.25]);
      final half = t.toFp16();
      expect(half.dtype, DType.fp16);
      expect(half.shape, [2, 2]);
      final round = half.toFp32();
      expect(round.dtype, DType.fp32);
      final r = round.toList();
      expect(r[0], closeTo(1.0, 1e-3));
      expect(r[1], closeTo(2.0, 1e-3));
      expect(r[2], closeTo(-0.5, 1e-3));
      expect(r[3], closeTo(0.25, 1e-3));
    });

    test('assign into an fp16 tensor throws', () {
      final t = Tensor.fromFp16Bits([
        2,
      ], encodeFp16Bulk(Float32List.fromList([1.0, 2.0])));
      final src = Tensor.fromList([2], [3.0, 4.0]);
      expect(() => t.assign(src), throwsA(isA<StateError>()));
    });

    test('assign from an fp16 tensor throws', () {
      final dst = Tensor.fromList([2], [1.0, 2.0]);
      final src = Tensor.fromFp16Bits([
        2,
      ], encodeFp16Bulk(Float32List.fromList([3.0, 4.0])));
      expect(() => dst.assign(src), throwsA(isA<StateError>()));
    });

    test('clone of fp16 stays fp16 and independent', () {
      final t = Tensor.fromFp16Bits([
        3,
      ], encodeFp16Bulk(Float32List.fromList([1.0, 2.0, 3.0])));
      final c = t.clone();
      expect(c.dtype, DType.fp16);
      expect(c.toList()[0], closeTo(1.0, 1e-3));
    });
  });

  group('fp16 weight participation in ops', () {
    test('matmul with fp16 weight matches fp32 baseline', () {
      // x: [1, 4] fp32 activation; W: [4, 3] weight — compare fp32 vs fp16.
      final x = Tensor.fromList([1, 4], [1.0, 2.0, 3.0, 4.0]);
      final wVals = <double>[
        0.5,
        -0.25,
        0.125,
        1.0,
        0.0,
        -0.5,
        0.25,
        0.75,
        -1.0,
        -0.5,
        0.5,
        0.25,
      ];
      final wFp32 = Tensor.fromList([4, 3], wVals);
      final wFp16 = wFp32.toFp16();
      expect(wFp16.dtype, DType.fp16);

      final yFp32 = x.matmul(wFp32).toList();
      final yFp16 = x.matmul(wFp16).toList();
      expect(yFp16.length, yFp32.length);
      for (int i = 0; i < yFp32.length; i++) {
        expect(yFp16[i], closeTo(yFp32[i], yFp32[i].abs() * 1e-3 + 1e-4));
      }
    });

    test('elementwise add with fp16 rhs matches fp32', () {
      final a = Tensor.fromList([3], [1.0, 2.0, 3.0]);
      final b = Tensor.fromList([3], [0.5, -0.25, 0.125]);
      final bFp16 = b.toFp16();
      final r1 = (a + b).toList();
      final r2 = (a + bFp16).toList();
      for (int i = 0; i < r1.length; i++) {
        expect(r2[i], closeTo(r1[i], 1e-3));
      }
    });
  });

  group('safetensors keepFp16', () {
    test('F16 entry stays as DType.fp16 when keepFp16=true', () {
      // Build a minimal safetensors blob in memory: one F16 tensor of
      // shape [4] with values [1, 2, -1, 0.5].
      final vals = Float32List.fromList([1.0, 2.0, -1.0, 0.5]);
      final bits = encodeFp16Bulk(vals);
      final payload = Uint8List(bits.length * 2);
      final view = ByteData.sublistView(payload);
      for (int i = 0; i < bits.length; i++) {
        view.setUint16(i * 2, bits[i], Endian.little);
      }

      final headerJson = jsonEncode({
        'w': {
          'dtype': 'F16',
          'shape': [4],
          'data_offsets': [0, payload.length],
        },
      });
      final headerBytes = utf8.encode(headerJson);
      final headerLen = headerBytes.length;
      final blob = BytesBuilder();
      final lenBuf = ByteData(8)..setUint64(0, headerLen, Endian.little);
      blob.add(lenBuf.buffer.asUint8List());
      blob.add(headerBytes);
      blob.add(payload);
      final bytes = blob.toBytes();

      // Default: promotes to fp32.
      final promoted = SafeTensors.loadBytes(bytes);
      expect(promoted['w']!.dtype, DType.fp32);
      expect(promoted['w']!.toList()[0], closeTo(1.0, 1e-3));

      // keepFp16 preserves storage.
      final kept = SafeTensors.loadBytes(bytes, keepFp16: true);
      expect(kept['w']!.dtype, DType.fp16);
      expect(kept['w']!.length, 4);
      expect(kept['w']!.toList()[1], closeTo(2.0, 1e-3));
    });
  });

  group('adoptCpuStorageFrom', () {
    test('swaps a CPU parameter to fp16 backing in place', () {
      // Start with a fp32 CPU "parameter" — matches what a Module
      // constructor allocates for its weights.
      final param = Tensor.fromList([2, 2], [1.0, 2.0, 3.0, 4.0]);
      expect(param.dtype, DType.fp32);

      final src = Tensor.fromFp16Bits([
        2,
        2,
      ], encodeFp16Bulk(Float32List.fromList([10.0, 20.0, 30.0, 40.0])));
      param.adoptCpuStorageFrom(src);

      expect(param.dtype, DType.fp16);
      expect(param.shape, [2, 2]);
      expect(param.length, 4);
      expect(param.toList()[0], closeTo(10.0, 1e-2));
      expect(param.toList()[3], closeTo(40.0, 1e-2));

      // Source is now a zombie — accessing storage throws.
      expect(() => src.toList(), throwsA(anything));
    });

    test('matmul with an adopted-fp16 param matches fp32 baseline', () {
      final wRef = Tensor.fromList([2, 2], [0.5, -0.25, 1.0, 0.75]);
      final wParam = Tensor.fromList([2, 2], [0.0, 0.0, 0.0, 0.0]);
      wParam.adoptCpuStorageFrom(wRef.toFp16());
      final x = Tensor.fromList([1, 2], [3.0, 4.0]);
      final y1 = x.matmul(wRef).toList();
      final y2 = x.matmul(wParam).toList();
      for (int i = 0; i < y1.length; i++) {
        expect(y2[i], closeTo(y1[i], y1[i].abs() * 1e-3 + 1e-4));
      }
    });

    test('rejects length mismatch', () {
      final dst = Tensor.fromList([4], [1.0, 2.0, 3.0, 4.0]);
      final src = Tensor.fromList([3], [1.0, 2.0, 3.0]);
      expect(() => dst.adoptCpuStorageFrom(src), throwsA(isA<ArgumentError>()));
    });
  });

  group('fp16-preserving ops', () {
    test('transpose keeps fp16 storage', () {
      final w = Tensor.fromList([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
      final wFp16 = w.toFp16();
      final wt = wFp16.transpose();
      expect(wt.dtype, DType.fp16);
      expect(wt.shape, [3, 2]);
      // Row-major [2,3] = [[1,2,3],[4,5,6]] transposed = [[1,4],[2,5],[3,6]].
      final host = wt.toList();
      expect(host[0], closeTo(1.0, 1e-3));
      expect(host[1], closeTo(4.0, 1e-3));
      expect(host[2], closeTo(2.0, 1e-3));
      expect(host[3], closeTo(5.0, 1e-3));
      expect(host[4], closeTo(3.0, 1e-3));
      expect(host[5], closeTo(6.0, 1e-3));
    });

    test('sliceRows keeps fp16 storage', () {
      // Simulate a `[H*hd, D]` = `[4, 3]` weight and slice per-head.
      final vals = <double>[
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
      ];
      final w = Tensor.fromList([4, 3], vals).toFp16();
      final head0 = w.sliceRows(0, 2);
      final head1 = w.sliceRows(2, 4);
      expect(head0.dtype, DType.fp16);
      expect(head1.dtype, DType.fp16);
      expect(head0.shape, [2, 3]);
      expect(head1.shape, [2, 3]);
      final h0 = head0.toList();
      final h1 = head1.toList();
      expect(h0[0], closeTo(1.0, 1e-3));
      expect(h0[5], closeTo(6.0, 1e-3));
      expect(h1[0], closeTo(7.0, 1e-3));
      expect(h1[5], closeTo(12.0, 1e-3));
    });

    test('sliceRows fp32 fallback', () {
      final w = Tensor.fromList([3, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
      final s = w.sliceRows(1, 3);
      expect(s.dtype, DType.fp32);
      expect(s.shape, [2, 2]);
      expect(s.toList(), [3.0, 4.0, 5.0, 6.0]);
    });

    test('sliceRows out-of-range throws', () {
      final w = Tensor.fromList([2, 2], [1.0, 2.0, 3.0, 4.0]);
      expect(() => w.sliceRows(0, 3), throwsA(isA<ArgumentError>()));
      expect(() => w.sliceRows(-1, 1), throwsA(isA<ArgumentError>()));
    });

    test('Linear-style x @ W.T + b with fp16 weight matches fp32', () {
      final w = Tensor.fromList(
        [3, 4],
        [0.5, -0.25, 0.125, 0.75, 1.0, 0.0, -0.5, 0.25, 0.25, 0.75, -1.0, 0.5],
      );
      final b = Tensor.fromList([1, 3], [0.1, -0.1, 0.2]);
      final x = Tensor.fromList([1, 4], [1.0, 2.0, 3.0, 4.0]);
      final yFp32 = (x.matmul(w.transpose()) + b).toList();
      final wHalf = w.toFp16();
      final bHalf = b.toFp16();
      final yFp16 = (x.matmul(wHalf.transpose()) + bHalf).toList();
      for (int i = 0; i < yFp32.length; i++) {
        expect(yFp16[i], closeTo(yFp32[i], yFp32[i].abs() * 1e-3 + 1e-3));
      }
    });
  });

  group('Tensor.residentBytes', () {
    test('fp32 tensor reports length * 4', () {
      final t = Tensor.fromList([2, 3], [1, 2, 3, 4, 5, 6]);
      expect(t.residentBytes, 24);
    });

    test('fp16 tensor reports length * 2', () {
      final t = Tensor.fromFp16Bits([2, 3], Uint16List(6));
      expect(t.residentBytes, 12);
    });

    test('fp16 halves resident bytes vs fp32 counterpart', () {
      final t = Tensor.fromList([16], List<double>.filled(16, 1.0));
      final tHalf = t.toFp16();
      expect(tHalf.residentBytes, t.residentBytes ~/ 2);
    });
  });

  group('SafeTensors sharded loader', () {
    late Directory tmpDir;

    setUp(() {
      tmpDir = Directory.systemTemp.createTempSync('dart_pytorch_shard_');
    });

    tearDown(() {
      if (tmpDir.existsSync()) tmpDir.deleteSync(recursive: true);
    });

    // Build a minimal single-tensor F16 safetensors blob for a shard.
    Uint8List buildF16Blob(String name, List<int> shape, Float32List vals) {
      final bits = encodeFp16Bulk(vals);
      final payload = Uint8List(bits.length * 2);
      final view = ByteData.sublistView(payload);
      for (int i = 0; i < bits.length; i++) {
        view.setUint16(i * 2, bits[i], Endian.little);
      }
      final headerJson = jsonEncode({
        name: {
          'dtype': 'F16',
          'shape': shape,
          'data_offsets': [0, payload.length],
        },
      });
      final headerBytes = utf8.encode(headerJson);
      final blob = BytesBuilder();
      final lenBuf = ByteData(8)
        ..setUint64(0, headerBytes.length, Endian.little);
      blob.add(lenBuf.buffer.asUint8List());
      blob.add(headerBytes);
      blob.add(payload);
      return blob.toBytes();
    }

    test('readShardIndex parses HF index.json', () {
      final idx = File('${tmpDir.path}/model.safetensors.index.json');
      idx.writeAsStringSync(
        jsonEncode({
          'metadata': {'total_size': 12345},
          'weight_map': {
            'model.embed.weight': 'model-00001-of-00002.safetensors',
            'model.layer.0.w': 'model-00001-of-00002.safetensors',
            'model.layer.1.w': 'model-00002-of-00002.safetensors',
          },
        }),
      );
      final parsed = SafeTensors.readShardIndex(idx.path);
      expect(parsed.weightMap.length, 3);
      expect(
        parsed.weightMap['model.layer.1.w'],
        'model-00002-of-00002.safetensors',
      );
      expect(parsed.metadata['total_size'], 12345);
      expect(parsed.shardFiles, [
        'model-00001-of-00002.safetensors',
        'model-00002-of-00002.safetensors',
      ]);
    });

    test('loadSharded merges tensors across shards, keepFp16 preserved', () {
      final s1 = buildF16Blob('a', [2], Float32List.fromList([1.0, 2.0]));
      final s2 = buildF16Blob('b', [3], Float32List.fromList([3.0, 4.0, 5.0]));
      File(
        '${tmpDir.path}/model-00001-of-00002.safetensors',
      ).writeAsBytesSync(s1);
      File(
        '${tmpDir.path}/model-00002-of-00002.safetensors',
      ).writeAsBytesSync(s2);

      final idxPath = '${tmpDir.path}/model.safetensors.index.json';
      File(idxPath).writeAsStringSync(
        jsonEncode({
          'metadata': {},
          'weight_map': {
            'a': 'model-00001-of-00002.safetensors',
            'b': 'model-00002-of-00002.safetensors',
          },
        }),
      );

      final merged = SafeTensors.loadSharded(idxPath, keepFp16: true);
      expect(merged.keys.toSet(), {'a', 'b'});
      expect(merged['a']!.dtype, DType.fp16);
      expect(merged['b']!.dtype, DType.fp16);
      expect(merged['a']!.toList(), [
        closeTo(1.0, 1e-3),
        closeTo(2.0, 1e-3),
      ]);
      expect(merged['b']!.toList(), [
        closeTo(3.0, 1e-3),
        closeTo(4.0, 1e-3),
        closeTo(5.0, 1e-3),
      ]);
    });

    test('streamSharded invokes callback once per shard', () {
      final s1 = buildF16Blob('a', [1], Float32List.fromList([7.0]));
      final s2 = buildF16Blob('b', [1], Float32List.fromList([8.0]));
      File('${tmpDir.path}/s1.safetensors').writeAsBytesSync(s1);
      File('${tmpDir.path}/s2.safetensors').writeAsBytesSync(s2);
      final idxPath = '${tmpDir.path}/model.safetensors.index.json';
      File(idxPath).writeAsStringSync(
        jsonEncode({
          'weight_map': {'a': 's1.safetensors', 'b': 's2.safetensors'},
        }),
      );

      final shardCalls = <Set<String>>[];
      SafeTensors.streamSharded(idxPath, (m) {
        shardCalls.add(m.keys.toSet());
      }, keepFp16: true);
      expect(shardCalls, [
        {'a'},
        {'b'},
      ]);
    });
  });
}
