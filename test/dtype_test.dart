import 'dart:convert';
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

      final src = Tensor.fromFp16Bits(
        [2, 2],
        encodeFp16Bulk(Float32List.fromList([10.0, 20.0, 30.0, 40.0])),
      );
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
      expect(
        () => dst.adoptCpuStorageFrom(src),
        throwsA(isA<ArgumentError>()),
      );
    });
  });
}
