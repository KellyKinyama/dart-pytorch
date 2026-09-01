import 'dart:convert';
import 'dart:typed_data';

import 'package:dart_pytorch/core/nn/nnue_proto.dart';
import 'package:test/test.dart';

Uint8List _buildHeader({
  required int version,
  required int hash,
  required String description,
}) {
  final descBytes = utf8.encode(description);
  final out = BytesBuilder();
  final u32 = ByteData(4);
  u32.setUint32(0, version, Endian.little);
  out.add(u32.buffer.asUint8List());
  u32.setUint32(0, hash, Endian.little);
  out.add(u32.buffer.asUint8List());
  u32.setUint32(0, descBytes.length, Endian.little);
  out.add(u32.buffer.asUint8List());
  out.add(descBytes);
  return out.takeBytes();
}

void main() {
  group('NnueReader.parseHeader', () {
    test('decodes a valid HalfKAv2_hm header', () {
      final bytes = _buildHeader(
        version: 0x7AF32F20,
        hash: 0xDEADBEEF,
        description: 'Features=HalfKAv2_hm(Friend); Network=example',
      );
      final h = NnueReader.parseHeader(bytes);
      expect(h.version, 0x7AF32F20);
      expect(h.architecture, NnueArchitecture.halfKAv2Hm);
      expect(h.hashValue, 0xDEADBEEF);
      expect(h.description, contains('HalfKAv2_hm'));
      expect(h.byteLength, bytes.length);
    });

    test('rejects unknown version tag', () {
      final bytes = _buildHeader(
        version: 0xDEADC0DE,
        hash: 0,
        description: 'unknown',
      );
      expect(
        () => NnueReader.parseHeader(bytes),
        throwsA(
          isA<FormatException>().having(
            (e) => e.message,
            'message',
            contains('unrecognised version'),
          ),
        ),
      );
    });

    test('rejects short buffer', () {
      final bytes = Uint8List(8); // less than 12
      expect(
        () => NnueReader.parseHeader(bytes),
        throwsA(isA<FormatException>()),
      );
    });

    test('rejects implausible description length', () {
      final u32 = ByteData(12);
      u32.setUint32(0, 0x7AF32F20, Endian.little);
      u32.setUint32(4, 0, Endian.little);
      u32.setUint32(8, 0xFFFFFFF, Endian.little); // ~256 MB desc
      expect(
        () => NnueReader.parseHeader(u32.buffer.asUint8List()),
        throwsA(isA<FormatException>()),
      );
    });
  });
}
