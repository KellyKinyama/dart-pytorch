import 'dart:convert';
import 'dart:typed_data';

import 'package:dart_pytorch/core/nn/needle/needle_tokenizer.dart';
import 'package:test/test.dart';

/// Build a synthetic RAW tokenizer blob.
Uint8List _blob(
  List<(String, int, double)> pieces, {
  int pad = 0,
  int eos = 1,
  int bos = 2,
  int unk = 3,
  bool addDummy = false,
  bool byteFallback = true,
}) {
  final bb = BytesBuilder();
  final hdr = ByteData(24);
  hdr.setUint32(0, pieces.length, Endian.little);
  hdr.setUint32(4, pad, Endian.little);
  hdr.setUint32(8, eos, Endian.little);
  hdr.setUint32(12, bos, Endian.little);
  hdr.setUint32(16, unk, Endian.little);
  hdr.setUint8(20, addDummy ? 1 : 0);
  hdr.setUint8(21, byteFallback ? 1 : 0);
  bb.add(hdr.buffer.asUint8List());
  for (final (surface, type, score) in pieces) {
    final b = utf8.encode(surface);
    final rec = ByteData(7);
    rec.setFloat32(0, score, Endian.little);
    rec.setUint8(4, type);
    rec.setUint16(5, b.length, Endian.little);
    bb.add(rec.buffer.asUint8List());
    bb.add(b);
  }
  return bb.toBytes();
}

const _m = kMetaspace;

NeedleTokenizer _tok({bool addDummy = false}) => NeedleTokenizer.fromBlob(_blob([
      ('<pad>', tkControl, 0),
      ('</s>', tkControl, 0),
      ('<s>', tkControl, 0),
      ('<unk>', tkUnknown, 0),
      ('<|im_start|>', tkUserDefined, 0), // id 4
      ('a', tkNormal, -1), // id 5
      ('b', tkNormal, -1), // id 6
      ('ab', tkNormal, 5), // id 7 (high score → merge)
      ('<0x63>', tkByte, 0), // id 8 (byte 'c')
      (_m, tkNormal, -2), // id 9 (metaspace)
    ], addDummy: addDummy));

void main() {
  group('NeedleTokenizer', () {
    test('BPE merges by highest score', () {
      expect(_tok().encode('ab'), [7]);
      expect(_tok().encode('a'), [5]);
    });

    test('byte fallback for unknown characters', () {
      expect(_tok().encode('c'), [8]); // via <0x63>
    });

    test('markers are recognised and split the stream', () {
      expect(_tok().encode('a<|im_start|>b'), [5, 4, 6]);
    });

    test('spaces become metaspace', () {
      expect(_tok().encode('a b'), [5, 9, 6]);
    });

    test('round-trips through decode', () {
      final t = _tok();
      for (final s in ['ab', 'a b', 'a<|im_start|>b']) {
        expect(t.decode(t.encode(s)), s);
      }
    });

    test('byte piece decodes to its raw byte', () {
      expect(_tok().decode([8]), 'c');
    });

    test('add_dummy_prefix prepends a metaspace and is stripped on decode', () {
      final t = _tok(addDummy: true);
      // 'a' -> ▁ + a ; ▁a not a piece → [▁, a] = [9, 5]
      expect(t.encode('a'), [9, 5]);
      expect(t.decode(t.encode('a')), 'a');
    });
  });

  group('buildToolPrompt', () {
    test('starts with BOS and contains the im_start marker', () {
      final t = _tok();
      final ids = buildToolPrompt(t, toolsJson: '', user: 'ab', date: null);
      expect(ids.first, idBos);
      expect(ids.contains(idImStart), isTrue);
    });
  });
}
