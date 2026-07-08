/// Tests for the character-level tokenizer.
library;

import 'dart:io';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

void main() {
  group('CharTokenizer', () {
    test('fromText builds sorted unique vocabulary', () {
      final tok = CharTokenizer.fromText('banana');
      expect(tok.chars, ['a', 'b', 'n']);
      expect(tok.vocabSize, 3);
    });

    test('encode / decode round-trip', () {
      final tok = CharTokenizer.fromText('hello, shakespeare!');
      const s = 'hello, shakespeare!';
      final ids = tok.encode(s);
      expect(ids.length, s.length);
      expect(tok.decode(ids), s);
    });

    test('unknown chars encode to 0 and decode is safe on OOB ids', () {
      final tok = CharTokenizer.fromText('abc');
      // 'z' is not in vocab -> id 0 -> 'a'
      expect(tok.encode('zzz'), [0, 0, 0]);
      // Out-of-range ids are dropped.
      expect(tok.decode([0, 1, 999, 2, -1]), 'abc');
    });

    test('JSON round-trip preserves vocab', () {
      final tok = CharTokenizer.fromText('the quick brown fox');
      final restored = CharTokenizer.fromJson(tok.toJson());
      expect(restored.chars, tok.chars);
      expect(restored.encode('quick'), tok.encode('quick'));
    });

    test('saveFile / loadFile round-trip', () {
      final tok = CharTokenizer.fromText('romeo and juliet');
      final dir = Directory.systemTemp.createTempSync('chartok_test_');
      try {
        final path = '${dir.path}/tok.json';
        tok.saveFile(path);
        final loaded = CharTokenizer.loadFile(path);
        expect(loaded.chars, tok.chars);
        expect(loaded.decode(loaded.encode('romeo')), 'romeo');
      } finally {
        dir.deleteSync(recursive: true);
      }
    });
  });
}
