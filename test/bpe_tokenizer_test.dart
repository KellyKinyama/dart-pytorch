import 'dart:io';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

void main() {
  group('BpeTokenizer.train', () {
    test('rejects target vocab < 256', () {
      expect(
        () => BpeTokenizer.train('abcabc', targetVocabSize: 100),
        throwsArgumentError,
      );
    });

    test('rejects empty corpus', () {
      expect(
        () => BpeTokenizer.train('', targetVocabSize: 300),
        throwsArgumentError,
      );
    });

    test('learns merges up to targetVocabSize (or stops when no gain)', () {
      // A tiny repetitive corpus. Base vocab is 256; each merge
      // adds one id, so plenty of headroom before we run out of
      // profitable pairs.
      final t = BpeTokenizer.train(
        'the quick brown fox jumps over the lazy dog. ' * 20,
        targetVocabSize: 280,
      );
      expect(t.vocabSize, lessThanOrEqualTo(280));
      expect(t.vocabSize, greaterThan(256));
      expect(t.merges.length, equals(t.vocabSize - 256));
    });

    test('is deterministic for the same input', () {
      const corpus = 'ababababab ababcabc ababcabc ababcabc';
      final a = BpeTokenizer.train(corpus, targetVocabSize: 270);
      final b = BpeTokenizer.train(corpus, targetVocabSize: 270);
      expect(a.merges, equals(b.merges));
      expect(a.vocabSize, equals(b.vocabSize));
    });

    test('honours minCount by stopping early on tiny corpora', () {
      // Each character occurs only once in a short unique string —
      // no adjacent pair occurs >= minCount times, so no merges.
      final t = BpeTokenizer.train('abcdef', targetVocabSize: 300, minCount: 2);
      expect(t.merges, isEmpty);
      expect(t.vocabSize, equals(256));
    });
  });

  group('encode / decode', () {
    test('base tokenizer (no merges) is byte-identity', () {
      final t = BpeTokenizer.train(
        'abc',
        targetVocabSize: 300,
        minCount: 100, // guarantees no merges get made
      );
      const s = 'hello, world!';
      final ids = t.encode(s);
      expect(ids, everyElement(lessThan(256)));
      expect(t.decode(ids), equals(s));
    });

    test('trained tokenizer round-trips ASCII', () {
      final t = BpeTokenizer.train(
        'the quick brown fox jumps over the lazy dog ' * 30,
        targetVocabSize: 320,
      );
      const s = 'the lazy fox jumps over the quick brown dog';
      expect(t.decode(t.encode(s)), equals(s));
    });

    test('trained tokenizer round-trips multi-byte UTF-8', () {
      // Include some non-ASCII text so encoding must go through
      // raw bytes correctly.
      const corpus = 'café — naïve façade. ';
      final t = BpeTokenizer.train(corpus * 30, targetVocabSize: 320);
      const probe = 'a naïve façade in a café';
      expect(t.decode(t.encode(probe)), equals(probe));
    });

    test('encoded length shrinks vs raw bytes for repetitive text', () {
      final corpus = 'abcabcabcabc ' * 50;
      final t = BpeTokenizer.train(corpus, targetVocabSize: 300);
      final rawLen = corpus.codeUnits.length;
      final tokLen = t.encode(corpus).length;
      expect(tokLen, lessThan(rawLen));
    });
  });

  group('persistence', () {
    test('JSON round-trip preserves merges and encode output', () {
      final t = BpeTokenizer.train(
        'the rain in spain stays mainly in the plain ' * 20,
        targetVocabSize: 320,
      );
      final restored = BpeTokenizer.fromJson(t.saveJson());
      expect(restored.vocabSize, equals(t.vocabSize));
      expect(restored.merges, equals(t.merges));
      const probe = 'the plain rain in spain';
      expect(restored.encode(probe), equals(t.encode(probe)));
    });

    test('file save/load round-trips', () {
      final t = BpeTokenizer.train(
        'hello hello hello world world ' * 20,
        targetVocabSize: 300,
      );
      final dir = Directory.systemTemp.createTempSync('bpe_test_');
      try {
        final path = '${dir.path}/tok.json';
        t.saveFile(path);
        final restored = BpeTokenizer.loadFile(path);
        expect(restored.merges, equals(t.merges));
        expect(restored.encode('hello world'), equals(t.encode('hello world')));
      } finally {
        dir.deleteSync(recursive: true);
      }
    });

    test('rejects unknown kind / version', () {
      expect(
        () => BpeTokenizer.fromJson(
          '{"kind":"not-bpe","version":1,'
          '"vocabSize":256,"merges":[]}',
        ),
        throwsArgumentError,
      );
      expect(
        () => BpeTokenizer.fromJson(
          '{"kind":"byte-bpe","version":99,'
          '"vocabSize":256,"merges":[]}',
        ),
        throwsArgumentError,
      );
    });
  });
}
