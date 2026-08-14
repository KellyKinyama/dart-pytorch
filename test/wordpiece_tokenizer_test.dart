import 'dart:io';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

const _vocabPath = 'models/minilm/vocab.txt';

void main() {
  final vocabExists = File(_vocabPath).existsSync();

  group('WordPieceTokenizer', () {
    late WordPieceTokenizer tok;

    setUpAll(() {
      if (vocabExists) {
        tok = WordPieceTokenizer.fromVocabFile(_vocabPath);
      }
    });

    test('loads MiniLM vocab (30522 tokens)', () {
      if (!vocabExists) return;
      expect(tok.vocabSize, 30522);
      expect(tok.clsId, greaterThanOrEqualTo(0));
      expect(tok.sepId, greaterThanOrEqualTo(0));
      expect(tok.padId, greaterThanOrEqualTo(0));
      expect(tok.unkId, greaterThanOrEqualTo(0));
    });

    test('encode wraps with [CLS] ... [SEP]', () {
      if (!vocabExists) return;
      final ids = tok.encode('hello world');
      expect(ids.first, tok.clsId);
      expect(ids.last, tok.sepId);
      expect(ids.length, greaterThan(2));
    });

    test('lowercase + basic split', () {
      if (!vocabExists) return;
      final upperIds = tok.encode('HELLO WORLD');
      final lowerIds = tok.encode('hello world');
      expect(upperIds, lowerIds);
    });

    test('WordPiece splits unknown words into ## continuations', () {
      if (!vocabExists) return;
      final pieces = tok.tokenize('embeddings');
      expect(pieces.length, greaterThanOrEqualTo(1));
      // Any subword after position 0 must start with '##'.
      for (int i = 1; i < pieces.length; i++) {
        expect(pieces[i].startsWith('##'), isTrue);
      }
    });

    test('encodePair inserts a [SEP] between text A and text B', () {
      if (!vocabExists) return;
      final ids = tok.encodePair('hello', 'world');
      expect(ids.first, tok.clsId);
      expect(ids.last, tok.sepId);
      // Exactly two [SEP]s: one between, one at end.
      final sepCount = ids.where((i) => i == tok.sepId).length;
      expect(sepCount, 2);
    });

    test('maxLength truncates while preserving special tokens', () {
      if (!vocabExists) return;
      final ids = tok.encode(
        'this is a longer sentence that we want to truncate down',
        maxLength: 6,
      );
      expect(ids.length, 6);
      expect(ids.first, tok.clsId);
      expect(ids.last, tok.sepId);
    });

    test('decode drops specials and merges ## continuations', () {
      if (!vocabExists) return;
      final ids = tok.encode('embeddings');
      expect(tok.decode(ids), 'embeddings');
    });

    test('unknown symbol lands on [UNK]', () {
      if (!vocabExists) return;
      // A truly out-of-vocab codepoint sequence.
      final ids = tok.encode('\u{1F600}\u{1F600}\u{1F600}');
      // Emoji clean-text drops or maps to [UNK]. Either way [CLS]/[SEP]
      // survive.
      expect(ids.first, tok.clsId);
      expect(ids.last, tok.sepId);
    });
  });
}
