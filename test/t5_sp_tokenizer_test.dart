import 'dart:io';

import 'package:dart_pytorch/core/data/t5_sp_tokenizer.dart';
import 'package:test/test.dart';

const _realTokenizer = 'models/flan-t5-small/tokenizer.json';

void main() {
  group('T5SpTokenizer (synthetic)', () {
    late T5SpTokenizer tok;

    setUp(() {
      // Minimal tokenizer.json in-memory: a handful of pieces + the
      // three specials. Enough to exercise Viterbi and metaspace
      // preprocessing.
      final raw = {
        'model': {
          'type': 'Unigram',
          'vocab': [
            ['<pad>', 0.0],
            ['</s>', 0.0],
            ['<unk>', 0.0],
            ['\u2581', -1.0],
            ['\u2581hello', -2.0],
            ['\u2581world', -2.0],
            ['\u2581the', -1.5],
            ['\u2581quick', -3.0],
            ['h', -5.0],
            ['e', -5.0],
            ['l', -5.0],
            ['o', -5.0],
            ['w', -5.0],
            ['r', -5.0],
            ['d', -5.0],
            ['t', -5.0],
            ['.', -3.0],
            [',', -3.0],
          ],
        },
        'added_tokens': [
          {'id': 0, 'content': '<pad>'},
          {'id': 1, 'content': '</s>'},
          {'id': 2, 'content': '<unk>'},
        ],
      };
      tok = T5SpTokenizer.fromJson(raw);
    });

    test('specials at fixed low ids', () {
      expect(tok.padId, equals(0));
      expect(tok.eosId, equals(1));
      expect(tok.unkId, equals(2));
    });

    test('vocab size matches piece count', () {
      expect(tok.vocabSize, greaterThanOrEqualTo(18));
    });

    test('encode prefers longer piece via Viterbi', () {
      // "hello" alone should snap to ▁hello + </s>.
      final ids = tok.encode('hello');
      expect(ids.length, equals(2));
      expect(ids.last, equals(tok.eosId));
      // The single ▁hello token is id 4 in our synthetic vocab.
      expect(ids.first, equals(4));
    });

    test('encode falls back to single chars when no long piece fits', () {
      // "hw" — no piece matches "▁hw"; Viterbi walks per-char with
      // ▁ + h + w. Should end with </s>.
      final ids = tok.encode('hw');
      expect(ids.last, equals(tok.eosId));
    });

    test('decode roundtrips a known-good sequence', () {
      final ids = tok.encode('hello world');
      final text = tok.decode(ids);
      expect(text, equals('hello world'));
    });

    test('decode skips special tokens by default', () {
      final ids = tok.encode('hello');
      final withSpecials = tok.decode(ids, skipSpecial: false);
      expect(withSpecials.contains('</s>'), isTrue);
      expect(tok.decode(ids), equals('hello'));
    });

    test('encode(addEos: false) omits </s>', () {
      final ids = tok.encode('hello', addEos: false);
      expect(ids.last, isNot(equals(tok.eosId)));
    });
  });

  group('T5SpTokenizer (google/flan-t5-small)', () {
    late T5SpTokenizer tok;

    setUpAll(() {
      if (!File(_realTokenizer).existsSync()) {
        return;
      }
      tok = T5SpTokenizer.loadFile(_realTokenizer);
    });

    test(
      'vocab size + special ids match HF',
      () {
        expect(tok.vocabSize, equals(32100));
        expect(tok.padId, equals(0));
        expect(tok.eosId, equals(1));
        expect(tok.unkId, equals(2));
      },
      skip: File(_realTokenizer).existsSync()
          ? null
          : 'flan-t5-small tokenizer not downloaded',
    );

    test(
      '"Hello, world." matches HF: [8774, 6, 296, 5, 1]',
      () {
        expect(tok.encode('Hello, world.'), equals([8774, 6, 296, 5, 1]));
      },
      skip: File(_realTokenizer).existsSync()
          ? null
          : 'flan-t5-small tokenizer not downloaded',
    );

    test(
      '"summarize: ..." matches HF golden',
      () {
        expect(
          tok.encode('summarize: The quick brown fox jumps over the lazy dog.'),
          equals([
            21603,
            10,
            37,
            1704,
            4216,
            3,
            20400,
            4418,
            7,
            147,
            8,
            19743,
            1782,
            5,
            1,
          ]),
        );
      },
      skip: File(_realTokenizer).existsSync()
          ? null
          : 'flan-t5-small tokenizer not downloaded',
    );

    test(
      '"translate ..." matches HF golden',
      () {
        expect(
          tok.encode('translate English to German: I love machine learning.'),
          equals([13959, 1566, 12, 2968, 10, 27, 333, 1437, 1036, 5, 1]),
        );
      },
      skip: File(_realTokenizer).existsSync()
          ? null
          : 'flan-t5-small tokenizer not downloaded',
    );

    test(
      'roundtrip: encode -> decode preserves lowercased/no-punct text',
      () {
        const s = 'the quick brown fox';
        final ids = tok.encode(s);
        expect(tok.decode(ids), equals(s));
      },
      skip: File(_realTokenizer).existsSync()
          ? null
          : 'flan-t5-small tokenizer not downloaded',
    );
  });
}
