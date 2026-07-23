import 'dart:io';

import 'package:dart_pytorch/core/data/hf_bpe_tokenizer.dart';
import 'package:test/test.dart';

void main() {
  // Use the Pythia tokenizer.json we already have on disk. If not
  // downloaded, skip the whole suite so CI without weights still works.
  const path = 'models/pythia-160m/tokenizer.json';
  final present = File(path).existsSync();

  group(
    'HFBpeTokenizer (Pythia)',
    skip: present
        ? null
        : 'skipped: models/pythia-160m/tokenizer.json not present',
    () {
      late HFBpeTokenizer tok;
      setUpAll(() {
        tok = HFBpeTokenizer.loadFile(path);
      });

      test('encodes and decodes basic ASCII', () {
        final ids = tok.encode('The world is');
        // Sanity: known ids for these three tokens on GPT-NeoX vocab.
        //   "The"     -> 510
        //   " world"  -> 1533
        //   " is"     -> 310
        expect(ids, equals(<int>[510, 1533, 310]));
        expect(tok.decode(ids), equals('The world is'));
      });

      test('round-trips a longer sentence', () {
        const s = 'Hello, world! This is a test of the emergency system.';
        final ids = tok.encode(s);
        expect(ids, isNotEmpty);
        expect(tok.decode(ids), equals(s));
      });

      test('round-trips unicode + emoji', () {
        const s = 'café — naïve façade — 你好, 世界! 🚀';
        final ids = tok.encode(s);
        expect(tok.decode(ids), equals(s));
      });

      test('handles leading whitespace correctly', () {
        // " The" (with space) tokenizes differently from "The".
        final withSpace = tok.encode(' The');
        final without = tok.encode('The');
        expect(withSpace, isNot(equals(without)));
        expect(tok.decode(withSpace), equals(' The'));
      });

      test('recognizes <|endoftext|> as a special token', () {
        final eot = tok.endOfTextId;
        expect(eot, isNotNull);
        expect(eot, equals(0));
        final ids = tok.encode('hi<|endoftext|>bye');
        expect(ids, contains(eot));
        // Decode preserves the special token literal.
        expect(tok.decode(ids), equals('hi<|endoftext|>bye'));
      });

      test('vocabSize is close to 50257', () {
        expect(tok.vocabSize, greaterThan(50000));
      });
    },
  );
}
