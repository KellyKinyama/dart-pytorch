import 'dart:convert';

import 'package:dart_pytorch/core/data/hf_bpe_tokenizer.dart';
import 'package:test/test.dart';

/// Build a synthetic Llama-3-style `tokenizer.json` map covering just
/// enough of the ASCII byte-mapped alphabet to encode/decode short
/// probes, plus the Llama-3 pre-tokenizer Sequence(Split + ByteLevel)
/// and the canonical specials at their real ids.
///
/// The BPE has no merges — every base byte-mapped character is a
/// standalone token. That's enough to exercise the pipeline
/// (byte-level encode → pre-tokenize with the Llama regex → BPE
/// lookup → id list).
Map<String, dynamic> _synthLlama3Tokenizer() {
  // Base byte-mapped alphabet: printable ASCII + Ġ (space) — matches
  // GPT-2's `bytes_to_unicode` for the ASCII range. Every byte-mapped
  // character gets a unique id in `[0, N)`.
  final vocab = <String, int>{};
  int nextId = 0;
  for (int i = 0x21; i <= 0x7E; i++) {
    vocab[String.fromCharCode(i)] = nextId++;
  }
  // Ġ (byte-mapped space) = U+0120.
  vocab[String.fromCharCode(0x120)] = nextId++;
  // Newline byte 0x0A → byte-mapped codepoint is 0x100 + 0x0A = 0x10A.
  vocab[String.fromCharCode(0x10A)] = nextId++;

  return <String, dynamic>{
    'model': {
      'type': 'BPE',
      'vocab': vocab,
      'merges': <List<String>>[],
    },
    'pre_tokenizer': {
      'type': 'Sequence',
      'pretokenizers': [
        {
          'type': 'Split',
          // The real Llama-3 pattern with the leading `(?i:…)` inline
          // flag we translate into `caseSensitive: false`.
          'pattern': {
            'Regex':
                r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
          },
          'behavior': 'Isolated',
          'invert': false,
        },
        {
          'type': 'ByteLevel',
          'add_prefix_space': false,
          'trim_offsets': true,
          'use_regex': false,
        },
      ],
    },
    'added_tokens': [
      {
        'id': 128000,
        'content': '<|begin_of_text|>',
        'single_word': false,
        'lstrip': false,
        'rstrip': false,
        'normalized': false,
        'special': true,
      },
      {
        'id': 128001,
        'content': '<|end_of_text|>',
        'single_word': false,
        'lstrip': false,
        'rstrip': false,
        'normalized': false,
        'special': true,
      },
      {
        'id': 128009,
        'content': '<|eot_id|>',
        'single_word': false,
        'lstrip': false,
        'rstrip': false,
        'normalized': false,
        'special': true,
      },
    ],
  };
}

void main() {
  group('HFBpeTokenizer (Llama-3 synthetic)', () {
    late HFBpeTokenizer tok;

    setUpAll(() {
      tok = HFBpeTokenizer.fromJson(_synthLlama3Tokenizer());
    });

    test('loads pre_tokenizer.Sequence(Split, ByteLevel) without error',
        () {
      // If we got here, the (?i:…) group survived _compilePyRegex.
      expect(tok.vocabSize, greaterThan(90));
    });

    test('encodes and decodes ASCII (single-byte fallback path)', () {
      final ids = tok.encode('Hello world');
      expect(ids, isNotEmpty);
      expect(tok.decode(ids), 'Hello world');
    });

    test('encodes and decodes ASCII with newline', () {
      final s = 'a\nb';
      final ids = tok.encode(s);
      expect(tok.decode(ids), s);
    });

    test('llamaBeginOfTextId and llamaEotId are wired', () {
      expect(tok.llamaBeginOfTextId, 128000);
      expect(tok.llamaEotId, 128009);
      expect(tok.tokenId('<|end_of_text|>'), 128001);
      expect(tok.tokenId('<|not-a-real-token|>'), isNull);
    });

    test('specials round-trip in the middle of a prompt', () {
      final s = 'hi<|eot_id|>bye';
      final ids = tok.encode(s);
      expect(ids, contains(128009));
      expect(tok.decode(ids), s);
    });

    test('endOfTextId is null when only <|end_of_text|> is present', () {
      // Llama-3 uses <|end_of_text|>, NOT <|endoftext|> — the existing
      // GPT-2 getter should return null on Llama.
      expect(tok.endOfTextId, isNull);
    });

    test(
        'JSON string load path also works (round-trips through '
        'jsonDecode)', () {
      final asString = jsonEncode(_synthLlama3Tokenizer());
      final parsed = jsonDecode(asString) as Map<String, dynamic>;
      final tok2 = HFBpeTokenizer.fromJson(parsed);
      expect(tok2.encode('abc'), tok.encode('abc'));
    });
  });

  group('HFBpeTokenizer regex compilation helpers', () {
    test('accepts pattern with no (?i:…) prefix', () {
      final j = _synthLlama3Tokenizer();
      (j['pre_tokenizer']['pretokenizers'][0]['pattern']
          as Map)['Regex'] = r'\p{L}+|\p{N}+|\s+';
      // Should not throw.
      final t = HFBpeTokenizer.fromJson(j);
      expect(t.encode('abc'), isNotEmpty);
    });

    test('falls back to GPT-2 regex when pre_tokenizer is ByteLevel-only',
        () {
      final j = _synthLlama3Tokenizer();
      j['pre_tokenizer'] = {
        'type': 'ByteLevel',
        'add_prefix_space': false,
        'trim_offsets': true,
        'use_regex': true,
      };
      // Should not throw — falls back to the hardcoded GPT-2 default.
      final t = HFBpeTokenizer.fromJson(j);
      expect(t.encode('a').isNotEmpty, isTrue);
    });

    test('falls back when pre_tokenizer is absent', () {
      final j = _synthLlama3Tokenizer();
      j.remove('pre_tokenizer');
      final t = HFBpeTokenizer.fromJson(j);
      expect(t.encode('a').isNotEmpty, isTrue);
    });
  });
}
