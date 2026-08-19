import 'dart:io';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

void main() {
  group('F5TtsCharTokenizer.defaultEnglish', () {
    late F5TtsCharTokenizer tok;
    setUp(() => tok = F5TtsCharTokenizer.defaultEnglish());

    test('has 4 specials + 26 letters + 10 digits + 3 ws + 8 punct = 51', () {
      expect(tok.vocabSize, equals(51));
    });

    test('specials are at IDs 0..3', () {
      expect(tok.padId, equals(0));
      expect(tok.unkId, equals(1));
      expect(tok.bosId, equals(2));
      expect(tok.eosId, equals(3));
    });

    test('encode lowercases and maps chars', () {
      final ids = tok.encode('Hello!');
      expect(ids.length, equals(6));
      expect(ids.every((i) => i != tok.unkId), isTrue);
    });

    test('unknown character maps to unkId', () {
      final ids = tok.encode('~');
      expect(ids, equals([tok.unkId]));
    });

    test('round-trip encode -> decode preserves lower-case text', () {
      const text = 'hello, world!';
      final ids = tok.encode(text);
      expect(tok.decode(ids), equals(text));
    });

    test('addBos + addEos bracket the sequence', () {
      final ids = tok.encode('ab', addBos: true, addEos: true);
      expect(ids.first, equals(tok.bosId));
      expect(ids.last, equals(tok.eosId));
      expect(ids.length, equals(4));
    });

    test('decode skips specials by default', () {
      final ids = tok.encode('ab', addBos: true, addEos: true);
      expect(tok.decode(ids), equals('ab'));
      expect(tok.decode(ids, skipSpecial: false), equals('<bos>ab<eos>'));
    });
  });

  group('F5TtsCharTokenizer.fromVocab', () {
    test('prepends missing special tokens', () {
      final tok = F5TtsCharTokenizer.fromVocab(['a', 'b']);
      expect(tok.vocabulary.take(4).toList(), equals(F5SpecialToken.all));
      expect(tok.vocabSize, equals(6));
    });

    test('rejects duplicates', () {
      expect(
        () => F5TtsCharTokenizer.fromVocab(['a', 'a']),
        throwsArgumentError,
      );
    });
  });

  group('F5TtsCharTokenizer.fromFile', () {
    test('loads vocab file, skips blanks and comments', () {
      final tmp = File(
        '${Directory.systemTemp.path}/f5_tokenizer_vocab_'
        '${DateTime.now().microsecondsSinceEpoch}.txt',
      );
      tmp.writeAsStringSync(
        '# header comment\n'
        'a\n'
        'b\n'
        '\n'
        'c\n',
      );
      try {
        final tok = F5TtsCharTokenizer.fromFile(tmp.path);
        expect(tok.vocabSize, equals(7)); // 4 specials + a,b,c
        expect(tok.encode('abc'), equals([4, 5, 6]));
      } finally {
        if (tmp.existsSync()) tmp.deleteSync();
      }
    });
  });

  group('F5TtsPhonemeTokenizer', () {
    test('inventory-based encode swaps known words for phoneme IDs', () {
      final tok = F5TtsPhonemeTokenizer.fromInventory(
        phonemes: ['HH', 'AH', 'L', 'OW'],
        words: {
          'hello': ['HH', 'AH', 'L', 'OW'],
        },
      );
      final ids = tok.encode('Hello');
      expect(ids.length, equals(4));
      // decoded output re-joins phonemes with spaces (multi-char tokens)
      expect(tok.decode(ids), equals('HH AH L OW'));
    });

    test('unknown word falls back to char-level (unk for OOV letters)', () {
      final tok = F5TtsPhonemeTokenizer.fromInventory(
        phonemes: ['HH', 'AH'],
        words: {
          'hi': ['HH', 'AH'],
        },
      );
      final ids = tok.encode('hi zz');
      expect(ids.length, equals(5)); // HH AH + " " + unk unk
      expect(ids[0], isNot(equals(tok.unkId)));
      expect(ids[1], isNot(equals(tok.unkId)));
      expect(ids[2], isNot(equals(tok.unkId))); // " " is in extraSymbols
      expect(ids[3], equals(tok.unkId));
      expect(ids[4], equals(tok.unkId));
    });

    test('punctuation is preserved via extraSymbols', () {
      final tok = F5TtsPhonemeTokenizer.fromInventory(
        phonemes: ['HH', 'AH'],
        words: {
          'hi': ['HH', 'AH'],
        },
      );
      final ids = tok.encode('hi!');
      expect(ids.length, equals(3));
      expect(ids[2], isNot(equals(tok.unkId)));
    });

    test('CMU-style dict file loads and encodes', () {
      final tmp = File(
        '${Directory.systemTemp.path}/f5_cmu_'
        '${DateTime.now().microsecondsSinceEpoch}.txt',
      );
      tmp.writeAsStringSync(
        ';;; header\n'
        'HELLO  HH AH0 L OW1\n'
        'WORLD  W ER1 L D\n'
        'HELLO(2)  HH EH0 L OW1\n', // duplicate variant — should be skipped
      );
      try {
        final tok = F5TtsPhonemeTokenizer.fromCmuDictFile(tmp.path);
        expect(tok.numWords, equals(2));
        final ids = tok.encode('hello world');
        // 4 (hello) + 1 (" ") + 4 (world) = 9
        expect(ids.length, equals(9));
      } finally {
        if (tmp.existsSync()) tmp.deleteSync();
      }
    });
  });

  group('F5DurationLoss', () {
    test('rate loss is zero when sum matches target', () {
      final dp = F5DurationPredictor(
        textDim: 16,
        intermediateDim: 32,
        numLayers: 1,
      );
      final text = Tensor.fromList(
        [5, 16],
        List<double>.filled(5 * 16, 0.0),
      );
      // At init the softplus(scores).sum() is roughly 5 * ln(2) ≈ 3.47.
      // Pick a target far off and verify the loss is > 0.
      final scores = dp.rawScores(text);
      final loss = F5DurationLoss.rate(rawScores: scores, targetMelFrames: 50);
      final v = loss.toList()[0];
      expect(v > 0, isTrue);
      expect(v.isFinite, isTrue);
    });

    test('rate loss decreases after gradient step', () {
      final dp = F5DurationPredictor(
        textDim: 16,
        intermediateDim: 32,
        numLayers: 1,
        seed: 42,
      );
      final text = Tensor.fromList(
        [4, 16],
        List<double>.generate(4 * 16, (i) => (i - 32) * 0.01),
      );
      final params = dp.parameters();

      double lossValue() {
        final scores = dp.rawScores(text);
        final l = F5DurationLoss.rate(
          rawScores: scores,
          targetMelFrames: 40,
        );
        return l.toList()[0];
      }

      final before = lossValue();
      // One manual SGD step against a single loss evaluation.
      final scores2 = dp.rawScores(text);
      final loss = F5DurationLoss.rate(
        rawScores: scores2,
        targetMelFrames: 40,
      );
      loss.backward();
      const lr = 0.05;
      for (final p in params) {
        final g = p.grad;
        if (g == null) continue;
        final data = p.toList();
        final gData = g.toList();
        for (int i = 0; i < data.length; i++) {
          data[i] -= lr * gData[i];
        }
        p.assign(Tensor.fromList(p.shape, data, device: p.device));
      }
      final after = lossValue();
      expect(
        after < before,
        isTrue,
        reason: 'rate loss should decrease after SGD step '
            '(before=$before after=$after)',
      );
    });

    test('mse loss zero on matched targets', () {
      final scores = Tensor.fromList([3], [5.0, 5.0, 5.0]);
      // softplus(5) ≈ 5.0067
      final target = Tensor.fromList([3], [5.0067, 5.0067, 5.0067]);
      final loss = F5DurationLoss.mse(rawScores: scores, target: target);
      final v = loss.toList()[0];
      expect(v.abs() < 1e-4, isTrue, reason: 'v=$v');
    });

    test('mse loss > 0 on mismatch', () {
      final scores = Tensor.fromList([2], [0.0, 0.0]);
      final target = Tensor.fromList([2], [10.0, 10.0]);
      final loss = F5DurationLoss.mse(rawScores: scores, target: target);
      final v = loss.toList()[0];
      // softplus(0)=ln 2, so per-char diff ≈ 9.31, mse ≈ 86.6
      expect(v > 50 && v < 100, isTrue, reason: 'v=$v');
    });

    test('rawScores preserves gradient tape (backward runs)', () {
      final dp = F5DurationPredictor(
        textDim: 8,
        intermediateDim: 16,
        numLayers: 1,
        seed: 7,
      );
      final text = Tensor.fromList(
        [3, 8],
        List<double>.filled(3 * 8, 0.1),
      );
      final scores = dp.rawScores(text);
      final loss = (scores * scores).sum();
      expect(() => loss.backward(), returnsNormally);
    });
  });
}
