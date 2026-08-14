import 'dart:io';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

const _weightsPath = 'models/minilm/model.safetensors';
const _vocabPath = 'models/minilm/vocab.txt';

Tensor _toTokens(List<int> ids) =>
    Tensor.fromList([ids.length], ids.map((i) => i.toDouble()).toList());

void main() {
  final available =
      File(_weightsPath).existsSync() && File(_vocabPath).existsSync();

  group('BertHFLoader on all-MiniLM-L6-v2 (skipped if weights absent)', () {
    late BertModel model;
    late WordPieceTokenizer tok;
    late SentenceEncoder encoder;

    setUpAll(() {
      if (!available) return;
      model = BertModel(BertHFLoader.miniLmL6V2Config());
      final report = BertHFLoader.loadFile(model, _weightsPath);
      expect(
        report.unusedKeys,
        isEmpty,
        reason: 'Unhandled keys: ${report.unusedKeys}',
      );
      tok = WordPieceTokenizer.fromVocabFile(_vocabPath);
      encoder = SentenceEncoder.wrap(model);
      encoder.eval();
    });

    test('config sanity — 6 layers, 384 hidden, 12 heads', () {
      if (!available) return;
      expect(model.config.numLayers, 6);
      expect(model.config.embedDim, 384);
      expect(model.config.numHeads, 12);
      expect(model.config.intermediateSize, 1536);
    });

    test('forward on "Hello world" produces a finite unit vector', () {
      if (!available) return;
      final ids = tok.encode('Hello world');
      Tensor.noGrad(() {
        final emb = encoder(_toTokens(ids)).toList();
        expect(emb.length, 384);
        var s = 0.0;
        for (final v in emb) {
          expect(v.isFinite, isTrue);
          s += v * v;
        }
        expect(s, closeTo(1.0, 1e-3));
      });
    });

    test('semantic ordering: paraphrase > topical > unrelated', () {
      if (!available) return;
      final anchor = _toTokens(tok.encode('A man is playing guitar.'));
      final paraphrase = _toTokens(tok.encode('Someone is playing a guitar.'));
      final topical = _toTokens(tok.encode('A woman is playing violin.'));
      final unrelated = _toTokens(
        tok.encode('The stock market rose today.'),
      );

      final scores = Tensor.noGrad(() {
        final a = encoder(anchor);
        return {
          'paraphrase': Similarity.cosSim(a, encoder(paraphrase)),
          'topical': Similarity.cosSim(a, encoder(topical)),
          'unrelated': Similarity.cosSim(a, encoder(unrelated)),
        };
      });

      expect(scores['paraphrase']!, greaterThan(scores['topical']!));
      expect(scores['topical']!, greaterThan(scores['unrelated']!));
    });
  });
}
