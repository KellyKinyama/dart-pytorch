import 'dart:io';

import 'package:dart_pytorch/core/data/marian_vocab_tokenizer.dart';
import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

MarianConfig _tinyCfg() => MarianConfig(
  vocabSize: 32,
  dModel: 16,
  ffnDim: 32,
  numLayers: 2,
  numDecoderLayers: 2,
  numHeads: 4,
  maxPositionEmbeddings: 32,
  padTokenId: 30,
  eosTokenId: 0,
  decoderStartTokenId: 30,
  scaleEmbeddings: true,
  activation: MarianActivation.silu,
  seed: 7,
);

Tensor _ids(List<int> ids) =>
    Tensor.fromList([ids.length], ids.map((i) => i.toDouble()).toList());

void main() {
  group('MarianConfig', () {
    test('opus-mt-en-de: 6+6 layers, dModel=512, silu FFN', () {
      final c = MarianHFLoader.opusMtEnDeConfig();
      expect(c.vocabSize, 58101);
      expect(c.dModel, 512);
      expect(c.ffnDim, 2048);
      expect(c.numLayers, 6);
      expect(c.numDecoderLayers, 6);
      expect(c.numHeads, 8);
      expect(c.headDim, 64);
      expect(c.padTokenId, 58100);
      expect(c.eosTokenId, 0);
      expect(c.decoderStartTokenId, 58100);
      expect(c.activation, MarianActivation.silu);
    });
  });

  group('Marian forward (shape only)', () {
    test('encoder returns [N, dModel]', () {
      final m = MarianModel(_tinyCfg());
      expect(m.encoder(_ids([0, 1, 2, 3])).shape, equals([4, 16]));
    });

    test('decoder returns [Nq, dModel]', () {
      final m = MarianModel(_tinyCfg());
      final memory = m.encoder(_ids([0, 1, 2, 3]));
      final tgt = _ids([30, 5, 6]);
      expect(m.decoder(tgt, memory: memory).shape, equals([3, 16]));
    });

    test('logitsLastToken returns [vocab]', () {
      final m = MarianModel(_tinyCfg());
      final memory = m.encoder(_ids([0, 1, 2, 3]));
      final logits = m.logitsLastToken([30, 5], memory);
      expect(logits.shape, equals([32]));
    });

    test('generate returns non-empty', () {
      final m = MarianModel(_tinyCfg());
      final out = m.generate([1, 2, 3, 4], maxNewTokens: 5);
      expect(out.length, greaterThan(1));
      expect(out.first, equals(30));
    });

    test('cached and non-cached generate produce identical outputs', () {
      final m = MarianModel(_tinyCfg());
      final cached = m.generate([1, 2, 3, 4], maxNewTokens: 8);
      final recomp = m.generate([1, 2, 3, 4], maxNewTokens: 8, useCache: false);
      expect(cached, equals(recomp));
    });

    test('beam=1 matches greedy', () {
      final m = MarianModel(_tinyCfg());
      final greedy = m.generate([1, 2, 3, 4], maxNewTokens: 8);
      final beam1 = m.generateBeam(
        [1, 2, 3, 4],
        numBeams: 1,
        maxNewTokens: 8,
      );
      expect(beam1, equals(greedy));
    });

    test('beam=4 returns non-empty sequence starting with decoder_start', () {
      final m = MarianModel(_tinyCfg());
      final out = m.generateBeam(
        [1, 2, 3, 4],
        numBeams: 4,
        maxNewTokens: 8,
      );
      expect(out.length, greaterThan(1));
      expect(out.first, equals(30));
    });
  });

  group('MarianVocabTokenizer', () {
    const vocabPath = 'models/opus-mt-en-de/vocab.json';
    final hasVocab = File(vocabPath).existsSync();
    late MarianVocabTokenizer tok;

    setUpAll(() {
      if (hasVocab) tok = MarianVocabTokenizer.loadFile(vocabPath);
    });

    test(
      'vocab size 58101, </s>=0 as eos',
      () {
        expect(tok.vocabSize, equals(58101));
        expect(tok.eosId, equals(0));
      },
      skip: hasVocab ? null : 'opus-mt-en-de vocab.json not downloaded',
    );

    test(
      'encode "Hello world." matches HF greedy: [16816, 360, 3, 0]',
      () {
        expect(tok.encode('Hello world.'), equals([16816, 360, 3, 0]));
      },
      skip: hasVocab ? null : 'opus-mt-en-de vocab.json not downloaded',
    );

    test(
      'roundtrip decode preserves basic text',
      () {
        const s = 'the quick brown fox';
        final ids = tok.encode(s);
        expect(tok.decode(ids), equals(s));
      },
      skip: hasVocab ? null : 'opus-mt-en-de vocab.json not downloaded',
    );

    test(
      'decode strips special tokens by default',
      () {
        final ids = tok.encode('Hello');
        expect(ids.last, equals(tok.eosId));
        expect(tok.decode(ids), isNot(contains('</s>')));
      },
      skip: hasVocab ? null : 'opus-mt-en-de vocab.json not downloaded',
    );
  });
}
