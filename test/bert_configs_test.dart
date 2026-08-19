import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

void main() {
  group('BertHFLoader config shape checks', () {
    test('gteSmallConfig matches thenlper/gte-small', () {
      final cfg = BertHFLoader.gteSmallConfig();
      expect(cfg.vocabSize, 30522);
      expect(cfg.embedDim, 384);
      expect(cfg.numLayers, 12);
      expect(cfg.numHeads, 12);
      expect(cfg.intermediateSize, 1536);
      expect(cfg.maxPositionEmbeddings, 512);
      expect(cfg.layerNormEps, 1e-12);
    });

    test('gteBaseConfig matches thenlper/gte-base', () {
      final cfg = BertHFLoader.gteBaseConfig();
      expect(cfg.embedDim, 768);
      expect(cfg.numLayers, 12);
      expect(cfg.numHeads, 12);
      expect(cfg.intermediateSize, 3072);
    });

    test('bgeSmallEnConfig unchanged', () {
      final cfg = BertHFLoader.bgeSmallEnConfig();
      expect(cfg.embedDim, 384);
      expect(cfg.numLayers, 12);
      expect(cfg.intermediateSize, 1536);
    });

    test('bertBaseUncasedConfig unchanged', () {
      final cfg = BertHFLoader.bertBaseUncasedConfig();
      expect(cfg.embedDim, 768);
      expect(cfg.numLayers, 12);
      expect(cfg.intermediateSize, 3072);
    });
  });
}
