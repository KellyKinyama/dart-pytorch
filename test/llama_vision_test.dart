import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

LlamaConfig _cfg({int vocab = 32, int maxCtx = 64, int embedDim = 16}) =>
    LlamaConfig(
      vocabSize: vocab,
      maxCtx: maxCtx,
      embedDim: embedDim,
      numLayers: 2,
      numHeads: 4,
      ffnDim: 32,
    );

ViTBackbone _vit({int embedDim = 12}) => ViTBackbone(
  imageSize: 8,
  patchSize: 4,
  numChannels: 3,
  embedDim: embedDim,
  numLayers: 2,
  numHeads: 3,
);

Tensor _fakeImage(ViTBackbone vit) {
  final expected = vit.patchSize * vit.patchSize * vit.numChannels;
  final data = List<double>.generate(
    vit.numPatches * expected,
    (i) => 0.01 * ((i % 7) - 3),
  );
  return Tensor.fromList([vit.numPatches, expected], data);
}

void main() {
  group('VisionProjector', () {
    test('shape [N, inDim] -> [N, outDim]', () {
      final proj = VisionProjector(6, 10);
      final x = Tensor.fromList([
        3,
        6,
      ], List<double>.generate(18, (i) => (i - 9) * 0.1));
      final y = proj(x);
      expect(y.shape, [3, 10]);
    });

    test('rejects wrong inner dim', () {
      final proj = VisionProjector(6, 10);
      final x = Tensor.fromList([2, 7], List<double>.filled(14, 0.0));
      expect(() => proj(x), throwsArgumentError);
    });

    test('exposes trainable parameters', () {
      final proj = VisionProjector(6, 10, hiddenDim: 20);
      // 2 Linears with bias => 4 params.
      expect(proj.parameters().length, 4);
    });
  });

  group('Llama.forwardFromEmbeddings', () {
    test('shape matches token-input forward', () {
      final llama = Llama(_cfg());
      final tokens = Tensor.fromList([5], [1.0, 2.0, 3.0, 4.0, 5.0]);
      final logitsTok = llama(tokens);
      final emb = llama.embedIn(tokens);
      final logitsEmb = llama.forwardFromEmbeddings(emb);
      expect(logitsEmb.shape, logitsTok.shape);
      expect(logitsEmb.shape, [5, llama.config.vocabSize]);
      // Values should also match — same math path.
      final a = logitsTok.toList();
      final b = logitsEmb.toList();
      for (int i = 0; i < a.length; i++) {
        expect(b[i], closeTo(a[i], 1e-5));
      }
    });

    test('rejects wrong embedDim', () {
      final llama = Llama(_cfg(embedDim: 16));
      final wrong = Tensor.fromList([2, 8], List<double>.filled(16, 0.0));
      expect(() => llama.forwardFromEmbeddings(wrong), throwsArgumentError);
    });
  });

  group('LlamaVision', () {
    test('build ties dims and reports numImageTokens', () {
      final vit = _vit(embedDim: 12);
      final llama = Llama(_cfg(embedDim: 16));
      final lv = LlamaVision.build(vit: vit, llama: llama);
      expect(lv.projector.inDim, 12);
      expect(lv.projector.outDim, 16);
      // 8/4 = 2 patches per side => 4 patches + CLS = 5 image tokens.
      expect(lv.numImageTokens, 5);
    });

    test('encodeImage shape [P+1, llamaEmbedDim]', () {
      final vit = _vit(embedDim: 12);
      final llama = Llama(_cfg(embedDim: 16));
      final lv = LlamaVision.build(vit: vit, llama: llama);
      final img = _fakeImage(vit);
      final toks = lv.encodeImage(img);
      expect(toks.shape, [5, 16]);
    });

    test('forward returns [P+1+T, vocab] logits', () {
      final vit = _vit(embedDim: 12);
      final llama = Llama(_cfg(embedDim: 16, vocab: 32));
      final lv = LlamaVision.build(vit: vit, llama: llama);
      final img = _fakeImage(vit);
      final text = Tensor.fromList([3], [1.0, 2.0, 3.0]);
      final y = lv(img, text);
      expect(y.shape, [5 + 3, 32]);
    });

    test('generate returns prompt + N new tokens', () {
      final vit = _vit(embedDim: 12);
      final llama = Llama(_cfg(embedDim: 16, vocab: 32, maxCtx: 64));
      final lv = LlamaVision.build(vit: vit, llama: llama);
      final img = _fakeImage(vit);
      final prompt = <double>[1.0, 2.0];
      final out = lv.generate(img, prompt, maxNewTokens: 4, temperature: 0.0);
      // Greedy, so deterministic; output is prompt + 4 new tokens.
      expect(out.length, prompt.length + 4);
      expect(out.sublist(0, prompt.length), prompt);
      for (final t in out.sublist(prompt.length)) {
        expect(t.toInt() >= 0 && t.toInt() < 32, isTrue);
      }
    });

    test('generate stops early at stopId', () {
      final vit = _vit(embedDim: 12);
      final llama = Llama(_cfg(embedDim: 16, vocab: 8, maxCtx: 64));
      final lv = LlamaVision.build(vit: vit, llama: llama);
      final img = _fakeImage(vit);
      // Greedy on a tiny vocab: pick some token, then set it as stop
      // and confirm the loop ends immediately.
      final probe = lv.generate(img, [1.0], maxNewTokens: 1, temperature: 0.0);
      final first = probe.last.toInt();
      final out = lv.generate(
        img,
        [1.0],
        maxNewTokens: 10,
        temperature: 0.0,
        stopId: first,
      );
      // First produced token equals `first`, then loop stops.
      expect(out.length, 2);
      expect(out.last.toInt(), first);
    });

    test('parameters include vit + projector + llama', () {
      final vit = _vit(embedDim: 12);
      final llama = Llama(_cfg(embedDim: 16));
      final lv = LlamaVision.build(vit: vit, llama: llama);
      expect(
        lv.parameters().length,
        vit.parameters().length +
            lv.projector.parameters().length +
            llama.parameters().length,
      );
    });
  });
}
