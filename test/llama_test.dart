import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

void main() {
  group('LlamaConfig', () {
    test('numKvHeads defaults to numHeads', () {
      final cfg = LlamaConfig(
        vocabSize: 32,
        maxCtx: 16,
        embedDim: 8,
        numLayers: 2,
        numHeads: 4,
        ffnDim: 16,
      );
      expect(cfg.numKvHeads, 4);
    });

    test('numKvHeads=1 (MQA) accepted', () {
      final cfg = LlamaConfig(
        vocabSize: 32,
        maxCtx: 16,
        embedDim: 8,
        numLayers: 2,
        numHeads: 4,
        numKvHeads: 1,
        ffnDim: 16,
      );
      expect(cfg.numKvHeads, 1);
    });

    test('rejects embedDim not divisible by numHeads', () {
      expect(
        () => LlamaConfig(
          vocabSize: 32,
          maxCtx: 16,
          embedDim: 9,
          numLayers: 2,
          numHeads: 4,
          ffnDim: 16,
        ),
        throwsArgumentError,
      );
    });

    test('rejects numHeads not divisible by numKvHeads', () {
      expect(
        () => LlamaConfig(
          vocabSize: 32,
          maxCtx: 16,
          embedDim: 12,
          numLayers: 2,
          numHeads: 3,
          numKvHeads: 2,
          ffnDim: 16,
        ),
        throwsArgumentError,
      );
    });
  });

  group('LlamaBlock', () {
    test(
      'parameters and submodules present (attnNorm, ffnNorm, attn, ffn)',
      () {
        final rope = RopeCache(maxCtx: 8, headDim: 4);
        final blk = LlamaBlock(8, 2, numKvHeads: 2, ffnDim: 16, rope: rope);
        // 2 RMSNorm gammas + attn (wq*2 + wk*2 + wv*2 + wo, all bias=false)
        // + swiglu (3 Linears bias=false)
        // = 2 + 7 + 3 = 12 params
        expect(blk.parameters().length, 12);
        expect(blk.submodules().length, 4);
        expect(blk.attn.rope, isNotNull);
      },
    );

    test('forward preserves shape [N, D]', () {
      final rope = RopeCache(maxCtx: 8, headDim: 4);
      final blk = LlamaBlock(8, 2, numKvHeads: 2, ffnDim: 16, rope: rope);
      final x = Tensor.fromList(
        [3, 8],
        [for (int i = 0; i < 24; i++) 0.1 * i - 0.5],
      );
      final y = blk(x);
      expect(y.shape, [3, 8]);
    });
  });

  group('Llama model', () {
    test('forward output shape is [seqLen, vocab] (tied head)', () {
      final cfg = LlamaConfig(
        vocabSize: 24,
        maxCtx: 16,
        embedDim: 8,
        numLayers: 2,
        numHeads: 4,
        numKvHeads: 2,
        ffnDim: 16,
        seed: 42,
      );
      final m = Llama(cfg);
      expect(m.untiedHead, isNull);
      final tokens = Tensor.fromList([5], [0.0, 1.0, 2.0, 3.0, 4.0]);
      final logits = m(tokens);
      expect(logits.shape, [5, 24]);
    });

    test('forward output shape is [seqLen, vocab] (untied head)', () {
      final cfg = LlamaConfig(
        vocabSize: 24,
        maxCtx: 16,
        embedDim: 8,
        numLayers: 1,
        numHeads: 2,
        numKvHeads: 1,
        ffnDim: 16,
        tieWeights: false,
        seed: 7,
      );
      final m = Llama(cfg);
      expect(m.untiedHead, isNotNull);
      final tokens = Tensor.fromList([4], [0.0, 3.0, 1.0, 2.0]);
      final logits = m(tokens);
      expect(logits.shape, [4, 24]);
    });

    test('token-by-token generate with cache matches no-cache prompt-fill', () {
      final cfg = LlamaConfig(
        vocabSize: 20,
        maxCtx: 16,
        embedDim: 8,
        numLayers: 2,
        numHeads: 4,
        numKvHeads: 2,
        ffnDim: 16,
        seed: 3,
      );
      final m = Llama(cfg);
      final prompt = [0.0, 1.0, 2.0, 3.0];

      // Greedy (temperature=0) so both paths deterministically produce
      // the same next tokens.
      final withCache = m.generate(
        prompt,
        maxNewTokens: 5,
        temperature: 0.0,
        useCache: true,
      );
      final withoutCache = m.generate(
        prompt,
        maxNewTokens: 5,
        temperature: 0.0,
        useCache: false,
      );
      expect(withCache, withoutCache);
      expect(withCache.length, prompt.length + 5);
    });

    test('rejects 2D input (batched not supported)', () {
      final cfg = LlamaConfig(
        vocabSize: 16,
        maxCtx: 8,
        embedDim: 4,
        numLayers: 1,
        numHeads: 2,
        numKvHeads: 1,
        ffnDim: 8,
      );
      final m = Llama(cfg);
      final tokens = Tensor.fromList([2, 3], [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]);
      expect(() => m(tokens), throwsArgumentError);
    });

    test('rejects seqLen exceeding maxCtx', () {
      final cfg = LlamaConfig(
        vocabSize: 16,
        maxCtx: 4,
        embedDim: 4,
        numLayers: 1,
        numHeads: 2,
        numKvHeads: 1,
        ffnDim: 8,
      );
      final m = Llama(cfg);
      final tokens = Tensor.fromList([5], [0.0, 1.0, 2.0, 3.0, 0.0]);
      expect(() => m(tokens), throwsArgumentError);
    });

    test('parameter count includes final RMSNorm and no lm_head when tied', () {
      final cfg = LlamaConfig(
        vocabSize: 24,
        maxCtx: 8,
        embedDim: 8,
        numLayers: 1,
        numHeads: 2,
        numKvHeads: 1,
        ffnDim: 16,
      );
      final tied = Llama(cfg);
      final untied = Llama(
        LlamaConfig(
          vocabSize: 24,
          maxCtx: 8,
          embedDim: 8,
          numLayers: 1,
          numHeads: 2,
          numKvHeads: 1,
          ffnDim: 16,
          tieWeights: false,
        ),
      );
      // untied adds one Linear (weight only, bias=false)
      expect(untied.parameters().length, tied.parameters().length + 1);
    });
  });
}
