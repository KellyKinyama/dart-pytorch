import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

void main() {
  group('MHACache', () {
    test('empty cache reports seqLen 0 and null buffers', () {
      final c = MHACache.empty(3);
      expect(c.seqLen, 0);
      expect(c.k, [null, null, null]);
      expect(c.v, [null, null, null]);
    });

    test('append grows K/V along axis 0 and updates seqLen', () {
      final c = MHACache.empty(1);
      final k1 = Tensor.fromList([2, 3], [1, 2, 3, 4, 5, 6]);
      final k2 = Tensor.fromList([1, 3], [7, 8, 9]);
      final merged = c.appendK(0, k1);
      expect(merged.shape, [2, 3]);
      expect(c.seqLen, 2);
      final merged2 = c.appendK(0, k2);
      expect(merged2.shape, [3, 3]);
      expect(merged2.toList(), [1, 2, 3, 4, 5, 6, 7, 8, 9]);
      expect(c.seqLen, 3);
    });
  });

  group('EncoderCache', () {
    test('factory sizes to [numLayers, numKvHeads]', () {
      final ec = EncoderCache.empty(2, 4);
      expect(ec.layers.length, 2);
      expect(ec.layers[0].numKvHeads, 4);
      expect(ec.seqLen, 0);
    });
  });

  group('MultiHeadAttention with cache', () {
    test('empty cache + mask: matches non-cached MHA output', () {
      // Same weights, same input, cache-empty prompt-fill should
      // reproduce the classic (no-cache) forward exactly.
      final baseline = MultiHeadAttention(8, 2, seed: 1);
      final withCache = MultiHeadAttention(8, 2, seed: 1);
      final x = Tensor.fromList([
        4,
        8,
      ], List<double>.generate(32, (i) => math.sin(i * 0.31)));
      final mask = causalMask(4);
      final y1 = baseline(x, mask: mask).toList();
      final cache = MHACache.empty(2);
      final y2 = withCache(x, mask: mask, cache: cache).toList();
      for (int i = 0; i < y1.length; i++) {
        expect((y1[i] - y2[i]).abs() < 1e-5, isTrue);
      }
      // Cache is now populated with the K/V of all 4 positions.
      expect(cache.seqLen, 4);
    });

    test('rejects mask when appending to a non-empty cache', () {
      final mha = MultiHeadAttention(8, 2, seed: 2);
      final cache = MHACache.empty(2);
      final x1 = Tensor.fromList([
        3,
        8,
      ], List<double>.generate(24, (i) => math.cos(i * 0.2)));
      mha(x1, mask: causalMask(3), cache: cache); // fill cache
      final xNew = Tensor.fromList([
        1,
        8,
      ], List<double>.generate(8, (i) => i * 0.1));
      expect(
        () => mha(xNew, mask: causalMask(1), cache: cache),
        throwsArgumentError,
      );
    });
  });

  group('GPT KV-cache parity', () {
    test('_forward: cached step at position N equals uncached last-row', () {
      // Run a length-5 forward without cache; compare its last row to
      // the sequence of "fill first 4, then append 5th one" using the
      // cache. Both should give the same 5th-row logits.
      final gpt = GPT(
        GPTConfig(
          vocabSize: 6,
          maxCtx: 16,
          embedDim: 8,
          numLayers: 2,
          numHeads: 2,
          seed: 3,
        ),
      );
      gpt.eval();
      final seq = [0.0, 1.0, 2.0, 3.0, 4.0];
      final full = gpt(Tensor.fromList([seq.length], seq)).toList();
      final v = gpt.config.vocabSize;
      final lastRowNoCache = full.sublist((seq.length - 1) * v);

      // Cached path: fill with the first 4, then append the 5th.
      // Use two separate models with identical seeds so grads
      // and dropout RNG state can't diverge — actually same model
      // is fine since we're in eval() and dropout=0.
      final cache = EncoderCache.empty(
        gpt.config.numLayers,
        gpt.config.numHeads,
      );
      final fill = seq.sublist(0, 4);
      gpt.callForward(
        Tensor.fromList([fill.length], fill),
        startPos: 0,
        cache: cache,
      );
      final stepLogits = gpt
          .callForward(
            Tensor.fromList([1], [seq.last]),
            startPos: 4,
            cache: cache,
          )
          .toList();
      // stepLogits has shape [1, V] — it *is* the last row.
      for (int i = 0; i < v; i++) {
        expect(
          (lastRowNoCache[i] - stepLogits[i]).abs() < 1e-4,
          isTrue,
          reason:
              'mismatch at vocab index $i: no-cache=${lastRowNoCache[i]} '
              'cached=${stepLogits[i]}',
        );
      }
    });

    test('generate: useCache=true reproduces useCache=false (greedy)', () {
      final gpt = GPT(
        GPTConfig(
          vocabSize: 7,
          maxCtx: 32,
          embedDim: 8,
          numLayers: 2,
          numHeads: 2,
          seed: 4,
        ),
      );
      final prompt = [1.0, 3.0, 5.0];
      final greedy = gpt.generate(
        prompt,
        maxNewTokens: 12,
        temperature: 0.0,
        useCache: false,
      );
      final greedyCached = gpt.generate(
        prompt,
        maxNewTokens: 12,
        temperature: 0.0,
        useCache: true,
      );
      expect(greedyCached, greedy);
    });

    test(
      'generate: useCache=true reproduces useCache=false with fixed RNG',
      () {
        final gpt = GPT(
          GPTConfig(
            vocabSize: 6,
            maxCtx: 24,
            embedDim: 8,
            numLayers: 2,
            numHeads: 2,
            seed: 5,
          ),
        );
        final prompt = [0.0, 1.0];
        final noCache = gpt.generate(
          prompt,
          maxNewTokens: 8,
          temperature: 1.0,
          rng: math.Random(7),
          useCache: false,
        );
        final cached = gpt.generate(
          prompt,
          maxNewTokens: 8,
          temperature: 1.0,
          rng: math.Random(7),
          useCache: true,
        );
        expect(cached, noCache);
      },
    );

    test(
      'overfit-then-generate still reproduces the memorized continuation',
      () {
        // Train briefly, then check that cache-mode sampling matches
        // no-cache sampling on a real (overfit) model.
        final gpt = GPT(
          GPTConfig(
            vocabSize: 5,
            maxCtx: 16,
            embedDim: 8,
            numLayers: 2,
            numHeads: 2,
            seed: 6,
          ),
        );
        final opt = Adam(gpt.parameters(), lr: 0.05);
        final x = Tensor.fromList([8], [0, 1, 2, 3, 4, 0, 1, 2]);
        final y = Tensor.fromList([8], [1, 2, 3, 4, 0, 1, 2, 3]);
        for (int i = 0; i < 150; i++) {
          opt.zeroGrad();
          gpt(x).crossEntropy(y).mean().backward();
          opt.step();
        }
        final prompt = [0.0, 1.0, 2.0];
        final a = gpt.generate(
          prompt,
          maxNewTokens: 6,
          temperature: 0.0,
          useCache: false,
        );
        final b = gpt.generate(
          prompt,
          maxNewTokens: 6,
          temperature: 0.0,
          useCache: true,
        );
        expect(b, a);
      },
    );
  });
}

/// Test-only convenience shim — exposes the private `_forward` so we
/// can compare cached and non-cached forwards without going through
/// the sampler.
extension _GPTTestAccess on GPT {
  Tensor callForward(
    Tensor tokens, {
    required int startPos,
    EncoderCache? cache,
  }) {
    // Re-implement the tiny wrapper (identical to the private one)
    // so the test file doesn't need library-private access.
    var h = tokenEmb(tokens);
    h = posEmb(h, startPos: startPos);
    h = embedDrop(h);
    final n = tokens.shape[0];
    final mask = n > 1 ? causalMask(n, device: h.device) : null;
    h = encoder(h, mask: mask, cache: cache);
    return config.tieWeights
        ? h.matmul(tokenEmb.weight.transpose())
        : untiedHead!(h);
  }
}
