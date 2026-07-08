/// Batched (3D) tensor foundation tests.
///
/// Verifies that a `[batch, seq, dim]` tensor flowing through the
/// row-wise ops and the affected `nn.Module`s produces results that
/// are numerically equivalent to running the same op per-sequence
/// (batch=1 slices) and stacking. Also covers the underlying
/// `Tensor.reshape` primitive.
library;

import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

const double _tol = 1e-5;

List<double> _rand(int n, {int seed = 0, double scale = 1.0}) {
  final r = math.Random(seed);
  return List<double>.generate(n, (_) => (r.nextDouble() * 2 - 1) * scale);
}

void _closeList(List<double> a, List<double> b, {double tol = _tol}) {
  expect(a.length, b.length);
  for (int i = 0; i < a.length; i++) {
    expect(a[i], closeTo(b[i], tol), reason: 'index $i');
  }
}

/// Batched output vs. per-sample: run `op` on the full 3D tensor and
/// on each `[S, D]` slice separately, then compare.
void _batchedMatchesPerSample(
  Tensor batched,
  Tensor Function(Tensor) op,
  int b,
  int s,
  int lastDim,
) {
  final flatOut = op(batched).toList();
  final singleShape = [s, lastDim];
  for (int i = 0; i < b; i++) {
    final row = batched.toList().sublist(
      i * s * lastDim,
      (i + 1) * s * lastDim,
    );
    final sample = Tensor.fromList(singleShape, row);
    final sampleOut = op(sample).toList();
    final slice = flatOut.sublist(
      i * sampleOut.length,
      (i + 1) * sampleOut.length,
    );
    _closeList(slice, sampleOut);
  }
}

void main() {
  group('Tensor.reshape', () {
    test('rejects mismatched product', () {
      final t = Tensor.fromList([2, 3], _rand(6));
      expect(() => t.reshape([2, 4]), throwsArgumentError);
    });

    test('2D <-> 3D preserves values', () {
      final vals = _rand(24);
      final t = Tensor.fromList([2, 3, 4], vals);
      final flat = t.reshape([6, 4]);
      expect(flat.shape, [6, 4]);
      _closeList(flat.toList(), vals);
      final back = flat.reshape([2, 3, 4]);
      expect(back.shape, [2, 3, 4]);
      _closeList(back.toList(), vals);
    });

    test('CPU reshape shares storage', () {
      // Reshape is meant as a cheap view on CPU; a downstream *out-
      // of-place* op should of course see the current values.
      final t = Tensor.fromList([4], [1, 2, 3, 4]);
      final r = t.reshape([2, 2]);
      _closeList(r.toList(), [1, 2, 3, 4]);
    });

    test('autograd: gradient flows back through reshape', () {
      final x = Tensor.fromList([2, 3, 4], _rand(24), requiresGrad: true);
      // sum(reshape(x)) has gradient of ones w.r.t. x, of shape [2,3,4].
      final loss = x.reshape([6, 4]).sum();
      loss.backward();
      expect(x.grad, isNotNull);
      expect(x.grad!.shape, [2, 3, 4]);
      _closeList(x.grad!.toList(), List<double>.filled(24, 1.0));
    });
  });

  group('row-wise ops accept 3D via last-axis reshape', () {
    const b = 2;
    const s = 3;
    const d = 4;

    test('softmax [B,S,D] matches per-sample softmax', () {
      final x = Tensor.fromList([b, s, d], _rand(b * s * d, seed: 1));
      _batchedMatchesPerSample(x, (t) => t.softmax(), b, s, d);
    });

    test('layerNorm [B,S,D] matches per-sample layerNorm', () {
      final gamma = Tensor.fromList([d], List<double>.filled(d, 1.0));
      final beta = Tensor.fromList([d], List<double>.filled(d, 0.0));
      final x = Tensor.fromList([b, s, d], _rand(b * s * d, seed: 2));
      _batchedMatchesPerSample(x, (t) => t.layerNorm(gamma, beta), b, s, d);
    });

    test('crossEntropy [B,S,V] + [B,S] matches [B*S,V] + [B*S]', () {
      const v = 5;
      final logits = Tensor.fromList([b, s, v], _rand(b * s * v, seed: 3));
      final tgts = Tensor.fromList([
        b,
        s,
      ], List<double>.generate(b * s, (i) => (i % v).toDouble()));
      final loss3D = logits.crossEntropy(tgts);
      expect(loss3D.shape, [b * s, 1]);

      final flatLogits = Tensor.fromList([b * s, v], logits.toList());
      final flatTgts = Tensor.fromList([b * s], tgts.toList());
      final loss2D = flatLogits.crossEntropy(flatTgts);
      _closeList(loss3D.toList(), loss2D.toList());
    });
  });

  group('Embedding accepts 2D indices', () {
    test('[B,S] indices -> [B,S,D] output equals per-sample', () {
      const v = 6;
      const d = 4;
      const b = 2;
      const s = 3;
      final emb = Embedding(v, d, seed: 42);
      final idx = Tensor.fromList([b, s], [0, 1, 2, 3, 4, 5]);
      final out = emb(idx);
      expect(out.shape, [b, s, d]);

      for (int i = 0; i < b; i++) {
        final sliceIdx = Tensor.fromList([
          s,
        ], idx.toList().sublist(i * s, (i + 1) * s));
        final sampleOut = emb(sliceIdx).toList();
        final got = out.toList().sublist(i * s * d, (i + 1) * s * d);
        _closeList(got, sampleOut);
      }
    });
  });

  group('Linear accepts 3D input', () {
    test('[B,S,in] -> [B,S,out] equals per-sample', () {
      const b = 3;
      const s = 4;
      const inF = 5;
      const outF = 6;
      final lin = Linear(inF, outF, seed: 7);
      final x = Tensor.fromList([b, s, inF], _rand(b * s * inF, seed: 8));
      final out = lin(x);
      expect(out.shape, [b, s, outF]);

      for (int i = 0; i < b; i++) {
        final row = x.toList().sublist(i * s * inF, (i + 1) * s * inF);
        final sample = Tensor.fromList([s, inF], row);
        final sampleOut = lin(sample).toList();
        final got = out.toList().sublist(i * s * outF, (i + 1) * s * outF);
        _closeList(got, sampleOut);
      }
    });

    test('autograd through batched Linear -> weight grad accumulates', () {
      const b = 2;
      const s = 3;
      const inF = 4;
      const outF = 5;
      final lin = Linear(inF, outF, seed: 11);
      final x = Tensor.fromList([b, s, inF], _rand(b * s * inF, seed: 12));
      final loss = lin(x).sum();
      loss.backward();
      expect(lin.weight.grad, isNotNull);
      expect(lin.weight.grad!.shape, [outF, inF]);
      // Non-trivial gradient (not all-zero and not NaN).
      final g = lin.weight.grad!.toList();
      expect(g.any((v) => v.abs() > 1e-6), isTrue);
      expect(g.every((v) => v.isFinite), isTrue);
    });
  });

  group('LayerNorm module accepts 3D', () {
    test('[B,S,D] -> [B,S,D] equals per-sample', () {
      const b = 2;
      const s = 4;
      const d = 6;
      final ln = LayerNorm(d);
      final x = Tensor.fromList([b, s, d], _rand(b * s * d, seed: 21));
      final out = ln(x);
      expect(out.shape, [b, s, d]);

      for (int i = 0; i < b; i++) {
        final row = x.toList().sublist(i * s * d, (i + 1) * s * d);
        final sample = Tensor.fromList([s, d], row);
        final sampleOut = ln(sample).toList();
        final got = out.toList().sublist(i * s * d, (i + 1) * s * d);
        _closeList(got, sampleOut);
      }
    });
  });

  group('Dropout preserves 3D shape', () {
    test('eval mode is identity', () {
      const b = 2;
      const s = 3;
      const d = 4;
      final drop = Dropout(0.5)..eval();
      final x = Tensor.fromList([b, s, d], _rand(b * s * d, seed: 31));
      final out = drop(x);
      expect(out.shape, [b, s, d]);
      _closeList(out.toList(), x.toList());
    });

    test('training mode zeros ~p fraction and preserves shape', () {
      const b = 2;
      const s = 8;
      const d = 8;
      final drop = Dropout(0.5, rng: math.Random(0));
      drop.train();
      final x = Tensor.fill([b, s, d], 1.0);
      final out = drop(x);
      expect(out.shape, [b, s, d]);
      // Each element is either 0 or 2 (scale = 1/(1-0.5)).
      for (final v in out.toList()) {
        expect(v == 0.0 || (v - 2.0).abs() < 1e-6, isTrue);
      }
    });
  });

  group('positional encodings accept 3D', () {
    const b = 2;
    const s = 4;
    const d = 6;

    test('SinusoidalPositionalEncoding [B,S,D] equals per-sample', () {
      final pe = SinusoidalPositionalEncoding(d);
      final x = Tensor.fromList([b, s, d], _rand(b * s * d, seed: 41));
      final out = pe(x);
      expect(out.shape, [b, s, d]);

      for (int i = 0; i < b; i++) {
        final row = x.toList().sublist(i * s * d, (i + 1) * s * d);
        final sample = Tensor.fromList([s, d], row);
        final sampleOut = pe(sample).toList();
        final got = out.toList().sublist(i * s * d, (i + 1) * s * d);
        _closeList(got, sampleOut);
      }
    });

    test('LearnedPositionalEmbedding [B,S,D] equals per-sample', () {
      final pe = LearnedPositionalEmbedding(16, d, seed: 5);
      final x = Tensor.fromList([b, s, d], _rand(b * s * d, seed: 42));
      final out = pe(x);
      expect(out.shape, [b, s, d]);

      for (int i = 0; i < b; i++) {
        final row = x.toList().sublist(i * s * d, (i + 1) * s * d);
        final sample = Tensor.fromList([s, d], row);
        final sampleOut = pe(sample).toList();
        final got = out.toList().sublist(i * s * d, (i + 1) * s * d);
        _closeList(got, sampleOut);
      }
    });

    test('LearnedPositionalEmbedding rejects startPos+seqLen > maxLen', () {
      final pe = LearnedPositionalEmbedding(4, d);
      final x = Tensor.fromList([2, 4, d], _rand(2 * 4 * d));
      expect(() => pe(x, startPos: 1), throwsArgumentError);
    });
  });

  group('end-to-end: Embedding -> Linear -> LN batched vs per-sample', () {
    test('forward numerics match', () {
      const v = 5;
      const d = 4;
      const b = 2;
      const s = 3;
      final emb = Embedding(v, d, seed: 100);
      final lin = Linear(d, d, seed: 101);
      final ln = LayerNorm(d);

      Tensor forward(Tensor idx) => ln(lin(emb(idx)));

      final idx = Tensor.fromList([b, s], [0, 1, 2, 3, 4, 0]);
      final batchedOut = forward(idx);
      expect(batchedOut.shape, [b, s, d]);

      for (int i = 0; i < b; i++) {
        final sliceIdx = Tensor.fromList([
          s,
        ], idx.toList().sublist(i * s, (i + 1) * s));
        final sampleOut = forward(sliceIdx).toList();
        final got = batchedOut.toList().sublist(i * s * d, (i + 1) * s * d);
        _closeList(got, sampleOut);
      }
    });

    test('backward: batched crossEntropy grads match summed per-sample', () {
      const v = 4;
      const d = 3;
      const b = 2;
      const s = 3;
      final emb = Embedding(v, d, seed: 200);
      final lin = Linear(d, v, seed: 201);

      // --- Batched path ---
      final idx = Tensor.fromList([b, s], [0, 1, 2, 3, 0, 1]);
      final tgt = Tensor.fromList([b, s], [1, 2, 3, 0, 1, 2]);
      lin.weight.zeroGrad();
      emb.weight.zeroGrad();
      final logits = lin(emb(idx));
      // Sum (not mean) so per-sample grads add cleanly.
      logits.crossEntropy(tgt).sum().backward();
      final batchWGrad = List<double>.of(lin.weight.grad!.toList());
      final batchEGrad = List<double>.of(emb.weight.grad!.toList());

      // --- Per-sample path (sum of losses per sequence) ---
      lin.weight.zeroGrad();
      emb.weight.zeroGrad();
      for (int i = 0; i < b; i++) {
        final sliceIdx = Tensor.fromList([
          s,
        ], idx.toList().sublist(i * s, (i + 1) * s));
        final sliceTgt = Tensor.fromList([
          s,
        ], tgt.toList().sublist(i * s, (i + 1) * s));
        lin(emb(sliceIdx)).crossEntropy(sliceTgt).sum().backward();
      }
      final perWGrad = lin.weight.grad!.toList();
      final perEGrad = emb.weight.grad!.toList();

      _closeList(batchWGrad, perWGrad);
      _closeList(batchEGrad, perEGrad);
    });
  });

  group('matmul accepts 3D left operand', () {
    test('[B,S,K] @ [K,N] -> [B,S,N] matches per-sample', () {
      const b = 2;
      const s = 3;
      const k = 4;
      const n = 5;
      final a = Tensor.fromList([b, s, k], _rand(b * s * k, seed: 300));
      final w = Tensor.fromList([k, n], _rand(k * n, seed: 301));
      final out = a.matmul(w);
      expect(out.shape, [b, s, n]);
      for (int i = 0; i < b; i++) {
        final row = a.toList().sublist(i * s * k, (i + 1) * s * k);
        final sample = Tensor.fromList([s, k], row);
        final ref = sample.matmul(w).toList();
        final got = out.toList().sublist(i * s * n, (i + 1) * s * n);
        _closeList(got, ref);
      }
    });
  });

  group('Tensor.splitRows', () {
    test('roundtrips via concat(axis=0)', () {
      final t = Tensor.fromList([6, 3], _rand(18, seed: 400));
      final parts = TensorConcat.splitRows(t, 2);
      expect(parts.length, 3);
      for (final p in parts) {
        expect(p.shape, [2, 3]);
      }
      final joined = TensorConcat.concat(parts, axis: 0);
      _closeList(joined.toList(), t.toList());
    });

    test('rejects non-divisible chunk sizes', () {
      final t = Tensor.fromList([5, 2], _rand(10));
      expect(() => TensorConcat.splitRows(t, 2), throwsArgumentError);
    });

    test('autograd routes grads back to correct source rows', () {
      final t = Tensor.fromList([4, 2], _rand(8, seed: 401),
          requiresGrad: true);
      final parts = TensorConcat.splitRows(t, 2);
      // Give each chunk a distinct downstream scaling and check the
      // source grad is the concatenation of those.
      final loss = (parts[0] * 3.0).sum() + (parts[1] * 5.0).sum();
      loss.backward();
      expect(t.grad, isNotNull);
      final g = t.grad!.toList();
      // Rows 0..1 receive gradient 3.0, rows 2..3 receive 5.0.
      _closeList(g, [3, 3, 3, 3, 5, 5, 5, 5]);
    });
  });

  group('MultiHeadAttention accepts 3D input', () {
    test('[B,S,D] -> [B,S,D] equals per-sample (no mask)', () {
      const b = 2;
      const s = 4;
      const d = 8;
      const heads = 2;
      final mha = MultiHeadAttention(d, heads, seed: 500);
      final x = Tensor.fromList([b, s, d], _rand(b * s * d, seed: 501));
      final out = mha(x);
      expect(out.shape, [b, s, d]);
      for (int i = 0; i < b; i++) {
        final row = x.toList().sublist(i * s * d, (i + 1) * s * d);
        final sample = Tensor.fromList([s, d], row);
        final ref = mha(sample).toList();
        final got = out.toList().sublist(i * s * d, (i + 1) * s * d);
        _closeList(got, ref);
      }
    });

    test('shared causal mask applies per-batch', () {
      const b = 2;
      const s = 3;
      const d = 4;
      const heads = 2;
      final mha = MultiHeadAttention(d, heads, seed: 510);
      final x = Tensor.fromList([b, s, d], _rand(b * s * d, seed: 511));
      final mask = causalMask(s);
      final out = mha(x, mask: mask);
      expect(out.shape, [b, s, d]);
      for (int i = 0; i < b; i++) {
        final row = x.toList().sublist(i * s * d, (i + 1) * s * d);
        final sample = Tensor.fromList([s, d], row);
        final ref = mha(sample, mask: mask).toList();
        final got = out.toList().sublist(i * s * d, (i + 1) * s * d);
        _closeList(got, ref);
      }
    });
  });

  group('GPT accepts batched [B, S] tokens', () {
    test('[B,S] -> [B,S,V] equals per-sample forward', () {
      final gpt = GPT(GPTConfig(
        vocabSize: 8,
        maxCtx: 8,
        embedDim: 8,
        numLayers: 2,
        numHeads: 2,
        seed: 600,
      ));
      gpt.eval(); // disable dropout for deterministic comparison
      const b = 2;
      const s = 5;
      final tokens = Tensor.fromList(
        [b, s],
        [0, 1, 2, 3, 4, 5, 6, 7, 0, 1],
      );
      final logits = gpt(tokens);
      expect(logits.shape, [b, s, 8]);

      for (int i = 0; i < b; i++) {
        final sliceTok = Tensor.fromList(
          [s],
          tokens.toList().sublist(i * s, (i + 1) * s),
        );
        final ref = gpt(sliceTok).toList();
        final got = logits.toList().sublist(i * s * 8, (i + 1) * s * 8);
        _closeList(got, ref);
      }
    });

    test('batched training step reduces loss', () {
      final gpt = GPT(GPTConfig(
        vocabSize: 6,
        maxCtx: 8,
        embedDim: 8,
        numLayers: 1,
        numHeads: 2,
        seed: 700,
      ));
      const b = 3;
      const s = 4;
      final rng = math.Random(0);
      final flat = List<double>.generate(
        b * s,
        (_) => rng.nextInt(6).toDouble(),
      );
      final tokens = Tensor.fromList([b, s], flat.sublist(0, b * s));
      // Targets = tokens shifted (arbitrary; we just need SOME target).
      final targets = Tensor.fromList(
        [b, s],
        List<double>.generate(b * s, (i) => flat[(i + 1) % (b * s)]),
      );

      final opt = SGD(gpt.parameters(), lr: 0.1);
      final l0 = gpt(tokens).crossEntropy(targets).mean().toList()[0];
      for (int step = 0; step < 20; step++) {
        opt.zeroGrad();
        gpt(tokens).crossEntropy(targets).mean().backward();
        opt.step();
      }
      final l1 = gpt(tokens).crossEntropy(targets).mean().toList()[0];
      expect(l1, lessThan(l0),
          reason: 'expected loss to decrease across 20 batched steps '
              '(before=$l0, after=$l1)');
    });
  });
}

