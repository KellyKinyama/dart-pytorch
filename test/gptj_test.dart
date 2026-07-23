/// Tests for GPT-J.
///
/// The riskiest bit of the GPT-J port is the rotary-embedding
/// convention: HF GPT-J uses **interleaved-pair** rotary, our
/// [RopeCache] uses **half-split**. The loader converts by
/// permuting the first `rotary_dim` output rows of q_proj and
/// k_proj per head. This test validates that the two rotary
/// flavours produce identical `Q @ K^T` scores after the
/// permutation trick.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

/// Reference GPT-J-style interleaved-pair rotary applied to a
/// `[N, headDim]` matrix. Only the first `rDim` columns get
/// rotated; the tail is passed through. This matches HF's
/// `modeling_gptj.py::apply_rotary_pos_emb`.
List<double> _interleavedRotary(
  List<double> q, // flat row-major [N, headDim]
  int n,
  int headDim,
  int rDim, {
  double base = 10000.0,
}) {
  final halfR = rDim ~/ 2;
  final invFreq = List<double>.generate(
    halfR,
    (i) => 1.0 / math.pow(base, 2.0 * i / rDim),
  );
  final out = Float64List(n * headDim);
  for (int m = 0; m < n; m++) {
    final base0 = m * headDim;
    for (int i = 0; i < halfR; i++) {
      final ang = m * invFreq[i];
      final c = math.cos(ang);
      final s = math.sin(ang);
      final q0 = q[base0 + 2 * i];
      final q1 = q[base0 + 2 * i + 1];
      out[base0 + 2 * i] = q0 * c - q1 * s;
      out[base0 + 2 * i + 1] = q1 * c + q0 * s;
    }
    for (int j = rDim; j < headDim; j++) {
      out[base0 + j] = q[base0 + j];
    }
  }
  return out.toList();
}

/// Apply the loader's permutation to a `[N, headDim]` row-major
/// vector — same rule as [GPTJHFLoader._permuteRotaryRows] but
/// applied along the *column* (head-dim) axis.
List<double> _permuteHeadDim(List<double> q, int n, int headDim, int rDim) {
  final halfR = rDim ~/ 2;
  final out = Float64List(n * headDim);
  for (int m = 0; m < n; m++) {
    final base0 = m * headDim;
    for (int i = 0; i < halfR; i++) {
      out[base0 + i] = q[base0 + 2 * i];
      out[base0 + i + halfR] = q[base0 + 2 * i + 1];
    }
    for (int j = rDim; j < headDim; j++) {
      out[base0 + j] = q[base0 + j];
    }
  }
  return out.toList();
}

void main() {
  group('GPT-J rotary permutation', () {
    test(
      'half-split(π(q)) @ half-split(π(k))^T == interleaved(q) @ interleaved(k)^T',
      () {
        const headDim = 8;
        const rDim = 4; // rotate first 4 dims, pass through last 4
        const n = 3;
        final rng = math.Random(7);
        final qRaw = List<double>.generate(
          n * headDim,
          (_) => rng.nextDouble() * 2 - 1,
        );
        final kRaw = List<double>.generate(
          n * headDim,
          (_) => rng.nextDouble() * 2 - 1,
        );

        // Reference: GPT-J interleaved rotary.
        final qJ = _interleavedRotary(qRaw, n, headDim, rDim);
        final kJ = _interleavedRotary(kRaw, n, headDim, rDim);
        // Ours: permute first, then half-split rotary (via RopeCache).
        final qPerm = _permuteHeadDim(qRaw, n, headDim, rDim);
        final kPerm = _permuteHeadDim(kRaw, n, headDim, rDim);
        final rope = RopeCache(
          maxCtx: n,
          headDim: headDim,
          rotaryDim: rDim,
        );
        final qTens = Tensor.fromList([n, headDim], qPerm);
        final kTens = Tensor.fromList([n, headDim], kPerm);
        final qRot = rope.apply(qTens).toList();
        final kRot = rope.apply(kTens).toList();

        // Compare score matrices Q @ K^T (n x n).
        List<double> scores(List<double> Q, List<double> K) {
          final s = Float64List(n * n);
          for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
              var acc = 0.0;
              for (int d = 0; d < headDim; d++) {
                acc += Q[i * headDim + d] * K[j * headDim + d];
              }
              s[i * n + j] = acc;
            }
          }
          return s.toList();
        }

        final sJ = scores(qJ, kJ);
        final sHalf = scores(qRot, kRot);
        for (int i = 0; i < n * n; i++) {
          expect((sJ[i] - sHalf[i]).abs(), lessThan(1e-6),
              reason: 'score[$i] mismatch: GPT-J=${sJ[i]}, ours=${sHalf[i]}');
        }
      },
    );
  });

  group('GPT-J model', () {
    test('forward on tiny config gives shape [seqLen, vocabSize]', () {
      final cfg = GPTJConfig(
        vocabSize: 32,
        maxCtx: 16,
        embedDim: 16,
        numLayers: 2,
        numHeads: 2, // headDim=8
        rotaryDim: 4,
        seed: 3,
      );
      final m = GPTJModel(cfg);
      final tokens = Tensor.fromList([5], [0, 1, 2, 3, 4]);
      final logits = m(tokens);
      expect(logits.shape, [5, 32]);
    });

    test('gpt-j-6b config presets are sane', () {
      final c = GPTJHFLoader.gptJ6bConfig();
      expect(c.vocabSize, 50400);
      expect(c.maxCtx, 2048);
      expect(c.embedDim, 4096);
      expect(c.numLayers, 28);
      expect(c.numHeads, 16);
      expect(c.rotaryDim, 64);
      expect(c.ffnDim, 16384);
    });

    test('greedy generate with useCache is deterministic', () {
      final cfg = GPTJConfig(
        vocabSize: 16,
        maxCtx: 12,
        embedDim: 8,
        numLayers: 1,
        numHeads: 2,
        rotaryDim: 2,
        seed: 42,
      );
      final m = GPTJModel(cfg);
      final a = m.generate(
        [1.0, 2.0, 3.0],
        maxNewTokens: 4,
        temperature: 0.0,
      );
      final b = m.generate(
        [1.0, 2.0, 3.0],
        maxNewTokens: 4,
        temperature: 0.0,
      );
      expect(a, b);
      expect(a.length, 7);
    });
  });
}
