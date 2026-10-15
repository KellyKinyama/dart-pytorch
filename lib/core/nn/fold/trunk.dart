/// ESMFold **folding trunk** block (`TriangularSelfAttentionBlock`): the
/// Evoformer-style stack that refines a single representation `s` `[L, cS]`
/// and a pair representation `z` `[L, L, cZ]` before the structure module.
///
/// Faithfully mirrors `esm.esmfold.v1.tri_self_attn_block` and the OpenFold
/// triangle primitives it composes, so a pretrained checkpoint maps onto these
/// modules later:
///   sequence:  pair→sequence bias → gated self-attention → residue MLP
///   pair:      sequence→pair outer update → triangle mult (out/in) →
///              triangle self-attention (start/end) → residue MLP
///
/// Plain Dart (double precision, explicit loops) on top of [Dense] from
/// `structure_module.dart`. All operations are equivariant to a joint
/// permutation of the residue axis — the property the tests assert.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'structure_module.dart' show Dense;

double _sigmoid(double x) => 1.0 / (1.0 + math.exp(-x));

/// LayerNorm a single vector in place-free fashion → new list.
Float32List _layerNorm(
    Float32List x, int off, int d, Float32List gamma, Float32List beta,
    {double eps = 1e-5}) {
  var mean = 0.0;
  for (var i = 0; i < d; i++) {
    mean += x[off + i];
  }
  mean /= d;
  var v = 0.0;
  for (var i = 0; i < d; i++) {
    final dv = x[off + i] - mean;
    v += dv * dv;
  }
  v /= d;
  final inv = 1.0 / math.sqrt(v + eps);
  final out = Float32List(d);
  for (var i = 0; i < d; i++) {
    out[i] = (x[off + i] - mean) * inv * gamma[i] + beta[i];
  }
  return out;
}

/// LayerNorm every row of `[n, d]` → new `[n, d]`.
Float32List _layerNormRows(
    Float32List x, int n, int d, Float32List gamma, Float32List beta,
    {double eps = 1e-5}) {
  final out = Float32List(n * d);
  for (var r = 0; r < n; r++) {
    final v = _layerNorm(x, r * d, d, gamma, beta, eps: eps);
    for (var i = 0; i < d; i++) {
      out[r * d + i] = v[i];
    }
  }
  return out;
}

Float32List _ones(int d) => Float32List(d)..fillRange(0, d, 1.0);

/// Gated self-attention with an external pairwise bias
/// (`esm.esmfold.v1.misc.Attention`). Q/K/V share one projection.
class SeqAttention {
  final int cS, heads, headWidth;
  final Dense proj; // cS -> 3*cS, no bias
  final Dense oProj; // cS -> cS, bias
  final Dense gProj; // cS -> cS, bias (gate)
  final double rescale;

  SeqAttention(this.cS, this.heads, this.headWidth, this.proj, this.oProj,
      this.gProj)
      : rescale = 1.0 / math.sqrt(headWidth);

  factory SeqAttention.random(int cS, int heads, int headWidth, {int seed = 0}) {
    final g = Dense.random(cS, cS, seed: seed + 3);
    g.bias!.fillRange(0, cS, 1.0); // gating: open at init
    return SeqAttention(
      cS,
      heads,
      headWidth,
      Dense.random(cS, 3 * cS, seed: seed + 1, bias: false),
      Dense.random(cS, cS, seed: seed + 2),
      g,
    );
  }

  /// `x` is `[L, cS]`, `bias` is `[L, L, heads]`. Returns `[L, cS]`.
  Float32List call(Float32List x, int n, Float32List bias) {
    final t = proj.applyRows(x, n); // [L, 3*cS]
    final hw = headWidth;
    final block = 3 * hw; // per-head stride in t
    final y = Float32List(n * cS); // [L, heads*hw] laid out as (h hw)
    final logits = Float32List(n);
    for (var h = 0; h < heads; h++) {
      for (var qi = 0; qi < n; qi++) {
        final qo = qi * (3 * cS) + h * block; // q at offset +0
        for (var ki = 0; ki < n; ki++) {
          final ko = ki * (3 * cS) + h * block + hw; // k at offset +hw
          var dot = 0.0;
          for (var c = 0; c < hw; c++) {
            dot += t[qo + c] * t[ko + c];
          }
          logits[ki] = dot * rescale + bias[(qi * n + ki) * heads + h];
        }
        _softmax(logits, n);
        final yo = qi * cS + h * hw;
        for (var ki = 0; ki < n; ki++) {
          final vo = ki * (3 * cS) + h * block + 2 * hw; // v at offset +2hw
          final w = logits[ki];
          for (var c = 0; c < hw; c++) {
            y[yo + c] += w * t[vo + c];
          }
        }
      }
    }
    // Gate then output projection.
    final g = gProj.applyRows(x, n);
    for (var i = 0; i < y.length; i++) {
      y[i] *= _sigmoid(g[i]);
    }
    return oProj.applyRows(y, n);
  }
}

/// Outer-product-style update from the sequence track into the pair track.
class SequenceToPair {
  final int cS, inner, cZ;
  final Float32List lnGamma, lnBeta;
  final Dense proj; // cS -> 2*inner
  final Dense oProj; // 2*inner -> cZ

  SequenceToPair(this.cS, this.inner, this.cZ, this.lnGamma, this.lnBeta,
      this.proj, this.oProj);

  factory SequenceToPair.random(int cS, int inner, int cZ, {int seed = 0}) =>
      SequenceToPair(
        cS,
        inner,
        cZ,
        _ones(cS),
        Float32List(cS),
        Dense.random(cS, 2 * inner, seed: seed + 1),
        Dense.random(2 * inner, cZ, seed: seed + 2),
      );

  /// `s` is `[L, cS]`. Returns pair update `[L, L, cZ]`.
  Float32List call(Float32List s, int n) {
    final ln = _layerNormRows(s, n, cS, lnGamma, lnBeta);
    final p = proj.applyRows(ln, n); // [L, 2*inner]
    final cat = Float32List(n * n * 2 * inner);
    for (var i = 0; i < n; i++) {
      for (var j = 0; j < n; j++) {
        final o = (i * n + j) * 2 * inner;
        final qj = j * 2 * inner; // q = first `inner` of row j
        final ki = i * 2 * inner + inner; // k = last `inner` of row i
        for (var c = 0; c < inner; c++) {
          final q = p[qj + c], k = p[ki + c];
          cat[o + c] = q * k; // prod[i,j] = q[j] * k[i]
          cat[o + inner + c] = q - k; // diff[i,j] = q[j] - k[i]
        }
      }
    }
    return oProj.applyRows(cat, n * n);
  }
}

/// Produces a per-head attention bias `[L, L, heads]` from the pair track.
class PairToSequence {
  final int cZ, heads;
  final Float32List lnGamma, lnBeta;
  final Dense linear; // cZ -> heads, no bias

  PairToSequence(this.cZ, this.heads, this.lnGamma, this.lnBeta, this.linear);

  factory PairToSequence.random(int cZ, int heads, {int seed = 0}) =>
      PairToSequence(cZ, heads, _ones(cZ), Float32List(cZ),
          Dense.random(cZ, heads, seed: seed, bias: false));

  Float32List call(Float32List z, int n) {
    final ln = _layerNormRows(z, n * n, cZ, lnGamma, lnBeta);
    return linear.applyRows(ln, n * n);
  }
}

/// `x + MLP(LayerNorm(x))` applied row-wise.
class ResidueMLP {
  final int dim, inner;
  final Float32List lnGamma, lnBeta;
  final Dense l1, l2;

  ResidueMLP(this.dim, this.inner, this.lnGamma, this.lnBeta, this.l1, this.l2);

  factory ResidueMLP.random(int dim, int inner, {int seed = 0}) => ResidueMLP(
        dim,
        inner,
        _ones(dim),
        Float32List(dim),
        Dense.random(dim, inner, seed: seed + 1),
        Dense.random(inner, dim, seed: seed + 2),
      );

  Float32List call(Float32List x, int n) {
    final ln = _layerNormRows(x, n, dim, lnGamma, lnBeta);
    final h = l1.applyRows(ln, n);
    for (var i = 0; i < h.length; i++) {
      if (h[i] < 0) h[i] = 0;
    }
    final o = l2.applyRows(h, n);
    final out = Float32List(n * dim);
    for (var i = 0; i < out.length; i++) {
      out[i] = x[i] + o[i];
    }
    return out;
  }
}

/// Triangle multiplicative update (OpenFold), outgoing or incoming edges.
class TriangleMultiplication {
  final int cZ;
  final bool outgoing;
  final Float32List lnInG, lnInB, lnOutG, lnOutB;
  final Dense aP, aG, bP, bG, g, z; // projections

  TriangleMultiplication(this.cZ, this.outgoing, this.lnInG, this.lnInB,
      this.lnOutG, this.lnOutB, this.aP, this.aG, this.bP, this.bG, this.g,
      this.z);

  factory TriangleMultiplication.random(int cZ, bool outgoing, {int seed = 0}) =>
      TriangleMultiplication(
        cZ,
        outgoing,
        _ones(cZ), Float32List(cZ),
        _ones(cZ), Float32List(cZ),
        Dense.random(cZ, cZ, seed: seed + 1),
        Dense.random(cZ, cZ, seed: seed + 2),
        Dense.random(cZ, cZ, seed: seed + 3),
        Dense.random(cZ, cZ, seed: seed + 4),
        Dense.random(cZ, cZ, seed: seed + 5),
        Dense.random(cZ, cZ, seed: seed + 6),
      );

  Float32List call(Float32List pair, int n) {
    final zln = _layerNormRows(pair, n * n, cZ, lnInG, lnInB);
    // a, b gated projections over all pairs.
    final a = Float32List(n * n * cZ);
    final b = Float32List(n * n * cZ);
    final ap = aP.applyRows(zln, n * n);
    final ag = aG.applyRows(zln, n * n);
    final bp = bP.applyRows(zln, n * n);
    final bg = bG.applyRows(zln, n * n);
    for (var i = 0; i < n * n * cZ; i++) {
      a[i] = ap[i] * _sigmoid(ag[i]);
      b[i] = bp[i] * _sigmoid(bg[i]);
    }
    // Combine over the shared index k.
    final x = Float32List(n * n * cZ);
    for (var i = 0; i < n; i++) {
      for (var j = 0; j < n; j++) {
        final o = (i * n + j) * cZ;
        for (var k = 0; k < n; k++) {
          final ao = outgoing ? (i * n + k) * cZ : (k * n + i) * cZ;
          final bo = outgoing ? (j * n + k) * cZ : (k * n + j) * cZ;
          for (var c = 0; c < cZ; c++) {
            x[o + c] += a[ao + c] * b[bo + c];
          }
        }
      }
    }
    // Normalize, project, gate.
    final gate = g.applyRows(zln, n * n);
    final xln = _layerNormRows(x, n * n, cZ, lnOutG, lnOutB);
    final proj = z.applyRows(xln, n * n);
    final out = Float32List(n * n * cZ);
    for (var i = 0; i < out.length; i++) {
      out[i] = proj[i] * _sigmoid(gate[i]);
    }
    return out;
  }
}

/// Triangle self-attention (OpenFold), starting-node or ending-node.
class TriangleAttention {
  final int cZ, heads, headWidth;
  final bool starting;
  final Float32List lnG, lnB;
  final Dense linearBias; // cZ -> heads, no bias
  final Dense q, k, v, o; // o has bias
  final Dense gate; // cZ -> cZ, gate
  final double rescale;

  TriangleAttention(this.cZ, this.heads, this.headWidth, this.starting,
      this.lnG, this.lnB, this.linearBias, this.q, this.k, this.v, this.o,
      this.gate)
      : rescale = 1.0 / math.sqrt(headWidth);

  factory TriangleAttention.random(int cZ, int heads, int headWidth,
      bool starting,
      {int seed = 0}) {
    final gt = Dense.random(cZ, cZ, seed: seed + 6);
    gt.bias!.fillRange(0, cZ, 1.0);
    return TriangleAttention(
      cZ,
      heads,
      headWidth,
      starting,
      _ones(cZ),
      Float32List(cZ),
      Dense.random(cZ, heads, seed: seed + 1, bias: false),
      Dense.random(cZ, cZ, seed: seed + 2, bias: false),
      Dense.random(cZ, cZ, seed: seed + 3, bias: false),
      Dense.random(cZ, cZ, seed: seed + 4, bias: false),
      Dense.random(cZ, cZ, seed: seed + 5),
      gt,
    );
  }

  Float32List call(Float32List pair, int n) {
    // Ending-node attention operates on the transpose of the pair matrix.
    final z = starting ? pair : _transpose(pair, n, cZ);
    final zln = _layerNormRows(z, n * n, cZ, lnG, lnB);
    final bias = linearBias.applyRows(zln, n * n); // [n,n,heads]
    final qp = q.applyRows(zln, n * n);
    final kp = k.applyRows(zln, n * n);
    final vp = v.applyRows(zln, n * n);
    final gp = gate.applyRows(zln, n * n);

    final hw = headWidth;
    final out = Float32List(n * n * cZ);
    final logits = Float32List(n);
    for (var row = 0; row < n; row++) {
      for (var h = 0; h < heads; h++) {
        for (var qi = 0; qi < n; qi++) {
          final qo = (row * n + qi) * cZ + h * hw;
          for (var ki = 0; ki < n; ki++) {
            final ko = (row * n + ki) * cZ + h * hw;
            var dot = 0.0;
            for (var c = 0; c < hw; c++) {
              dot += qp[qo + c] * kp[ko + c];
            }
            // Triangle bias b[qi, ki, h] shared across the row batch.
            logits[ki] = dot * rescale + bias[(qi * n + ki) * heads + h];
          }
          _softmax(logits, n);
          final oo = (row * n + qi) * cZ + h * hw;
          for (var ki = 0; ki < n; ki++) {
            final vo = (row * n + ki) * cZ + h * hw;
            final w = logits[ki];
            for (var c = 0; c < hw; c++) {
              out[oo + c] += w * vp[vo + c];
            }
          }
        }
      }
    }
    // Gate and output projection per pair.
    for (var i = 0; i < out.length; i++) {
      out[i] *= _sigmoid(gp[i]);
    }
    final proj = o.applyRows(out, n * n);
    return starting ? proj : _transpose(proj, n, cZ);
  }
}

Float32List _transpose(Float32List x, int n, int c) {
  final out = Float32List(n * n * c);
  for (var i = 0; i < n; i++) {
    for (var j = 0; j < n; j++) {
      final src = (i * n + j) * c;
      final dst = (j * n + i) * c;
      for (var k = 0; k < c; k++) {
        out[dst + k] = x[src + k];
      }
    }
  }
  return out;
}

void _softmax(Float32List a, int n) {
  var mx = a[0];
  for (var j = 1; j < n; j++) {
    if (a[j] > mx) mx = a[j];
  }
  var sum = 0.0;
  for (var j = 0; j < n; j++) {
    final e = math.exp(a[j] - mx);
    a[j] = e;
    sum += e;
  }
  final inv = 1.0 / sum;
  for (var j = 0; j < n; j++) {
    a[j] *= inv;
  }
}

/// Configuration for one trunk block / the trunk stack.
class FoldingTrunkConfig {
  final int cS; // sequence_state_dim
  final int cZ; // pairwise_state_dim
  final int seqHeadWidth;
  final int pairHeadWidth;

  const FoldingTrunkConfig({
    this.cS = 1024,
    this.cZ = 128,
    this.seqHeadWidth = 32,
    this.pairHeadWidth = 32,
  });

  int get seqHeads => cS ~/ seqHeadWidth;
  int get pairHeads => cZ ~/ pairHeadWidth;
}

/// One `TriangularSelfAttentionBlock`: refines `(s, z)` and returns the
/// updated pair. The sequence update is applied in place to [s].
class TriangularSelfAttentionBlock {
  final FoldingTrunkConfig cfg;
  final Float32List ln1G, ln1B; // layernorm_1 (sequence)
  final SeqAttention seqAttention;
  final SequenceToPair sequenceToPair;
  final PairToSequence pairToSequence;
  final TriangleMultiplication triMulOut, triMulIn;
  final TriangleAttention triAttStart, triAttEnd;
  final ResidueMLP mlpSeq, mlpPair;

  TriangularSelfAttentionBlock(
    this.cfg,
    this.ln1G,
    this.ln1B,
    this.seqAttention,
    this.sequenceToPair,
    this.pairToSequence,
    this.triMulOut,
    this.triMulIn,
    this.triAttStart,
    this.triAttEnd,
    this.mlpSeq,
    this.mlpPair,
  );

  factory TriangularSelfAttentionBlock.random(FoldingTrunkConfig cfg,
      {int seed = 0}) {
    return TriangularSelfAttentionBlock(
      cfg,
      _ones(cfg.cS),
      Float32List(cfg.cS),
      SeqAttention.random(cfg.cS, cfg.seqHeads, cfg.seqHeadWidth, seed: seed + 1),
      SequenceToPair.random(cfg.cS, cfg.cZ ~/ 2, cfg.cZ, seed: seed + 2),
      PairToSequence.random(cfg.cZ, cfg.seqHeads, seed: seed + 3),
      TriangleMultiplication.random(cfg.cZ, true, seed: seed + 4),
      TriangleMultiplication.random(cfg.cZ, false, seed: seed + 5),
      TriangleAttention.random(cfg.cZ, cfg.pairHeads, cfg.pairHeadWidth, true,
          seed: seed + 6),
      TriangleAttention.random(cfg.cZ, cfg.pairHeads, cfg.pairHeadWidth, false,
          seed: seed + 7),
      ResidueMLP.random(cfg.cS, 4 * cfg.cS, seed: seed + 8),
      ResidueMLP.random(cfg.cZ, 4 * cfg.cZ, seed: seed + 9),
    );
  }

  /// Returns updated `(s, z)`.
  ({Float32List s, Float32List z}) call(Float32List s, Float32List z, int n) {
    // --- sequence track ---
    final bias = pairToSequence.call(z, n); // [n,n,seqHeads]
    final yln = _layerNormRows(s, n, cfg.cS, ln1G, ln1B);
    final y = seqAttention.call(yln, n, bias);
    var sOut = Float32List(n * cfg.cS);
    for (var i = 0; i < sOut.length; i++) {
      sOut[i] = s[i] + y[i];
    }
    sOut = mlpSeq.call(sOut, n);

    // --- pair track ---
    final pairUpd = sequenceToPair.call(sOut, n); // [n,n,cZ]
    var zOut = Float32List(n * n * cfg.cZ);
    for (var i = 0; i < zOut.length; i++) {
      zOut[i] = z[i] + pairUpd[i];
    }
    void addInPlace(Float32List upd) {
      for (var i = 0; i < zOut.length; i++) {
        zOut[i] += upd[i];
      }
    }

    addInPlace(triMulOut.call(zOut, n));
    addInPlace(triMulIn.call(zOut, n));
    addInPlace(triAttStart.call(zOut, n));
    addInPlace(triAttEnd.call(zOut, n));
    zOut = mlpPair.call(zOut, n * n);

    return (s: sOut, z: zOut);
  }
}
