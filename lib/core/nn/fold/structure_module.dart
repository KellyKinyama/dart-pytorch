/// AlphaFold2 / ESMFold **structure module**: Invariant Point Attention (IPA),
/// the backbone-frame update, and the shared-weight block stack that turns a
/// single representation `s` and pair representation `z` into per-residue
/// rigid frames (and hence backbone atom coordinates).
///
/// The whole module is *equivariant*: applying a global rigid motion to the
/// input frames transforms the output coordinates by the same motion, while
/// the learned single representation `s` stays invariant. That property —
/// verified by this file's tests — is what lets ESMFold reason about geometry
/// without a canonical global orientation.
///
/// Implemented in plain Dart (double precision, explicit loops) on top of
/// [geometry.dart] so it is deterministic and unit-testable without weights.
/// It mirrors OpenFold's `InvariantPointAttention` / `StructureModule` tensor
/// layouts so a pretrained ESMFold checkpoint can be loaded later.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'geometry.dart';

/// A dense layer `y = W·x + b`, weight stored row-major as `[outDim, inDim]`.
class Dense {
  final int inDim, outDim;
  final Float32List weight; // [outDim * inDim]
  final Float32List? bias; // [outDim] or null

  Dense(this.inDim, this.outDim, this.weight, this.bias)
      : assert(weight.length == inDim * outDim),
        assert(bias == null || bias.length == outDim);

  /// LeCun-uniform random init (bias zero unless [bias] is false → no bias).
  factory Dense.random(
    int inDim,
    int outDim, {
    int seed = 0,
    bool bias = true,
    double? scale,
  }) {
    final rng = math.Random(seed);
    final s = scale ?? (1.0 / math.sqrt(inDim));
    final w = Float32List(inDim * outDim);
    for (var i = 0; i < w.length; i++) {
      w[i] = (rng.nextDouble() * 2 - 1) * s;
    }
    return Dense(inDim, outDim, w, bias ? Float32List(outDim) : null);
  }

  /// Apply row-wise to `x` shaped `[n, inDim]` → `[n, outDim]`.
  Float32List applyRows(Float32List x, int n) {
    final out = Float32List(n * outDim);
    for (var r = 0; r < n; r++) {
      final xo = r * inDim;
      final oo = r * outDim;
      for (var o = 0; o < outDim; o++) {
        var acc = bias == null ? 0.0 : bias![o];
        final wo = o * inDim;
        for (var i = 0; i < inDim; i++) {
          acc += weight[wo + i] * x[xo + i];
        }
        out[oo + o] = acc;
      }
    }
    return out;
  }
}

/// Configuration for [InvariantPointAttention]. Defaults match the ESMFold /
/// AF2 structure-module head (c_s=384, c_z=128, 12 heads, 4 query/key points,
/// 8 value points).
class IpaConfig {
  final int cS; // single-representation channels
  final int cZ; // pair-representation channels
  final int cHidden; // per-head scalar channels
  final int noHeads;
  final int noQkPoints;
  final int noVPoints;
  final double eps;

  const IpaConfig({
    this.cS = 384,
    this.cZ = 128,
    this.cHidden = 16,
    this.noHeads = 12,
    this.noQkPoints = 4,
    this.noVPoints = 8,
    this.eps = 1e-8,
  });
}

/// Invariant Point Attention (Jumper et al. 2021, Alg. 22).
class InvariantPointAttention {
  final IpaConfig cfg;
  final Dense linearQ, linearKv; // scalar queries / keys+values
  final Dense linearQPoints, linearKvPoints; // geometric points
  final Dense linearB; // pair bias
  final Dense linearOut; // output projection
  final Float32List headWeights; // per-head gamma (softplus applied)

  InvariantPointAttention(
    this.cfg,
    this.linearQ,
    this.linearKv,
    this.linearQPoints,
    this.linearKvPoints,
    this.linearB,
    this.linearOut,
    this.headWeights,
  );

  factory InvariantPointAttention.random(IpaConfig cfg, {int seed = 0}) {
    final hc = cfg.cHidden * cfg.noHeads;
    final concatDim =
        cfg.noHeads * (cfg.cZ + cfg.cHidden + cfg.noVPoints * 4);
    return InvariantPointAttention(
      cfg,
      Dense.random(cfg.cS, hc, seed: seed + 1),
      Dense.random(cfg.cS, 2 * hc, seed: seed + 2),
      Dense.random(cfg.cS, cfg.noHeads * cfg.noQkPoints * 3, seed: seed + 3),
      Dense.random(cfg.cS,
          cfg.noHeads * (cfg.noQkPoints + cfg.noVPoints) * 3,
          seed: seed + 4),
      Dense.random(cfg.cZ, cfg.noHeads, seed: seed + 5),
      Dense.random(concatDim, cfg.cS, seed: seed + 6),
      Float32List(cfg.noHeads), // gamma = 0 → softplus(0) = ln 2
    );
  }

  /// Forward pass. `s` is `[n, cS]`, `z` is `[n, n, cZ]`, `frames` has length
  /// `n`. Returns the single-representation update `[n, cS]`.
  Float32List call(Float32List s, Float32List z, List<Rigid> frames) {
    final n = frames.length;
    final H = cfg.noHeads;
    final C = cfg.cHidden;
    final Pq = cfg.noQkPoints;
    final Pv = cfg.noVPoints;

    final q = linearQ.applyRows(s, n); // [n, H*C]
    final kv = linearKv.applyRows(s, n); // [n, 2*H*C]
    final qpRaw = linearQPoints.applyRows(s, n); // [n, H*Pq*3]
    final kvpRaw = linearKvPoints.applyRows(s, n); // [n, H*(Pq+Pv)*3]
    final b = linearB.applyRows(z, n * n); // [n*n, H]

    // Place query/key/value points into the global frame of their residue.
    final qpts = _framePoints(qpRaw, frames, n, H, Pq, 0, Pq);
    final kpts = _framePoints(kvpRaw, frames, n, H, Pq + Pv, 0, Pq);
    final vpts = _framePoints(kvpRaw, frames, n, H, Pq + Pv, Pq, Pv);

    // Softplus head weights, scaled per AF2 (variance normalization).
    final wScalar = math.sqrt(1.0 / (3 * C));
    const wBias = 0.5773502691896258; // sqrt(1/3)
    final wPointHead = Float32List(H);
    final pointScale = math.sqrt(1.0 / (3 * (Pq * 9.0 / 2)));
    for (var h = 0; h < H; h++) {
      wPointHead[h] = _softplus(headWeights[h]) * pointScale;
    }

    final hcHalf = H * C;
    final out = Float32List(n * cfg.cS);
    final logits = Float32List(n);
    final concat = Float32List(H * (C + 4 * Pv + cfg.cZ));

    for (var i = 0; i < n; i++) {
      for (var h = 0; h < H; h++) {
        // Attention logits over all j for head h.
        for (var j = 0; j < n; j++) {
          var scalar = 0.0;
          final qo = i * hcHalf + h * C;
          final ko = j * (2 * hcHalf) + h * C; // keys are first half of kv
          for (var c = 0; c < C; c++) {
            scalar += q[qo + c] * kv[ko + c];
          }
          var pt = 0.0;
          for (var p = 0; p < Pq; p++) {
            final a = qpts[(i * H + h) * Pq + p];
            final bk = kpts[(j * H + h) * Pq + p];
            final dx = a.x - bk.x, dy = a.y - bk.y, dz = a.z - bk.z;
            pt += dx * dx + dy * dy + dz * dz;
          }
          logits[j] = scalar * wScalar +
              b[(i * n + j) * H + h] * wBias +
              (-0.5) * wPointHead[h] * pt;
        }
        _softmaxInPlace(logits, n);

        // --- Outputs for (i, h) ---
        // Scalar value output o ∈ R^C.
        final oBase = h * C;
        for (var c = 0; c < C; c++) {
          var acc = 0.0;
          for (var j = 0; j < n; j++) {
            acc += logits[j] * kv[j * (2 * hcHalf) + hcHalf + h * C + c];
          }
          concat[oBase + c] = acc;
        }
        // Point value output, summed in global space then mapped to local.
        final ptXBase = H * C + h * Pv;
        final ptYBase = H * C + H * Pv + h * Pv;
        final ptZBase = H * C + 2 * H * Pv + h * Pv;
        final ptNBase = H * C + 3 * H * Pv + h * Pv;
        for (var p = 0; p < Pv; p++) {
          var gx = 0.0, gy = 0.0, gz = 0.0;
          for (var j = 0; j < n; j++) {
            final v = vpts[(j * H + h) * Pv + p];
            final w = logits[j];
            gx += w * v.x;
            gy += w * v.y;
            gz += w * v.z;
          }
          final local = frames[i].applyInverse(Vec3(gx, gy, gz));
          concat[ptXBase + p] = local.x;
          concat[ptYBase + p] = local.y;
          concat[ptZBase + p] = local.z;
          concat[ptNBase + p] = math.sqrt(
              local.x * local.x + local.y * local.y + local.z * local.z +
                  cfg.eps);
        }
        // Pair output o_pair ∈ R^cZ.
        final pairBase = H * (C + 4 * Pv) + h * cfg.cZ;
        for (var c = 0; c < cfg.cZ; c++) {
          var acc = 0.0;
          for (var j = 0; j < n; j++) {
            acc += logits[j] * z[(i * n + j) * cfg.cZ + c];
          }
          concat[pairBase + c] = acc;
        }
      }
      // Project the concatenated per-residue features back to cS.
      final proj = linearOut.applyRows(concat, 1);
      for (var c = 0; c < cfg.cS; c++) {
        out[i * cfg.cS + c] = proj[c];
      }
    }
    return out;
  }

  /// Extract `[n, H, count]` points starting at point-offset [start] from a
  /// raw `[n, H*stride*3]` projection and move each into its residue frame.
  List<Vec3> _framePoints(Float32List raw, List<Rigid> frames, int n, int H,
      int stride, int start, int count) {
    final out = List<Vec3>.filled(n * H * count, Vec3.zero);
    for (var i = 0; i < n; i++) {
      for (var h = 0; h < H; h++) {
        final base = i * H * stride * 3 + h * stride * 3 + start * 3;
        for (var p = 0; p < count; p++) {
          final o = base + p * 3;
          final local = Vec3(raw[o], raw[o + 1], raw[o + 2]);
          out[(i * H + h) * count + p] = frames[i].apply(local);
        }
      }
    }
    return out;
  }
}

double _softplus(double x) =>
    x > 20 ? x : math.log(1 + math.exp(x)); // numerically stable

void _softmaxInPlace(Float32List a, int n) {
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

/// LayerNorm over the last dimension (channels).
void _layerNormRows(
    Float32List x, int n, int d, Float32List gamma, Float32List beta,
    {double eps = 1e-5}) {
  for (var r = 0; r < n; r++) {
    final o = r * d;
    var mean = 0.0;
    for (var i = 0; i < d; i++) {
      mean += x[o + i];
    }
    mean /= d;
    var varr = 0.0;
    for (var i = 0; i < d; i++) {
      final dv = x[o + i] - mean;
      varr += dv * dv;
    }
    varr /= d;
    final inv = 1.0 / math.sqrt(varr + eps);
    for (var i = 0; i < d; i++) {
      x[o + i] = (x[o + i] - mean) * inv * gamma[i] + beta[i];
    }
  }
}

/// The 3-layer residual transition applied to `s` after IPA.
class StructureTransition {
  final Dense l1, l2, l3;
  final Float32List lnGamma, lnBeta;
  StructureTransition(this.l1, this.l2, this.l3, this.lnGamma, this.lnBeta);

  factory StructureTransition.random(int cS, {int seed = 0}) =>
      StructureTransition(
        Dense.random(cS, cS, seed: seed + 1),
        Dense.random(cS, cS, seed: seed + 2),
        Dense.random(cS, cS, seed: seed + 3),
        Float32List(cS)..fillRange(0, cS, 1.0),
        Float32List(cS),
      );

  Float32List call(Float32List s, int n, int cS) {
    var h = l1.applyRows(s, n);
    _reluInPlace(h);
    h = l2.applyRows(h, n);
    _reluInPlace(h);
    h = l3.applyRows(h, n);
    for (var i = 0; i < h.length; i++) {
      h[i] += s[i];
    }
    _layerNormRows(h, n, cS, lnGamma, lnBeta);
    return h;
  }
}

void _reluInPlace(Float32List x) {
  for (var i = 0; i < x.length; i++) {
    if (x[i] < 0) x[i] = 0;
  }
}

/// Predicts a per-residue rigid update `(quaternion, translation)` from `s`
/// and composes it onto the current frame.
class BackboneUpdate {
  final Dense linear; // cS -> 6
  BackboneUpdate(this.linear);

  factory BackboneUpdate.random(int cS, {int seed = 0}) =>
      BackboneUpdate(Dense.random(cS, 6, seed: seed));

  List<Rigid> call(Float32List s, int n, int cS, List<Rigid> frames) {
    final u = linear.applyRows(s, n); // [n, 6]
    final out = List<Rigid>.filled(n, Rigid.identity);
    for (var i = 0; i < n; i++) {
      final o = i * 6;
      // (b, c, d) with implicit real part a = 1, then normalized.
      final rot = Rot3.fromQuaternion(1.0, u[o], u[o + 1], u[o + 2]);
      final update = Rigid(rot, Vec3(u[o + 3], u[o + 4], u[o + 5]));
      out[i] = frames[i].compose(update);
    }
    return out;
  }
}

/// Full structure module: shared-weight IPA + transition + backbone update,
/// iterated [noBlocks] times, producing final backbone frames and atoms.
class StructureModule {
  final IpaConfig cfg;
  final int noBlocks;
  final double transScale;
  final InvariantPointAttention ipa;
  final StructureTransition transition;
  final BackboneUpdate bbUpdate;
  final Float32List lnSGamma, lnSBeta, lnZGamma, lnZBeta, lnIpaGamma, lnIpaBeta;
  final Dense linearIn;

  StructureModule(
    this.cfg,
    this.ipa,
    this.transition,
    this.bbUpdate,
    this.linearIn,
    this.lnSGamma,
    this.lnSBeta,
    this.lnZGamma,
    this.lnZBeta,
    this.lnIpaGamma,
    this.lnIpaBeta, {
    this.noBlocks = 8,
    this.transScale = 10.0,
  });

  factory StructureModule.random(IpaConfig cfg,
      {int noBlocks = 8, int seed = 0}) {
    Float32List ones(int d) => Float32List(d)..fillRange(0, d, 1.0);
    return StructureModule(
      cfg,
      InvariantPointAttention.random(cfg, seed: seed + 10),
      StructureTransition.random(cfg.cS, seed: seed + 20),
      BackboneUpdate.random(cfg.cS, seed: seed + 30),
      Dense.random(cfg.cS, cfg.cS, seed: seed + 40),
      ones(cfg.cS), Float32List(cfg.cS),
      ones(cfg.cZ), Float32List(cfg.cZ),
      ones(cfg.cS), Float32List(cfg.cS),
      noBlocks: noBlocks,
    );
  }

  /// Run the module. `s` is `[n, cS]`, `z` is `[n, n, cZ]`. Frames are
  /// initialized to [initial] (default: identity "black hole" per residue).
  /// Returns the final per-residue frames (translations in Å) and the
  /// placed backbone atoms.
  ({List<Rigid> frames, List<({Vec3 n, Vec3 cAlpha, Vec3 c})> atoms}) call(
    Float32List s,
    Float32List z, {
    Rigid? initial,
  }) {
    final n = s.length ~/ cfg.cS;
    final sN = Float32List.fromList(s);
    final zN = Float32List.fromList(z);
    _layerNormRows(sN, n, cfg.cS, lnSGamma, lnSBeta);
    _layerNormRows(zN, n * n, cfg.cZ, lnZGamma, lnZBeta);

    var h = linearIn.applyRows(sN, n);
    var frames = List<Rigid>.filled(n, initial ?? Rigid.identity);

    for (var blk = 0; blk < noBlocks; blk++) {
      final upd = ipa.call(h, zN, frames);
      for (var i = 0; i < h.length; i++) {
        h[i] += upd[i];
      }
      _layerNormRows(h, n, cfg.cS, lnIpaGamma, lnIpaBeta);
      h = transition.call(h, n, cfg.cS);
      frames = bbUpdate.call(h, n, cfg.cS, frames);
    }

    // Scale translations from trunk units to Ångström and place atoms.
    final scaled = [
      for (final f in frames) Rigid(f.rot, f.trans.scale(transScale))
    ];
    return (frames: scaled, atoms: [for (final f in scaled) backboneAtoms(f)]);
  }
}
