/// Needle 3 forward pass (float path), matching the reference
/// `SimpleAttentionNetwork` in `architecture.py`. Operates on a single
/// sequence (batch 1) with row-major [Float64List] buffers. Weights are
/// supplied already dequantised, so this module is independent of the
/// `.cact` byte layout and testable with synthetic weights.
///
/// Per block: engram fusion → gated GQA attention (conv taps, QK-norm,
/// RoPE, sliding/global window) → Hadamard MLP, wrapped by the 4-lane
/// multi-lane hyper-connection (mHC) mixer.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'hadamard.dart';
import 'needle_config.dart';
import 'needle_engram.dart';
import 'needle_math.dart';

/// `y = x · Wᵀ` for `x [rows, inDim]`, `W [outDim, inDim]` → `y [rows, outDim]`.
Float64List matmulXWt(
  Float64List x,
  int rows,
  int inDim,
  Float64List w,
  int outDim,
) {
  final y = Float64List(rows * outDim);
  for (var r = 0; r < rows; r++) {
    final xb = r * inDim;
    for (var o = 0; o < outDim; o++) {
      final wb = o * inDim;
      var s = 0.0;
      for (var i = 0; i < inDim; i++) {
        s += x[xb + i] * w[wb + i];
      }
      y[r * outDim + o] = s;
    }
  }
  return y;
}

/// Causal depthwise conv over time: `out[t,ch] = Σ_j taps[j,ch]·in[t-j,ch]`
/// with `in[t-j] = 0` for `t < j` (shift-right zero pad), [dilation]-spaced.
Float64List _causalConv(
  Float64List x,
  int rows,
  int dim,
  Float64List taps,
  int nTaps, {
  int dilation = 1,
}) {
  final out = Float64List(rows * dim);
  for (var t = 0; t < rows; t++) {
    for (var ch = 0; ch < dim; ch++) {
      var s = 0.0;
      for (var j = 0; j < nTaps; j++) {
        final src = t - j * dilation;
        if (src >= 0) s += taps[j * dim + ch] * x[src * dim + ch];
      }
      out[t * dim + ch] = s;
    }
  }
  return out;
}

/// Attention sublayer weights (all dequantised, row-major `[out, in]`).
class AttnW {
  final Float64List qProj, kProj, vProj; // [nh*qk,d], [nkv*qk,d], [nkv*vd,d]
  final Float64List qTaps, kTaps, vTaps; // [taps, nh*qk], [taps, nkv*qk], ...
  final Float64List qNorm, kNorm; // [qk]
  final Float64List gateProj, outProj; // [nh*vd, d], [d, nh*vd]
  const AttnW({
    required this.qProj,
    required this.kProj,
    required this.vProj,
    required this.qTaps,
    required this.kTaps,
    required this.vTaps,
    required this.qNorm,
    required this.kNorm,
    required this.gateProj,
    required this.outProj,
  });
}

/// Hadamard-MLP weights.
class MlpW {
  final Float64List condV, condU; // [d, 8], [8, n]
  final Float64List d1, d2, b2, d3, d4; // [n]
  final Float64List w1a, w1b, w2a, w2b, w3a, w3b; // [ba*ba]/[bb*bb]
  final Int32List p1, p2; // [n] permutations
  const MlpW({
    required this.condV,
    required this.condU,
    required this.d1,
    required this.d2,
    required this.b2,
    required this.d3,
    required this.d4,
    required this.w1a,
    required this.w1b,
    required this.w2a,
    required this.w2b,
    required this.w3a,
    required this.w3b,
    required this.p1,
    required this.p2,
  });
}

/// One block's weights.
class BlockW {
  final Float64List normIn, postNorm, preHada; // [d]
  final double attnGate; // scalar
  final AttnW attn;
  final MlpW mlp;
  const BlockW({
    required this.normIn,
    required this.postNorm,
    required this.preHada,
    required this.attnGate,
    required this.attn,
    required this.mlp,
  });
}

/// One engram site's weights.
class EngramW {
  final int site;
  final Float64List tables; // [numTables, slots, subDim]
  final Float64List keyProj, valueProj; // [d, numTables*subDim]
  final Float64List taps; // [engramTaps, d]
  const EngramW({
    required this.site,
    required this.tables,
    required this.keyProj,
    required this.valueProj,
    required this.taps,
  });
}

/// Per-layer mHC parameters.
class MhcW {
  final Float64List aPre, aPost, aRes; // [L]
  final Float64List bPre, bPost; // [L, n]
  final Float64List bRes; // [L, n, n]
  final Float64List phiPre, phiPost; // [L, n*d, n]
  final Float64List phiRes; // [L, n*d, n*n]
  const MhcW({
    required this.aPre,
    required this.aPost,
    required this.aRes,
    required this.bPre,
    required this.bPost,
    required this.bRes,
    required this.phiPre,
    required this.phiPost,
    required this.phiRes,
  });
}

const int kEngramConvTaps = 4;

/// A fully-dequantised Needle 3 model that can produce logits for a token
/// sequence. CPU, single sequence.
class NeedleModelF {
  final NeedleConfig config;
  final Float64List embedding; // [vocab, d]
  final List<BlockW> blocks;
  final MhcW mhc;
  final List<EngramW> engram; // one per active site
  final Float64List finalNorm; // [d]

  const NeedleModelF({
    required this.config,
    required this.embedding,
    required this.blocks,
    required this.mhc,
    required this.engram,
    required this.finalNorm,
  });

  int get _d => config.dModel;
  int get _nLanes => config.numLanes;

  /// Logits `[T, vocab]` for [tokens].
  Float64List logits(List<int> tokens) {
    final t = tokens.length;
    final d = _d;

    // token embedding * sqrt(d)
    final embScale = math.sqrt(d);
    var x = Float64List(t * d);
    for (var p = 0; p < t; p++) {
      final row = tokens[p] * d;
      for (var c = 0; c < d; c++) {
        x[p * d + c] = embedding[row + c] * embScale;
      }
    }

    // engram k/v per active site (computed once over the sequence)
    final engKv = <int, (Float64List, Float64List)>{};
    for (final e in engram) {
      engKv[e.site] = _engramKv(tokens, e);
    }

    // RoPE tables for the q/k head dim
    final (cos, sin) = ropeTables(t, config.headDimQ, config.ropeTheta);
    final globals = config.activeGlobalLayers().toSet();

    // expand to lanes: stream [T, n, d]
    final n = _nLanes;
    var stream = Float64List(t * n * d);
    for (var p = 0; p < t; p++) {
      for (var lane = 0; lane < n; lane++) {
        for (var c = 0; c < d; c++) {
          stream[(p * n + lane) * d + c] = x[p * d + c];
        }
      }
    }

    for (var l = 0; l < blocks.length; l++) {
      stream = _mhcAdvance(stream, tokens, l, cos, sin, globals, engKv);
    }

    // mean over lanes, final norm, tied head
    final merged = Float64List(t * d);
    for (var p = 0; p < t; p++) {
      for (var c = 0; c < d; c++) {
        var s = 0.0;
        for (var lane = 0; lane < n; lane++) {
          s += stream[(p * n + lane) * d + c];
        }
        merged[p * d + c] = s / n;
      }
    }
    zcRmsNorm(merged, t, d, finalNorm);
    return matmulXWt(merged, t, d, embedding, config.vocab);
  }

  /// mHC wrapper around block [l]: combine lanes → block → redistribute.
  Float64List _mhcAdvance(
    Float64List stream,
    List<int> tokens,
    int l,
    Float64List cos,
    Float64List sin,
    Set<int> globals,
    Map<int, (Float64List, Float64List)> engKv,
  ) {
    final t = tokens.length;
    final d = _d;
    final n = _nLanes;
    final nd = n * d;

    // owned lane for this layer and its offsets
    final owned = l % n;
    final nx = rmsUnit(stream, t, nd); // [T, n*d]

    // hpre [T, n] = sigmoid(a_pre*(nx@phi_pre) + b_pre + pre_off)
    final phiPreBase = l * nd * n;
    final bPreBase = l * n;
    final hpre = Float64List(t * n);
    for (var p = 0; p < t; p++) {
      for (var j = 0; j < n; j++) {
        var s = 0.0;
        for (var k = 0; k < nd; k++) {
          s += nx[p * nd + k] * mhc.phiPre[phiPreBase + k * n + j];
        }
        final preOff = 8.0 * (j == owned ? 1.0 : 0.0) - 4.0;
        hpre[p * n + j] =
            sigmoid(mhc.aPre[l] * s + mhc.bPre[bPreBase + j] + preOff);
      }
    }

    // u [T, d] = Σ_lane hpre[:,lane] * stream[:,lane,:]
    final u = Float64List(t * d);
    for (var p = 0; p < t; p++) {
      for (var c = 0; c < d; c++) {
        var s = 0.0;
        for (var lane = 0; lane < n; lane++) {
          s += hpre[p * n + lane] * stream[(p * n + lane) * d + c];
        }
        u[p * d + c] = s;
      }
    }

    // y = block(u) - u
    final blockOut = _blockForward(u, tokens, l, cos, sin, globals, engKv);
    final y = Float64List(t * d);
    for (var i = 0; i < t * d; i++) {
      y[i] = blockOut[i] - u[i];
    }

    // hpost [T, n] = 2*sigmoid(a_post*(nx@phi_post)+b_post+post_off)
    final phiPostBase = l * nd * n;
    final bPostBase = l * n;
    final hpost = Float64List(t * n);
    for (var p = 0; p < t; p++) {
      for (var j = 0; j < n; j++) {
        var s = 0.0;
        for (var k = 0; k < nd; k++) {
          s += nx[p * nd + k] * mhc.phiPost[phiPostBase + k * n + j];
        }
        final postOff = -4.0 * (j == owned ? 0.0 : 1.0);
        hpost[p * n + j] =
            2.0 * sigmoid(mhc.aPost[l] * s + mhc.bPost[bPostBase + j] + postOff);
      }
    }

    // res [T, n*n] = nx @ phi_res; hres = sinkhorn(a_res*res + b_res)
    final phiResBase = l * nd * (n * n);
    final bResBase = l * n * n;
    final out = Float64List(t * n * d);
    final logits = Float64List(n * n);
    for (var p = 0; p < t; p++) {
      for (var ij = 0; ij < n * n; ij++) {
        var s = 0.0;
        for (var k = 0; k < nd; k++) {
          s += nx[p * nd + k] * mhc.phiRes[phiResBase + k * (n * n) + ij];
        }
        logits[ij] = mhc.aRes[l] * s + mhc.bRes[bResBase + ij];
      }
      final hres = _sinkhorn(logits, n);
      // new[i,c] = Σ_j hres[i,j]*stream[j,c] + hpost[i]*y[c]
      for (var i = 0; i < n; i++) {
        for (var c = 0; c < d; c++) {
          var s = 0.0;
          for (var j = 0; j < n; j++) {
            s += hres[i * n + j] * stream[(p * n + j) * d + c];
          }
          out[(p * n + i) * d + c] = s + hpost[p * n + i] * y[p * d + c];
        }
      }
    }
    return out;
  }

  /// Block forward on the combined stream [T, d].
  Float64List _blockForward(
    Float64List uIn,
    List<int> tokens,
    int l,
    Float64List cos,
    Float64List sin,
    Set<int> globals,
    Map<int, (Float64List, Float64List)> engKv,
  ) {
    final t = tokens.length;
    final d = _d;
    final bw = blocks[l];
    var x = Float64List.fromList(uIn);

    // engram fusion if this layer hosts a site
    final kv = engKv[l];
    if (kv != null) {
      final (ek, ev) = kv;
      final rx = rmsUnit(x, t, d);
      final rek = rmsUnit(ek, t, d);
      final scaleD = math.sqrt(d);
      for (var p = 0; p < t; p++) {
        var dot = 0.0;
        for (var c = 0; c < d; c++) {
          dot += rx[p * d + c] * rek[p * d + c];
        }
        final alpha = sigmoid(dot / scaleD);
        for (var c = 0; c < d; c++) {
          x[p * d + c] += alpha * ev[p * d + c];
        }
      }
    }

    // attention sublayer: skip + sigmoid(attn_gate) * attn(norm(x))
    final skip1 = Float64List.fromList(x);
    final h1 = Float64List.fromList(x);
    zcRmsNorm(h1, t, d, bw.normIn);
    final attnOut = _attention(h1, t, l, bw.attn, cos, sin, globals);
    zcRmsNorm(attnOut, t, d, bw.postNorm);
    final g = sigmoid(bw.attnGate);
    for (var i = 0; i < t * d; i++) {
      x[i] = skip1[i] + g * attnOut[i];
    }

    // MLP sublayer: skip + hadamard_mlp(norm(x))
    final skip2 = Float64List.fromList(x);
    final h2 = Float64List.fromList(x);
    zcRmsNorm(h2, t, d, bw.preHada);
    final mlpOut = _hadamardMlp(h2, t, bw.mlp);
    for (var i = 0; i < t * d; i++) {
      x[i] = skip2[i] + mlpOut[i];
    }
    return x;
  }

  Float64List _attention(
    Float64List x,
    int t,
    int l,
    AttnW w,
    Float64List cos,
    Float64List sin,
    Set<int> globals,
  ) {
    final d = _d;
    final nh = config.numQHeads;
    final nkv = config.numKvHeads;
    final qk = config.headDimQ;
    final vd = config.headDimV;
    final qDim = nh * qk;
    final kDim = nkv * qk;
    final vDim = nkv * vd;
    final taps = config.convTaps;

    var q = matmulXWt(x, t, d, w.qProj, qDim);
    var k = matmulXWt(x, t, d, w.kProj, kDim);
    var v = matmulXWt(x, t, d, w.vProj, vDim);
    q = _causalConv(q, t, qDim, w.qTaps, taps);
    k = _causalConv(k, t, kDim, w.kTaps, taps);
    v = _causalConv(v, t, vDim, w.vTaps, taps);

    // QK-norm per head (ZCRMSNorm over qk dims) + RoPE per head
    for (var h = 0; h < nh; h++) {
      final head = _extractHead(q, t, qDim, h, qk);
      zcRmsNorm(head, t, qk, w.qNorm);
      applyRopeHead(head, t, qk, cos, sin);
      _writeHead(q, t, qDim, h, qk, head);
    }
    for (var h = 0; h < nkv; h++) {
      final head = _extractHead(k, t, kDim, h, qk);
      zcRmsNorm(head, t, qk, w.kNorm);
      applyRopeHead(head, t, qk, cos, sin);
      _writeHead(k, t, kDim, h, qk, head);
    }

    final window = globals.contains(l) ? (1 << 30) : config.window;
    final scale = 1.0 / math.sqrt(qk);
    final repeats = nh ~/ nkv;
    final out = Float64List(t * (nh * vd));
    final scores = Float64List(t);
    for (var h = 0; h < nh; h++) {
      final kvh = h ~/ repeats;
      for (var ti = 0; ti < t; ti++) {
        var lo = ti - window + 1;
        if (lo < 0) lo = 0;
        var mx = double.negativeInfinity;
        for (var s = lo; s <= ti; s++) {
          var dot = 0.0;
          for (var i = 0; i < qk; i++) {
            dot += q[ti * qDim + h * qk + i] * k[s * kDim + kvh * qk + i];
          }
          dot *= scale;
          scores[s] = dot;
          if (dot > mx) mx = dot;
        }
        var sum = 0.0;
        for (var s = lo; s <= ti; s++) {
          final e = math.exp(scores[s] - mx);
          scores[s] = e;
          sum += e;
        }
        final inv = 1.0 / sum;
        for (var c = 0; c < vd; c++) {
          var acc = 0.0;
          for (var s = lo; s <= ti; s++) {
            acc += scores[s] * inv * v[s * vDim + kvh * vd + c];
          }
          out[ti * (nh * vd) + h * vd + c] = acc;
        }
      }
    }

    // gate: out *= sigmoid(gate_proj(x)); then out_proj
    final gate = matmulXWt(x, t, d, w.gateProj, nh * vd);
    for (var i = 0; i < t * (nh * vd); i++) {
      out[i] *= sigmoid(gate[i]);
    }
    return matmulXWt(out, t, nh * vd, w.outProj, d);
  }

  Float64List _hadamardMlp(Float64List x, int t, MlpW w) {
    final d = _d;
    final n = config.hadamardSize;
    final (ba, bb) = hadaBlocks(n);

    // cond [T, n] = 1 + softmax(x @ cond_v) @ cond_u
    const rank = 8;
    final proj = matmulXWt(x, t, d, _transpose(w.condV, d, rank), rank);
    softmaxRows(proj, t, rank);
    final out = Float64List(t * d);
    final z = Float64List(n);
    final cond = Float64List(n);
    for (var p = 0; p < t; p++) {
      for (var c = 0; c < n; c++) {
        var s = 0.0;
        for (var r = 0; r < rank; r++) {
          s += proj[p * rank + r] * w.condU[r * n + c];
        }
        cond[c] = 1.0 + s;
      }
      // z = pad(x, n)
      for (var c = 0; c < n; c++) {
        z[c] = c < d ? x[p * d + c] : 0.0;
      }
      // stage 1: kron(d1*z) then permute p1
      for (var c = 0; c < n; c++) {
        z[c] *= w.d1[c];
      }
      var zz = monarchApplyRect(z, w.w1a, ba, w.w1b, bb);
      zz = _permute(zz, w.p1);
      // stage 2: silu(d2*cond*zz + b2), kron, permute p2
      for (var c = 0; c < n; c++) {
        zz[c] = silu(w.d2[c] * cond[c] * zz[c] + w.b2[c]);
      }
      zz = monarchApplyRect(zz, w.w2a, ba, w.w2b, bb);
      zz = _permute(zz, w.p2);
      // stage 3: kron(d3*zz)
      for (var c = 0; c < n; c++) {
        zz[c] *= w.d3[c];
      }
      zz = monarchApplyRect(zz, w.w3a, ba, w.w3b, bb);
      // out = (d4 * zz)[:d]
      for (var c = 0; c < d; c++) {
        out[p * d + c] = w.d4[c] * zz[c];
      }
    }
    return out;
  }

  (Float64List, Float64List) _engramKv(List<int> tokens, EngramW e) {
    final t = tokens.length;
    final d = _d;
    final orders = config.engramOrders;
    final numTables = engramTableCount(orders, numEngramHeads());
    final subDim = config.engramSubDim;
    final slots = config.engramSlots;
    final idx = engramIndices(tokens,
        orders: orders, heads: numEngramHeads(), slots: slots);

    // fetch + ngram_ok mask + concat -> e [T, numTables*subDim]
    final feat = Float64List(t * numTables * subDim);
    for (var p = 0; p < t; p++) {
      var col = 0;
      for (var oi = 0; oi < orders.length; oi++) {
        final order = orders[oi];
        final ok = p >= order - 1 ? 1.0 : 0.0; // ngram_ok (mask_diag order-1)
        for (var hh = 0; hh < numEngramHeads(); hh++) {
          final table = col;
          final rowIdx = idx[p][table];
          final tb = (table * slots + rowIdx) * subDim;
          for (var c = 0; c < subDim; c++) {
            feat[(p * numTables + table) * subDim + c] = e.tables[tb + c] * ok;
          }
          col++;
        }
      }
    }

    final ed = numTables * subDim;
    final kk = matmulXWt(feat, t, ed, e.keyProj, d);
    var vv = matmulXWt(feat, t, ed, e.valueProj, d);
    // value conv: 4 taps, dilation = max(orders)
    final dilation = orders.reduce(math.max);
    vv = _causalConv(vv, t, d, e.taps, kEngramConvTaps, dilation: dilation);
    return (kk, vv);
  }

  /// Engram head count for this geometry (`d / (orders * subDim)`, min 1).
  int numEngramHeads() {
    final h = config.dModel ~/ (config.engramOrders.length * config.engramSubDim);
    return h < 1 ? 1 : h;
  }
}

Float64List _extractHead(Float64List x, int t, int rowDim, int h, int hd) {
  final out = Float64List(t * hd);
  for (var p = 0; p < t; p++) {
    for (var i = 0; i < hd; i++) {
      out[p * hd + i] = x[p * rowDim + h * hd + i];
    }
  }
  return out;
}

void _writeHead(Float64List x, int t, int rowDim, int h, int hd,
    Float64List head) {
  for (var p = 0; p < t; p++) {
    for (var i = 0; i < hd; i++) {
      x[p * rowDim + h * hd + i] = head[p * hd + i];
    }
  }
}

Float64List _permute(Float64List z, Int32List perm) {
  final out = Float64List(z.length);
  for (var i = 0; i < perm.length; i++) {
    out[i] = z[perm[i]];
  }
  return out;
}

Float64List _transpose(Float64List m, int rows, int cols) {
  final out = Float64List(rows * cols);
  for (var r = 0; r < rows; r++) {
    for (var c = 0; c < cols; c++) {
      out[c * rows + r] = m[r * cols + c];
    }
  }
  return out;
}

/// Sinkhorn normalisation to a doubly-stochastic `n×n` matrix, 20 iterations
/// in log space (alternating row/col log-sum-exp), returning `exp(logK)`.
Float64List _sinkhorn(Float64List logits, int n, {int iters = 20}) {
  final logK = Float64List.fromList(logits);
  for (var it = 0; it < iters; it++) {
    // rows
    for (var i = 0; i < n; i++) {
      var mx = double.negativeInfinity;
      for (var j = 0; j < n; j++) {
        if (logK[i * n + j] > mx) mx = logK[i * n + j];
      }
      var sum = 0.0;
      for (var j = 0; j < n; j++) {
        sum += math.exp(logK[i * n + j] - mx);
      }
      final lse = mx + math.log(sum);
      for (var j = 0; j < n; j++) {
        logK[i * n + j] -= lse;
      }
    }
    // cols
    for (var j = 0; j < n; j++) {
      var mx = double.negativeInfinity;
      for (var i = 0; i < n; i++) {
        if (logK[i * n + j] > mx) mx = logK[i * n + j];
      }
      var sum = 0.0;
      for (var i = 0; i < n; i++) {
        sum += math.exp(logK[i * n + j] - mx);
      }
      final lse = mx + math.log(sum);
      for (var i = 0; i < n; i++) {
        logK[i * n + j] -= lse;
      }
    }
  }
  final out = Float64List(n * n);
  for (var i = 0; i < n * n; i++) {
    out[i] = math.exp(logK[i]);
  }
  return out;
}
