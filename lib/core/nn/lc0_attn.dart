/// LC0 attention-body (transformer) net reader + CPU forward — a pure-Dart port
/// of lc0's BLAS reference (src/neural/backends/blas/network_blas.cc) for
/// NETWORK_ATTENTIONBODY_WITH_HEADFORMAT nets (attention policy + WDL value +
/// MLH, with smolgen). Classical nets are handled by [Lc0Net] in lc0.dart.
///
/// Built incrementally: M2 = reader + input embedding. Encoder/smolgen, the
/// attention policy head, and value/MLH heads land in later milestones.
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'lc0_attn_policy_map.dart';
import 'lc0_pos_encoding.dart';

const int _kSquares = 64;
const int _kInputPlanes = 112;

double _mish(double x) {
  // x * tanh(softplus(x)); softplus guarded for large x.
  final sp = x > 20.0 ? x : math.log(1.0 + math.exp(x));
  final t = _tanh(sp);
  return x * t;
}

double _tanh(double x) {
  if (x > 20) return 1.0;
  if (x < -20) return -1.0;
  final e2 = math.exp(2 * x);
  return (e2 - 1) / (e2 + 1);
}

/// One encoder layer's weights (flat, lc0 [out,in] row-major).
class Lc0AttnEncoder {
  final Float32List qW, qB, kW, kB, vW, vB, denseW, denseB;
  final Float32List ln1G, ln1B, ln2G, ln2B;
  final Float32List ffn1W, ffn1B, ffn2W, ffn2B;
  // Smolgen (nullable if absent).
  final Float32List? smCompress, smD1W, smD1B, smLn1G, smLn1B, smD2W, smD2B,
      smLn2G, smLn2B;
  Lc0AttnEncoder({
    required this.qW,
    required this.qB,
    required this.kW,
    required this.kB,
    required this.vW,
    required this.vB,
    required this.denseW,
    required this.denseB,
    required this.ln1G,
    required this.ln1B,
    required this.ln2G,
    required this.ln2B,
    required this.ffn1W,
    required this.ffn1B,
    required this.ffn2W,
    required this.ffn2B,
    this.smCompress,
    this.smD1W,
    this.smD1B,
    this.smLn1G,
    this.smLn1B,
    this.smD2W,
    this.smD2B,
    this.smLn2G,
    this.smLn2B,
  });

  bool get hasSmolgen => smCompress != null;
}

/// Attention-net forward result.
class Lc0AttnOutput {
  final Float32List policy; // 1858 classical move logits (unmasked)
  final List<double> wdl; // [W, D, L] softmaxed
  final double movesLeft;
  const Lc0AttnOutput(this.policy, this.wdl, this.movesLeft);
  double get value => wdl[0] - wdl[2];
}

/// Parsed attention-body weights.
class Lc0AttnWeights {
  final int embDim; // 256
  final int heads; // 8
  final int dff; // 1024
  final int wdl; // 3

  final Float32List ipEmbW, ipEmbB; // [embDim, 176], [embDim]
  final Float32List? ipMultGate, ipAddGate; // [embDim, 64] channel-major
  final List<Lc0AttnEncoder> encoders;
  final Float32List? smolgenW, smolgenB; // global [64*64, gen/heads]

  // Attention policy.
  final Float32List ipPolW, ipPolB; // embedding [polEmb, embDim]
  final Float32List ip2PolW, ip2PolB; // wq [dModel, polEmb]
  final Float32List ip3PolW, ip3PolB; // wk [dModel, polEmb]
  final Float32List ip4PolW; // ppo [4, dModel]

  // Value head (attention).
  final Float32List ipValW, ipValB; // [valPlanes, embDim]
  final Float32List ip1ValW, ip1ValB; // [valChannels, valPlanes*64]
  final Float32List ip2ValW, ip2ValB; // [wdl, valChannels]

  // Moves-left head (attention).
  final Float32List? ipMovW, ipMovB, ip1MovW, ip1MovB, ip2MovW, ip2MovB;

  Lc0AttnWeights({
    required this.embDim,
    required this.heads,
    required this.dff,
    required this.wdl,
    required this.ipEmbW,
    required this.ipEmbB,
    required this.ipMultGate,
    required this.ipAddGate,
    required this.encoders,
    required this.smolgenW,
    required this.smolgenB,
    required this.ipPolW,
    required this.ipPolB,
    required this.ip2PolW,
    required this.ip2PolB,
    required this.ip3PolW,
    required this.ip3PolB,
    required this.ip4PolW,
    required this.ipValW,
    required this.ipValB,
    required this.ip1ValW,
    required this.ip1ValB,
    required this.ip2ValW,
    required this.ip2ValB,
    required this.ipMovW,
    required this.ipMovB,
    required this.ip1MovW,
    required this.ip1MovB,
    required this.ip2MovW,
    required this.ip2MovB,
  });

  /// True if [path] is an attention-body net (has encoder/embedding fields).
  static bool isAttentionNet(String path) {
    final d = Uint8List.fromList(gzip.decode(File(path).readAsBytesSync()));
    final w = _field(d, _Range(0, d.length), 10);
    if (w == null) return false;
    return _field(d, w, 25) != null && _field(d, w, 27) != null;
  }
}

class Lc0AttnReader {
  static Lc0AttnWeights readFile(String path) {
    final d = Uint8List.fromList(gzip.decode(File(path).readAsBytesSync()));
    final w = _field(d, _Range(0, d.length), 10);
    if (w == null) throw ArgumentError('lc0-attn: no Weights (field 10)');

    final ipEmbW = _layer(d, _field(d, w, 25)!);
    final ipEmbB = _layer(d, _field(d, w, 26)!);
    final embDim = ipEmbB.length;
    final heads = _varintField(d, w, 28) ?? 0;

    final encRanges = _fields(d, w, 27);
    final encoders = <Lc0AttnEncoder>[
      for (final e in encRanges) _parseEncoder(d, e)
    ];
    final dff = encoders.isEmpty ? 0 : encoders.first.ffn1B.length;

    Float32List? optLayer(int f) {
      final r = _field(d, w, f);
      return r == null ? null : _layer(d, r);
    }

    final ip2ValW = _layer(d, _field(d, w, 9)!);
    final ip2ValB = _layer(d, _field(d, w, 10)!);

    return Lc0AttnWeights(
      embDim: embDim,
      heads: heads,
      dff: dff,
      wdl: ip2ValB.length,
      ipEmbW: ipEmbW,
      ipEmbB: ipEmbB,
      ipMultGate: optLayer(33),
      ipAddGate: optLayer(34),
      encoders: encoders,
      smolgenW: optLayer(35),
      smolgenB: optLayer(36),
      ipPolW: _layer(d, _field(d, w, 4)!),
      ipPolB: _layer(d, _field(d, w, 5)!),
      ip2PolW: _layer(d, _field(d, w, 17)!),
      ip2PolB: _layer(d, _field(d, w, 18)!),
      ip3PolW: _layer(d, _field(d, w, 19)!),
      ip3PolB: _layer(d, _field(d, w, 20)!),
      ip4PolW: _layer(d, _field(d, w, 22)!),
      ipValW: _layer(d, _field(d, w, 29)!),
      ipValB: _layer(d, _field(d, w, 30)!),
      ip1ValW: _layer(d, _field(d, w, 7)!),
      ip1ValB: _layer(d, _field(d, w, 8)!),
      ip2ValW: ip2ValW,
      ip2ValB: ip2ValB,
      ipMovW: optLayer(31),
      ipMovB: optLayer(32),
      ip1MovW: optLayer(13),
      ip1MovB: optLayer(14),
      ip2MovW: optLayer(15),
      ip2MovB: optLayer(16),
    );
  }

  static Lc0AttnEncoder _parseEncoder(Uint8List d, _Range e) {
    final mha = _field(d, e, 1)!;
    final sm = _field(d, mha, 9);
    Float32List? smL(int f) => sm == null ? null : _layer(d, _field(d, sm, f)!);
    final ffn = _field(d, e, 4)!;
    return Lc0AttnEncoder(
      qW: _layer(d, _field(d, mha, 1)!),
      qB: _layer(d, _field(d, mha, 2)!),
      kW: _layer(d, _field(d, mha, 3)!),
      kB: _layer(d, _field(d, mha, 4)!),
      vW: _layer(d, _field(d, mha, 5)!),
      vB: _layer(d, _field(d, mha, 6)!),
      denseW: _layer(d, _field(d, mha, 7)!),
      denseB: _layer(d, _field(d, mha, 8)!),
      ln1G: _layer(d, _field(d, e, 2)!),
      ln1B: _layer(d, _field(d, e, 3)!),
      ln2G: _layer(d, _field(d, e, 5)!),
      ln2B: _layer(d, _field(d, e, 6)!),
      ffn1W: _layer(d, _field(d, ffn, 1)!),
      ffn1B: _layer(d, _field(d, ffn, 2)!),
      ffn2W: _layer(d, _field(d, ffn, 3)!),
      ffn2B: _layer(d, _field(d, ffn, 4)!),
      smCompress: smL(1),
      smD1W: smL(2),
      smD1B: smL(3),
      smLn1G: smL(4),
      smLn1B: smL(5),
      smD2W: smL(6),
      smD2B: smL(7),
      smLn2G: smL(8),
      smLn2B: smL(9),
    );
  }
}

/// Attention-body network (CPU, pure Dart).
class Lc0AttnNet {
  final Lc0AttnWeights w;
  Lc0AttnNet(this.w);

  /// Input embedding (M2): NCHW-flat input planes `[112*64]` (plane*64+square)
  /// -> `[64*embDim]` square-major embedding after gating.
  Float32List embed(Float32List inputNCHW) {
    final e = w.embDim;
    final inSize = _kInputPlanes + 64; // 176
    final out = Float32List(_kSquares * e);
    final row = Float32List(inSize);
    for (var s = 0; s < _kSquares; s++) {
      for (var p = 0; p < _kInputPlanes; p++) {
        row[p] = inputNCHW[p * _kSquares + s];
      }
      for (var c = 0; c < 64; c++) {
        row[_kInputPlanes + c] = kLc0PosEncoding[s * 64 + c];
      }
      for (var o = 0; o < e; o++) {
        var sum = w.ipEmbB[o];
        final base = o * inSize;
        for (var k = 0; k < inSize; k++) {
          sum += row[k] * w.ipEmbW[base + k];
        }
        var v = _mish(sum);
        if (w.ipMultGate != null) {
          v = v * w.ipMultGate![o * _kSquares + s] +
              w.ipAddGate![o * _kSquares + s];
        }
        out[s * e + o] = v;
      }
    }
    return out;
  }

  /// Full attention body (M3): input planes -> encoder-stack output `[64*embDim]`.
  Float32List encode(Float32List inputNCHW) {
    var x = embed(inputNCHW);
    final alpha = math.pow(2.0 * w.encoders.length, -0.25).toDouble();
    for (final layer in w.encoders) {
      x = _encoder(x, layer, alpha);
    }
    return x;
  }

  /// One transformer encoder layer (MHA + smolgen, FFN, two LayerNorm+skip).
  Float32List _encoder(Float32List x, Lc0AttnEncoder l, double alpha) {
    final e = w.embDim;
    final heads = w.heads;
    final depth = e ~/ heads;
    final scaling = 1.0 / math.sqrt(depth);

    // Smolgen: produces a per-head [64*64] attention-logit bias.
    Float32List? smBias;
    if (l.hasSmolgen) {
      final hc = l.smCompress!.length ~/ e; // hidden channels (32)
      final comp = _fc(x, l.smCompress!, null, false, 64, e, hc); // [64,hc]
      final hidden = l.smD1B!.length;
      final d1 = _fc(comp, l.smD1W!, l.smD1B, true, 1, 64 * hc, hidden);
      _layerNorm(d1, 1.0, null, l.smLn1G!, l.smLn1B!, 1e-3, 1, hidden);
      final genOut = l.smD2B!.length;
      final d2 = _fc(d1, l.smD2W!, l.smD2B, true, 1, hidden, genOut);
      _layerNorm(d2, 1.0, null, l.smLn2G!, l.smLn2B!, 1e-3, 1, genOut);
      final perHead = genOut ~/ heads;
      smBias = _fc(d2, w.smolgenW!, null, false, heads, perHead, 64 * 64);
    }

    final q = _fc(x, l.qW, l.qB, false, 64, e, e);
    final k = _fc(x, l.kW, l.kB, false, 64, e, e);
    final v = _fc(x, l.vW, l.vB, false, 64, e, e);

    final attn = Float32List(64 * e);
    final row = Float32List(64);
    for (var h = 0; h < heads; h++) {
      final ho = h * depth;
      for (var i = 0; i < 64; i++) {
        var maxL = double.negativeInfinity;
        for (var j = 0; j < 64; j++) {
          var dot = 0.0;
          for (var dd = 0; dd < depth; dd++) {
            dot += q[i * e + ho + dd] * k[j * e + ho + dd];
          }
          var lg = dot * scaling;
          if (smBias != null) lg += smBias[h * 4096 + i * 64 + j];
          row[j] = lg;
          if (lg > maxL) maxL = lg;
        }
        var denom = 0.0;
        for (var j = 0; j < 64; j++) {
          final ex = math.exp(row[j] - maxL);
          row[j] = ex;
          denom += ex;
        }
        final inv = 1.0 / denom;
        for (var dd = 0; dd < depth; dd++) {
          var acc = 0.0;
          for (var j = 0; j < 64; j++) {
            acc += row[j] * v[j * e + ho + dd];
          }
          attn[i * e + ho + dd] = acc * inv;
        }
      }
    }

    final mhaOut = _fc(attn, l.denseW, l.denseB, false, 64, e, e);
    _layerNorm(mhaOut, alpha, x, l.ln1G, l.ln1B, 1e-6, 64, e); // LN(alpha*mha + x)
    final y = mhaOut;

    final h1 = _fc(y, l.ffn1W, l.ffn1B, true, 64, e, w.dff);
    final ffnOut = _fc(h1, l.ffn2W, l.ffn2B, false, 64, w.dff, e);
    _layerNorm(ffnOut, alpha, y, l.ln2G, l.ln2B, 1e-6, 64, e); // LN(alpha*ffn + y)
    return ffnOut;
  }

  /// lc0 FullyConnectedLayer::Forward1D: out[m,n] = act(sum_k in[m,k]*w[n,k] + b[n]).
  Float32List _fc(Float32List input, Float32List weight, Float32List? bias,
      bool mish, int m, int k, int n) {
    final out = Float32List(m * n);
    for (var mi = 0; mi < m; mi++) {
      final ib = mi * k;
      final ob = mi * n;
      for (var ni = 0; ni < n; ni++) {
        var sum = bias != null ? bias[ni] : 0.0;
        final wb = ni * k;
        for (var ki = 0; ki < k; ki++) {
          sum += input[ib + ki] * weight[wb + ki];
        }
        out[ob + ni] = mish ? _mish(sum) : sum;
      }
    }
    return out;
  }

  /// lc0 LayerNorm2DWithSkipConnection (in place): combined = alpha*data + skip,
  /// then normalize over [ch] per row and scale/shift by gamma/beta.
  void _layerNorm(Float32List data, double alpha, Float32List? skip,
      Float32List gamma, Float32List beta, double eps, int rows, int ch) {
    for (var i = 0; i < rows; i++) {
      final base = i * ch;
      var mean = 0.0;
      if (skip != null) {
        for (var c = 0; c < ch; c++) {
          final val = data[base + c] * alpha + skip[base + c];
          data[base + c] = val;
          mean += val;
        }
      } else {
        for (var c = 0; c < ch; c++) {
          final val = data[base + c] * alpha;
          data[base + c] = val;
          mean += val;
        }
      }
      mean /= ch;
      var variance = 0.0;
      for (var c = 0; c < ch; c++) {
        final diff = data[base + c] - mean;
        variance += diff * diff;
      }
      variance /= ch;
      final den = 1.0 / math.sqrt(variance + eps);
      for (var c = 0; c < ch; c++) {
        data[base + c] = beta[c] + gamma[c] * (data[base + c] - mean) * den;
      }
    }
  }

  /// Full forward: input planes `[112*64]` -> policy(1858) + WDL + moves-left.
  Lc0AttnOutput forward(Float32List inputNCHW) {
    final body = encode(inputNCHW);
    return Lc0AttnOutput(_policy(body), _value(body), _movesLeft(body));
  }

  /// Attention policy head -> 1858 classical move logits (via kAttnPolicyMap).
  Float32List _policy(Float32List body) {
    final e = w.embDim;
    final polEmb = w.ipPolB.length;
    final emb = _fc(body, w.ipPolW, w.ipPolB, true, 64, e, polEmb); // MISH
    final dModel = w.ip2PolB.length;
    final q = _fc(emb, w.ip2PolW, w.ip2PolB, false, 64, polEmb, dModel);
    final k = _fc(emb, w.ip3PolW, w.ip3PolB, false, 64, polEmb, dModel);
    final scaling = 1.0 / math.sqrt(dModel);

    final hb = Float32List(64 * 64 + 8 * 24);
    for (var m = 0; m < 64; m++) {
      for (var n = 0; n < 64; n++) {
        var dot = 0.0;
        for (var c = 0; c < dModel; c++) {
          dot += q[m * dModel + c] * k[n * dModel + c];
        }
        hb[m * 64 + n] = dot * scaling;
      }
    }
    // Promotion offsets from the rank-8 keys and the ppo weight [4, dModel].
    final promo = [for (var i = 0; i < 4; i++) Float32List(8)];
    for (var i = 0; i < 4; i++) {
      for (var j = 0; j < 8; j++) {
        var sum = 0.0;
        for (var c = 0; c < dModel; c++) {
          sum += k[(56 + j) * dModel + c] * w.ip4PolW[i * dModel + c];
        }
        promo[i][j] = sum;
      }
    }
    for (var i = 0; i < 3; i++) {
      for (var j = 0; j < 8; j++) {
        promo[i][j] += promo[3][j];
      }
    }
    for (var kk = 0; kk < 8; kk++) {
      for (var j = 0; j < 8; j++) {
        for (var i = 0; i < 3; i++) {
          hb[4096 + 24 * kk + 3 * j + i] =
              hb[(48 + kk) * 64 + 56 + j] + promo[i][j];
        }
      }
    }
    final pol = Float32List(1858);
    for (var idx = 0; idx < kAttnPolicyMap.length; idx++) {
      final j = kAttnPolicyMap[idx];
      if (j >= 0) pol[j] = hb[idx];
    }
    return pol;
  }

  /// Attention value head -> WDL (softmaxed).
  List<double> _value(Float32List body) {
    final vp = w.ipValB.length;
    final emb = _fc(body, w.ipValW, w.ipValB, true, 64, w.embDim, vp); // MISH
    final vc = w.ip1ValB.length;
    final h1 = _fc(emb, w.ip1ValW, w.ip1ValB, true, 1, 64 * vp, vc); // MISH
    final logits = _fc(h1, w.ip2ValW, w.ip2ValB, false, 1, vc, w.wdl);
    return _softmax(logits);
  }

  double _movesLeft(Float32List body) {
    if (w.ipMovW == null) return 0.0;
    final mp = w.ipMovB!.length;
    final emb = _fc(body, w.ipMovW!, w.ipMovB, true, 64, w.embDim, mp); // MISH
    final mc = w.ip1MovB!.length;
    final h1 = _fc(emb, w.ip1MovW!, w.ip1MovB, true, 1, 64 * mp, mc); // MISH
    final out = _fc(h1, w.ip2MovW!, w.ip2MovB, false, 1, mc, 1);
    return out[0] > 0 ? out[0] : 0.0; // lc0: ip2_mov uses RELU
  }

  List<double> _softmax(Float32List logits) {
    var mx = double.negativeInfinity;
    for (final v in logits) {
      if (v > mx) mx = v;
    }
    var denom = 0.0;
    final out = List<double>.filled(logits.length, 0.0);
    for (var i = 0; i < logits.length; i++) {
      final e = math.exp(logits[i] - mx);
      out[i] = e;
      denom += e;
    }
    for (var i = 0; i < out.length; i++) {
      out[i] /= denom;
    }
    return out;
  }
}

// ---------------- minimal protobuf helpers ----------------

class _Range {
  final int start, end;
  const _Range(this.start, this.end);
}

int? _varintField(Uint8List d, _Range m, int field) {
  final r = _rawField(d, m, field);
  return r == null ? null : _varint(d, r.start).value;
}

_Range? _field(Uint8List d, _Range m, int field) {
  final r = _rawField(d, m, field);
  return r == null ? null : _Range(r.cs, r.ce);
}

List<_Range> _fields(Uint8List d, _Range m, int field) {
  final out = <_Range>[];
  var i = m.start;
  while (i < m.end) {
    final f = _read(d, i);
    if (f == null) break;
    if (f.number == field) out.add(_Range(f.cs, f.ce));
    i = f.ce;
  }
  return out;
}

class _Raw {
  final int start, cs, ce;
  _Raw(this.start, this.cs, this.ce);
}

_Raw? _rawField(Uint8List d, _Range m, int field) {
  var i = m.start;
  while (i < m.end) {
    final f = _read(d, i);
    if (f == null) return null;
    if (f.number == field) return _Raw(f.cs, f.cs, f.ce);
    i = f.ce;
  }
  return null;
}

/// Dequantize a Layer message (min_val=1,max_val=2,params=3,encoding=4).
Float32List _layer(Uint8List d, _Range layer) {
  double minVal = 0, maxVal = 0;
  int enc = 1, ps = 0, pe = 0;
  var i = layer.start;
  while (i < layer.end) {
    final f = _read(d, i)!;
    switch (f.number) {
      case 1:
        minVal = _f32(d, f.cs);
      case 2:
        maxVal = _f32(d, f.cs);
      case 3:
        ps = f.cs;
        pe = f.ce;
      case 4:
        enc = _varint(d, f.cs).value;
    }
    i = f.ce;
  }
  final n = (pe - ps) ~/ 2;
  final out = Float32List(n);
  if (enc == 1) {
    final range = maxVal - minVal;
    var j = 0;
    for (var p = ps; p < pe; p += 2) {
      final v = d[p] | (d[p + 1] << 8);
      out[j++] = minVal + (v / 65535.0) * range;
    }
  } else if (enc == 2) {
    var j = 0;
    for (var p = ps; p < pe; p += 2) {
      out[j++] = _half2float(d[p] | (d[p + 1] << 8));
    }
  } else {
    throw ArgumentError('lc0-attn: unsupported Layer encoding $enc');
  }
  return out;
}

class _F {
  final int number, wire, cs, ce;
  _F(this.number, this.wire, this.cs, this.ce);
}

_F? _read(Uint8List d, int i) {
  final tag = _varint(d, i);
  final number = tag.value >> 3;
  final wire = tag.value & 7;
  final p = tag.next;
  switch (wire) {
    case 0:
      return _F(number, wire, p, _varint(d, p).next);
    case 1:
      return _F(number, wire, p, p + 8);
    case 2:
      final len = _varint(d, p);
      return _F(number, wire, len.next, len.next + len.value);
    case 5:
      return _F(number, wire, p, p + 4);
    default:
      return null;
  }
}

class _V {
  final int value, next;
  _V(this.value, this.next);
}

_V _varint(Uint8List d, int i) {
  var shift = 0, result = 0;
  while (i < d.length) {
    final b = d[i++];
    result |= (b & 0x7f) << shift;
    if ((b & 0x80) == 0) return _V(result, i);
    shift += 7;
  }
  return _V(result, i);
}

double _f32(Uint8List d, int i) =>
    ByteData.sublistView(d, i, i + 4).getFloat32(0, Endian.little);

double _half2float(int h) {
  final sign = (h & 0x8000) << 16;
  final exp = (h >> 10) & 0x1f;
  final mant = h & 0x3ff;
  int bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      var e = -1, m = mant;
      do {
        e++;
        m <<= 1;
      } while ((m & 0x400) == 0);
      m &= 0x3ff;
      bits = sign | ((112 - e) << 23) | (m << 13);
    }
  } else if (exp == 0x1f) {
    bits = sign | 0x7f800000 | (mant << 13);
  } else {
    bits = sign | ((exp + 112) << 23) | (mant << 13);
  }
  final bd = ByteData(4)..setUint32(0, bits, Endian.little);
  return bd.getFloat32(0, Endian.little);
}
