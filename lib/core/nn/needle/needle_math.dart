/// Low-level numeric primitives for the Needle 3 forward pass, matching the
/// reference `architecture.py`. Operate on row-major [Float64List] buffers in
/// the repo's single-sequence convention (rows = sequence positions).
library;

import 'dart:math' as math;
import 'dart:typed_data';

/// Zero-centred RMSNorm (`ZCRMSNorm` in the reference): for each of [rows]
/// rows of width [dim], `out = (1 + scale) * x / rms`, where
/// `rms = sqrt(mean(x^2) + eps)`. [scale] has length [dim] (init 0 → gain 1).
/// In place.
void zcRmsNorm(
  Float64List x,
  int rows,
  int dim,
  Float64List scale, {
  double eps = 1e-6,
}) {
  for (var r = 0; r < rows; r++) {
    final base = r * dim;
    var ss = 0.0;
    for (var i = 0; i < dim; i++) {
      final v = x[base + i];
      ss += v * v;
    }
    final inv = 1.0 / math.sqrt(ss / dim + eps);
    for (var i = 0; i < dim; i++) {
      x[base + i] = (1.0 + scale[i]) * x[base + i] * inv;
    }
  }
}

/// `rms_unit`: normalise each row to unit RMS with no learned scale.
/// Returns a new buffer.
Float64List rmsUnit(
  Float64List x,
  int rows,
  int dim, {
  double eps = 1e-6,
}) {
  final out = Float64List(rows * dim);
  for (var r = 0; r < rows; r++) {
    final base = r * dim;
    var ss = 0.0;
    for (var i = 0; i < dim; i++) {
      final v = x[base + i];
      ss += v * v;
    }
    final inv = 1.0 / math.sqrt(ss / dim + eps);
    for (var i = 0; i < dim; i++) {
      out[base + i] = x[base + i] * inv;
    }
  }
  return out;
}

/// SiLU / swish: `x * sigmoid(x)`.
double silu(double x) => x / (1.0 + math.exp(-x));

/// Logistic sigmoid.
double sigmoid(double x) => 1.0 / (1.0 + math.exp(-x));

/// In-place row-wise softmax over [cols] columns for [rows] rows.
void softmaxRows(Float64List x, int rows, int cols) {
  for (var r = 0; r < rows; r++) {
    final base = r * cols;
    var mx = double.negativeInfinity;
    for (var c = 0; c < cols; c++) {
      if (x[base + c] > mx) mx = x[base + c];
    }
    var sum = 0.0;
    for (var c = 0; c < cols; c++) {
      final e = math.exp(x[base + c] - mx);
      x[base + c] = e;
      sum += e;
    }
    final inv = 1.0 / sum;
    for (var c = 0; c < cols; c++) {
      x[base + c] *= inv;
    }
  }
}

/// RoPE cos/sin tables of shape `[seqLen, headDim/2]` for the half-split
/// rotary convention: `freqs = 1 / theta^(2i/headDim)`.
(Float64List, Float64List) ropeTables(
  int seqLen,
  int headDim,
  double theta,
) {
  final half = headDim ~/ 2;
  final cos = Float64List(seqLen * half);
  final sin = Float64List(seqLen * half);
  for (var t = 0; t < seqLen; t++) {
    for (var i = 0; i < half; i++) {
      final freq = 1.0 / math.pow(theta, (2.0 * i) / headDim);
      final a = t * freq;
      cos[t * half + i] = math.cos(a);
      sin[t * half + i] = math.sin(a);
    }
  }
  return (cos, sin);
}

/// Apply half-split RoPE in place to one head's `[T, headDim]` buffer.
/// `x1 = x[:half]`, `x2 = x[half:]`;
/// `out = concat(x1·cos − x2·sin, x2·cos + x1·sin)`.
void applyRopeHead(
  Float64List x,
  int seqLen,
  int headDim,
  Float64List cos,
  Float64List sin,
) {
  final half = headDim ~/ 2;
  for (var t = 0; t < seqLen; t++) {
    final xb = t * headDim;
    final cb = t * half;
    for (var i = 0; i < half; i++) {
      final x1 = x[xb + i];
      final x2 = x[xb + half + i];
      final c = cos[cb + i];
      final s = sin[cb + i];
      x[xb + i] = x1 * c - x2 * s;
      x[xb + half + i] = x2 * c + x1 * s;
    }
  }
}
