/// Walsh-Hadamard + Monarch primitives for the Needle 3 port.
///
/// Two consumers:
///   * CQ dequantisation rotates each 128-weight group by the normalised
///     Walsh-Hadamard matrix `H` (orthonormal, symmetric, `H·H = I`).
///   * the Hadamard MLP applies learnable Monarch mixes `M = A⊗B` that are
///     initialised to `H32⊗H32 = H1024`.
///
/// See https://cactuscompute.com/blog/hadamard-mlp.
library;

import 'dart:math' as math;
import 'dart:typed_data';

/// In-place **orthonormal** fast Walsh-Hadamard transform of [a], whose
/// length must be a power of two. Self-inverse: applying it twice returns
/// the original vector (up to float error).
void fwhtInPlace(Float64List a) {
  final n = a.length;
  if (n == 0 || (n & (n - 1)) != 0) {
    throw ArgumentError('fwht length must be a power of two; got $n');
  }
  for (var len = 1; len < n; len <<= 1) {
    for (var i = 0; i < n; i += len << 1) {
      for (var j = i; j < i + len; j++) {
        final x = a[j];
        final y = a[j + len];
        a[j] = x + y;
        a[j + len] = x - y;
      }
    }
  }
  final inv = 1.0 / math.sqrt(n);
  for (var i = 0; i < n; i++) {
    a[i] *= inv;
  }
}

/// Convenience: orthonormal Walsh-Hadamard transform returning a new list.
Float64List fwht(List<double> x) {
  final a = Float64List.fromList(x);
  fwhtInPlace(a);
  return a;
}

/// One Monarch stage `y = (A⊗B)·x`, applied by folding the length-`t*t`
/// vector [x] into a `t×t` tile `Z` (row-major), computing `Aᵀ Z B`, and
/// flattening back. [a] and [b] are `t×t` row-major matrices.
///
/// `(A⊗B) vec(Z) = vec(Aᵀ Z B)` — see the Hadamard-MLP post.
Float64List monarchApply(Float64List x, Float64List a, Float64List b, int t) {
  if (x.length != t * t) {
    throw ArgumentError('monarch: x length ${x.length} != $t*$t');
  }
  // tmp = Aᵀ Z  (t×t)
  final tmp = Float64List(t * t);
  for (var r = 0; r < t; r++) {
    for (var c = 0; c < t; c++) {
      var s = 0.0;
      for (var k = 0; k < t; k++) {
        s += a[k * t + r] * x[k * t + c]; // Aᵀ[r,k] = A[k,r]
      }
      tmp[r * t + c] = s;
    }
  }
  // y = tmp B  (t×t)
  final y = Float64List(t * t);
  for (var r = 0; r < t; r++) {
    for (var c = 0; c < t; c++) {
      var s = 0.0;
      for (var k = 0; k < t; k++) {
        s += tmp[r * t + k] * b[k * t + c];
      }
      y[r * t + c] = s;
    }
  }
  return y;
}

/// The `t×t` normalised Walsh-Hadamard matrix (row-major), the init value of
/// every Monarch factor. `t` must be a power of two.
Float64List hadamardMatrix(int t) {
  if (t == 0 || (t & (t - 1)) != 0) {
    throw ArgumentError('hadamard size must be a power of two; got $t');
  }
  final m = Float64List(t * t);
  final inv = 1.0 / math.sqrt(t);
  for (var i = 0; i < t; i++) {
    for (var j = 0; j < t; j++) {
      // (-1)^popcount(i & j) gives the Walsh-Hadamard sign pattern.
      final sign = (_popcount(i & j) & 1) == 0 ? 1.0 : -1.0;
      m[i * t + j] = sign * inv;
    }
  }
  return m;
}

/// Monarch tile factorisation of [n] (a power of two): `(ba, bb)` with
/// `ba * bb == n`, matching the reference `_hada_blocks`.
(int, int) hadaBlocks(int n) {
  final b = 1 << ((n - 1).bitLength ~/ 2);
  return (b, n ~/ b);
}

/// One Monarch stage on a length-`aDim*bDim` vector with **rectangular**
/// tile: fold [z] into an `aDim×bDim` tile `Z`, compute `Aᵀ Z B`
/// (`A` is `aDim×aDim`, `B` is `bDim×bDim`), flatten back.
Float64List monarchApplyRect(
  Float64List z,
  Float64List a,
  int aDim,
  Float64List b,
  int bDim,
) {
  if (z.length != aDim * bDim) {
    throw ArgumentError('monarch: z length ${z.length} != $aDim*$bDim');
  }
  // tmp[k,j] = (Aᵀ Z)[k,j] = Σ_i A[i,k] Z[i,j]
  final tmp = Float64List(aDim * bDim);
  for (var k = 0; k < aDim; k++) {
    for (var j = 0; j < bDim; j++) {
      var s = 0.0;
      for (var i = 0; i < aDim; i++) {
        s += a[i * aDim + k] * z[i * bDim + j];
      }
      tmp[k * bDim + j] = s;
    }
  }
  // y[k,l] = Σ_j tmp[k,j] B[j,l]
  final y = Float64List(aDim * bDim);
  for (var k = 0; k < aDim; k++) {
    for (var l = 0; l < bDim; l++) {
      var s = 0.0;
      for (var j = 0; j < bDim; j++) {
        s += tmp[k * bDim + j] * b[j * bDim + l];
      }
      y[k * bDim + l] = s;
    }
  }
  return y;
}

int _popcount(int x) {
  var c = 0;
  var v = x;
  while (v != 0) {
    v &= v - 1;
    c++;
  }
  return c;
}
