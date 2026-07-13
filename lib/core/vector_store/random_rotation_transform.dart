/// `RandomRotationTransform` — multiplies each vector by a fixed random
/// orthogonal matrix `R ∈ ℝ^{d×d}`. Sampled once from `Random(seed)`
/// via QR of a Gaussian matrix, so distances and inner products are
/// preserved exactly (up to floating-point rounding).
///
/// Useful upstream of quantization (PQ, LSH) to spread information
/// evenly across dimensions, and as a component of OPQ.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'index_io.dart';
import 'vector_transform.dart';

class RandomRotationTransform extends VectorTransform {
  RandomRotationTransform({required int d, this.seed = 1234})
    : _r = Float32List(d * d),
      super(d, d) {
    _generate();
    isTrained = true;
  }

  final int seed;
  final Float32List _r; // row-major d × d rotation matrix

  Float32List get rotation => _r;

  void _generate() {
    // Modified Gram–Schmidt on a Gaussian d×d matrix produces a random
    // orthogonal Q, which is what we store as the rotation. Suitable
    // for moderate d; O(d³) but only runs once at construction.
    final rng = math.Random(seed);
    final a = List<Float64List>.generate(dIn, (_) => Float64List(dIn));
    for (var i = 0; i < dIn; i++) {
      for (var j = 0; j < dIn; j++) {
        // Box–Muller.
        final u1 = math.max(rng.nextDouble(), 1e-12);
        final u2 = rng.nextDouble();
        a[i][j] = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2);
      }
    }
    // Modified Gram–Schmidt: orthonormalize rows of `a`.
    for (var i = 0; i < dIn; i++) {
      final row = a[i];
      for (var k = 0; k < i; k++) {
        final prev = a[k];
        var dot = 0.0;
        for (var j = 0; j < dIn; j++) {
          dot += row[j] * prev[j];
        }
        for (var j = 0; j < dIn; j++) {
          row[j] -= dot * prev[j];
        }
      }
      var norm = 0.0;
      for (var j = 0; j < dIn; j++) {
        norm += row[j] * row[j];
      }
      norm = math.sqrt(norm);
      if (norm < 1e-12) {
        // Extremely unlikely for random gaussians; regenerate this row.
        for (var j = 0; j < dIn; j++) {
          row[j] = (j == i) ? 1.0 : 0.0;
        }
      } else {
        final inv = 1.0 / norm;
        for (var j = 0; j < dIn; j++) {
          row[j] *= inv;
        }
      }
    }
    for (var i = 0; i < dIn; i++) {
      for (var j = 0; j < dIn; j++) {
        _r[i * dIn + j] = a[i][j];
      }
    }
  }

  @override
  List<Float32List> apply(List<Float32List> xs) {
    return List<Float32List>.generate(xs.length, (i) {
      final row = xs[i];
      if (row.length != dIn) {
        throw ArgumentError('vector $i length ${row.length} != $dIn');
      }
      final o = Float32List(dIn);
      for (var r = 0; r < dIn; r++) {
        var s = 0.0;
        final base = r * dIn;
        for (var c = 0; c < dIn; c++) {
          s += _r[base + c] * row[c];
        }
        o[r] = s;
      }
      return o;
    });
  }

  @override
  List<Float32List> reverseTransform(List<Float32List> xs) {
    // R is orthogonal, so R⁻¹ = Rᵀ.
    return List<Float32List>.generate(xs.length, (i) {
      final row = xs[i];
      if (row.length != dIn) {
        throw ArgumentError('vector $i length ${row.length} != $dIn');
      }
      final o = Float32List(dIn);
      for (var c = 0; c < dIn; c++) {
        var s = 0.0;
        for (var r = 0; r < dIn; r++) {
          s += _r[r * dIn + c] * row[r];
        }
        o[c] = s;
      }
      return o;
    });
  }

  @override
  void writeTo(IoWriter w) {
    w.writeU32(TransformKind.randomRotation);
    w.writeU32(dIn);
    w.writeU32(seed);
    w.writeF32List(_r);
  }

  static RandomRotationTransform readFrom(IoReader r) {
    final d = r.readU32();
    final seed = r.readU32();
    final t = RandomRotationTransform(d: d, seed: seed);
    final mat = r.readF32List(d * d);
    for (var i = 0; i < mat.length; i++) {
      t._r[i] = mat[i];
    }
    return t;
  }
}
