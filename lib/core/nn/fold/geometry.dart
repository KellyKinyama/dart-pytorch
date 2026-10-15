/// Rigid-body geometry for protein structure prediction (ESMFold /
/// AlphaFold2 structure module).
///
/// The structure module reasons about each residue as a local coordinate
/// frame — a **rigid transform** `(R, t)` with a 3x3 rotation `R` and a
/// translation `t` — anchored at the Cα atom. Backbone atoms live at fixed
/// *ideal* positions inside that frame; applying the frame places them in
/// global space. Invariant Point Attention (IPA) and the backbone update
/// operate entirely in terms of these frames, which is what makes ESMFold
/// equivariant to global rotation/translation.
///
/// This file is pure Dart (no [Tensor]/autograd) because the geometry is
/// inference-only, double-precision, and far clearer as plain arrays. It
/// deliberately mirrors OpenFold's `Rotation`/`Rigid` and
/// `residue_constants` conventions so pretrained weights transfer exactly.
library;

import 'dart:math' as math;

/// A 3-vector `(x, y, z)`.
class Vec3 {
  final double x, y, z;
  const Vec3(this.x, this.y, this.z);

  static const zero = Vec3(0, 0, 0);

  Vec3 operator +(Vec3 o) => Vec3(x + o.x, y + o.y, z + o.z);
  Vec3 operator -(Vec3 o) => Vec3(x - o.x, y - o.y, z - o.z);
  Vec3 scale(double s) => Vec3(x * s, y * s, z * s);

  double dot(Vec3 o) => x * o.x + y * o.y + z * o.z;
  Vec3 cross(Vec3 o) =>
      Vec3(y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x);

  double get norm => math.sqrt(x * x + y * y + z * z);

  Vec3 normalized({double eps = 1e-8}) {
    final n = norm;
    return n < eps ? this : scale(1.0 / n);
  }

  List<double> toList() => [x, y, z];

  @override
  String toString() =>
      'Vec3(${x.toStringAsFixed(3)}, ${y.toStringAsFixed(3)}, '
      '${z.toStringAsFixed(3)})';
}

/// A 3x3 rotation matrix, stored row-major as nine doubles.
///
/// Columns are the frame's basis vectors in global coordinates, so
/// [apply] computes `R · v` (local → global) and [applyTranspose]
/// computes `Rᵀ · v` (global → local), which equals the inverse for an
/// orthonormal rotation.
class Rot3 {
  // Row-major: [m00 m01 m02 m10 m11 m12 m20 m21 m22].
  final double m00, m01, m02, m10, m11, m12, m20, m21, m22;

  const Rot3(
    this.m00,
    this.m01,
    this.m02,
    this.m10,
    this.m11,
    this.m12,
    this.m20,
    this.m21,
    this.m22,
  );

  static const identity = Rot3(1, 0, 0, 0, 1, 0, 0, 0, 1);

  /// Build a rotation from its column basis vectors.
  factory Rot3.fromColumns(Vec3 e1, Vec3 e2, Vec3 e3) => Rot3(
        e1.x, e2.x, e3.x, //
        e1.y, e2.y, e3.y, //
        e1.z, e2.z, e3.z,
      );

  /// Rotation from a (not necessarily normalized) quaternion `(a, b, c, d)`
  /// with real part `a`. Matches the AlphaFold2 backbone-update convention
  /// where IPA predicts `(b, c, d)` with an implicit `a = 1`, then the
  /// quaternion is normalized here.
  factory Rot3.fromQuaternion(double a, double b, double c, double d) {
    final n = math.sqrt(a * a + b * b + c * c + d * d);
    final inv = n < 1e-12 ? 0.0 : 1.0 / n;
    a *= inv;
    b *= inv;
    c *= inv;
    d *= inv;
    final aa = a * a, bb = b * b, cc = c * c, dd = d * d;
    final ab = a * b, ac = a * c, ad = a * d;
    final bc = b * c, bd = b * d, cd = c * d;
    return Rot3(
      aa + bb - cc - dd, 2 * (bc - ad), 2 * (bd + ac), //
      2 * (bc + ad), aa - bb + cc - dd, 2 * (cd - ab), //
      2 * (bd - ac), 2 * (cd + ab), aa - bb - cc + dd,
    );
  }

  /// `R · v` — rotate a local vector into global coordinates.
  Vec3 apply(Vec3 v) => Vec3(
        m00 * v.x + m01 * v.y + m02 * v.z,
        m10 * v.x + m11 * v.y + m12 * v.z,
        m20 * v.x + m21 * v.y + m22 * v.z,
      );

  /// `Rᵀ · v` — rotate a global vector into this frame (inverse rotation).
  Vec3 applyTranspose(Vec3 v) => Vec3(
        m00 * v.x + m10 * v.y + m20 * v.z,
        m01 * v.x + m11 * v.y + m21 * v.z,
        m02 * v.x + m12 * v.y + m22 * v.z,
      );

  /// Matrix product `this · other`.
  Rot3 compose(Rot3 o) => Rot3(
        m00 * o.m00 + m01 * o.m10 + m02 * o.m20,
        m00 * o.m01 + m01 * o.m11 + m02 * o.m21,
        m00 * o.m02 + m01 * o.m12 + m02 * o.m22,
        m10 * o.m00 + m11 * o.m10 + m12 * o.m20,
        m10 * o.m01 + m11 * o.m11 + m12 * o.m21,
        m10 * o.m02 + m11 * o.m12 + m12 * o.m22,
        m20 * o.m00 + m21 * o.m10 + m22 * o.m20,
        m20 * o.m01 + m21 * o.m11 + m22 * o.m21,
        m20 * o.m02 + m21 * o.m12 + m22 * o.m22,
      );

  /// Transpose, which is the inverse for an orthonormal rotation.
  Rot3 get transpose =>
      Rot3(m00, m10, m20, m01, m11, m21, m02, m12, m22);

  double get determinant =>
      m00 * (m11 * m22 - m12 * m21) -
      m01 * (m10 * m22 - m12 * m20) +
      m02 * (m10 * m21 - m11 * m20);
}

/// A rigid transform `(rotation, translation)` — a residue's local frame.
class Rigid {
  final Rot3 rot;
  final Vec3 trans;
  const Rigid(this.rot, this.trans);

  static const identity = Rigid(Rot3.identity, Vec3.zero);

  /// Map a local point into global coordinates: `R · v + t`.
  Vec3 apply(Vec3 v) => rot.apply(v) + trans;

  /// Map a global point into this frame: `Rᵀ · (v - t)`.
  Vec3 applyInverse(Vec3 v) => rot.applyTranspose(v - trans);

  /// Compose transforms: `this ∘ other`, applied right-to-left.
  Rigid compose(Rigid o) =>
      Rigid(rot.compose(o.rot), rot.apply(o.trans) + trans);

  /// Inverse transform `(Rᵀ, -Rᵀ·t)`.
  Rigid get inverse {
    final rt = rot.transpose;
    return Rigid(rt, rt.apply(trans).scale(-1));
  }
}

/// Build a residue frame from three backbone atoms using Gram-Schmidt,
/// following AlphaFold2's `rigid_from_3_points` (OpenFold convention):
///
///   * translation = `cAlpha` (the Cα origin),
///   * x-axis `e1` along `c - cAlpha`,
///   * `e2` = component of `n - cAlpha` orthogonal to `e1`, normalized,
///   * `e3 = e1 × e2`.
///
/// The resulting rotation is orthonormal with determinant +1.
Rigid rigidFromThreePoints(Vec3 n, Vec3 cAlpha, Vec3 c) {
  final v1 = c - cAlpha;
  final v2 = n - cAlpha;
  final e1 = v1.normalized();
  final u2 = v2 - e1.scale(e1.dot(v2));
  final e2 = u2.normalized();
  final e3 = e1.cross(e2);
  return Rigid(Rot3.fromColumns(e1, e2, e3), cAlpha);
}

/// Ideal backbone atom positions (Å) in the residue's local frame, matching
/// AlphaFold2 `residue_constants.rigid_group_atom_positions` for the
/// backbone group. Cα sits at the origin and C on the +x axis, consistent
/// with [rigidFromThreePoints].
class IdealBackbone {
  static const n = Vec3(-0.525, 1.363, 0.0);
  static const cAlpha = Vec3(0.0, 0.0, 0.0);
  static const c = Vec3(1.526, 0.0, 0.0);
  // Cβ direction (used for a crude Gly/non-Gly placement if needed).
  static const cBeta = Vec3(-0.529, -0.774, -1.205);
}

/// Place the ideal backbone atoms (N, Cα, C) of a residue into global
/// coordinates given its frame.
({Vec3 n, Vec3 cAlpha, Vec3 c}) backboneAtoms(Rigid frame) => (
      n: frame.apply(IdealBackbone.n),
      cAlpha: frame.apply(IdealBackbone.cAlpha),
      c: frame.apply(IdealBackbone.c),
    );
