library;

import 'package:dart_pytorch/core/nn/fold/geometry.dart';
import 'package:test/test.dart';

/// Assert two vectors agree within [tol] per component.
void _expectVec(Vec3 a, Vec3 b, {double tol = 1e-6}) {
  expect(a.x, closeTo(b.x, tol), reason: 'x');
  expect(a.y, closeTo(b.y, tol), reason: 'y');
  expect(a.z, closeTo(b.z, tol), reason: 'z');
}

void main() {
  group('Rot3', () {
    test('identity quaternion yields identity rotation', () {
      final r = Rot3.fromQuaternion(1, 0, 0, 0);
      _expectVec(r.apply(const Vec3(1, 2, 3)), const Vec3(1, 2, 3));
    });

    test('quaternion rotation is orthonormal (RᵀR = I, det = +1)', () {
      final r = Rot3.fromQuaternion(0.3, -0.7, 0.4, 0.9);
      final rtr = r.transpose.compose(r);
      _expectVec(rtr.apply(const Vec3(1, 0, 0)), const Vec3(1, 0, 0));
      _expectVec(rtr.apply(const Vec3(0, 1, 0)), const Vec3(0, 1, 0));
      _expectVec(rtr.apply(const Vec3(0, 0, 1)), const Vec3(0, 0, 1));
      expect(r.determinant, closeTo(1.0, 1e-9));
    });

    test('rotation preserves vector length', () {
      final r = Rot3.fromQuaternion(0.1, 0.2, -0.5, 0.8);
      const v = Vec3(1.5, -2.0, 0.75);
      expect(r.apply(v).norm, closeTo(v.norm, 1e-9));
    });

    test('transpose is the inverse rotation', () {
      final r = Rot3.fromQuaternion(0.6, 0.1, -0.2, 0.4);
      const v = Vec3(2, -1, 3);
      _expectVec(r.applyTranspose(r.apply(v)), v);
    });
  });

  group('Rigid', () {
    final frame = Rigid(
      Rot3.fromQuaternion(0.5, -0.3, 0.7, 0.2),
      const Vec3(4, -1, 2),
    );

    test('applyInverse undoes apply', () {
      const v = Vec3(1, 2, 3);
      _expectVec(frame.applyInverse(frame.apply(v)), v);
    });

    test('compose with inverse gives identity transform', () {
      final id = frame.compose(frame.inverse);
      const v = Vec3(-2, 5, 1);
      _expectVec(id.apply(v), v, tol: 1e-6);
    });

    test('compose matches sequential application', () {
      final a = Rigid(
        Rot3.fromQuaternion(0.9, 0.1, 0.2, -0.3),
        const Vec3(1, 0, -1),
      );
      final b = Rigid(
        Rot3.fromQuaternion(0.4, -0.6, 0.5, 0.1),
        const Vec3(-2, 3, 0),
      );
      const v = Vec3(0.5, -0.5, 2);
      _expectVec(a.compose(b).apply(v), a.apply(b.apply(v)));
    });
  });

  group('rigidFromThreePoints', () {
    test('translation is Cα and rotation is orthonormal', () {
      const n = Vec3(1, 2, 3);
      const ca = Vec3(0.5, -1, 2);
      const c = Vec3(-1, 0, 1);
      final f = rigidFromThreePoints(n, ca, c);
      _expectVec(f.trans, ca);
      expect(f.rot.determinant, closeTo(1.0, 1e-9));
      // e1 (first column) points along (c - ca).
      final e1 = f.rot.apply(const Vec3(1, 0, 0));
      _expectVec(e1, (c - ca).normalized());
    });

    test('frame maps ideal backbone to a valid local layout', () {
      // Build a frame from arbitrary atoms, then map global atoms back into
      // the local frame: Cα -> origin, C -> +x axis, N -> xy-plane (z≈0).
      const n = Vec3(2.1, -0.4, 1.7);
      const ca = Vec3(0.0, 0.0, 0.0);
      const c = Vec3(1.3, 0.9, -0.2);
      final f = rigidFromThreePoints(n, ca, c);
      final localC = f.applyInverse(c);
      final localN = f.applyInverse(n);
      _expectVec(f.applyInverse(ca), Vec3.zero);
      expect(localC.y, closeTo(0, 1e-6), reason: 'C on x-axis: y=0');
      expect(localC.z, closeTo(0, 1e-6), reason: 'C on x-axis: z=0');
      expect(localC.x, greaterThan(0), reason: 'C on +x');
      expect(localN.z, closeTo(0, 1e-6), reason: 'N in xy-plane: z=0');
    });

    test('backboneAtoms reproduces ideal geometry up to the frame', () {
      final f = Rigid(
        Rot3.fromQuaternion(0.7, 0.2, -0.1, 0.5),
        const Vec3(3, 3, 3),
      );
      final atoms = backboneAtoms(f);
      // Cα lands exactly on the translation.
      _expectVec(atoms.cAlpha, const Vec3(3, 3, 3));
      // Ideal N–Cα and C–Cα distances are preserved by the rigid motion.
      expect((atoms.n - atoms.cAlpha).norm,
          closeTo(IdealBackbone.n.norm, 1e-9));
      expect((atoms.c - atoms.cAlpha).norm,
          closeTo(IdealBackbone.c.norm, 1e-9));
    });
  });

  test('roundtrip: recover a frame from the atoms it generated', () {
    // A frame generates ideal backbone atoms; rebuilding a frame from those
    // atoms must recover the original rotation and translation.
    final f = Rigid(
      Rot3.fromQuaternion(0.55, -0.4, 0.3, 0.66),
      const Vec3(-2, 5, 1),
    );
    final a = backboneAtoms(f);
    final g = rigidFromThreePoints(a.n, a.cAlpha, a.c);
    _expectVec(g.trans, f.trans);
    for (final v in const [Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)]) {
      _expectVec(g.rot.apply(v), f.rot.apply(v), tol: 1e-6);
    }
  });

  test('frames are equivariant under a global rigid motion', () {
    // Applying a global transform G to the input atoms should transform the
    // derived frame by G as well (left-composition) — the property IPA relies
    // on for rotation/translation equivariance.
    const n = Vec3(1.2, 0.3, -0.7);
    const ca = Vec3(0.1, 0.2, 0.3);
    const c = Vec3(-0.8, 1.1, 0.4);
    final f = rigidFromThreePoints(n, ca, c);

    final g = Rigid(
      Rot3.fromQuaternion(0.3, 0.5, -0.2, 0.77),
      const Vec3(10, -4, 6),
    );
    final f2 = rigidFromThreePoints(g.apply(n), g.apply(ca), g.apply(c));
    final expected = g.compose(f);

    _expectVec(f2.trans, expected.trans, tol: 1e-6);
    for (final v in const [Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)]) {
      _expectVec(f2.rot.apply(v), expected.rot.apply(v), tol: 1e-6);
    }
  });
}
