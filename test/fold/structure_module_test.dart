library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/core/nn/fold/geometry.dart';
import 'package:dart_pytorch/core/nn/fold/structure_module.dart';
import 'package:test/test.dart';

Float32List _randTensor(int len, int seed) {
  final rng = math.Random(seed);
  final t = Float32List(len);
  for (var i = 0; i < len; i++) {
    t[i] = rng.nextDouble() * 2 - 1;
  }
  return t;
}

List<Rigid> _randFrames(int n, int seed) {
  final rng = math.Random(seed);
  return [
    for (var i = 0; i < n; i++)
      Rigid(
        Rot3.fromQuaternion(rng.nextDouble(), rng.nextDouble() - 0.5,
            rng.nextDouble() - 0.5, rng.nextDouble() - 0.5),
        Vec3((rng.nextDouble() - 0.5) * 4, (rng.nextDouble() - 0.5) * 4,
            (rng.nextDouble() - 0.5) * 4),
      )
  ];
}

void main() {
  // Small dims keep the tests fast while exercising every code path.
  const cfg = IpaConfig(
    cS: 16,
    cZ: 8,
    cHidden: 4,
    noHeads: 2,
    noQkPoints: 3,
    noVPoints: 4,
  );
  const n = 5;

  group('InvariantPointAttention', () {
    test('output has shape [n, cS]', () {
      final ipa = InvariantPointAttention.random(cfg, seed: 1);
      final s = _randTensor(n * cfg.cS, 10);
      final z = _randTensor(n * n * cfg.cZ, 11);
      final frames = _randFrames(n, 12);
      final out = ipa.call(s, z, frames);
      expect(out.length, n * cfg.cS);
      expect(out.every((v) => v.isFinite), isTrue);
    });

    test('single-rep update is invariant to a global rigid motion', () {
      final ipa = InvariantPointAttention.random(cfg, seed: 2);
      final s = _randTensor(n * cfg.cS, 20);
      final z = _randTensor(n * n * cfg.cZ, 21);
      final frames = _randFrames(n, 22);

      final base = ipa.call(s, z, frames);

      // Compose every residue frame with the same global transform G.
      final g = Rigid(
        Rot3.fromQuaternion(0.3, 0.5, -0.2, 0.77),
        const Vec3(7, -3, 2),
      );
      final moved = [for (final f in frames) g.compose(f)];
      final after = ipa.call(s, z, moved);

      for (var i = 0; i < base.length; i++) {
        expect(after[i], closeTo(base[i], 1e-4), reason: 'channel $i');
      }
    });
  });

  group('StructureModule', () {
    test('produces n frames and finite backbone atoms', () {
      final sm = StructureModule.random(cfg, noBlocks: 3, seed: 3);
      final s = _randTensor(n * cfg.cS, 30);
      final z = _randTensor(n * n * cfg.cZ, 31);
      final out = sm.call(s, z);
      expect(out.frames.length, n);
      expect(out.atoms.length, n);
      for (final a in out.atoms) {
        for (final v in [a.n, a.cAlpha, a.c]) {
          expect(v.x.isFinite && v.y.isFinite && v.z.isFinite, isTrue);
        }
      }
    });

    test('ideal N–Cα and Cα–C bond lengths are preserved', () {
      final sm = StructureModule.random(cfg, noBlocks: 4, seed: 4);
      final s = _randTensor(n * cfg.cS, 40);
      final z = _randTensor(n * n * cfg.cZ, 41);
      final out = sm.call(s, z);
      for (final a in out.atoms) {
        expect((a.n - a.cAlpha).norm, closeTo(IdealBackbone.n.norm, 1e-4));
        expect((a.c - a.cAlpha).norm, closeTo(IdealBackbone.c.norm, 1e-4));
      }
    });

    test('backbone coordinates are equivariant under a global rotation', () {
      final sm = StructureModule.random(cfg, noBlocks: 4, seed: 5);
      final s = _randTensor(n * cfg.cS, 50);
      final z = _randTensor(n * n * cfg.cZ, 51);

      final ref = sm.call(s, z); // black-hole (identity) init
      // A pure rotation commutes with the final translation scaling, so it is
      // the clean equivariance probe (a translation would be rescaled by
      // transScale). Frames start rotated by G; all atoms must rotate by G.
      final g = Rigid(
        Rot3.fromQuaternion(0.6, -0.4, 0.3, 0.5),
        Vec3.zero,
      );
      final moved = sm.call(s, z, initial: g);

      for (var i = 0; i < n; i++) {
        _expectVec(moved.atoms[i].n, g.apply(ref.atoms[i].n), tol: 1e-3);
        _expectVec(moved.atoms[i].cAlpha, g.apply(ref.atoms[i].cAlpha),
            tol: 1e-3);
        _expectVec(moved.atoms[i].c, g.apply(ref.atoms[i].c), tol: 1e-3);
      }
    });
  });
}

void _expectVec(Vec3 a, Vec3 b, {double tol = 1e-6}) {
  expect(a.x, closeTo(b.x, tol), reason: 'x');
  expect(a.y, closeTo(b.y, tol), reason: 'y');
  expect(a.z, closeTo(b.z, tol), reason: 'z');
}
