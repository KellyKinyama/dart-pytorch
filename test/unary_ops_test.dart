/// Gradcheck coverage for the extended unary math ops (B2 of
/// `doc/parity-plan.md`): exp, sqrt, sin, cos, erf, clamp, and the
/// composed helpers neg/reciprocal/rsqrt/silu/softplus/gelu.
library;

import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

import 'gradcheck.dart';

void main() {
  group('B2 unary primitives (CPU)', () {
    test('exp', () {
      expectGradCheck(
        (t) => t[0].exp(),
        [
          [0.5, -1.0, 1.2, -0.3],
        ],
        [
          [2, 2],
        ],
      );
    });

    test('sqrt (positive domain)', () {
      expectGradCheck(
        (t) => t[0].sqrt(),
        [
          [0.5, 1.0, 2.0, 3.0],
        ],
        [
          [2, 2],
        ],
      );
    });

    test('sin', () {
      expectGradCheck(
        (t) => t[0].sin(),
        [
          [0.5, -1.0, 1.2, -0.3],
        ],
        [
          [2, 2],
        ],
      );
    });

    test('cos', () {
      expectGradCheck(
        (t) => t[0].cos(),
        [
          [0.5, -1.0, 1.2, -0.3],
        ],
        [
          [2, 2],
        ],
      );
    });

    test('erf', () {
      expectGradCheck(
        (t) => t[0].erf(),
        [
          [0.5, -1.0, 1.2, -0.3],
        ],
        [
          [2, 2],
        ],
      );
    });

    test('clamp (seeds kept away from the bounds)', () {
      expectGradCheck(
        (t) => t[0].clamp(-1.0, 1.0),
        [
          [0.5, -0.5, 0.8, -0.8],
        ],
        [
          [2, 2],
        ],
      );
    });
  });

  group('B2 composed helpers (CPU)', () {
    test('neg', () {
      expectGradCheck(
        (t) => t[0].neg(),
        [
          [0.5, -1.0, 1.2, -0.3],
        ],
        [
          [2, 2],
        ],
      );
    });

    test('reciprocal (nonzero)', () {
      expectGradCheck(
        (t) => t[0].reciprocal(),
        [
          [0.5, 1.0, 2.0, -0.8],
        ],
        [
          [2, 2],
        ],
      );
    });

    test('rsqrt (positive domain)', () {
      expectGradCheck(
        (t) => t[0].rsqrt(),
        [
          [0.5, 1.0, 2.0, 3.0],
        ],
        [
          [2, 2],
        ],
      );
    });

    test('silu', () {
      expectGradCheck(
        (t) => t[0].silu(),
        [
          [0.5, -1.0, 1.2, -0.3],
        ],
        [
          [2, 2],
        ],
      );
    });

    test('softplus', () {
      expectGradCheck(
        (t) => t[0].softplus(),
        [
          [0.5, -1.0, 1.2, -0.3],
        ],
        [
          [2, 2],
        ],
      );
    });

    test('geluTanh', () {
      expectGradCheck(
        (t) => t[0].geluTanh(),
        [
          [0.5, -1.0, 1.2, -0.3],
        ],
        [
          [2, 2],
        ],
      );
    });

    test('gelu (exact)', () {
      expectGradCheck(
        (t) => t[0].gelu(),
        [
          [0.5, -1.0, 1.2, -0.3],
        ],
        [
          [2, 2],
        ],
      );
    });
  });

  group('B2 forward values (CPU)', () {
    test('exp / sqrt / sin / cos match dart:math', () {
      final xs = [0.5, -1.0, 1.2, 2.0];
      final x = Tensor.fromList([4], xs);
      _expectClose(x.exp().toList(), [for (final v in xs) math.exp(v)]);
      final pos = Tensor.fromList([4], [0.5, 1.0, 2.0, 3.0]);
      _expectClose(pos.sqrt().toList(), [0.5, 1.0, 2.0, 3.0].map(math.sqrt).toList());
      _expectClose(x.sin().toList(), [for (final v in xs) math.sin(v)]);
      _expectClose(x.cos().toList(), [for (final v in xs) math.cos(v)]);
    });

    test('silu matches x*sigmoid(x)', () {
      final xs = [0.5, -1.0, 1.2, 2.0];
      final x = Tensor.fromList([4], xs);
      final want = [for (final v in xs) v * (1.0 / (1.0 + math.exp(-v)))];
      _expectClose(x.silu().toList(), want);
    });
  });
}

void _expectClose(List<double> a, List<double> b, {double tol = 1e-4}) {
  expect(a.length, b.length);
  for (int i = 0; i < a.length; i++) {
    expect((a[i] - b[i]).abs() < tol, isTrue,
        reason: 'index $i: ${a[i]} vs ${b[i]}');
  }
}
