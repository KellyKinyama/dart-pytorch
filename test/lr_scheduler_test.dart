import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

// Any real Tensor parameter list will do; a single scalar parameter
// is enough — the scheduler only reads/writes `optimizer.lr`.
List<Tensor> _params() => [Tensor.fill([1, 1], 0.0)];

void main() {
  group('StepLR', () {
    test('rejects non-positive stepSize', () {
      expect(
        () => StepLR(
          SGD(_params(), lr: 0.1),
          initialLr: 0.1,
          stepSize: 0,
        ),
        throwsArgumentError,
      );
    });

    test('sets initial LR on construction', () {
      final opt = SGD(_params(), lr: 999.0); // constructor LR ignored
      StepLR(opt, initialLr: 0.5, stepSize: 3, gamma: 0.1);
      expect(opt.lr, closeTo(0.5, 1e-12));
    });

    test('holds LR flat within an interval and drops at boundary', () {
      final opt = SGD(_params(), lr: 0.0);
      final s = StepLR(opt, initialLr: 1.0, stepSize: 3, gamma: 0.5);

      // step() advances BEFORE reading, and lrAt uses floor(t/stepSize):
      // t=1..2 => k=0 => 1.0
      // t=3..5 => k=1 => 0.5
      // t=6..8 => k=2 => 0.25
      // t=9    => k=3 => 0.125
      final trace = <double>[];
      for (int i = 0; i < 9; i++) {
        s.step();
        trace.add(opt.lr);
      }
      const expected = [1.0, 1.0, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.125];
      for (int i = 0; i < 9; i++) {
        expect(trace[i], closeTo(expected[i], 1e-12), reason: 't=${i + 1}');
      }
    });

    test('lastLr matches optimizer.lr after each step', () {
      final opt = SGD(_params(), lr: 0.0);
      final s = StepLR(opt, initialLr: 0.4, stepSize: 2, gamma: 0.1);
      expect(s.lastLr, isNull);
      s.step();
      expect(s.lastLr, closeTo(opt.lr, 1e-12));
      // stepSize=2 gamma=0.1: t=1->0.4, t=2->0.04, t=3->0.04, t=4->0.004.
      s.step();
      s.step();
      s.step();
      expect(s.lastLr, closeTo(0.004, 1e-12));
    });
  });

  group('LinearWarmupCosineDecay', () {
    test('rejects invalid arguments', () {
      final opt = SGD(_params(), lr: 0.0);
      expect(
        () => LinearWarmupCosineDecay(opt,
            warmupSteps: -1, totalSteps: 10, maxLr: 1.0),
        throwsArgumentError,
      );
      expect(
        () => LinearWarmupCosineDecay(opt,
            warmupSteps: 20, totalSteps: 10, maxLr: 1.0),
        throwsArgumentError,
      );
      expect(
        () => LinearWarmupCosineDecay(opt,
            warmupSteps: 2, totalSteps: 10, maxLr: 0.5, minLr: 0.9),
        throwsArgumentError,
      );
    });

    test('linear warmup reaches maxLr at exactly warmupSteps', () {
      final opt = Adam(_params(), lr: 0.0);
      final s = LinearWarmupCosineDecay(
        opt,
        warmupSteps: 4,
        totalSteps: 20,
        maxLr: 1.0,
      );
      // At t=1: 0.25, t=2: 0.5, t=3: 0.75, t=4: 1.0
      final expected = [0.25, 0.5, 0.75, 1.0];
      for (int i = 0; i < 4; i++) {
        s.step();
        expect(opt.lr, closeTo(expected[i], 1e-12));
      }
    });

    test('cosine decay lands exactly at minLr at totalSteps', () {
      final opt = SGD(_params(), lr: 0.0);
      final s = LinearWarmupCosineDecay(
        opt,
        warmupSteps: 0,
        totalSteps: 10,
        maxLr: 1.0,
        minLr: 0.1,
      );
      double last = double.nan;
      for (int i = 0; i < 10; i++) {
        s.step();
        last = opt.lr;
      }
      expect(last, closeTo(0.1, 1e-12));
    });

    test('clamps to minLr past totalSteps', () {
      final opt = SGD(_params(), lr: 0.0);
      final s = LinearWarmupCosineDecay(
        opt,
        warmupSteps: 2,
        totalSteps: 10,
        maxLr: 1.0,
        minLr: 0.05,
      );
      for (int i = 0; i < 50; i++) {
        s.step();
      }
      expect(opt.lr, closeTo(0.05, 1e-12));
    });

    test('midpoint of cosine phase == halfway between minLr and maxLr', () {
      final opt = SGD(_params(), lr: 0.0);
      // warmup=0, total=20 => cosine over t in [1, 20].
      // midpoint of cosine phase: (0 + 20) / 2 = 10, progress = 0.5 => cos(pi/2) = 0
      // => lr = minLr + 0.5*(maxLr-minLr) = 0.55
      final s = LinearWarmupCosineDecay(
        opt,
        warmupSteps: 0,
        totalSteps: 20,
        maxLr: 1.0,
        minLr: 0.1,
      );
      for (int i = 0; i < 10; i++) {
        s.step();
      }
      expect(opt.lr, closeTo(0.55, 1e-9));
    });

    test('curve is monotone within each phase', () {
      final opt = SGD(_params(), lr: 0.0);
      final s = LinearWarmupCosineDecay(
        opt,
        warmupSteps: 10,
        totalSteps: 100,
        maxLr: 3e-4,
        minLr: 3e-5,
      );
      double prev = -double.infinity;
      // Warmup: strictly increasing
      for (int i = 0; i < 10; i++) {
        s.step();
        expect(opt.lr, greaterThan(prev));
        prev = opt.lr;
      }
      // Cosine: strictly decreasing
      prev = double.infinity;
      for (int i = 0; i < 90; i++) {
        s.step();
        expect(opt.lr, lessThan(prev));
        prev = opt.lr;
      }
    });
  });

  test('scheduler works polymorphically over SGD and Adam', () {
    for (final opt in <Optimizer>[
      SGD(_params(), lr: 0.0),
      Adam(_params(), lr: 0.0),
    ]) {
      final s = LinearWarmupCosineDecay(
        opt,
        warmupSteps: 3,
        totalSteps: 30,
        maxLr: 0.5,
      );
      for (int i = 0; i < 15; i++) {
        s.step();
      }
      // Definitely past warmup, definitely not at end.
      expect(opt.lr, lessThan(0.5));
      expect(opt.lr, greaterThan(0.0));
      // And matches the analytic formula.
      const t = 15;
      const w = 3;
      const total = 30;
      final progress = (t - w) / (total - w);
      final expected = 0.5 * (1 + math.cos(math.pi * progress)) * 0.5;
      expect(opt.lr, closeTo(expected, 1e-9));
    }
  });
}
