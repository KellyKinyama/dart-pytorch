import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

void main() {
  group('Tensor.dropout', () {
    test('eval mode / training=false is identity', () {
      final xv = [1.0, 2.0, 3.0, 4.0];
      final x = Tensor.fromList([4], xv);
      final y = x.dropout(0.5, training: false);
      expect(y.toList(), xv);
      expect(
        identical(y, x),
        isTrue,
        reason: 'eval-mode dropout should return `this`',
      );
    });

    test('p=0 is identity in training mode', () {
      final xv = [1.0, 2.0, 3.0];
      final x = Tensor.fromList([3], xv);
      final y = x.dropout(0.0, training: true);
      expect(y.toList(), xv);
    });

    test('drops elements and scales survivors by 1/(1-p)', () {
      // With a seeded RNG and p=0.5, verify each output is either 0 or
      // exactly 2x the input.
      final xv = List<double>.generate(1000, (i) => (i + 1).toDouble());
      final x = Tensor.fromList([1000], xv);
      final y = x.dropout(0.5, training: true, rng: math.Random(0)).toList();

      int dropped = 0;
      for (int i = 0; i < 1000; i++) {
        if (y[i] == 0.0) {
          dropped++;
        } else {
          expect(
            (y[i] - 2.0 * xv[i]).abs() < 1e-4,
            isTrue,
            reason: 'i=$i got=${y[i]} expected 2*${xv[i]}',
          );
        }
      }
      // Rough drop-rate check: ~500 ± 60 for 1000 samples at p=0.5.
      expect(
        dropped > 440 && dropped < 560,
        isTrue,
        reason: 'dropped=$dropped, expected around 500',
      );
    });

    test('expected mean preserved (inverted dropout)', () {
      // E[y] should equal x elementwise. Average a bunch of draws.
      final xv = List<double>.generate(20, (i) => (i + 1).toDouble());
      final x = Tensor.fromList([20], xv);
      const trials = 400;
      final acc = List<double>.filled(20, 0);
      final rng = math.Random(1);
      for (int t = 0; t < trials; t++) {
        final y = x.dropout(0.3, training: true, rng: rng).toList();
        for (int i = 0; i < 20; i++) {
          acc[i] += y[i];
        }
      }
      for (int i = 0; i < 20; i++) {
        final mean = acc[i] / trials;
        // Std of Bernoulli-scaled ~ x * sqrt(p/(1-p)/N). For p=0.3,
        // x=20, N=400 this is ~0.65 * 20 / 20 ~= 0.65 per elem.
        expect(
          (mean - xv[i]).abs() < 1.5,
          isTrue,
          reason: 'i=$i mean=$mean expected ~${xv[i]}',
        );
      }
    });

    test('gradient flows through survivors, zero for drops', () {
      final xv = [1.0, 2.0, 3.0, 4.0];
      final x = Tensor.fromList([4], xv, requiresGrad: true);
      final rng = math.Random(0);
      // Loss = sum(dropout(x, 0.5))
      x.dropout(0.5, training: true, rng: rng).sum().backward();

      // Grad of each element is exactly the mask entry (0 or 2.0).
      final g = x.grad!.toList();
      for (int i = 0; i < 4; i++) {
        expect(
          g[i] == 0.0 || (g[i] - 2.0).abs() < 1e-4,
          isTrue,
          reason: 'i=$i g=${g[i]}',
        );
      }
    });

    test('rejects out-of-range p', () {
      final x = Tensor.fromList([1], [1.0]);
      expect(() => x.dropout(-0.1), throwsArgumentError);
      expect(() => x.dropout(1.0), throwsArgumentError);
      expect(() => x.dropout(1.5), throwsArgumentError);
    });

    test('works on GPU', () {
      // Use a size >= autoDeviceThreshold to exercise the GPU mask path.
      final xv = List<double>.generate(4096, (_) => 1.0);
      final x = Tensor.fromList([64, 64], xv, device: Device.GPU);
      final y = x.dropout(0.5, training: true, rng: math.Random(0)).toList();
      int dropped = 0;
      for (int i = 0; i < 4096; i++) {
        if (y[i] == 0.0) {
          dropped++;
        } else {
          expect((y[i] - 2.0).abs() < 1e-4, isTrue, reason: 'i=$i got=${y[i]}');
        }
      }
      expect(
        dropped > 1800 && dropped < 2300,
        isTrue,
        reason: 'dropped=$dropped, expected around 2048',
      );
    });
  });

  group('nn.Dropout module', () {
    test('training mode drops, eval mode is identity', () {
      final drop = Dropout(0.5, rng: math.Random(0));
      expect(drop.training, isTrue);
      expect(drop.parameters(), isEmpty);

      final x = Tensor.fromList([100], List<double>.generate(100, (i) => 1.0));
      final trainOut = drop(x).toList();
      int dropped = 0;
      for (final v in trainOut) {
        if (v == 0.0) dropped++;
      }
      expect(
        dropped > 30,
        isTrue,
        reason: 'expected some dropouts in training mode',
      );

      drop.eval();
      final evalOut = drop(x).toList();
      for (int i = 0; i < 100; i++) {
        expect(evalOut[i], 1.0);
      }

      drop.train();
      expect(drop.training, isTrue);
    });
  });

  group('Module.train() / eval() propagation', () {
    test('parent train/eval reaches submodules', () {
      final child = Dropout(0.5);
      final parent = _WrapperModule(child);
      expect(parent.training, isTrue);
      expect(child.training, isTrue);

      parent.eval();
      expect(parent.training, isFalse);
      expect(child.training, isFalse);

      parent.train();
      expect(parent.training, isTrue);
      expect(child.training, isTrue);
    });
  });

  group('clipGradNorm', () {
    test('no-op when total norm <= maxNorm', () {
      final p = Tensor.fromList([4], [1.0, 2.0, 3.0, 4.0], requiresGrad: true);
      p.pow(2).sum().backward(); // grad = 2p = [2,4,6,8]; norm = sqrt(120)
      final before = p.grad!.toList();
      final norm = clipGradNorm([p], 100.0);
      final after = p.grad!.toList();
      expect(after, before);
      expect((norm - math.sqrt(120)).abs() < 1e-4, isTrue);
    });

    test('scales grads so total norm equals maxNorm', () {
      final p = Tensor.fromList([4], [1.0, 2.0, 3.0, 4.0], requiresGrad: true);
      p.pow(2).sum().backward();
      final preNorm = math.sqrt(120.0);
      const maxNorm = 1.0;
      final returned = clipGradNorm([p], maxNorm);
      expect((returned - preNorm).abs() < 1e-4, isTrue);

      final g = p.grad!.toList();
      double sq = 0;
      for (final v in g) {
        sq += v * v;
      }
      final postNorm = math.sqrt(sq);
      expect(
        (postNorm - maxNorm).abs() < 1e-4,
        isTrue,
        reason: 'postNorm=$postNorm',
      );

      // Direction preserved: g_after == g_before * (maxNorm / preNorm).
      final scale = maxNorm / preNorm;
      final expected = [2.0, 4.0, 6.0, 8.0].map((v) => v * scale).toList();
      for (int i = 0; i < 4; i++) {
        expect(
          (g[i] - expected[i]).abs() < 1e-4,
          isTrue,
          reason: 'i=$i g=${g[i]} expected=${expected[i]}',
        );
      }
    });

    test('handles multi-parameter global norm', () {
      final a = Tensor.fromList([2], [3.0, 4.0], requiresGrad: true);
      final b = Tensor.fromList([1], [12.0], requiresGrad: true);
      // Grads via backward: loss = a.sum() + b.sum() -> grad_a=[1,1], grad_b=[1]
      // Not useful. Set manually via a scaled loss.
      (a.pow(2).sum() + b.pow(2).sum()).backward();
      // grad_a = [6, 8], grad_b = [24]. Norm = sqrt(36+64+576) = sqrt(676)=26.
      final norm = clipGradNorm([a, b], 13.0);
      expect((norm - 26.0).abs() < 1e-3, isTrue, reason: 'preclip norm=$norm');
      final ga = a.grad!.toList();
      final gb = b.grad!.toList();
      // Each grad scaled by 0.5.
      expectCloseList(ga, [3.0, 4.0]);
      expectCloseList(gb, [12.0]);
    });

    test('skips parameters without gradients', () {
      final p1 = Tensor.fromList([2], [1.0, 2.0], requiresGrad: true);
      final p2 = Tensor.fromList([2], [3.0, 4.0], requiresGrad: true);
      p1.pow(2).sum().backward(); // p1.grad set, p2.grad null
      expect(p2.grad, isNull);
      // Should not throw.
      clipGradNorm([p1, p2], 1.0);
      expect(p2.grad, isNull);
    });

    test('rejects non-positive maxNorm', () {
      final p = Tensor.fromList([1], [1.0]);
      expect(() => clipGradNorm([p], 0.0), throwsArgumentError);
      expect(() => clipGradNorm([p], -1.0), throwsArgumentError);
    });
  });
}

void expectCloseList(List<double> got, List<double> want, {double tol = 1e-4}) {
  expect(got.length, want.length);
  for (int i = 0; i < got.length; i++) {
    expect(
      (got[i] - want[i]).abs() < tol,
      isTrue,
      reason: 'i=$i got=${got[i]} want=${want[i]}',
    );
  }
}

class _WrapperModule extends Module {
  final Dropout child;
  _WrapperModule(this.child);
  @override
  List<Tensor> parameters() => const [];
  @override
  List<Module> submodules() => [child];
}
