import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

void main() {
  group('HopfieldNetwork.storeHebb', () {
    test('weights are symmetric with zero diagonal', () {
      final net = HopfieldNetwork(6);
      net.storeHebb([
        Int8List.fromList([1, -1, 1, -1, 1, -1]),
        Int8List.fromList([1, 1, -1, -1, 1, 1]),
      ]);
      final I = net.numNeurons;
      for (int i = 0; i < I; i++) {
        expect(net.weights[i * I + i], 0.0, reason: 'diag($i)');
        for (int j = i + 1; j < I; j++) {
          expect(
            net.weights[i * I + j],
            net.weights[j * I + i],
            reason: 'w[$i,$j] vs w[$j,$i]',
          );
        }
      }
    });

    test('a single stored pattern is an exact fixed point', () {
      final net = HopfieldNetwork(8);
      final p = Int8List.fromList([1, 1, -1, -1, 1, -1, 1, -1]);
      net.storeHebb([p]);
      final r = net.recallDiscrete(p, update: HopfieldUpdate.synchronous);
      expect(r.converged, isTrue);
      expect(r.sweeps, lessThanOrEqualTo(1));
      expect(r.state, orderedEquals(p));
    });

    test('a few well-separated patterns are all fixed points', () {
      final rng = math.Random(1);
      final I = 40;
      final patterns = <Int8List>[
        for (int k = 0; k < 4; k++)
          Int8List.fromList([
            for (int i = 0; i < I; i++) rng.nextBool() ? 1 : -1,
          ]),
      ];
      final net = HopfieldNetwork(I)..storeHebb(patterns);
      for (final p in patterns) {
        final r = net.recallDiscrete(
          p,
          update: HopfieldUpdate.asynchronous,
          rng: math.Random(2),
        );
        expect(r.converged, isTrue);
        expect(bipolarHamming(r.state, p), 0);
      }
    });
  });

  group('HopfieldNetwork.recallDiscrete', () {
    test('async updates monotonically decrease energy', () {
      final rng = math.Random(3);
      final I = 30;
      final patterns = <Int8List>[
        for (int k = 0; k < 3; k++)
          Int8List.fromList([
            for (int i = 0; i < I; i++) rng.nextBool() ? 1 : -1,
          ]),
      ];
      final net = HopfieldNetwork(I)..storeHebb(patterns);
      final noisy = flipRandomBits(patterns[0], 4, rng: math.Random(4));
      final energies = <double>[net.energy(noisy)];
      var x = Int8List.fromList(noisy);
      for (int sweep = 0; sweep < 20; sweep++) {
        final r = net.recallDiscrete(
          x,
          maxSweeps: 1,
          update: HopfieldUpdate.asynchronous,
          rng: math.Random(5 + sweep),
        );
        energies.add(r.finalEnergy);
        if (r.converged) break;
        x = r.state;
      }
      for (int i = 1; i < energies.length; i++) {
        expect(
          energies[i],
          lessThanOrEqualTo(energies[i - 1] + 1e-6),
          reason: 'energy went up at step $i',
        );
      }
    });

    test('recovers pattern from small corruption', () {
      final rng = math.Random(6);
      final I = 60;
      final patterns = <Int8List>[
        for (int k = 0; k < 4; k++)
          Int8List.fromList([
            for (int i = 0; i < I; i++) rng.nextBool() ? 1 : -1,
          ]),
      ];
      final net = HopfieldNetwork(I)..storeHebb(patterns);
      final target = patterns[2];
      final noisy = flipRandomBits(target, 3, rng: math.Random(7));
      final r = net.recallDiscrete(
        noisy,
        update: HopfieldUpdate.asynchronous,
        rng: math.Random(8),
      );
      expect(r.converged, isTrue);
      expect(bipolarHamming(r.state, target), 0);
    });
  });

  group('HopfieldNetwork.relaxContinuous', () {
    test('state stays in (-1, 1) and drifts toward a stored pattern', () {
      final net = HopfieldNetwork(12);
      final p = Int8List.fromList([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]);
      net.storeHebb([p]);
      final x0 = Float32List.fromList([
        for (int i = 0; i < 12; i++) 0.05 * (p[i]).toDouble(),
      ]);
      final xFinal = net.relaxContinuous(
        x0,
        beta: 1.5,
        tau: 1.0,
        dt: 0.1,
        steps: 400,
      );
      for (int i = 0; i < 12; i++) {
        expect(xFinal[i].abs(), lessThan(1.0 + 1e-6));
      }
      final signed = Int8List(12);
      for (int i = 0; i < 12; i++) {
        signed[i] = xFinal[i] > 0 ? 1 : -1;
      }
      expect(bipolarHamming(signed, p), 0);
    });
  });

  group('capacity sanity', () {
    test('below N/I = 0.05 recall works reliably', () {
      final rng = math.Random(9);
      final I = 100;
      final N = 4; // N/I = 0.04, well below MacKay's 0.138 threshold
      final patterns = <Int8List>[
        for (int k = 0; k < N; k++)
          Int8List.fromList([
            for (int i = 0; i < I; i++) rng.nextBool() ? 1 : -1,
          ]),
      ];
      final net = HopfieldNetwork(I)..storeHebb(patterns);
      int okExact = 0;
      for (final p in patterns) {
        final noisy = flipRandomBits(p, 5, rng: math.Random(10));
        final r = net.recallDiscrete(
          noisy,
          update: HopfieldUpdate.asynchronous,
          rng: math.Random(11),
        );
        if (bipolarHamming(r.state, p) == 0) okExact++;
      }
      expect(okExact, N);
    });
  });

  group('HopfieldNetwork.trainPerceptron', () {
    test('makes all six correlated 5x5 memories stable (Fig. 42.5)', () {
      Int8List grid(List<String> rows) {
        final out = Int8List(25);
        for (int r = 0; r < 5; r++) {
          for (int c = 0; c < 5; c++) {
            out[r * 5 + c] = rows[r][c] == '#' ? 1 : -1;
          }
        }
        return out;
      }

      final memories = <Int8List>[
        grid(['..#..', '..#..', '#####', '..#..', '..#..']),
        grid(['#####', '#...#', '#...#', '#...#', '#####']),
        grid(['#....', '.#...', '..#..', '...#.', '....#']),
        grid(['#####', '.....', '#####', '.....', '#####']),
        grid(['#.#.#', '#.#.#', '#.#.#', '#.#.#', '#.#.#']),
        grid(['..#..', '..#..', '#####', '..#..', '#####']),
      ];

      final net = HopfieldNetwork(25);
      final stable = net.trainPerceptron(memories, steps: 4000, eta: 0.02);
      expect(stable, memories.length);
      expect(net.lastTrainLoss, isNotNull);
      expect(net.lastTrainLoss!, lessThan(0.05));
      for (int i = 0; i < 25; i++) {
        expect(net.weights[i * 25 + i], 0.0);
        for (int j = i + 1; j < 25; j++) {
          expect(
            net.weights[i * 25 + j],
            closeTo(net.weights[j * 25 + i], 1e-6),
          );
        }
      }
    });

    test('reduces loss monotonically over successive chunks', () {
      final rng = math.Random(101);
      final I = 32;
      final N = 8; // N/I = 0.25, above Hebb capacity
      final memories = <Int8List>[
        for (int k = 0; k < N; k++)
          Int8List.fromList([
            for (int i = 0; i < I; i++) rng.nextBool() ? 1 : -1,
          ]),
      ];
      final net = HopfieldNetwork(I);
      net.storeHebb(memories);
      final losses = <double>[];
      for (int chunk = 0; chunk < 6; chunk++) {
        net.trainPerceptron(memories, steps: 200, eta: 0.05, initHebb: false);
        losses.add(net.lastTrainLoss!);
      }
      for (int i = 1; i < losses.length; i++) {
        expect(
          losses[i],
          lessThanOrEqualTo(losses[i - 1] + 1e-4),
          reason: 'loss went up at chunk $i',
        );
      }
      expect(losses.last, lessThan(losses.first));
    });
  });
}
