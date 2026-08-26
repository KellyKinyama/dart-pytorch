/// MacKay ch. 42.8 demo: train a Hopfield network with the
/// logistic-perceptron rule (algorithm 42.9) to overcome the Hebb
/// rule's capacity limit.
///
/// Reproduces Fig. 42.5: six correlated memories that Hebb cannot store
/// simultaneously. After training with algorithm 42.9 every memory
/// becomes an exact fixed point and small corruptions are corrected.
///
/// Run:
///   dart run bin/hopfield/train_perceptron.dart
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

const int _rows = 5;
const int _cols = 5;
const int _size = _rows * _cols;

const List<String> _memoryNames = [
  'cross',
  'frame',
  'diag',
  'hbars',
  'vbars',
  'plus5',
];

const List<List<String>> _memoryGrids = [
  ['..#..', '..#..', '#####', '..#..', '..#..'],
  ['#####', '#...#', '#...#', '#...#', '#####'],
  ['#....', '.#...', '..#..', '...#.', '....#'],
  ['#####', '.....', '#####', '.....', '#####'],
  ['#.#.#', '#.#.#', '#.#.#', '#.#.#', '#.#.#'],
  ['..#..', '..#..', '#####', '..#..', '#####'],
];

Int8List _gridToPattern(List<String> grid) {
  final out = Int8List(_size);
  for (int r = 0; r < _rows; r++) {
    final row = grid[r];
    for (int c = 0; c < _cols; c++) {
      out[r * _cols + c] = row[c] == '#' ? 1 : -1;
    }
  }
  return out;
}

String _patternToAscii(Int8List x) {
  final sb = StringBuffer();
  for (int r = 0; r < _rows; r++) {
    for (int c = 0; c < _cols; c++) {
      sb.write(x[r * _cols + c] > 0 ? '#' : '.');
    }
    sb.write('\n');
  }
  return sb.toString();
}

void _printSideBySide(String title, List<String> labels, List<Int8List> pats) {
  print(title);
  final blocks = pats.map(_patternToAscii).map((s) => s.split('\n')).toList();
  final labelPad = labels.map((s) => s.padRight(_cols)).toList();
  print('  ${labelPad.join('   ')}');
  for (int r = 0; r < _rows; r++) {
    final row = blocks.map((b) => b[r]).join('   ');
    print('  $row');
  }
  print('');
}

void main() {
  print('=== MacKay Ch. 42.8 Hopfield perceptron training demo ===\n');

  final memories = _memoryGrids.map(_gridToPattern).toList();
  final N = memories.length;

  _printSideBySide('Six desired memories (Fig. 42.5):', _memoryNames, memories);

  print('--- Hebb rule baseline ---');
  final hebb = HopfieldNetwork(_size)..storeHebb(memories);
  final hebbStable = hebb.countStableSynchronous(memories);
  print('  stable memories: $hebbStable / $N');
  final hebbFinals = <Int8List>[];
  int hebbRecovered = 0;
  for (int k = 0; k < N; k++) {
    final r = hebb.recallDiscrete(
      memories[k],
      update: HopfieldUpdate.asynchronous,
      rng: math.Random(100 + k),
      maxSweeps: 64,
    );
    hebbFinals.add(r.state);
    if (bipolarHamming(r.state, memories[k]) == 0) hebbRecovered++;
  }
  print('  memories recalled from clean input: $hebbRecovered / $N\n');
  _printSideBySide(
    'Hebb attractors when started from each memory:',
    _memoryNames,
    hebbFinals,
  );

  print('--- Training with algorithm 42.9 ---');
  final net = HopfieldNetwork(_size);
  const eta = 0.02;
  const alpha = 0.0;
  const totalSteps = 4000;
  const reportEvery = 500;
  net.storeHebb(memories);
  print(
    '  I=$_size, N=$N, eta=$eta, alpha=$alpha, '
    'init=Hebb, steps=$totalSteps',
  );
  print('  step         loss   stable  W-norm');
  int accSteps = 0;
  while (accSteps < totalSteps) {
    final chunk = math.min(reportEvery, totalSteps - accSteps);
    final stable = net.trainPerceptron(
      memories,
      steps: chunk,
      eta: eta,
      alpha: alpha,
      initHebb: false,
    );
    accSteps += chunk;
    double wsq = 0.0;
    for (final w in net.weights) {
      wsq += w * w;
    }
    final wnorm = math.sqrt(wsq);
    print(
      '  ${accSteps.toString().padLeft(5)}  '
      '${net.lastTrainLoss!.toStringAsFixed(3).padLeft(10)}  '
      '${stable.toString().padLeft(4)}/$N  '
      '${wnorm.toStringAsFixed(2)}',
    );
  }
  print('');

  print('--- Trained network recall from clean memories ---');
  int trainedRecovered = 0;
  final trainedFinals = <Int8List>[];
  for (int k = 0; k < N; k++) {
    final r = net.recallDiscrete(
      memories[k],
      update: HopfieldUpdate.asynchronous,
      rng: math.Random(200 + k),
      maxSweeps: 64,
    );
    trainedFinals.add(r.state);
    if (bipolarHamming(r.state, memories[k]) == 0) trainedRecovered++;
  }
  print('  memories recalled: $trainedRecovered / $N\n');
  _printSideBySide(
    'Trained attractors when started from each memory:',
    _memoryNames,
    trainedFinals,
  );

  print('--- Trained recall from 3-bit noise ---');
  final rng = math.Random(2026);
  int noisyRecovered = 0;
  for (int k = 0; k < N; k++) {
    final target = memories[k];
    final noisy = flipRandomBits(target, 3, rng: rng);
    final r = net.recallDiscrete(
      noisy,
      update: HopfieldUpdate.asynchronous,
      rng: math.Random(300 + k),
      maxSweeps: 64,
    );
    final finalD = bipolarHamming(r.state, target);
    if (finalD == 0) noisyRecovered++;
    _printSideBySide(
      '${_memoryNames[k]}  (noise=3, final Hamming=$finalD, '
      'sweeps=${r.sweeps}, converged=${r.converged})',
      ['target', 'noisy', 'recall'],
      [target, noisy, r.state],
    );
  }
  print(
    'summary: $noisyRecovered / $N patterns exactly recalled '
    'from 3-bit noise (Hebb: $hebbRecovered / $N from clean input).',
  );
}
