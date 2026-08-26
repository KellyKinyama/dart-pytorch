/// MacKay ch. 42 demo: store five 5x5 binary memories in a Hopfield
/// network via the Hebb rule, corrupt a few bits, and recover them by
/// running discrete asynchronous dynamics — the pedagogical setup of
/// Fig. 42.4 (five memories, small-noise recall) and the continuous
/// relaxation of §42.6.
///
/// Run:
///   dart run bin/hopfield/basic_recall.dart
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

const int _rows = 5;
const int _cols = 5;
const int _size = _rows * _cols;

const List<String> _memoryNames = ['cross', 'frame', 'diag', 'hbars', 'vbars'];

const List<List<String>> _memoryGrids = [
  // cross
  ['..#..', '..#..', '#####', '..#..', '..#..'],
  // frame
  ['#####', '#...#', '#...#', '#...#', '#####'],
  // diag
  ['#....', '.#...', '..#..', '...#.', '....#'],
  // hbars
  ['#####', '.....', '#####', '.....', '#####'],
  // vbars
  ['#.#.#', '#.#.#', '#.#.#', '#.#.#', '#.#.#'],
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
  print('=== MacKay Ch. 42 Hopfield associative memory demo ===\n');

  final memories = _memoryGrids.map(_gridToPattern).toList();
  final net = HopfieldNetwork(_size)..storeHebb(memories);

  print(
    'Stored ${memories.length} memories on ${_size} neurons '
    '(N/I = ${(memories.length / _size).toStringAsFixed(3)}, '
    'MacKay critical N/I = 0.138)\n',
  );

  _printSideBySide('Desired memories:', _memoryNames, memories);

  print('Energies of stored memories:');
  for (int k = 0; k < memories.length; k++) {
    print(
      '  ${_memoryNames[k].padRight(7)} E = '
      '${net.energy(memories[k]).toStringAsFixed(2)}',
    );
  }
  print('');

  print('--- Discrete asynchronous recall from 3-bit noise (Fig. 42.4) ---\n');
  final rng = math.Random(42);
  for (int k = 0; k < memories.length; k++) {
    final target = memories[k];
    final noisy = flipRandomBits(target, 3, rng: rng);
    final r = net.recallDiscrete(
      noisy,
      update: HopfieldUpdate.asynchronous,
      rng: math.Random(100 + k),
      maxSweeps: 64,
    );
    final flippedInNoise = bipolarHamming(noisy, target);
    final flippedFinal = bipolarHamming(r.state, target);
    _printSideBySide(
      '${_memoryNames[k]}  '
      '(noise=$flippedInNoise flips, converged=${r.converged}, '
      'sweeps=${r.sweeps}, '
      'final Hamming=$flippedFinal, '
      'E=${r.finalEnergy.toStringAsFixed(2)})',
      ['target', 'noisy', 'recall'],
      [target, noisy, r.state],
    );
  }

  print('--- Continuous-time relaxation (\u00a742.6) ---\n');
  final targetC = memories[0];
  final noisyC = flipRandomBits(targetC, 3, rng: math.Random(7));
  final x0 = Float32List(_size);
  for (int i = 0; i < _size; i++) {
    x0[i] = 0.3 * noisyC[i];
  }
  final xFinal = net.relaxContinuous(
    x0,
    beta: 2.0,
    tau: 1.0,
    dt: 0.1,
    steps: 400,
  );
  final signed = Int8List(_size);
  for (int i = 0; i < _size; i++) {
    signed[i] = xFinal[i] > 0 ? 1 : -1;
  }
  print(
    'cross: 3-bit noisy input relaxed continuously with \u03b2=2, dt=0.1, '
    'T=40',
  );
  print('  final Hamming to target: ${bipolarHamming(signed, targetC)}');
  print(
    '  final continuous energy: '
    '${net.energyContinuous(xFinal).toStringAsFixed(2)}',
  );
  print('');
  _printSideBySide(
    'Continuous recall (signed):',
    ['target', 'noisy', 'sign(x_T)'],
    [targetC, noisyC, signed],
  );

  print('--- Overload demo (\u00a742.7) ---');
  final rng2 = math.Random(0);
  final loadFractions = <double>[0.05, 0.10, 0.138, 0.20, 0.30];
  const trials = 20;
  print(
    '  (each row: N/I -> fraction of random patterns recalled exactly '
    'from 2-bit noise, averaged over $trials trials)',
  );
  for (final f in loadFractions) {
    final I = 60;
    final N = math.max(1, (f * I).round());
    int okAny = 0;
    int total = 0;
    for (int t = 0; t < trials; t++) {
      final ps = <Int8List>[
        for (int k = 0; k < N; k++)
          Int8List.fromList([
            for (int i = 0; i < I; i++) rng2.nextBool() ? 1 : -1,
          ]),
      ];
      final n2 = HopfieldNetwork(I)..storeHebb(ps);
      for (final p in ps) {
        final noisy = flipRandomBits(p, 2, rng: rng2);
        final r = n2.recallDiscrete(
          noisy,
          update: HopfieldUpdate.asynchronous,
          rng: rng2,
          maxSweeps: 32,
        );
        if (bipolarHamming(r.state, p) == 0) okAny++;
        total++;
      }
    }
    final acc = okAny / total;
    print(
      '  N/I = ${f.toStringAsFixed(3)}  '
      '(I=$I, N=$N)  exact-recall rate = '
      '${(100 * acc).toStringAsFixed(1)}%',
    );
  }
}
