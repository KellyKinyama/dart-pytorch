/// Hopfield application demo: denoise small black-and-white icons.
///
/// Four 8x8 icons (heart, star, smiley, arrow) are stored via the Hebb
/// rule. Each icon is corrupted with salt-and-pepper noise and the
/// network runs asynchronous discrete dynamics to recover the clean
/// version.
///
/// Run:
///   dart run bin/hopfield/image_denoise.dart
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

const int _side = 8;
const int _size = _side * _side;

const Map<String, List<String>> _icons = {
  'heart': [
    '.##..##.',
    '########',
    '########',
    '########',
    '.######.',
    '..####..',
    '...##...',
    '........',
  ],
  'star': [
    '...##...',
    '...##...',
    '########',
    '.######.',
    '..####..',
    '.##..##.',
    '##....##',
    '........',
  ],
  'smiley': [
    '..####..',
    '.######.',
    '##.##.##',
    '##.##.##',
    '########',
    '##....##',
    '.######.',
    '..####..',
  ],
  'arrow': [
    '...##...',
    '..####..',
    '.######.',
    '########',
    '...##...',
    '...##...',
    '...##...',
    '...##...',
  ],
};

Int8List _iconToPattern(List<String> grid) {
  final out = Int8List(_size);
  for (int r = 0; r < _side; r++) {
    final row = grid[r];
    for (int c = 0; c < _side; c++) {
      out[r * _side + c] = row[c] == '#' ? 1 : -1;
    }
  }
  return out;
}

String _patternToAscii(Int8List x) {
  final sb = StringBuffer();
  for (int r = 0; r < _side; r++) {
    for (int c = 0; c < _side; c++) {
      sb.write(x[r * _side + c] > 0 ? '#' : '.');
    }
    sb.write('\n');
  }
  return sb.toString();
}

void _printBlocks(String title, List<String> labels, List<Int8List> pats) {
  print(title);
  final blocks = pats.map(_patternToAscii).map((s) => s.split('\n')).toList();
  final labelPad = labels.map((s) => s.padRight(_side)).toList();
  print('  ${labelPad.join('   ')}');
  for (int r = 0; r < _side; r++) {
    final row = blocks.map((b) => b[r]).join('   ');
    print('  $row');
  }
  print('');
}

Int8List _saltAndPepper(Int8List x, double frac, math.Random rng) {
  final out = Int8List.fromList(x);
  for (int i = 0; i < x.length; i++) {
    if (rng.nextDouble() < frac) {
      out[i] = rng.nextBool() ? 1 : -1;
    }
  }
  return out;
}

void main() {
  print('=== Hopfield image-denoising demo ===\n');

  final names = _icons.keys.toList();
  final memories = names.map((n) => _iconToPattern(_icons[n]!)).toList();

  print(
    'Storing ${memories.length} $_side×$_side icons '
    '($_size neurons, N/I = '
    '${(memories.length / _size).toStringAsFixed(3)})\n',
  );
  final net = HopfieldNetwork(_size)..storeHebb(memories);

  _printBlocks('Clean icons:', names, memories);

  for (final noiseFrac in [0.10, 0.20, 0.30]) {
    print(
      '--- Salt-and-pepper noise: '
      '${(100 * noiseFrac).toStringAsFixed(0)}% of pixels flipped ---',
    );
    final rng = math.Random(2026 + (noiseFrac * 1000).round());
    int okBits = 0;
    int totalBits = 0;
    for (int k = 0; k < memories.length; k++) {
      final clean = memories[k];
      final noisy = _saltAndPepper(clean, noiseFrac, rng);
      final noiseBits = bipolarHamming(clean, noisy);
      final r = net.recallDiscrete(
        noisy,
        update: HopfieldUpdate.asynchronous,
        rng: math.Random(k * 17 + 1),
        maxSweeps: 64,
      );
      final finalBits = bipolarHamming(clean, r.state);
      okBits += _size - finalBits;
      totalBits += _size;
      _printBlocks(
        '${names[k]}  '
        '(noisy=$noiseBits flips → recovered with $finalBits errors, '
        'sweeps=${r.sweeps})',
        ['clean', 'noisy', 'denoised'],
        [clean, noisy, r.state],
      );
    }
    print(
      'overall pixel accuracy: '
      '${(100 * okBits / totalBits).toStringAsFixed(1)}%\n',
    );
  }
}
