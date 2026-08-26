/// Hopfield application demo: complete a pattern from a partial cue.
///
/// Same 8x8 icons as `image_denoise.dart`, but instead of noise we
/// blank out a rectangular region (setting those neurons to random
/// `±1`) and let the network fill it in. This is the "give me half a
/// picture, get back the whole thing" mode of associative memory.
///
/// Run:
///   dart run bin/hopfield/partial_cue.dart
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

String _patternToAscii(Int8List x, {Set<int>? maskedIdx}) {
  final sb = StringBuffer();
  for (int r = 0; r < _side; r++) {
    for (int c = 0; c < _side; c++) {
      final idx = r * _side + c;
      if (maskedIdx != null && maskedIdx.contains(idx)) {
        sb.write('?');
      } else {
        sb.write(x[idx] > 0 ? '#' : '.');
      }
    }
    sb.write('\n');
  }
  return sb.toString();
}

void _printBlocks(
  String title,
  List<String> labels,
  List<Int8List> pats, {
  Set<int>? maskForFirst,
}) {
  print(title);
  final blocks = <List<String>>[];
  for (int i = 0; i < pats.length; i++) {
    final mask = i == 1 ? maskForFirst : null;
    blocks.add(_patternToAscii(pats[i], maskedIdx: mask).split('\n'));
  }
  final labelPad = labels.map((s) => s.padRight(_side)).toList();
  print('  ${labelPad.join('   ')}');
  for (int r = 0; r < _side; r++) {
    final row = blocks.map((b) => b[r]).join('   ');
    print('  $row');
  }
  print('');
}

/// Blank out a rectangular region: entries inside it are set to random
/// `±1`. Returns the modified pattern and the set of masked indices.
({Int8List cue, Set<int> masked}) _maskRect(
  Int8List x, {
  required int r0,
  required int c0,
  required int h,
  required int w,
  required math.Random rng,
}) {
  final out = Int8List.fromList(x);
  final masked = <int>{};
  for (int r = r0; r < r0 + h; r++) {
    for (int c = c0; c < c0 + w; c++) {
      final idx = r * _side + c;
      out[idx] = rng.nextBool() ? 1 : -1;
      masked.add(idx);
    }
  }
  return (cue: out, masked: masked);
}

void main() {
  print('=== Hopfield partial-cue completion demo ===\n');

  final names = _icons.keys.toList();
  final memories = names.map((n) => _iconToPattern(_icons[n]!)).toList();

  final net = HopfieldNetwork(_size)..storeHebb(memories);
  print('Stored ${memories.length} icons on $_size neurons.\n');

  final rng = math.Random(2026);
  final maskConfigs = <({String name, int r0, int c0, int h, int w})>[
    (name: 'right half', r0: 0, c0: 4, h: 8, w: 4),
    (name: 'bottom half', r0: 4, c0: 0, h: 4, w: 8),
    (name: 'center 4×4', r0: 2, c0: 2, h: 4, w: 4),
  ];

  for (final cfg in maskConfigs) {
    print('--- Mask: ${cfg.name} ---');
    for (int k = 0; k < memories.length; k++) {
      final clean = memories[k];
      final m = _maskRect(
        clean,
        r0: cfg.r0,
        c0: cfg.c0,
        h: cfg.h,
        w: cfg.w,
        rng: rng,
      );
      final r = net.recallDiscrete(
        m.cue,
        update: HopfieldUpdate.asynchronous,
        rng: math.Random(k * 31 + 5),
        maxSweeps: 64,
      );
      final finalErrs = bipolarHamming(clean, r.state);
      _printBlocks(
        '${names[k]}  (mask=${m.masked.length} bits → '
        'reconstruction errors=$finalErrs, sweeps=${r.sweeps})',
        ['clean', 'cue', 'recalled'],
        [clean, m.cue, r.state],
        maskForFirst: m.masked,
      );
    }
  }
}
