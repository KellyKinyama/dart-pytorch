/// Hopfield application demo: correct one-letter typos in a small
/// fixed vocabulary.
///
/// Each word is encoded as `L * 26` bits using per-position one-hot
/// letter columns mapped to `±1`. A Hopfield network is trained on the
/// vocabulary; a mistyped word is fed in and the network settles onto
/// the nearest stored word.
///
/// Run:
///   dart run bin/hopfield/typo_correction.dart
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

const int _wordLen = 6;
const int _alphabet = 26;
const int _bits = _wordLen * _alphabet;

const List<String> _vocab = [
  'apple ',
  'brave ',
  'candle',
  'donkey',
  'elbow ',
  'forest',
  'grape ',
  'hollow',
];

Int8List _encode(String word) {
  if (word.length != _wordLen) {
    throw ArgumentError('word "$word" must be $_wordLen chars');
  }
  final out = Int8List(_bits)..fillRange(0, _bits, -1);
  for (int p = 0; p < _wordLen; p++) {
    final ch = word.codeUnitAt(p);
    if (ch == 0x20) continue; // space → all -1 in that block
    final letter = ch - 0x61;
    if (letter < 0 || letter >= _alphabet) {
      throw ArgumentError('non-lowercase char in "$word" at $p');
    }
    out[p * _alphabet + letter] = 1;
  }
  return out;
}

String _decode(Int8List bits) {
  final sb = StringBuffer();
  for (int p = 0; p < _wordLen; p++) {
    int best = -1;
    double bestScore = -1e30;
    for (int a = 0; a < _alphabet; a++) {
      final v = bits[p * _alphabet + a].toDouble();
      if (v > bestScore) {
        bestScore = v;
        best = a;
      }
    }
    // All entries in the block are -1 → space (no letter fired).
    int fired = 0;
    for (int a = 0; a < _alphabet; a++) {
      if (bits[p * _alphabet + a] > 0) fired++;
    }
    if (fired == 0) {
      sb.write(' ');
    } else {
      sb.writeCharCode(0x61 + best);
    }
  }
  return sb.toString();
}

String _corrupt(String word, int nEdits, math.Random rng) {
  final chars = word.split('');
  final positions = List<int>.generate(_wordLen, (i) => i);
  for (int k = _wordLen - 1; k > 0; k--) {
    final j = rng.nextInt(k + 1);
    final t = positions[k];
    positions[k] = positions[j];
    positions[j] = t;
  }
  for (int e = 0; e < nEdits; e++) {
    final p = positions[e];
    final orig = chars[p];
    String replacement;
    do {
      final code = 0x61 + rng.nextInt(_alphabet);
      replacement = String.fromCharCode(code);
    } while (replacement == orig);
    chars[p] = replacement;
  }
  return chars.join();
}

void main() {
  print('=== Hopfield typo-correction demo ===\n');
  print('Vocabulary (${_vocab.length} words, $_wordLen chars each):');
  for (final w in _vocab) {
    print('  "$w"');
  }
  print('');

  final memories = _vocab.map(_encode).toList();
  final net = HopfieldNetwork(_bits);
  print(
    'Network: $_bits neurons (${_wordLen} letters × $_alphabet '
    'one-hot bits), N/I = ${(memories.length / _bits).toStringAsFixed(3)}',
  );

  print('Phase 1: perceptron training on clean words (algorithm 42.9) …');
  final stableClean = net.trainPerceptron(memories, steps: 400, eta: 0.05);
  print(
    '  stable memories after phase 1: '
    '$stableClean / ${memories.length}',
  );
  print('  loss: ${net.lastTrainLoss!.toStringAsFixed(4)}');

  print('Phase 2: denoising training with typo-augmented pairs …');
  final rngAug = math.Random(0xBEEF);
  const copiesPerWord = 8;
  final inputs = <Int8List>[];
  final targets = <Int8List>[];
  for (final w in _vocab) {
    final clean = _encode(w);
    inputs.add(clean);
    targets.add(clean);
    for (int k = 0; k < copiesPerWord; k++) {
      final typo = _corrupt(w, 1 + rngAug.nextInt(2), rngAug);
      inputs.add(_encode(typo));
      targets.add(clean);
    }
  }
  net.trainDenoise(inputs, targets, steps: 400, eta: 0.02);
  final stableAfter = net.countStableSynchronous(memories);
  print(
    '  training pairs: ${inputs.length} '
    '(1 clean + $copiesPerWord noisy per word)',
  );
  print('  stable clean memories: $stableAfter / ${memories.length}');
  print('  loss: ${net.lastTrainLoss!.toStringAsFixed(4)}\n');

  final rng = math.Random(2026);
  print('Correcting single-letter typos:\n');
  int ok = 0;
  for (final word in _vocab) {
    final typo = _corrupt(word, 1, rng);
    final r = net.recallDiscrete(
      _encode(typo),
      update: HopfieldUpdate.asynchronous,
      rng: math.Random(word.hashCode),
      maxSweeps: 64,
    );
    final recovered = _decode(r.state);
    final correct = recovered == word;
    if (correct) ok++;
    print(
      '  typed  "$typo"   →   recalled  "$recovered"   '
      '${correct ? "OK" : "(wrong, wanted \"$word\")"}',
    );
  }
  print('\n$ok / ${_vocab.length} typos corrected.\n');

  print('Correcting two-letter typos:');
  int ok2 = 0;
  for (final word in _vocab) {
    final typo = _corrupt(word, 2, rng);
    final r = net.recallDiscrete(
      _encode(typo),
      update: HopfieldUpdate.asynchronous,
      rng: math.Random(word.hashCode ^ 0xABC),
      maxSweeps: 64,
    );
    final recovered = _decode(r.state);
    final correct = recovered == word;
    if (correct) ok2++;
    print(
      '  typed  "$typo"   →   recalled  "$recovered"   '
      '${correct ? "OK" : "(wrong, wanted \"$word\")"}',
    );
  }
  print('\n$ok2 / ${_vocab.length} two-letter typos corrected.');
}
