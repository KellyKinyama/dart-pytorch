/// Shared data helpers for the coop_* demos:
///   - character-level tokeniser and vocab
///   - training-window sampler that returns (x, y) sequence pairs
///   - built-in self-contained toy corpus so the demos don't require
///     data/tiny_shakespeare.txt
library;

import 'dart:math' as math;

import '../tensor/tensor.dart';

/// Small self-contained corpus for the demos. ~1500 chars, exercises
/// enough vocab to train a tiny GPT to visible convergence in seconds.
const String kToyCorpus =
    'to be or not to be that is the question '
    'whether tis nobler in the mind to suffer '
    'the slings and arrows of outrageous fortune '
    'or to take arms against a sea of troubles '
    'and by opposing end them to die to sleep '
    'no more and by a sleep to say we end '
    'the heart ache and the thousand natural shocks '
    'that flesh is heir to tis a consummation '
    'devoutly to be wished to die to sleep '
    'to sleep perchance to dream ay theres the rub '
    'for in that sleep of death what dreams may come '
    'when we have shuffled off this mortal coil '
    'must give us pause theres the respect '
    'that makes calamity of so long life ';

class CharVocab {
  CharVocab._(this.itos, this.stoi);
  final List<String> itos;
  final Map<String, int> stoi;

  int get size => itos.length;

  factory CharVocab.fromText(String text) {
    final chars = text.split('').toSet().toList()..sort();
    final stoi = <String, int>{
      for (var i = 0; i < chars.length; i++) chars[i]: i,
    };
    return CharVocab._(chars, stoi);
  }

  /// Rebuild a vocab from an already-computed `itos` list (e.g. one
  /// received from a training coordinator over the network).
  factory CharVocab.fromItos(List<String> itos) {
    final stoi = <String, int>{
      for (var i = 0; i < itos.length; i++) itos[i]: i,
    };
    return CharVocab._(List<String>.unmodifiable(itos), stoi);
  }

  List<double> encode(String s) =>
      s.split('').map((c) => (stoi[c] ?? 0).toDouble()).toList();

  String decode(List<double> ids) =>
      ids.map((v) => itos[v.toInt().clamp(0, size - 1)]).join();
}

/// Sample one `[blockSize]` window at a random start position and
/// return the (x, y = next-token) pair as rank-1 tensors on [device].
(Tensor, Tensor) sampleWindow(
  List<double> ids,
  int blockSize,
  math.Random rng, {
  required Device device,
}) {
  final maxStart = ids.length - blockSize - 1;
  if (maxStart <= 0) {
    throw ArgumentError(
      'sampleWindow: corpus (${ids.length}) too short for blockSize $blockSize',
    );
  }
  final start = rng.nextInt(maxStart);
  final x = List<double>.generate(blockSize, (i) => ids[start + i]);
  final y = List<double>.generate(blockSize, (i) => ids[start + i + 1]);
  return (
    Tensor.fromList([blockSize], x, device: device),
    Tensor.fromList([blockSize], y, device: device),
  );
}

/// Partition [ids] into [numShards] contiguous, non-overlapping slices
/// and return slice number [shardIndex] (0-based). Used by the demos
/// to give each worker/replica its own subset of the corpus.
List<double> shardSlice(List<double> ids, int shardIndex, int numShards) {
  if (numShards <= 0 || shardIndex < 0 || shardIndex >= numShards) {
    throw ArgumentError(
      'shardSlice: shardIndex=$shardIndex, numShards=$numShards',
    );
  }
  final size = ids.length ~/ numShards;
  final start = shardIndex * size;
  final end = shardIndex == numShards - 1 ? ids.length : start + size;
  return ids.sublist(start, end);
}
