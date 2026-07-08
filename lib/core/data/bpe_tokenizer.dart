/// Byte-level Byte Pair Encoding tokenizer.
///
/// A pure-Dart implementation that operates on the UTF-8 bytes of the
/// input text — every text is representable, and the initial vocab is
/// exactly the 256 possible byte values. Training then greedily
/// merges the most frequent adjacent pair of symbols until the vocab
/// reaches `targetVocabSize` (or no merges are possible).
///
/// This is not the fastest BPE trainer in existence — the training
/// loop is O(N * merges) in the worst case — but the intent is a
/// small, dependency-free reference implementation suitable for
/// toy-scale training (~1e6 bytes, vocab ~256..2048).
///
/// Wire format for `save` / `load`:
///
/// ```json
/// {
///   "version": 1,
///   "kind": "byte-bpe",
///   "vocabSize": <int>,
///   "merges": [[<int>, <int>], ...]   // in merge order, pair -> vocabSize-1
/// }
/// ```
library;

import 'dart:convert';
import 'dart:io';

class BpeTokenizer {
  /// Total vocabulary size (256 base bytes + number of merges).
  final int vocabSize;

  /// Ordered list of merges. `merges[i] = (a, b)` means "when
  /// encountering the pair `(a, b)`, replace it with token id `256 + i`".
  final List<List<int>> merges;

  /// Rank lookup for the encoder: `mergeRank[(a, b)] = rank`. Lower
  /// rank means "merge earlier".
  final Map<int, int> _mergeRank;

  BpeTokenizer._(this.vocabSize, this.merges)
    : _mergeRank = {
        for (int i = 0; i < merges.length; i++)
          _pairKey(merges[i][0], merges[i][1]): i,
      };

  /// Train a fresh tokenizer from a text corpus.
  ///
  /// `targetVocabSize` must be >= 256; training stops when the vocab
  /// reaches that size or when no more valid pairs exist. `minCount`
  /// filters out pair candidates that occur fewer than this many
  /// times — a small speedup / stability guard for tiny corpora.
  factory BpeTokenizer.train(
    String corpus, {
    required int targetVocabSize,
    int minCount = 2,
  }) {
    if (targetVocabSize < 256) {
      throw ArgumentError(
        'BpeTokenizer.train: targetVocabSize must be >= 256; got $targetVocabSize',
      );
    }
    final bytes = utf8.encode(corpus);
    if (bytes.isEmpty) {
      throw ArgumentError('BpeTokenizer.train: corpus is empty');
    }
    // Working sequence — each entry is a current token id.
    var seq = List<int>.of(bytes);
    final merges = <List<int>>[];
    var nextId = 256;

    while (nextId < targetVocabSize) {
      // Count adjacent pairs.
      final counts = <int, int>{};
      for (int i = 0; i + 1 < seq.length; i++) {
        final k = _pairKey(seq[i], seq[i + 1]);
        counts[k] = (counts[k] ?? 0) + 1;
      }
      if (counts.isEmpty) break;

      // Pick the most frequent pair (ties broken by lower key for
      // determinism across runs).
      int bestKey = -1;
      int bestCount = 0;
      counts.forEach((k, c) {
        if (c > bestCount || (c == bestCount && (bestKey == -1 || k < bestKey))) {
          bestCount = c;
          bestKey = k;
        }
      });
      if (bestCount < minCount) break;

      final a = bestKey >> 20;
      final b = bestKey & 0xFFFFF;
      merges.add([a, b]);

      // Apply the merge to `seq` in a single pass.
      final merged = <int>[];
      int i = 0;
      while (i < seq.length) {
        if (i + 1 < seq.length && seq[i] == a && seq[i + 1] == b) {
          merged.add(nextId);
          i += 2;
        } else {
          merged.add(seq[i]);
          i += 1;
        }
      }
      seq = merged;
      nextId += 1;
    }

    return BpeTokenizer._(nextId, merges);
  }

  /// Encode a string into a list of token ids. Applies merges in the
  /// order learned during training (lowest rank first).
  List<int> encode(String text) {
    var seq = List<int>.of(utf8.encode(text));
    if (seq.isEmpty) return seq;
    // Greedy pairwise merging using the merge-rank map.
    // Each pass finds the merge with the lowest rank among all
    // current adjacent pairs and applies it. Terminates when no
    // adjacent pair is in the vocab.
    while (seq.length >= 2) {
      int bestRank = 1 << 30;
      int bestI = -1;
      for (int i = 0; i + 1 < seq.length; i++) {
        final r = _mergeRank[_pairKey(seq[i], seq[i + 1])];
        if (r != null && r < bestRank) {
          bestRank = r;
          bestI = i;
        }
      }
      if (bestI == -1) break;
      final newId = 256 + bestRank;
      seq = [
        ...seq.sublist(0, bestI),
        newId,
        ...seq.sublist(bestI + 2),
      ];
    }
    return seq;
  }

  /// Decode a list of token ids back to a string. Unpacks each merged
  /// id recursively down to raw byte ids, then UTF-8-decodes.
  String decode(List<int> tokens) {
    final bytes = <int>[];
    for (final t in tokens) {
      _expandTo(bytes, t);
    }
    return utf8.decode(bytes, allowMalformed: true);
  }

  void _expandTo(List<int> out, int id) {
    if (id < 256) {
      out.add(id);
      return;
    }
    final rank = id - 256;
    if (rank >= merges.length) {
      throw ArgumentError('BpeTokenizer.decode: unknown token id $id');
    }
    final pair = merges[rank];
    _expandTo(out, pair[0]);
    _expandTo(out, pair[1]);
  }

  // -------- Persistence --------

  /// Serialize the tokenizer to a compact JSON string.
  String saveJson() => jsonEncode({
    'version': 1,
    'kind': 'byte-bpe',
    'vocabSize': vocabSize,
    'merges': merges,
  });

  /// Write the tokenizer to a file.
  void saveFile(String path) {
    final f = File(path);
    f.parent.createSync(recursive: true);
    f.writeAsStringSync(saveJson());
  }

  /// Parse a tokenizer from a JSON string produced by [saveJson].
  factory BpeTokenizer.fromJson(String json) {
    final data = jsonDecode(json) as Map<String, dynamic>;
    if (data['kind'] != 'byte-bpe') {
      throw ArgumentError(
        'BpeTokenizer.fromJson: expected kind "byte-bpe", got ${data['kind']}',
      );
    }
    if (data['version'] != 1) {
      throw ArgumentError(
        'BpeTokenizer.fromJson: unsupported version ${data['version']}',
      );
    }
    final vocab = data['vocabSize'] as int;
    final merges = (data['merges'] as List)
        .map((e) => (e as List).cast<int>())
        .toList();
    return BpeTokenizer._(vocab, merges);
  }

  /// Load a tokenizer from a file.
  factory BpeTokenizer.loadFile(String path) =>
      BpeTokenizer.fromJson(File(path).readAsStringSync());

  // -------- Utilities --------

  /// Pack a (a, b) pair of 20-bit-safe token ids into a single int
  /// suitable for use as a `Map<int, ...>` key. We use 20 bits per
  /// side which comfortably fits any vocab up to 2^20 = ~1e6.
  static int _pairKey(int a, int b) => (a << 20) | (b & 0xFFFFF);
}
