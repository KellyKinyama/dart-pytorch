/// Character-level tokenizer.
///
/// Builds its vocabulary from the sorted set of unique characters in
/// the training corpus (mirrors the reference `dart_cuda` tokenizer
/// used by the tiny-Shakespeare examples). Every char maps to a
/// dense integer id in `[0, vocabSize)`.
///
/// Unknown characters at `encode` time map to id 0. `decode` treats
/// out-of-range ids as the empty string so it stays safe on
/// hallucinated model output.
library;

import 'dart:convert';
import 'dart:io';

class CharTokenizer {
  /// Vocabulary in canonical order — `chars[i]` is the string for id `i`.
  final List<String> chars;

  /// Char -> id lookup used by [encode].
  final Map<String, int> _stoi;

  int get vocabSize => chars.length;

  CharTokenizer._(this.chars, this._stoi);

  /// Fit a tokenizer to `text`.
  ///
  /// Vocabulary is the set of characters in `text`, sorted for
  /// determinism (so the same text always yields the same id map
  /// across runs and machines).
  factory CharTokenizer.fromText(String text) {
    final chars = text.split('').toSet().toList()..sort();
    final stoi = <String, int>{
      for (int i = 0; i < chars.length; i++) chars[i]: i,
    };
    return CharTokenizer._(chars, stoi);
  }

  /// Restore from a previously-saved char list. `chars` must contain
  /// unique entries; order defines the id mapping.
  factory CharTokenizer.fromVocab(List<String> chars) {
    final stoi = <String, int>{
      for (int i = 0; i < chars.length; i++) chars[i]: i,
    };
    if (stoi.length != chars.length) {
      throw ArgumentError('CharTokenizer.fromVocab: chars must be unique');
    }
    return CharTokenizer._(List<String>.of(chars), stoi);
  }

  /// Encode a string to a list of int ids. Unknown chars → 0.
  List<int> encode(String s) {
    final out = List<int>.filled(s.length, 0);
    for (int i = 0; i < s.length; i++) {
      out[i] = _stoi[s[i]] ?? 0;
    }
    return out;
  }

  /// Decode a list of int ids back to a string. Out-of-range ids
  /// become the empty string.
  String decode(List<int> ids) {
    final sb = StringBuffer();
    for (final id in ids) {
      if (id >= 0 && id < chars.length) sb.write(chars[id]);
    }
    return sb.toString();
  }

  /// JSON wire format:
  /// ```json
  /// { "version": 1, "kind": "char", "chars": [...] }
  /// ```
  String toJson() => jsonEncode({'version': 1, 'kind': 'char', 'chars': chars});

  factory CharTokenizer.fromJson(String s) {
    final obj = jsonDecode(s) as Map<String, dynamic>;
    if (obj['kind'] != 'char') {
      throw FormatException('CharTokenizer.fromJson: kind is ${obj['kind']}');
    }
    final chars = (obj['chars'] as List).cast<String>();
    return CharTokenizer.fromVocab(chars);
  }

  void saveFile(String path) {
    File(path).writeAsStringSync(toJson());
  }

  factory CharTokenizer.loadFile(String path) =>
      CharTokenizer.fromJson(File(path).readAsStringSync());
}
