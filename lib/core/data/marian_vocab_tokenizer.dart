/// SentencePiece-style greedy-longest-match tokenizer for Marian /
/// Opus-MT models, driven by the HF `vocab.json` shipped alongside
/// the checkpoint.
///
/// This is a lightweight best-effort tokenizer — Marian's real
/// pipeline uses SentencePiece BPE via a `source.spm` binary, which
/// we don't parse here. For most well-formed inputs the greedy
/// longest-match with `▁` (U+2581) metaspace preprocessing produces
/// the same segmentation SPM would; the edge cases are rare
/// enough for casual translation demos.
///
/// Vocab format: a JSON dict `{ token: id }`. Keys with a leading
/// `▁` are word-initial pieces; keys without are word-medial. Ids
/// are dense (no gaps).
library;

import 'dart:convert';
import 'dart:io';

class MarianVocabTokenizer {
  final List<String> _idToToken;
  final int _padId;
  final int _eosId;
  final int _unkId;
  final _PieceTrie _trie;

  MarianVocabTokenizer._(
    this._idToToken,
    this._padId,
    this._eosId,
    this._unkId,
    this._trie,
  );

  int get vocabSize => _idToToken.length;
  int get padId => _padId;
  int get eosId => _eosId;
  int get unkId => _unkId;

  /// Load an HF Marian `vocab.json`.
  factory MarianVocabTokenizer.loadFile(String path) {
    final raw =
        jsonDecode(File(path).readAsStringSync()) as Map<String, dynamic>;
    var maxId = 0;
    raw.forEach((_, v) {
      final id = (v as num).toInt();
      if (id > maxId) maxId = id;
    });
    final idToToken = List<String>.filled(maxId + 1, '');
    raw.forEach((k, v) {
      idToToken[(v as num).toInt()] = k;
    });
    final trie = _PieceTrie();
    for (int i = 0; i < idToToken.length; i++) {
      final s = idToToken[i];
      if (s.isEmpty) continue;
      trie.insert(s, i);
    }
    final padId = _lookupInt(raw, '<pad>', fallback: maxId);
    final eosId = _lookupInt(raw, '</s>', fallback: 0);
    final unkId = _lookupInt(raw, '<unk>', fallback: 1);
    return MarianVocabTokenizer._(idToToken, padId, eosId, unkId, trie);
  }

  static int _lookupInt(
    Map<String, dynamic> raw,
    String key, {
    required int fallback,
  }) {
    final v = raw[key];
    if (v is num) return v.toInt();
    return fallback;
  }

  /// Encode `text` to a list of int piece ids.
  ///
  /// When `addEos` is true (default), appends `</s>` — matching what
  /// HF's `MarianTokenizer.encode(text)` does with
  /// `add_special_tokens=True`.
  List<int> encode(String text, {bool addEos = true}) {
    final normalized = _preprocess(text);
    final ids = _greedyMatch(normalized);
    if (addEos) ids.add(_eosId);
    return ids;
  }

  /// Decode a list of ids back into a string, dropping special
  /// tokens unless `skipSpecial` is false.
  String decode(List<int> ids, {bool skipSpecial = true}) {
    final buf = StringBuffer();
    for (final id in ids) {
      if (id < 0 || id >= _idToToken.length) continue;
      final p = _idToToken[id];
      if (p.isEmpty) continue;
      if (skipSpecial && (p == '<pad>' || p == '</s>' || p == '<unk>')) {
        continue;
      }
      buf.write(p);
    }
    var s = buf.toString();
    s = s.replaceAll('\u2581', ' ');
    if (s.startsWith(' ')) s = s.substring(1);
    return s;
  }

  /// Prepend `▁`, replace runs of whitespace with `▁`.
  String _preprocess(String text) {
    final compact = text.replaceAll(RegExp(r'\s+'), ' ').trim();
    if (compact.isEmpty) return '\u2581';
    return '\u2581${compact.replaceAll(' ', '\u2581')}';
  }

  /// Greedy longest-match against the vocab trie. Falls back to a
  /// single-character `<unk>` if no match starts at the current
  /// position.
  List<int> _greedyMatch(String text) {
    final ids = <int>[];
    var i = 0;
    while (i < text.length) {
      var node = _trie.root;
      var lastMatchEnd = -1;
      var lastMatchId = -1;
      var j = i;
      while (j < text.length) {
        final ch = text.codeUnitAt(j);
        final next = node.children[ch];
        if (next == null) break;
        node = next;
        j++;
        if (node.id >= 0) {
          lastMatchEnd = j;
          lastMatchId = node.id;
        }
      }
      if (lastMatchEnd >= 0) {
        ids.add(lastMatchId);
        i = lastMatchEnd;
      } else {
        ids.add(_unkId);
        i++;
      }
    }
    return ids;
  }

  /// Read-only view of the id -> token table.
  List<String> get vocabulary => List.unmodifiable(_idToToken);
}

class _PieceTrie {
  final _TrieNode root = _TrieNode();

  void insert(String piece, int id) {
    var node = root;
    for (int i = 0; i < piece.length; i++) {
      final ch = piece.codeUnitAt(i);
      node = node.children.putIfAbsent(ch, _TrieNode.new);
    }
    node.id = id;
  }
}

class _TrieNode {
  final Map<int, _TrieNode> children = {};
  int id = -1;
}
