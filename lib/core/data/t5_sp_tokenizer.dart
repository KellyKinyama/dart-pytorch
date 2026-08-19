/// Loads a HuggingFace T5-style `tokenizer.json` (SentencePiece
/// Unigram model) and provides encode/decode.
///
/// T5 uses SentencePiece with a Unigram language model, not BPE.
/// The pipeline is:
///
///   1. **Normalization**. Text is NFKC-normalized and whitespace
///      is compressed / stripped per T5's `precompiled_charsmap`.
///      We approximate this with a simple NFKC pass plus space
///      compression.
///   2. **Metaspace pre-tokenizer**. A leading space is prepended
///      to the text (T5 always treats the first token as
///      word-initial), then every space is replaced with `▁`
///      (U+2581, "lower one-eighth block"), the SentencePiece
///      whitespace marker.
///   3. **Unigram Viterbi**. Dynamic-programming search for the
///      single-best segmentation into pieces from the vocabulary,
///      maximizing the sum of log-scores. Uses a piece trie to keep
///      the inner loop O(text_length * max_piece_length).
///   4. **Vocab lookup** → int ids.
///
/// Decode reverses: vocab → piece strings → concatenate → replace
/// `▁` with space → strip leading space.
///
/// Supports the standard T5 special tokens: `<pad>` (id 0), `</s>`
/// (id 1), `<unk>` (id 2), and the `<extra_id_0>` .. `<extra_id_99>`
/// sentinel tokens used by T5's span-corruption objective.
library;

import 'dart:convert';
import 'dart:io';

class T5SpTokenizer {
  final List<String> _idToPiece;
  final int _padId;
  final int _eosId;
  final int _unkId;
  final _PieceTrie _trie;

  T5SpTokenizer._(
    this._idToPiece,
    this._padId,
    this._eosId,
    this._unkId,
    this._trie,
  );

  int get vocabSize => _idToPiece.length;
  int get padId => _padId;
  int get eosId => _eosId;
  int get unkId => _unkId;

  /// Load a T5 `tokenizer.json` (HuggingFace tokenizers-lib format
  /// with `model.type == "Unigram"`).
  factory T5SpTokenizer.loadFile(String path) {
    final raw = jsonDecode(File(path).readAsStringSync());
    return T5SpTokenizer.fromJson(raw as Map<String, dynamic>);
  }

  factory T5SpTokenizer.fromJson(Map<String, dynamic> raw) {
    final model = raw['model'] as Map<String, dynamic>?;
    if (model == null) {
      throw ArgumentError('T5SpTokenizer: tokenizer.json has no "model" key');
    }
    final type = model['type'];
    if (type != 'Unigram') {
      throw ArgumentError('T5SpTokenizer: expected Unigram model; got "$type"');
    }
    final pieces = model['vocab'];
    if (pieces is! List) {
      throw ArgumentError('T5SpTokenizer: model.vocab is not a list');
    }
    final idToPiece = <String>[];
    final idToScore = <double>[];
    final pieceToId = <String, int>{};
    for (int i = 0; i < pieces.length; i++) {
      final entry = pieces[i];
      if (entry is! List || entry.length < 2) {
        throw ArgumentError(
          'T5SpTokenizer: piece $i is not [string, score]; got $entry',
        );
      }
      final tok = entry[0] as String;
      final score = (entry[1] as num).toDouble();
      idToPiece.add(tok);
      idToScore.add(score);
      pieceToId[tok] = i;
    }

    // Merge in added tokens (specials + extra_ids). They override any
    // colliding vocab entry.
    final added = raw['added_tokens'];
    if (added is List) {
      for (final a in added) {
        final m = a as Map<String, dynamic>;
        final id = (m['id'] as num).toInt();
        final content = m['content'] as String;
        while (idToPiece.length <= id) {
          idToPiece.add('');
          idToScore.add(0.0);
        }
        idToPiece[id] = content;
        pieceToId[content] = id;
      }
    }

    final padId = pieceToId['<pad>'] ?? 0;
    final eosId = pieceToId['</s>'] ?? 1;
    final unkId = pieceToId['<unk>'] ?? 2;

    final trie = _PieceTrie();
    for (int i = 0; i < idToPiece.length; i++) {
      final s = idToPiece[i];
      if (s.isEmpty) continue;
      trie.insert(s, i, idToScore[i]);
    }

    return T5SpTokenizer._(idToPiece, padId, eosId, unkId, trie);
  }

  /// Encode `text` to a list of int piece ids.
  ///
  /// When `addEos` is true (default), appends `</s>` — matching what
  /// HF's `T5Tokenizer.encode(text)` does with `add_special_tokens=
  /// True`. Pass `addEos: false` for raw piece output.
  List<int> encode(String text, {bool addEos = true}) {
    final normalized = _preprocess(text);
    final ids = _viterbi(normalized);
    if (addEos) ids.add(_eosId);
    return ids;
  }

  /// Decode a list of ids back into a string, dropping special
  /// tokens (`<pad>`, `</s>`, `<unk>`, `<extra_id_*>`) unless
  /// `skipSpecial` is false.
  String decode(List<int> ids, {bool skipSpecial = true}) {
    final buf = StringBuffer();
    for (final id in ids) {
      if (id < 0 || id >= _idToPiece.length) continue;
      final p = _idToPiece[id];
      if (skipSpecial && _isSpecial(p)) continue;
      buf.write(p);
    }
    var s = buf.toString();
    s = s.replaceAll('\u2581', ' ');
    if (s.startsWith(' ')) s = s.substring(1);
    return s;
  }

  bool _isSpecial(String piece) {
    if (piece == '<pad>' || piece == '</s>' || piece == '<unk>') return true;
    if (piece.startsWith('<extra_id_') && piece.endsWith('>')) return true;
    return false;
  }

  /// Prepend `▁`, replace runs of whitespace with `▁`.
  String _preprocess(String text) {
    // Compress whitespace and strip leading/trailing.
    final compact = text.replaceAll(RegExp(r'\s+'), ' ').trim();
    if (compact.isEmpty) return '\u2581';
    return '\u2581${compact.replaceAll(' ', '\u2581')}';
  }

  /// Viterbi over the input string: dp[i] = (best_score, back_id,
  /// back_len) for the best segmentation covering codepoints [0, i).
  /// Falls back to `<unk>` for any codepoint that isn't reachable
  /// via a matching piece.
  List<int> _viterbi(String text) {
    final n = text.length;
    if (n == 0) return <int>[];
    final dp = List<double>.filled(n + 1, double.negativeInfinity);
    final backId = List<int>.filled(n + 1, -1);
    final backLen = List<int>.filled(n + 1, 0);
    dp[0] = 0;

    for (int i = 0; i < n; i++) {
      if (dp[i] == double.negativeInfinity) continue;
      // Walk the trie from position i, collecting every matching piece.
      var node = _trie.root;
      var j = i;
      var matched = false;
      while (j < n) {
        final ch = text.codeUnitAt(j);
        final next = node.children[ch];
        if (next == null) break;
        node = next;
        j++;
        if (node.id >= 0) {
          final cand = dp[i] + node.score;
          if (cand > dp[j]) {
            dp[j] = cand;
            backId[j] = node.id;
            backLen[j] = j - i;
          }
          matched = true;
        }
      }
      // If no piece matched at all from this position, emit a
      // single-codeunit `<unk>` fallback so we can still advance.
      if (!matched) {
        // Move 1 codeunit forward with a small penalty.
        const unkPenalty = -100.0;
        final cand = dp[i] + unkPenalty;
        final nxt = i + 1;
        if (cand > dp[nxt]) {
          dp[nxt] = cand;
          backId[nxt] = _unkId;
          backLen[nxt] = 1;
        }
      }
    }

    // Backtrack.
    final out = <int>[];
    var pos = n;
    while (pos > 0) {
      final id = backId[pos];
      final len = backLen[pos];
      if (id < 0 || len <= 0) {
        // Unreachable — shouldn't happen given the unk fallback, but
        // guard against infinite loop.
        break;
      }
      out.add(id);
      pos -= len;
    }
    return out.reversed.toList();
  }

  /// Convenience for tests: read-only view of the id -> piece table.
  List<String> get vocabulary => List.unmodifiable(_idToPiece);
}

class _PieceTrie {
  final _TrieNode root = _TrieNode();

  void insert(String piece, int id, double score) {
    var node = root;
    for (int i = 0; i < piece.length; i++) {
      final ch = piece.codeUnitAt(i);
      node = node.children.putIfAbsent(ch, _TrieNode.new);
    }
    node.id = id;
    node.score = score;
  }
}

class _TrieNode {
  final Map<int, _TrieNode> children = {};
  int id = -1;
  double score = 0.0;
}
