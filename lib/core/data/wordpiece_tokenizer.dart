/// BERT WordPiece tokenizer (uncased BERT / MiniLM / all-MiniLM-L6-v2
/// style). Faithful enough to reproduce HuggingFace `BertTokenizer`
/// output on the vast majority of English text.
///
/// Pipeline (matches `BertTokenizer` with `do_lower_case=true`):
///   1. Clean text: strip control chars, collapse whitespace.
///   2. Lowercase; strip common accents (NFD -> drop Mn).
///   3. Split punctuation into their own tokens.
///   4. Whitespace-split into words.
///   5. For each word: greedy-longest WordPiece match against the
///      vocab, with `##`-prefixed continuation pieces; unknown ->
///      `[UNK]`.
///
/// Not implemented (kept simple; falls back to sensible defaults):
///   * Chinese-character forced splitting.
///   * `never_split` list.
///   * Special-token protection (they are recovered by the vocab
///     lookup itself).
library;

import 'dart:convert';
import 'dart:io';

class WordPieceTokenizer {
  final Map<String, int> vocab;
  final List<String> inverse;
  final String unkToken;
  final String clsToken;
  final String sepToken;
  final String padToken;
  final bool doLowerCase;
  final int maxInputCharsPerWord;

  int get vocabSize => inverse.length;
  int get unkId => vocab[unkToken]!;
  int get clsId => vocab[clsToken]!;
  int get sepId => vocab[sepToken]!;
  int get padId => vocab[padToken]!;

  WordPieceTokenizer._({
    required this.vocab,
    required this.inverse,
    required this.unkToken,
    required this.clsToken,
    required this.sepToken,
    required this.padToken,
    required this.doLowerCase,
    this.maxInputCharsPerWord = 100,
  });

  /// Load a BERT-style `vocab.txt` (one token per line). Line index
  /// is the token id.
  factory WordPieceTokenizer.fromVocabFile(
    String path, {
    String unkToken = '[UNK]',
    String clsToken = '[CLS]',
    String sepToken = '[SEP]',
    String padToken = '[PAD]',
    bool doLowerCase = true,
    int maxInputCharsPerWord = 100,
  }) {
    final lines = File(path).readAsLinesSync();
    final inverse = <String>[];
    final vocab = <String, int>{};
    for (final l in lines) {
      // vocab.txt tokens are pure UTF-8; keep them verbatim — no
      // trimming (would break the `##` prefix).
      inverse.add(l);
      vocab[l] = inverse.length - 1;
    }
    for (final t in [unkToken, clsToken, sepToken, padToken]) {
      if (!vocab.containsKey(t)) {
        throw ArgumentError('WordPieceTokenizer: vocab missing "$t"');
      }
    }
    return WordPieceTokenizer._(
      vocab: vocab,
      inverse: inverse,
      unkToken: unkToken,
      clsToken: clsToken,
      sepToken: sepToken,
      padToken: padToken,
      doLowerCase: doLowerCase,
      maxInputCharsPerWord: maxInputCharsPerWord,
    );
  }

  /// Encode `text` into token ids. When [addSpecialTokens] is true
  /// (the default) the output is `[CLS] pieces... [SEP]`.
  ///
  /// If [maxLength] is set the result is truncated (special tokens
  /// preserved) to at most that many ids.
  List<int> encode(
    String text, {
    bool addSpecialTokens = true,
    int? maxLength,
  }) {
    final pieces = tokenize(text);
    final ids = <int>[];
    if (addSpecialTokens) ids.add(clsId);
    for (final p in pieces) {
      ids.add(vocab[p] ?? unkId);
    }
    if (addSpecialTokens) ids.add(sepId);

    if (maxLength != null && ids.length > maxLength) {
      if (addSpecialTokens) {
        // Keep [CLS] at 0 and [SEP] at end.
        final head = ids.sublist(0, maxLength - 1);
        head.add(sepId);
        return head;
      }
      return ids.sublist(0, maxLength);
    }
    return ids;
  }

  /// Encode a (query, passage) pair as `[CLS] q [SEP] p [SEP]`, the
  /// standard input for a [CrossEncoder].
  List<int> encodePair(String textA, String textB, {int? maxLength}) {
    final a = tokenize(textA);
    final b = tokenize(textB);
    final ids = <int>[clsId];
    for (final p in a) {
      ids.add(vocab[p] ?? unkId);
    }
    ids.add(sepId);
    for (final p in b) {
      ids.add(vocab[p] ?? unkId);
    }
    ids.add(sepId);
    if (maxLength != null && ids.length > maxLength) {
      final head = ids.sublist(0, maxLength - 1);
      head.add(sepId);
      return head;
    }
    return ids;
  }

  /// Decode ids back to a string. Concatenates WordPiece continuations
  /// (`##suffix` merges into the previous token). Special tokens are
  /// dropped from the output.
  String decode(List<int> ids) {
    final buf = StringBuffer();
    var needSpace = false;
    for (final id in ids) {
      if (id == clsId || id == sepId || id == padId) continue;
      final tok = id >= 0 && id < inverse.length ? inverse[id] : unkToken;
      if (tok.startsWith('##')) {
        buf.write(tok.substring(2));
      } else {
        if (needSpace) buf.write(' ');
        buf.write(tok);
      }
      needSpace = true;
    }
    return buf.toString();
  }

  /// Basic + WordPiece tokenization without special tokens.
  List<String> tokenize(String text) {
    final words = _basicTokenize(text);
    final out = <String>[];
    for (final w in words) {
      out.addAll(_wordpieceTokenize(w));
    }
    return out;
  }

  // ---------------- basic tokenizer ----------------

  List<String> _basicTokenize(String text) {
    var t = _cleanText(text);
    if (doLowerCase) {
      t = t.toLowerCase();
      t = _stripAccents(t);
    }
    final split = _splitOnPunc(t);
    return split.where((s) => s.isNotEmpty).toList();
  }

  String _cleanText(String text) {
    final buf = StringBuffer();
    for (final rune in text.runes) {
      if (rune == 0 || rune == 0xFFFD || _isControl(rune)) continue;
      if (_isWhitespace(rune)) {
        buf.write(' ');
      } else {
        buf.writeCharCode(rune);
      }
    }
    return buf.toString().split(RegExp(r'\s+')).where((s) => s.isNotEmpty).join(
      ' ',
    );
  }

  bool _isWhitespace(int r) {
    if (r == 0x20 || r == 0x09 || r == 0x0A || r == 0x0D) return true;
    return false;
  }

  bool _isControl(int r) {
    if (r == 0x09 || r == 0x0A || r == 0x0D) return false;
    return r < 0x20 || (r >= 0x7F && r < 0xA0);
  }

  bool _isPunctuation(int r) {
    // ASCII punctuation ranges.
    if ((r >= 33 && r <= 47) ||
        (r >= 58 && r <= 64) ||
        (r >= 91 && r <= 96) ||
        (r >= 123 && r <= 126)) {
      return true;
    }
    return false;
  }

  List<String> _splitOnPunc(String text) {
    final out = <String>[];
    final buf = StringBuffer();
    for (final rune in text.runes) {
      if (_isPunctuation(rune)) {
        if (buf.isNotEmpty) {
          out.addAll(buf.toString().split(' ').where((s) => s.isNotEmpty));
          buf.clear();
        }
        out.add(String.fromCharCode(rune));
      } else {
        buf.writeCharCode(rune);
      }
    }
    if (buf.isNotEmpty) {
      out.addAll(buf.toString().split(' ').where((s) => s.isNotEmpty));
    }
    return out;
  }

  // Approximate NFD decomposition + Mn drop for common Latin accents.
  // Full Unicode normalization would require importing `unorm_dart`
  // or building a large table; this covers the accented Latin chars
  // that show up in typical corpora and matches what the HF BERT
  // tokenizer does for those.
  String _stripAccents(String s) {
    const stripMap = <int, int>{
      0xE0: 0x61, 0xE1: 0x61, 0xE2: 0x61, 0xE3: 0x61, 0xE4: 0x61, 0xE5: 0x61,
      0xE7: 0x63,
      0xE8: 0x65, 0xE9: 0x65, 0xEA: 0x65, 0xEB: 0x65,
      0xEC: 0x69, 0xED: 0x69, 0xEE: 0x69, 0xEF: 0x69,
      0xF1: 0x6E,
      0xF2: 0x6F, 0xF3: 0x6F, 0xF4: 0x6F, 0xF5: 0x6F, 0xF6: 0x6F,
      0xF9: 0x75, 0xFA: 0x75, 0xFB: 0x75, 0xFC: 0x75,
      0xFD: 0x79, 0xFF: 0x79,
    };
    final buf = StringBuffer();
    for (final r in s.runes) {
      buf.writeCharCode(stripMap[r] ?? r);
    }
    return buf.toString();
  }

  // ---------------- WordPiece ----------------

  List<String> _wordpieceTokenize(String word) {
    if (word.length > maxInputCharsPerWord) {
      return [unkToken];
    }
    final sub = <String>[];
    var start = 0;
    while (start < word.length) {
      var end = word.length;
      String? cur;
      while (start < end) {
        var substr = word.substring(start, end);
        if (start > 0) substr = '##$substr';
        if (vocab.containsKey(substr)) {
          cur = substr;
          break;
        }
        end -= 1;
      }
      if (cur == null) {
        return [unkToken];
      }
      sub.add(cur);
      start = end;
    }
    return sub;
  }
}

/// Utility for callers: `List<int> -> jsonEncode(...)` doesn't hit
/// arg limits, but a helper avoids duplicating this everywhere in
/// bin/ demos.
String debugPreview(List<int> ids, int max) {
  if (ids.length <= max) return jsonEncode(ids);
  return '${jsonEncode(ids.sublist(0, max))} ...(${ids.length} total)';
}
