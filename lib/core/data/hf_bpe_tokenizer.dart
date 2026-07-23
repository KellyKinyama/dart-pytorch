/// Loads a HuggingFace `tokenizer.json` (byte-level BPE, e.g.
/// GPT-2, GPT-NeoX/Pythia, Llama-style) and provides encode/decode.
///
/// The algorithm mirrors what `tokenizers` does at runtime:
///
///   1. **UTF-8 byte encode** the input string.
///   2. **Byte→unicode map** (GPT-2's `bytes_to_unicode`): converts
///      each byte 0-255 to a printable single-code-point string,
///      dodging whitespace and control chars. Bytes we recognise as
///      "safe printable" (0x21-0x7e, 0xa1-0xac, 0xae-0xff) pass
///      through as-is; the rest are shifted into the private range
///      starting at 0x100. Ġ = U+0120 corresponds to byte 0x20 (SPACE).
///   3. **Pre-tokenizer regex** (GPT-2 default) to chop the
///      byte-mapped string into "words".
///   4. **BPE merges**: for each word (as a sequence of single-char
///      symbols), iteratively fuse the adjacent pair with the lowest
///      merge rank until no more merges apply.
///   5. **Vocab lookup** to turn each final symbol into its int id.
///
/// Decode reverses steps 5→2 and UTF-8 decodes the resulting bytes.
///
/// Supports:
///   * loading tokenizer.json for GPT-2, distilgpt2, Pythia (any
///     size), GPT-Neo — anything with `model.type == "BPE"` and a
///     `ByteLevel` pre-tokenizer.
///   * added special tokens (e.g. `<|endoftext|>`) — matched as
///     literal substrings before pre-tokenizing.
library;

import 'dart:convert';
import 'dart:io';

class HFBpeTokenizer {
  /// token string -> id.
  final Map<String, int> vocab;

  /// id -> token string.
  final Map<int, String> _idToTok;

  /// Merge rank lookup: `"a b" -> rank` (space-joined pair). Lower
  /// rank = merged earlier.
  final Map<String, int> _mergeRank;

  /// Byte value (0-255) -> single-code-point string.
  final List<String> _byteEncoder;

  /// Single-code-point string -> byte value (0-255). Populated as
  /// the inverse of [_byteEncoder].
  final Map<String, int> _byteDecoder;

  /// Ordered list of added / special tokens, longest first — used
  /// for greedy left-to-right split before regex pre-tokenizing.
  final List<_Special> _specials;

  /// Compiled GPT-2 style pre-tokenizer regex.
  static final RegExp _preTokenRe = RegExp(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
    unicode: true,
  );

  HFBpeTokenizer._(
    this.vocab,
    this._mergeRank,
    this._byteEncoder,
    this._byteDecoder,
    this._specials,
  ) : _idToTok = {for (final e in vocab.entries) e.value: e.key};

  /// Load from a `tokenizer.json` on disk.
  factory HFBpeTokenizer.loadFile(String path) {
    final raw = jsonDecode(File(path).readAsStringSync()) as Map<String, dynamic>;
    return HFBpeTokenizer.fromJson(raw);
  }

  /// Load from an already-parsed `tokenizer.json` map.
  factory HFBpeTokenizer.fromJson(Map<String, dynamic> raw) {
    final model = raw['model'] as Map<String, dynamic>?;
    if (model == null) {
      throw ArgumentError('tokenizer.json: missing "model"');
    }
    final type = model['type'];
    if (type != null && type != 'BPE') {
      throw ArgumentError(
        'tokenizer.json: unsupported model type "$type" (only BPE)',
      );
    }
    final rawVocab = model['vocab'];
    if (rawVocab is! Map) {
      throw ArgumentError('tokenizer.json: missing "model.vocab"');
    }
    final vocab = <String, int>{
      for (final e in rawVocab.entries) e.key.toString(): (e.value as num).toInt(),
    };
    final rawMerges = model['merges'];
    if (rawMerges is! List) {
      throw ArgumentError('tokenizer.json: missing "model.merges"');
    }
    final mergeRank = <String, int>{};
    for (int i = 0; i < rawMerges.length; i++) {
      final m = rawMerges[i];
      if (m is String) {
        mergeRank[m] = i;
      } else if (m is List && m.length == 2) {
        mergeRank['${m[0]} ${m[1]}'] = i;
      }
    }

    final byteEncoder = _bytesToUnicode();
    final byteDecoder = <String, int>{
      for (int b = 0; b < 256; b++) byteEncoder[b]: b,
    };

    final specials = <_Special>[];
    final rawAdded = raw['added_tokens'];
    if (rawAdded is List) {
      for (final t in rawAdded) {
        if (t is! Map) continue;
        final id = t['id'];
        final content = t['content'];
        if (id is num && content is String && content.isNotEmpty) {
          specials.add(_Special(content, id.toInt()));
        }
      }
      // Longest-first so greedy match doesn't miss e.g. `<|endoftext|>`
      // in favour of a partial match.
      specials.sort((a, b) => b.text.length - a.text.length);
    }

    return HFBpeTokenizer._(
      vocab,
      mergeRank,
      byteEncoder,
      byteDecoder,
      specials,
    );
  }

  /// GPT-2's `bytes_to_unicode`. Returns a length-256 list mapping
  /// each byte to a single-code-point Dart string.
  static List<String> _bytesToUnicode() {
    // Bytes that are already "safe" printable characters, taken
    // verbatim from the reference Python:
    //   list(range(ord('!'), ord('~')+1))
    // + list(range(ord('¡'), ord('¬')+1))
    // + list(range(ord('®'), ord('ÿ')+1))
    final bs = <int>[];
    for (int i = 0x21; i <= 0x7E; i++) bs.add(i);
    for (int i = 0xA1; i <= 0xAC; i++) bs.add(i);
    for (int i = 0xAE; i <= 0xFF; i++) bs.add(i);
    final cs = List<int>.of(bs);
    // Assign the missing bytes (control chars, whitespace, del) to
    // codepoints starting at 0x100 in insertion order.
    var n = 0;
    for (int b = 0; b < 256; b++) {
      if (!bs.contains(b)) {
        bs.add(b);
        cs.add(0x100 + n);
        n++;
      }
    }
    final out = List<String>.filled(256, '');
    for (int i = 0; i < bs.length; i++) {
      out[bs[i]] = String.fromCharCode(cs[i]);
    }
    return out;
  }

  /// Encode `text` to a list of BPE ids.
  List<int> encode(String text) {
    final out = <int>[];
    // Greedy split around special-token literals, then BPE the rest.
    var i = 0;
    while (i < text.length) {
      final match = _matchSpecial(text, i);
      if (match != null) {
        _encodeSpan(text.substring(i, match.$1), out);
        out.add(match.$2);
        i = match.$1 + match.$3;
      } else {
        // Consume up to next special (or end).
        final nextSpecial = _findNextSpecial(text, i);
        final end = nextSpecial ?? text.length;
        _encodeSpan(text.substring(i, end), out);
        i = end;
      }
    }
    return out;
  }

  /// Decode a list of BPE ids back into a string.
  String decode(List<int> ids) {
    final buf = StringBuffer();
    for (final id in ids) {
      final t = _idToTok[id];
      if (t == null) continue;
      buf.write(t);
    }
    final tokenString = buf.toString();
    // Reverse the byte-mapping character-by-character, then UTF-8
    // decode the resulting bytes.
    final bytes = <int>[];
    for (final ch in tokenString.runes) {
      final chStr = String.fromCharCode(ch);
      final b = _byteDecoder[chStr];
      if (b != null) {
        bytes.add(b);
      } else {
        // Unknown char (e.g. from a special-token string like `<|endoftext|>`)
        // — round-trip as UTF-8 bytes.
        bytes.addAll(utf8.encode(chStr));
      }
    }
    return utf8.decode(bytes, allowMalformed: true);
  }

  /// The special token used by GPT-2 / Pythia as end-of-text.
  /// Returns `null` if the vocab has no `<|endoftext|>` entry.
  int? get endOfTextId {
    for (final s in _specials) {
      if (s.text == '<|endoftext|>') return s.id;
    }
    return vocab['<|endoftext|>'];
  }

  int get vocabSize => vocab.length;

  // -------------------------------------------------------------------------

  /// If `text` starts with any special token at position `start`,
  /// return `(start, id, length)`.
  (int, int, int)? _matchSpecial(String text, int start) {
    for (final s in _specials) {
      if (text.startsWith(s.text, start)) {
        return (start, s.id, s.text.length);
      }
    }
    return null;
  }

  /// Return the index of the next special-token occurrence at or
  /// after `from`, or `null` if none.
  int? _findNextSpecial(String text, int from) {
    int? best;
    for (final s in _specials) {
      final idx = text.indexOf(s.text, from);
      if (idx >= 0 && (best == null || idx < best)) best = idx;
    }
    return best;
  }

  /// BPE-encode a plain (no special-token) substring into `out`.
  void _encodeSpan(String span, List<int> out) {
    if (span.isEmpty) return;
    // Map bytes → unicode chars.
    final bytes = utf8.encode(span);
    final mappedBuf = StringBuffer();
    for (final b in bytes) {
      mappedBuf.write(_byteEncoder[b]);
    }
    final mapped = mappedBuf.toString();

    // Pre-tokenize with GPT-2 regex.
    for (final m in _preTokenRe.allMatches(mapped)) {
      final word = m.group(0);
      if (word == null || word.isEmpty) continue;
      _bpeWord(word, out);
    }
  }

  /// Apply BPE merges to a single pre-token word (as a string of
  /// byte-mapped single-code-point characters) and append the
  /// resulting token ids to `out`.
  void _bpeWord(String word, List<int> out) {
    // Fast path: whole word already in vocab.
    final direct = vocab[word];
    if (direct != null) {
      out.add(direct);
      return;
    }
    // Start with each character as its own symbol.
    final symbols = <String>[
      for (final r in word.runes) String.fromCharCode(r),
    ];
    while (symbols.length > 1) {
      // Find the pair with the lowest merge rank.
      int bestIdx = -1;
      int bestRank = 1 << 30;
      for (int i = 0; i < symbols.length - 1; i++) {
        final rank = _mergeRank['${symbols[i]} ${symbols[i + 1]}'];
        if (rank != null && rank < bestRank) {
          bestRank = rank;
          bestIdx = i;
        }
      }
      if (bestIdx < 0) break;
      // Merge symbols[bestIdx..bestIdx+1] in place.
      final merged = symbols[bestIdx] + symbols[bestIdx + 1];
      symbols
        ..removeAt(bestIdx + 1)
        ..[bestIdx] = merged;
    }
    for (final s in symbols) {
      final id = vocab[s];
      if (id == null) {
        // Should never happen — every single byte-mapped char is in
        // the base vocab of any well-formed byte-level BPE tokenizer.
        throw StateError('HFBpeTokenizer: symbol "$s" not in vocab');
      }
      out.add(id);
    }
  }
}

class _Special {
  final String text;
  final int id;
  const _Special(this.text, this.id);
}
