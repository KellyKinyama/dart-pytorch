/// Tokenizers for F5-TTS.
///
/// Two flavours are provided:
///
///   * `F5TtsCharTokenizer` — character-level, ships with a small default
///     English vocabulary (lowercase a-z, digits, common punctuation and
///     whitespace). Suitable for demos and CI. Callers can pass a custom
///     vocab, or load one from the on-disk `vocab.txt` format used by
///     official F5-TTS releases (one token per line, first line has ID 0).
///
///   * `F5TtsPhonemeTokenizer` — dictionary-based word-to-phoneme mapping,
///     driven by a CMU-Pronouncing-Dictionary style file
///     (`WORD  PH1 PH2 PH3`, one word per line, case-insensitive). Unknown
///     words fall back to a character tokenizer so the pipeline never
///     rejects input.
///
/// Both tokenizers expose the same `.encode(String text) -> List<int>` and
/// `.decode(List<int> ids) -> String` surface, plus `.vocabSize`.
library;

import 'dart:convert';
import 'dart:io';

/// Special tokens that live at fixed low IDs in every F5-TTS vocabulary
/// we produce here. Kept intentionally short.
class F5SpecialToken {
  static const String pad = '<pad>';
  static const String unk = '<unk>';
  static const String bos = '<bos>';
  static const String eos = '<eos>';
  static const List<String> all = [pad, unk, bos, eos];
}

/// Minimum interface every F5-TTS tokenizer implements.
abstract class F5TtsTokenizer {
  int get vocabSize;
  int get padId;
  int get unkId;
  int get bosId;
  int get eosId;
  List<int> encode(String text, {bool addBos = false, bool addEos = false});
  String decode(List<int> ids, {bool skipSpecial = true});
}

/// Character-level tokenizer. The default vocabulary covers:
///   * The 4 special tokens (`<pad>`, `<unk>`, `<bos>`, `<eos>`)
///   * ASCII lowercase a-z
///   * Digits 0-9
///   * Whitespace (space, `\t`, `\n`)
///   * Common punctuation: `.,;:?!'-`
///
/// Input is lower-cased before lookup. Unknown code points map to `<unk>`.
class F5TtsCharTokenizer implements F5TtsTokenizer {
  @override
  final int padId;
  @override
  final int unkId;
  @override
  final int bosId;
  @override
  final int eosId;
  final List<String> _idToToken;
  final Map<String, int> _tokenToId;
  final bool lowerCase;

  F5TtsCharTokenizer._(
    this._idToToken,
    this._tokenToId, {
    this.lowerCase = true,
  }) : padId = _tokenToId[F5SpecialToken.pad]!,
       unkId = _tokenToId[F5SpecialToken.unk]!,
       bosId = _tokenToId[F5SpecialToken.bos]!,
       eosId = _tokenToId[F5SpecialToken.eos]!;

  /// Builds the default English character vocabulary described above.
  factory F5TtsCharTokenizer.defaultEnglish() {
    final tokens = <String>[...F5SpecialToken.all];
    for (int c = 'a'.codeUnitAt(0); c <= 'z'.codeUnitAt(0); c++) {
      tokens.add(String.fromCharCode(c));
    }
    for (int c = '0'.codeUnitAt(0); c <= '9'.codeUnitAt(0); c++) {
      tokens.add(String.fromCharCode(c));
    }
    tokens.addAll([' ', '\t', '\n']);
    tokens.addAll(['.', ',', ';', ':', '?', '!', "'", '-']);
    return F5TtsCharTokenizer.fromVocab(tokens);
  }

  /// Build from an explicit vocab list. The list must contain the four
  /// special tokens; if any are missing they are prepended.
  factory F5TtsCharTokenizer.fromVocab(
    List<String> vocab, {
    bool lowerCase = true,
  }) {
    final ordered = <String>[];
    for (final s in F5SpecialToken.all) {
      if (!vocab.contains(s)) ordered.add(s);
    }
    ordered.addAll(vocab);
    final map = <String, int>{};
    for (int i = 0; i < ordered.length; i++) {
      if (map.containsKey(ordered[i])) {
        throw ArgumentError(
          'F5TtsCharTokenizer: duplicate token "${ordered[i]}"',
        );
      }
      map[ordered[i]] = i;
    }
    return F5TtsCharTokenizer._(ordered, map, lowerCase: lowerCase);
  }

  /// Load an on-disk vocabulary file — one token per line, UTF-8. Blank
  /// lines and lines starting with `#` are skipped.
  factory F5TtsCharTokenizer.fromFile(String path, {bool lowerCase = true}) {
    final raw = File(path).readAsStringSync();
    final tokens = <String>[];
    for (final line in LineSplitter.split(raw)) {
      final t = line;
      if (t.isEmpty || t.startsWith('#')) continue;
      tokens.add(t);
    }
    return F5TtsCharTokenizer.fromVocab(tokens, lowerCase: lowerCase);
  }

  @override
  int get vocabSize => _idToToken.length;

  @override
  List<int> encode(String text, {bool addBos = false, bool addEos = false}) {
    final ids = <int>[];
    if (addBos) ids.add(bosId);
    final src = lowerCase ? text.toLowerCase() : text;
    for (int i = 0; i < src.length; i++) {
      final ch = src[i];
      ids.add(_tokenToId[ch] ?? unkId);
    }
    if (addEos) ids.add(eosId);
    return ids;
  }

  @override
  String decode(List<int> ids, {bool skipSpecial = true}) {
    final buf = StringBuffer();
    for (final id in ids) {
      if (id < 0 || id >= _idToToken.length) continue;
      final tok = _idToToken[id];
      if (skipSpecial && F5SpecialToken.all.contains(tok)) continue;
      buf.write(tok);
    }
    return buf.toString();
  }

  /// Read-only view of the id -> token table.
  List<String> get vocabulary => List.unmodifiable(_idToToken);
}

/// Dictionary-based English word-to-phoneme tokenizer.
///
/// The dictionary is a text file where each line has the form
///
///     WORD  PH1 PH2 PH3 ...
///
/// (whitespace-separated). Comments start with `;;;` (CMU-dict style)
/// or `#`. Words with multiple pronunciations may be suffixed with
/// `(1)`, `(2)` — only the first pronunciation is kept.
///
/// Words are matched case-insensitively. Unknown words fall back to a
/// character tokenizer whose vocabulary shares the phoneme vocab's
/// special-token slot IDs. Punctuation and whitespace are emitted as
/// themselves via the char fallback.
class F5TtsPhonemeTokenizer implements F5TtsTokenizer {
  @override
  final int padId;
  @override
  final int unkId;
  @override
  final int bosId;
  @override
  final int eosId;
  final List<String> _idToToken;
  final Map<String, int> _tokenToId;
  final Map<String, List<int>> _wordToPhonemeIds;
  final Set<int> _phonemeIds;

  F5TtsPhonemeTokenizer._(
    this._idToToken,
    this._tokenToId,
    this._wordToPhonemeIds,
    this._phonemeIds,
  ) : padId = _tokenToId[F5SpecialToken.pad]!,
      unkId = _tokenToId[F5SpecialToken.unk]!,
      bosId = _tokenToId[F5SpecialToken.bos]!,
      eosId = _tokenToId[F5SpecialToken.eos]!;

  /// Build from a phoneme inventory + a word -> phoneme map. The
  /// phoneme inventory is the list of distinct phoneme strings (e.g.
  /// `['AH', 'B', 'K', ...]`). Callers who need a full CMU-dict tokenizer
  /// should prefer `F5TtsPhonemeTokenizer.fromCmuDictFile`.
  factory F5TtsPhonemeTokenizer.fromInventory({
    required List<String> phonemes,
    required Map<String, List<String>> words,
    List<String> extraSymbols = const [' ', '.', ',', '?', '!', "'", '-'],
  }) {
    final tokens = <String>[
      ...F5SpecialToken.all,
      ...phonemes,
      ...extraSymbols,
    ];
    final map = <String, int>{};
    for (int i = 0; i < tokens.length; i++) {
      if (map.containsKey(tokens[i])) {
        throw ArgumentError(
          'F5TtsPhonemeTokenizer: duplicate token "${tokens[i]}"',
        );
      }
      map[tokens[i]] = i;
    }
    final phonemeIds = <int>{for (final p in phonemes) map[p]!};
    final wpi = <String, List<int>>{};
    words.forEach((w, phs) {
      final ids = <int>[];
      for (final p in phs) {
        final id = map[p];
        if (id == null) {
          throw ArgumentError(
            'F5TtsPhonemeTokenizer: phoneme "$p" for word "$w" not in inventory',
          );
        }
        ids.add(id);
      }
      wpi[w.toLowerCase()] = ids;
    });
    return F5TtsPhonemeTokenizer._(tokens, map, wpi, phonemeIds);
  }

  /// Parse a CMU-Pronouncing-Dictionary style file. Stress digits are
  /// stripped by default (`AH0` -> `AH`, `EY1` -> `EY`) so the phoneme
  /// vocabulary stays compact; pass `stripStress: false` to keep them.
  factory F5TtsPhonemeTokenizer.fromCmuDictFile(
    String path, {
    bool stripStress = true,
  }) {
    final phonemeSet = <String>{};
    final words = <String, List<String>>{};
    final raw = File(path).readAsStringSync();
    for (final rawLine in LineSplitter.split(raw)) {
      final line = rawLine.trim();
      if (line.isEmpty || line.startsWith(';;;') || line.startsWith('#'))
        continue;
      final parts = line.split(RegExp(r'\s+'));
      if (parts.length < 2) continue;
      var word = parts[0].toLowerCase();
      final variantMatch = RegExp(r'\((\d+)\)$').firstMatch(word);
      if (variantMatch != null) {
        if (variantMatch.group(1) != '1') continue; // keep only first form
        word = word.substring(0, variantMatch.start);
      }
      if (words.containsKey(word)) continue;
      final phs = <String>[];
      for (int i = 1; i < parts.length; i++) {
        var ph = parts[i];
        if (stripStress) ph = ph.replaceAll(RegExp(r'\d+$'), '');
        if (ph.isEmpty) continue;
        phs.add(ph);
        phonemeSet.add(ph);
      }
      if (phs.isNotEmpty) words[word] = phs;
    }
    final sortedPhonemes = phonemeSet.toList()..sort();
    return F5TtsPhonemeTokenizer.fromInventory(
      phonemes: sortedPhonemes,
      words: words,
    );
  }

  @override
  int get vocabSize => _idToToken.length;

  @override
  List<int> encode(String text, {bool addBos = false, bool addEos = false}) {
    final ids = <int>[];
    if (addBos) ids.add(bosId);
    final wordRe = RegExp(r"[A-Za-z']+|\s+|[^\sA-Za-z']");
    for (final m in wordRe.allMatches(text)) {
      final chunk = m.group(0)!;
      if (RegExp(r"^[A-Za-z']+$").hasMatch(chunk)) {
        final key = chunk.toLowerCase();
        final phIds = _wordToPhonemeIds[key];
        if (phIds != null) {
          ids.addAll(phIds);
        } else {
          for (int i = 0; i < chunk.length; i++) {
            ids.add(_tokenToId[chunk[i].toLowerCase()] ?? unkId);
          }
        }
      } else {
        for (int i = 0; i < chunk.length; i++) {
          ids.add(_tokenToId[chunk[i]] ?? unkId);
        }
      }
    }
    if (addEos) ids.add(eosId);
    return ids;
  }

  @override
  String decode(List<int> ids, {bool skipSpecial = true}) {
    final buf = StringBuffer();
    var prevWasPhoneme = false;
    for (final id in ids) {
      if (id < 0 || id >= _idToToken.length) continue;
      final tok = _idToToken[id];
      if (skipSpecial && F5SpecialToken.all.contains(tok)) continue;
      final isPhoneme = _phonemeIds.contains(id);
      if (isPhoneme && prevWasPhoneme) buf.write(' ');
      buf.write(tok);
      prevWasPhoneme = isPhoneme;
    }
    return buf.toString();
  }

  List<String> get vocabulary => List.unmodifiable(_idToToken);
  int get numWords => _wordToPhonemeIds.length;
}
