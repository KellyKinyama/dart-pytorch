/// SentencePiece-BPE tokenizer for Needle 3, matching `RefTokenizer` in the
/// reference `export.py`, plus the chat-prompt builder.
///
/// The tokenizer is carried in the `.cact` RAW attachment:
///   header  u32 n_pieces, u32 pad/eos/bos/unk id, u8 add_dummy_prefix,
///           u8 byte_fallback, u16 _pad;
///   then n_pieces records: f32 score, u8 type (0 NORMAL, 1 UNKNOWN,
///           2 CONTROL, 3 USER_DEFINED, 4 BYTE), u16 surface_len, bytes.
library;

import 'dart:convert';
import 'dart:typed_data';

/// SentencePiece metaspace (▁, U+2581) that stands in for a space.
const String kMetaspace = '\u2581';

const int tkNormal = 0,
    tkUnknown = 1,
    tkControl = 2,
    tkUserDefined = 3,
    tkByte = 4;

/// Fixed special-token ids (reserved 0..15) from the reference tokenizer.
const int idPad = 0,
    idEos = 1,
    idBos = 2,
    idUnk = 3,
    idImStart = 4,
    idImEnd = 5,
    idThinkStart = 6,
    idThinkEnd = 7,
    idToolsStart = 8,
    idToolsEnd = 9,
    idToolCallStart = 10,
    idToolCallEnd = 11,
    idToolResultStart = 12,
    idToolResultEnd = 13,
    idContextStart = 14,
    idContextEnd = 15;

/// Parsed tokenizer metadata.
class TokenizerMeta {
  final List<String> pieces;
  final Float32List scores;
  final Uint8List types;
  final int padId, eosId, bosId, unkId;
  final bool addDummyPrefix, byteFallback;
  const TokenizerMeta({
    required this.pieces,
    required this.scores,
    required this.types,
    required this.padId,
    required this.eosId,
    required this.bosId,
    required this.unkId,
    required this.addDummyPrefix,
    required this.byteFallback,
  });
}

/// Parse the RAW tokenizer blob.
TokenizerMeta parseTokenizerBlob(Uint8List blob) {
  final view = ByteData.view(blob.buffer, blob.offsetInBytes, blob.length);
  final n = view.getUint32(0, Endian.little);
  final pad = view.getUint32(4, Endian.little);
  final eos = view.getUint32(8, Endian.little);
  final bos = view.getUint32(12, Endian.little);
  final unk = view.getUint32(16, Endian.little);
  final addDummy = view.getUint8(20) != 0;
  final byteFb = view.getUint8(21) != 0;
  var off = 24;

  final pieces = <String>[];
  final scores = Float32List(n);
  final types = Uint8List(n);
  for (var i = 0; i < n; i++) {
    scores[i] = view.getFloat32(off, Endian.little);
    types[i] = view.getUint8(off + 4);
    final len = view.getUint16(off + 5, Endian.little);
    off += 7;
    pieces.add(utf8.decode(blob.sublist(off, off + len)));
    off += len;
  }
  return TokenizerMeta(
    pieces: pieces,
    scores: scores,
    types: types,
    padId: pad,
    eosId: eos,
    bosId: bos,
    unkId: unk,
    addDummyPrefix: addDummy,
    byteFallback: byteFb,
  );
}

/// A reference-faithful SentencePiece-BPE encoder/decoder.
class NeedleTokenizer {
  final TokenizerMeta meta;
  final Map<String, int> _p2id = {};
  final Map<int, int> _byteId = {}; // raw byte -> piece id
  late final List<String> _markers; // USER_DEFINED, longest first

  NeedleTokenizer(this.meta) {
    for (var i = 0; i < meta.pieces.length; i++) {
      _p2id[meta.pieces[i]] = i;
      if (meta.types[i] == tkByte) {
        // byte pieces look like "<0xAB>"
        _byteId[int.parse(meta.pieces[i].substring(3, 5), radix: 16)] = i;
      }
    }
    _markers = [
      for (var i = 0; i < meta.pieces.length; i++)
        if (meta.types[i] == tkUserDefined) meta.pieces[i],
    ]..sort((a, b) => b.length.compareTo(a.length));
  }

  factory NeedleTokenizer.fromBlob(Uint8List blob) =>
      NeedleTokenizer(parseTokenizerBlob(blob));

  /// Encode [text] to token ids.
  List<int> encode(String text) {
    if (text.isEmpty) return [];
    var esc = text.replaceAll(' ', kMetaspace);
    if (meta.addDummyPrefix) esc = kMetaspace + esc;

    final ids = <int>[];
    final buf = StringBuffer();
    var i = 0;
    final n = esc.length;
    while (i < n) {
      String? marker;
      for (final m in _markers) {
        if (esc.startsWith(m, i)) {
          marker = m;
          break;
        }
      }
      if (marker != null) {
        if (buf.isNotEmpty) {
          ids.addAll(_bpe(buf.toString()));
          buf.clear();
        }
        ids.add(_p2id[marker]!);
        i += marker.length;
      } else {
        final cu = esc.codeUnitAt(i);
        if (cu >= 0xD800 && cu <= 0xDBFF && i + 1 < n) {
          buf.write(esc.substring(i, i + 2));
          i += 2;
        } else {
          buf.write(esc[i]);
          i += 1;
        }
      }
    }
    if (buf.isNotEmpty) ids.addAll(_bpe(buf.toString()));
    return ids;
  }

  List<int> _bpe(String seg) {
    // split into code-point symbols
    final syms = <String>[];
    for (var k = 0; k < seg.length;) {
      final cu = seg.codeUnitAt(k);
      if (cu >= 0xD800 && cu <= 0xDBFF && k + 1 < seg.length) {
        syms.add(seg.substring(k, k + 2));
        k += 2;
      } else {
        syms.add(seg[k]);
        k += 1;
      }
    }
    // greedy highest-score merges
    while (syms.length > 1) {
      double? bestScore;
      var bestJ = -1;
      for (var j = 0; j < syms.length - 1; j++) {
        final idx = _p2id[syms[j] + syms[j + 1]];
        if (idx != null && (bestScore == null || meta.scores[idx] > bestScore)) {
          bestScore = meta.scores[idx];
          bestJ = j;
        }
      }
      if (bestJ < 0) break;
      syms[bestJ] = syms[bestJ] + syms[bestJ + 1];
      syms.removeAt(bestJ + 1);
    }
    final ids = <int>[];
    for (final s in syms) {
      final idx = _p2id[s];
      if (idx != null) {
        ids.add(idx);
      } else if (meta.byteFallback) {
        for (final b in utf8.encode(s)) {
          ids.add(_byteId[b] ?? meta.unkId);
        }
      } else {
        ids.add(meta.unkId);
      }
    }
    return ids;
  }

  /// Decode token ids back to text.
  String decode(List<int> ids) {
    final bytes = <int>[];
    for (final i in ids) {
      final t = meta.types[i];
      if (t == tkByte) {
        bytes.add(int.parse(meta.pieces[i].substring(3, 5), radix: 16));
      } else if (t == tkControl || t == tkUnknown) {
        continue;
      } else {
        bytes.addAll(utf8.encode(meta.pieces[i]));
      }
    }
    var text = utf8.decode(bytes, allowMalformed: true).replaceAll(
          kMetaspace,
          ' ',
        );
    if (meta.addDummyPrefix && text.startsWith(' ')) {
      text = text.substring(1);
    }
    return text;
  }
}

/// Build the Needle chat prefix token ids for a tool-calling turn, matching
/// the wire format in the porting guide:
///
///   <s><|im_start|>system\ndate: {date}<|im_end|>
///   <|im_start|>user\n<tools>{toolsJson}</tools>\n{user}<|im_end|>
///   <|im_start|>assistant\n
///
/// [toolsJson] is the minified JSON array of tool definitions; [date] is an
/// optional system line like `2026-09-18 Fri 09:41`. Returns BOS followed by
/// the encoded template.
List<int> buildToolPrompt(
  NeedleTokenizer tok, {
  required String toolsJson,
  required String user,
  String? date,
}) {
  final sb = StringBuffer();
  if (date != null) {
    sb.write('<|im_start|>system\ndate: $date<|im_end|>\n');
  }
  sb.write('<|im_start|>user\n');
  if (toolsJson.isNotEmpty) sb.write('<tools>$toolsJson</tools>\n');
  sb.write('$user<|im_end|>\n<|im_start|>assistant\n');
  return [tok.meta.bosId, ...tok.encode(sb.toString())];
}
