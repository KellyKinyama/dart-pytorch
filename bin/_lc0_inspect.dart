/// Debug tool: print which field numbers are present in a
/// `.pb.gz` LC0 net. Useful for figuring out whether a downloaded
/// net is classical (fields 1, 2, 3, 4, 11 all present) or an
/// attention-body net (fields ≥ 17 typical).
library;

import 'dart:io';
import 'dart:typed_data';

void main(List<String> args) {
  if (args.isEmpty) {
    stderr.writeln('usage: dart run bin/_lc0_inspect.dart <net.pb.gz>');
    exit(64);
  }
  final path = args[0];
  final bytes = Uint8List.fromList(gzip.decode(File(path).readAsBytesSync()));

  final counts = <int, int>{};
  final firstSizes = <int, int>{};
  var i = 0;
  int? weightsStart, weightsEnd;
  while (i < bytes.length) {
    final r = _readField(bytes, i);
    i = r.end;
    counts.update(r.number, (v) => v + 1, ifAbsent: () => 1);
    firstSizes.putIfAbsent(r.number, () => r.end - r.start);
    if (r.number == 10) {
      weightsStart = r.start;
      weightsEnd = r.end;
    }
  }

  stdout.writeln('=== Net message fields ===');
  final sorted = counts.keys.toList()..sort();
  for (final k in sorted) {
    stdout.writeln(
      '  field $k  x${counts[k]}  first-payload=${firstSizes[k]} bytes',
    );
  }

  if (weightsStart == null || weightsEnd == null) {
    stderr.writeln(
      'no Weights (field 10) found — top-level field mapping'
      ' may differ from the classical Net proto',
    );
    return;
  }

  final wCounts = <int, int>{};
  final wSizes = <int, int>{};
  var j = weightsStart;
  while (j < weightsEnd) {
    final r = _readField(bytes, j);
    j = r.end;
    wCounts.update(r.number, (v) => v + 1, ifAbsent: () => 1);
    wSizes.putIfAbsent(r.number, () => r.end - r.start);
  }
  stdout.writeln('\n=== Weights sub-message fields ===');
  final wSorted = wCounts.keys.toList()..sort();
  for (final k in wSorted) {
    stdout.writeln(
      '  field $k  x${wCounts[k]}  first-payload=${wSizes[k]} bytes  '
      '${_fieldName(k)}',
    );
  }

  final hasClassical =
      wCounts.containsKey(1) &&
      wCounts.containsKey(2) &&
      wCounts.containsKey(11);
  final hasAttention = wCounts.keys.any((k) => k >= 17);
  stdout.writeln(
    '\nverdict: ${hasClassical ? "classical" : "NOT classical"}'
    '${hasAttention ? " (has attention-body fields ≥17)" : ""}',
  );
}

String _fieldName(int n) {
  switch (n) {
    case 1:
      return 'input conv';
    case 2:
      return 'residual block';
    case 3:
      return 'policy conv';
    case 4:
      return 'value conv';
    case 5:
      return 'ip1_val_w';
    case 6:
      return 'ip1_val_b';
    case 7:
      return 'reserved';
    case 8:
      return 'reserved';
    case 9:
      return 'ip2_val_w';
    case 10:
      return 'ip2_val_b';
    case 11:
      return 'policy1 (classical marker)';
    case 12:
      return 'moves-left conv';
    case 13:
      return 'ip1_mov_w';
    case 14:
      return 'ip1_mov_b';
    case 15:
      return 'ip2_mov_w';
    case 16:
      return 'ip2_mov_b';
    case 17:
      return 'encoder (attention body)';
    case 18:
      return 'ip_emb_w';
    case 19:
      return 'ip_emb_b';
    default:
      return '';
  }
}

class _Rec {
  final int number, start, end;
  _Rec(this.number, this.start, this.end);
}

_Rec _readField(Uint8List d, int i) {
  final tag = _readVarint(d, i);
  final wire = tag.value & 7;
  final number = tag.value >> 3;
  var pos = tag.end;
  int end;
  if (wire == 0) {
    final v = _readVarint(d, pos);
    end = v.end;
  } else if (wire == 1) {
    end = pos + 8;
  } else if (wire == 2) {
    final len = _readVarint(d, pos);
    pos = len.end;
    end = pos + len.value;
  } else if (wire == 5) {
    end = pos + 4;
  } else {
    throw StateError('unsupported wire type $wire');
  }
  return _Rec(number, pos, end);
}

class _VR {
  final int value, end;
  _VR(this.value, this.end);
}

_VR _readVarint(Uint8List d, int i) {
  var r = 0;
  var s = 0;
  while (true) {
    final b = d[i++];
    r |= (b & 0x7f) << s;
    if ((b & 0x80) == 0) return _VR(r, i);
    s += 7;
  }
}
