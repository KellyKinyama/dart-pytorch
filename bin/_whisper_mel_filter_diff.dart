/// Diff our Dart Whisper mel filterbank vs openai-whisper's reference.
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

Future<void> main() async {
  const npzPath = '/tmp/mel_80.raw';
  if (!File(npzPath).existsSync()) {
    stderr.writeln(
      'missing $npzPath — run this Python one-liner first:\n'
      "  python3 -c \"import numpy as np; "
      "open('/tmp/mel_80.raw', 'wb').write("
      "np.load('/tmp/mel_filters.npz')['mel_80']"
      ".astype(np.float32).tobytes())\"",
    );
    exit(64);
  }

  final refFilters = _loadMel80(File(npzPath).readAsBytesSync());
  stdout.writeln(
    'ref shape: ${refFilters.length} × ${refFilters[0].length}  '
    'sum=${_sum(refFilters).toStringAsFixed(6)}',
  );

  // Our Dart filterbank.
  final ours = createMelFilterbank(
    sampleRate: 16000,
    nFft: 400,
    nMels: 80,
    fMin: 0.0,
  );
  stdout.writeln(
    'dart shape: ${ours.length} × ${ours[0].length}  '
    'sum=${_sumF64(ours).toStringAsFixed(6)}',
  );

  // Element-wise diff.
  var maxAbs = 0.0;
  var maxRel = 0.0;
  var count = 0;
  for (int m = 0; m < 80; m++) {
    for (int k = 0; k < 201; k++) {
      final r = refFilters[m][k];
      final o = ours[m][k];
      final d = (r - o).abs();
      if (d > maxAbs) maxAbs = d;
      if (r.abs() > 1e-8) {
        final rel = d / r.abs();
        if (rel > maxRel) maxRel = rel;
      }
      if (d > 0.001) count++;
    }
  }
  stdout.writeln(
    'max abs diff = $maxAbs, '
    'max rel diff = ${(maxRel * 100).toStringAsFixed(3)}%, '
    '#big diffs = $count',
  );
}

// Read mel_80 from a raw fp32 binary (rows-major, 80×201).
List<Float32List> _loadMel80(Uint8List bytes) {
  const rows = 80, cols = 201;
  if (bytes.length != rows * cols * 4) {
    throw StateError('expected ${rows * cols * 4} bytes, got ${bytes.length}');
  }
  final bd = ByteData.sublistView(bytes);
  final out = List.generate(rows, (_) => Float32List(cols));
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      out[r][c] = bd.getFloat32((r * cols + c) * 4, Endian.little);
    }
  }
  return out;
}

double _sum(List<Float32List> m) {
  var s = 0.0;
  for (final row in m) {
    for (final v in row) {
      s += v;
    }
  }
  return s;
}

double _sumF64(List<Float64List> m) {
  var s = 0.0;
  for (final row in m) {
    for (final v in row) {
      s += v;
    }
  }
  return s;
}
