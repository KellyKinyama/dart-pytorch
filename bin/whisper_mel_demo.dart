/// Whisper log-mel smoke test on the warmup WAV.
///
/// Verifies shape [80, 3000] and reasonable value range vs. what
/// openai-whisper produces on a real audio clip.
library;

import 'dart:io';

import 'package:dart_pytorch/dart_pytorch.dart';

Future<void> main(List<String> args) async {
  final wavPath = args.isNotEmpty
      ? args[0]
      : 'models/silero_vad/warmup_audio.wav';
  if (!File(wavPath).existsSync()) {
    stderr.writeln('missing $wavPath');
    exit(64);
  }

  final mel = WhisperMel();
  final sw = Stopwatch()..start();
  final flat = await mel.logMelFromFile(wavPath);
  final ms = sw.elapsedMilliseconds;

  final n = mel.cfg.nMels;
  final t = mel.cfg.nFrames;
  stdout.writeln(
    'log-mel: ${n} mels × ${t} frames  '
    '(${flat.length} floats, $ms ms)',
  );

  var minV = double.infinity, maxV = -double.infinity, sum = 0.0;
  for (int i = 0; i < flat.length; i++) {
    final v = flat[i];
    if (v < minV) minV = v;
    if (v > maxV) maxV = v;
    sum += v;
  }
  final mean = sum / flat.length;
  stdout.writeln(
    'stats: min=${minV.toStringAsFixed(4)}  '
    'max=${maxV.toStringAsFixed(4)}  '
    'mean=${mean.toStringAsFixed(4)}',
  );
  // openai-whisper's log_mel outputs values in roughly [-1, +0.5] on speech
  // with the (log10 - max, +4)/4 scaling. Anything wildly outside means the
  // frontend is off.
  if (maxV > 2.0 || minV < -5.0) {
    stderr.writeln('WARNING: value range looks wrong for whisper log-mel');
  }

  // ASCII heat preview: 20 rows × 60 cols.
  const rows = 20, cols = 60;
  final glyphs = ' .:-=+*#%@';
  stdout.writeln('\nlog-mel preview (low mel top, low frame left):');
  for (int r = 0; r < rows; r++) {
    final mIdx = n - 1 - (r * n ~/ rows);
    final sb = StringBuffer();
    for (int c = 0; c < cols; c++) {
      final tIdx = c * t ~/ cols;
      final v = flat[mIdx * t + tIdx];
      final norm = ((v - minV) / (maxV - minV)).clamp(0.0, 1.0);
      sb.write(glyphs[(norm * (glyphs.length - 1)).floor()]);
    }
    stdout.writeln('  $sb');
  }
}
