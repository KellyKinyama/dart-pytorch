/// Demo: load a 16 kHz WAV and dump both power and mel spectrograms
/// (as summaries), then save a PNG mel-spectrogram visualization.
///
///   dart run bin/spectrogram_demo.dart models/silero_vad/warmup_audio.wav
library;

import 'dart:io';

import 'package:dart_pytorch/dart_pytorch.dart';

Future<void> main(List<String> args) async {
  if (args.isEmpty) {
    stderr.writeln('usage: spectrogram_demo <in.wav>');
    exit(64);
  }
  final wavPath = args[0];
  const sampleRate = 16000;
  const nFft = 512;
  const nMels = 80;
  const hop = nFft ~/ 4; // 128 samples = 8 ms at 16 kHz

  stdout.writeln('Loading $wavPath (target ${sampleRate} Hz mono) ...');
  final audio = await loadAudio(wavPath, sampleRate);
  stdout.writeln(
    '  ${audio.samples.length} samples '
    '(${(audio.samples.length / sampleRate).toStringAsFixed(3)} s)',
  );

  final power = calculateSpectrogram(
    audio.samples,
    fftSize: nFft,
    hopSize: hop,
  );
  stdout.writeln(
    'Power spectrogram: ${power.length} frames × ${power[0].length} bins',
  );

  final fbank = createMelFilterbank(
    sampleRate: sampleRate,
    nFft: nFft,
    nMels: nMels,
  );
  final mel = applyMelFilterbank(power, fbank);
  final melDb = powerToDb(mel);
  stdout.writeln(
    'Mel spectrogram: ${melDb.length} mels × ${melDb[0].length} frames  '
    '(${nMels} mels, ${nFft} FFT, hop ${hop})',
  );

  // Summary stats.
  var minV = double.infinity, maxV = -double.infinity;
  for (final row in melDb) {
    for (final v in row) {
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
    }
  }
  stdout.writeln(
    '  dB range: ${minV.toStringAsFixed(1)} .. ${maxV.toStringAsFixed(1)}',
  );

  // Print a small ASCII heatmap: 20 columns × 20 rows.
  const rows = 20, cols = 60;
  final frames = melDb[0].length;
  final mels = melDb.length;
  final glyphs = ' .:-=+*#%@';
  stdout.writeln('\nMel-dB ASCII preview (low → high mel top → bottom):');
  for (int r = 0; r < rows; r++) {
    final mIdx = mels - 1 - (r * mels ~/ rows);
    final sb = StringBuffer();
    for (int c = 0; c < cols; c++) {
      final tIdx = c * frames ~/ cols;
      final v = melDb[mIdx][tIdx];
      final n = ((v - minV) / (maxV - minV)).clamp(0.0, 1.0);
      sb.write(glyphs[(n * (glyphs.length - 1)).floor()]);
    }
    stdout.writeln('  $sb');
  }

  final pngPath = wavPath.replaceAll(RegExp(r'\.wav$'), '.mel.png');
  await saveSpectrogramImage(
    melDb,
    pngPath,
    sampleRate: sampleRate,
    hopLength: hop,
  );
  stdout.writeln('\nSaved: $pngPath');
}
