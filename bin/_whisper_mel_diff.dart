/// Compare our Whisper log-mel output vs the Python reference we
/// just dumped to /tmp/mel_ref.raw.
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

Future<void> main() async {
  const refPath = '/tmp/mel_ref.raw';
  if (!File(refPath).existsSync()) {
    stderr.writeln(
      'missing $refPath — run: '
      'python3 scripts/whisper_mel_reference.py',
    );
    exit(64);
  }
  const wavPath = 'models/silero_vad/warmup_audio.wav';

  final mel = WhisperMel();
  final ours = await mel.logMelFromFile(wavPath);
  final refBytes = File(refPath).readAsBytesSync();
  final ref = Float32List.sublistView(refBytes);
  if (ref.length != ours.length) {
    stderr.writeln('shape mismatch: ${ref.length} vs ${ours.length}');
    exit(1);
  }

  var maxAbs = 0.0;
  var sumAbs = 0.0;
  var above1e3 = 0;
  var above1e2 = 0;
  for (int i = 0; i < ours.length; i++) {
    final d = (ours[i] - ref[i]).abs();
    if (d > maxAbs) maxAbs = d;
    sumAbs += d;
    if (d > 1e-3) above1e3++;
    if (d > 1e-2) above1e2++;
  }
  final meanAbs = sumAbs / ours.length;
  stdout.writeln('N = ${ours.length}');
  stdout.writeln('max abs diff:   $maxAbs');
  stdout.writeln('mean abs diff:  $meanAbs');
  stdout.writeln('# > 1e-3:       $above1e3');
  stdout.writeln('# > 1e-2:       $above1e2');
  stdout.writeln('sample dart[0..4]: ${ours.sublist(0, 5).toList()}');
  stdout.writeln('sample ref[0..4]:  ${ref.sublist(0, 5).toList()}');
}
