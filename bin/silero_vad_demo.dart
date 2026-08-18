/// Streaming Silero VAD demo — reads a 16 kHz mono 16-bit PCM WAV
/// file, feeds it to the VAD in 512-sample chunks, and prints a
/// timeline of speech probabilities.
///
///   dart run bin/silero_vad_demo.dart <in.wav>
///
/// Default input is `lib/../models/silero_vad/warmup.wav` if no
/// argument is given.
///
/// Weights are loaded from `models/silero_vad/silero_vad.dpt`; run
///   python3 scripts/extract_silero_vad.py \
///     models/silero_vad/silero_vad.onnx \
///     models/silero_vad/silero_vad.dpt
/// first.
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

const _weightsPath = 'models/silero_vad/silero_vad.dpt';

Future<void> main(List<String> args) async {
  if (!File(_weightsPath).existsSync()) {
    stderr.writeln(
      'missing $_weightsPath — run:\n'
      '  python3 scripts/extract_silero_vad.py \\\n'
      '    models/silero_vad/silero_vad.onnx \\\n'
      '    models/silero_vad/silero_vad.dpt',
    );
    exit(64);
  }
  if (args.isEmpty) {
    stderr.writeln('usage: silero_vad_demo <in.wav>');
    exit(64);
  }
  final wavPath = args[0];
  if (!File(wavPath).existsSync()) {
    stderr.writeln('missing $wavPath');
    exit(64);
  }

  final model = SileroVad();
  SileroVadReader.loadFile(model, _weightsPath);
  stdout.writeln('Loaded Silero VAD (${model.parameters().length} tensors)');

  final samples = _readWavMono16kHz(wavPath);
  stdout.writeln(
    'Loaded ${samples.length} samples '
    '(${(samples.length / 16000).toStringAsFixed(2)} s)',
  );

  var state = model.zeroState();
  var ctx = model.zeroContext();
  final chunks = samples.length ~/ SileroVad.chunkSize;
  const threshold = 0.5;
  var speechStart = -1.0;
  final speechSpans = <List<double>>[];
  stdout.writeln('\ntime (s)   p_speech   marker');
  for (int i = 0; i < chunks; i++) {
    final chunkData = Float32List.sublistView(
      samples,
      i * SileroVad.chunkSize,
      (i + 1) * SileroVad.chunkSize,
    );
    final input = Tensor.fromFloat32List(
      [1, SileroVad.chunkSize],
      Float32List.fromList(chunkData),
      device: Device.CPU,
    );
    final r = model.callChunk(input: input, state: state, context: ctx);
    state = r.state;
    ctx = r.context;
    final p = r.prob.toList()[0];

    final t = (i * SileroVad.chunkSize) / 16000.0;
    final bar = '#' * (p * 40).round();
    if (i % 5 == 0) {
      stdout.writeln(
        '${t.toStringAsFixed(2).padLeft(7)}   '
        '${p.toStringAsFixed(3)}   ${bar.isEmpty ? '.' : bar}',
      );
    }

    if (p >= threshold && speechStart < 0) {
      speechStart = t;
    } else if (p < threshold && speechStart >= 0) {
      speechSpans.add([speechStart, t]);
      speechStart = -1;
    }
  }
  if (speechStart >= 0) speechSpans.add([speechStart, chunks * SileroVad.chunkSize / 16000.0]);

  stdout.writeln('\nDetected speech spans (threshold p ≥ $threshold):');
  if (speechSpans.isEmpty) {
    stdout.writeln('  (none)');
  } else {
    for (final s in speechSpans) {
      stdout.writeln(
        '  ${s[0].toStringAsFixed(2)}s → ${s[1].toStringAsFixed(2)}s',
      );
    }
  }
}

Float32List _readWavMono16kHz(String path) {
  final bytes = File(path).readAsBytesSync();
  final bd = ByteData.sublistView(bytes);
  if (String.fromCharCodes(bytes.sublist(0, 4)) != 'RIFF') {
    throw StateError('$path: not a RIFF/WAV file');
  }
  if (String.fromCharCodes(bytes.sublist(8, 12)) != 'WAVE') {
    throw StateError('$path: not a WAVE file');
  }

  var off = 12;
  int numChannels = 0;
  int sampleRate = 0;
  int bitsPerSample = 0;
  int dataOff = -1;
  int dataLen = 0;
  while (off + 8 <= bytes.length) {
    final tag = String.fromCharCodes(bytes.sublist(off, off + 4));
    final size = bd.getUint32(off + 4, Endian.little);
    if (tag == 'fmt ') {
      numChannels = bd.getUint16(off + 10, Endian.little);
      sampleRate = bd.getUint32(off + 12, Endian.little);
      bitsPerSample = bd.getUint16(off + 22, Endian.little);
    } else if (tag == 'data') {
      dataOff = off + 8;
      dataLen = size;
      break;
    }
    off += 8 + size;
  }
  if (dataOff < 0) throw StateError('$path: no data chunk');
  if (sampleRate != 16000) {
    throw StateError('$path: expected 16 kHz, got $sampleRate');
  }
  if (bitsPerSample != 16) {
    throw StateError('$path: expected 16-bit PCM, got $bitsPerSample');
  }

  final samplesPerChan = dataLen ~/ (2 * numChannels);
  final out = Float32List(samplesPerChan);
  for (int i = 0; i < samplesPerChan; i++) {
    // Downmix to mono by taking channel 0.
    final s = bd.getInt16(dataOff + i * 2 * numChannels, Endian.little);
    out[i] = s / 32768.0;
  }
  return out;
}
