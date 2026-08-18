/// Silero VAD device-comparison demo.
///
/// Loads the same 16 kHz mono 16-bit PCM WAV, runs it through both a
/// CPU-hosted and a GPU-hosted `SileroVad`, and reports
///   * mean absolute probability difference per chunk
///   * wall-clock forward time
///
///   dart run bin/silero_vad_cpu_vs_gpu.dart <in.wav>
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

const _weightsPath = 'models/silero_vad/silero_vad.dpt';

Future<void> main(List<String> args) async {
  if (args.isEmpty) {
    stderr.writeln('usage: silero_vad_cpu_vs_gpu <in.wav>');
    exit(64);
  }
  final wavPath = args[0];
  if (!File(_weightsPath).existsSync()) {
    stderr.writeln('missing $_weightsPath — extract weights first');
    exit(64);
  }
  if (!File(wavPath).existsSync()) {
    stderr.writeln('missing $wavPath');
    exit(64);
  }

  final cpuModel = SileroVad();
  SileroVadReader.loadFile(cpuModel, _weightsPath);
  final gpuModel = SileroVad(device: Device.GPU);
  SileroVadReader.loadFile(gpuModel, _weightsPath);

  final samples = _readWavMono16kHz(wavPath);
  final chunks = samples.length ~/ SileroVad.chunkSize;
  stdout.writeln(
    'Loaded ${samples.length} samples '
    '(${(samples.length / 16000).toStringAsFixed(2)} s, $chunks chunks)',
  );

  final cpuProbs = _runAll(cpuModel, samples, chunks);
  final gpuProbs = _runAll(gpuModel, samples, chunks);

  var absDiff = 0.0;
  var maxDiff = 0.0;
  for (int i = 0; i < chunks; i++) {
    final d = (cpuProbs.probs[i] - gpuProbs.probs[i]).abs();
    absDiff += d;
    if (d > maxDiff) maxDiff = d;
  }
  final meanDiff = absDiff / chunks;

  stdout.writeln('\n            time    n_chunks    per-chunk');
  stdout.writeln(
    '  CPU  ${_pad('${cpuProbs.ms} ms', 8)}  '
    '  ${_pad('$chunks', 8)}   '
    '${(cpuProbs.ms / chunks).toStringAsFixed(2)} ms',
  );
  stdout.writeln(
    '  GPU  ${_pad('${gpuProbs.ms} ms', 8)}  '
    '  ${_pad('$chunks', 8)}   '
    '${(gpuProbs.ms / chunks).toStringAsFixed(2)} ms',
  );
  stdout.writeln(
    '\n  probability diff:  '
    'mean=${meanDiff.toStringAsFixed(6)}  '
    'max=${maxDiff.toStringAsFixed(6)}',
  );
}

String _pad(String s, int w) => s.padLeft(w);

class _Run {
  final List<double> probs;
  final int ms;
  const _Run(this.probs, this.ms);
}

_Run _runAll(SileroVad model, Float32List samples, int chunks) {
  var state = model.zeroState();
  var ctx = model.zeroContext();
  final probs = <double>[];
  final sw = Stopwatch()..start();
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
    probs.add(r.prob.toList()[0]);
  }
  return _Run(probs, sw.elapsedMilliseconds);
}

Float32List _readWavMono16kHz(String path) {
  final bytes = File(path).readAsBytesSync();
  final bd = ByteData.sublistView(bytes);
  if (String.fromCharCodes(bytes.sublist(0, 4)) != 'RIFF') {
    throw StateError('$path: not RIFF');
  }
  if (String.fromCharCodes(bytes.sublist(8, 12)) != 'WAVE') {
    throw StateError('$path: not WAVE');
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
  if (sampleRate != 16000 || bitsPerSample != 16 || dataOff < 0) {
    throw StateError('$path: expected 16 kHz 16-bit PCM WAV');
  }
  final samplesPerChan = dataLen ~/ (2 * numChannels);
  final out = Float32List(samplesPerChan);
  for (int i = 0; i < samplesPerChan; i++) {
    final s = bd.getInt16(dataOff + i * 2 * numChannels, Endian.little);
    out[i] = s / 32768.0;
  }
  return out;
}
