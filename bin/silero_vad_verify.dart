/// Dart-side reproduction of `scripts/silero_vad_reference.py`.
/// Same deterministic input, same 5-chunk sequence, print probs so
/// we can eyeball-compare vs the ONNX Runtime reference output.
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

Float32List makeChunk(int i, int n) {
  final out = Float32List(n);
  if (i == 0) return out;
  for (int k = 0; k < n; k++) {
    final t = (k + i * n) / 16000.0;
    out[k] = 0.3 * math.sin(2 * math.pi * 440 * t);
  }
  return out;
}

Future<void> main() async {
  const weightsPath = 'models/silero_vad/silero_vad.dpt';
  if (!File(weightsPath).existsSync()) {
    stderr.writeln('missing $weightsPath — run the extractor first');
    exit(64);
  }
  final model = SileroVad();
  SileroVadReader.loadFile(model, weightsPath);

  var state = model.zeroState();
  var ctx = model.zeroContext();
  stdout.writeln('chunks (dart_pytorch, deterministic sine at 440 Hz):');
  for (int i = 0; i < 6; i++) {
    final chunk = makeChunk(i, SileroVad.chunkSize);
    final input = Tensor.fromFloat32List(
      [1, SileroVad.chunkSize],
      chunk,
      device: Device.CPU,
    );
    final r = model.callChunk(input: input, state: state, context: ctx);
    state = r.state;
    ctx = r.context;
    final p = r.prob.toList()[0];
    stdout.writeln('  $i: p=${p.toStringAsFixed(6)}');
  }
}
