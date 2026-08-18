/// Load Silero VAD weights and feed silence — probability should be near zero.
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

Future<void> main() async {
  const weightsPath = 'models/silero_vad/silero_vad.dpt';
  if (!File(weightsPath).existsSync()) {
    stderr.writeln(
      'missing $weightsPath — run:\n'
      '  python3 scripts/extract_silero_vad.py \\\n'
      '    models/silero_vad/silero_vad.onnx \\\n'
      '    models/silero_vad/silero_vad.dpt',
    );
    exit(64);
  }

  final model = SileroVad();
  SileroVadReader.loadFile(model, weightsPath);
  stdout.writeln('Loaded ${model.parameters().length} parameter tensors');

  final silence = Float32List(SileroVad.chunkSize);
  final input = Tensor.fromFloat32List(
    [1, SileroVad.chunkSize],
    silence,
    device: Device.CPU,
  );

  var state = model.zeroState();
  var ctx = model.zeroContext();
  for (int i = 0; i < 5; i++) {
    final r = model.callChunk(input: input, state: state, context: ctx);
    state = r.state;
    ctx = r.context;
    final p = r.prob.toList()[0];
    stdout.writeln('  chunk $i (silence) → p_speech = ${p.toStringAsFixed(4)}');
  }
}
