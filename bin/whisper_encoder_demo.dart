/// End-to-end smoke test for the Whisper tiny.en encoder.
///
///   dart run bin/whisper_encoder_demo.dart \
///       [--wav PATH] [--weights PATH]
///
/// Defaults:
///   --wav      models/silero_vad/warmup_audio.wav
///   --weights  models/whisper-tiny.en/model.safetensors
///
/// Loads the wav, computes an 80-mel [1, 80, 3000] log-mel input,
/// then runs it through a HF-loaded `WhisperEncoder` (tiny.en) and
/// prints the output shape + basic stats.
library;

import 'dart:io';

import 'package:dart_pytorch/core/audio/whisper_mel.dart';
import 'package:dart_pytorch/core/nn/whisper.dart';
import 'package:dart_pytorch/core/nn/whisper_hf_loader.dart';
import 'package:dart_pytorch/core/tensor/tensor.dart';

Future<void> main(List<String> args) async {
  String wavPath = 'models/silero_vad/warmup_audio.wav';
  String weightsPath = 'models/whisper-tiny.en/model.safetensors';
  for (int i = 0; i < args.length; i++) {
    final a = args[i];
    if (a == '--wav' && i + 1 < args.length) {
      wavPath = args[++i];
    } else if (a == '--weights' && i + 1 < args.length) {
      weightsPath = args[++i];
    }
  }

  if (!File(wavPath).existsSync()) {
    stderr.writeln('missing wav: $wavPath');
    exit(2);
  }
  if (!File(weightsPath).existsSync()) {
    stderr.writeln('missing weights: $weightsPath');
    exit(2);
  }

  final swTotal = Stopwatch()..start();

  print('== log-mel ==');
  final swMel = Stopwatch()..start();
  final mel = await WhisperMel().logMelFromFile(wavPath);
  swMel.stop();
  print('  shape        = [1, 80, 3000] ($wavPath)');
  print('  bytes        = ${mel.lengthInBytes}');
  print('  wall         = ${swMel.elapsedMilliseconds} ms');

  // Wrap into a Tensor [1, 80, 3000].
  final melTensor = Tensor.fromFloat32List(
    [1, 80, 3000],
    mel,
    device: Device.CPU,
  );

  print('');
  print('== build encoder ==');
  final swBuild = Stopwatch()..start();
  final encoder = WhisperEncoder(
    nMels: 80,
    embedDim: 384,
    numHeads: 6,
    numLayers: 4,
    nCtx: 1500,
    device: Device.CPU,
  );
  swBuild.stop();
  print('  tiny.en (d=384, h=6, L=4)  built in ${swBuild.elapsedMilliseconds} ms');

  print('');
  print('== load weights ==');
  final swLoad = Stopwatch()..start();
  final report = WhisperHFLoader.loadFile(encoder, weightsPath);
  swLoad.stop();
  print('  path         = $weightsPath');
  print('  wall         = ${swLoad.elapsedMilliseconds} ms');
  print('  $report');
  if (report.unusedKeys.isNotEmpty) {
    print('  unused (up to 10):');
    for (final k in report.unusedKeys.take(10)) {
      print('    - $k');
    }
  }

  print('');
  print('== forward pass ==');
  final swFwd = Stopwatch()..start();
  final hidden = encoder(melTensor);
  swFwd.stop();
  print('  input shape  = [1, 80, 3000]');
  print('  output shape = ${hidden.shape}');
  print('  wall         = ${swFwd.elapsedMilliseconds} ms');

  final data = hidden.toList();
  double sum = 0.0;
  double sq = 0.0;
  double mn = double.infinity;
  double mx = -double.infinity;
  for (final v in data) {
    sum += v;
    sq += v * v;
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  final n = data.length;
  final mean = sum / n;
  final std = ((sq / n) - mean * mean).clamp(0.0, double.infinity);
  print('  stats        = mean=${mean.toStringAsFixed(6)} '
      'std=${(std * 0.5 + 0.5).toStringAsFixed(6)} '
      'min=${mn.toStringAsFixed(4)} max=${mx.toStringAsFixed(4)}');

  swTotal.stop();
  print('');
  print('total wall     = ${swTotal.elapsedMilliseconds} ms');
}
