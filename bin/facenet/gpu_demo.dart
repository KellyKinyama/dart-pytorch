/// FaceNet demo, GPU variant. Same CLI as `bin/facenet/demo.dart`.
///
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib \
///     dart run bin/facenet/gpu_demo.dart [--weights P] [--input P] [--ref P]
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:dart_pytorch/core/nn/vision/facenet.dart';
import 'package:dart_pytorch/core/nn/vision/facenet_loader.dart';
import 'package:dart_pytorch/core/tensor/tensor.dart';

Future<void> main(List<String> args) async {
  var weightsPath = 'models/facenet-vggface2/model.safetensors';
  var inputPath = '/tmp/facenet_input.raw';
  var refPath = '/tmp/facenet_ref.raw';
  for (int i = 0; i < args.length; i++) {
    final a = args[i];
    if (a == '--weights' && i + 1 < args.length) {
      weightsPath = args[++i];
    } else if (a == '--input' && i + 1 < args.length) {
      inputPath = args[++i];
    } else if (a == '--ref' && i + 1 < args.length) {
      refPath = args[++i];
    }
  }
  if (!File(weightsPath).existsSync()) {
    stderr.writeln('missing weights: $weightsPath');
    exit(2);
  }
  if (!File(inputPath).existsSync()) {
    stderr.writeln('missing input raw: $inputPath');
    exit(2);
  }

  final swTotal = Stopwatch()..start();
  const device = Device.GPU;

  print('== build + load (GPU) ==');
  final swLoad = Stopwatch()..start();
  final model = InceptionResnetV1(device: device);
  final report = FaceNetLoader.loadFile(model, weightsPath);
  model.eval();
  swLoad.stop();
  print('  ${swLoad.elapsedMilliseconds} ms  $report');

  print('');
  print('== read input ==');
  final bytes = File(inputPath).readAsBytesSync();
  const expectN = 3 * 160 * 160;
  final input = Float32List.view(bytes.buffer, bytes.offsetInBytes, expectN);
  final xT = Tensor.fromFloat32List([1, 3, 160, 160], input, device: device);

  print('');
  print('== forward (GPU) ==');
  final swFwd = Stopwatch()..start();
  final emb = model(xT);
  swFwd.stop();
  print('  ${swFwd.elapsedMilliseconds} ms  → ${emb.shape}');

  final e = emb.toList();
  double sq = 0.0, mn = double.infinity, mx = -double.infinity;
  for (final v in e) {
    sq += v * v;
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  final norm = _sqrt(sq);
  print(
    '  norm=${norm.toStringAsFixed(6)}  '
    'min=${mn.toStringAsFixed(4)}  max=${mx.toStringAsFixed(4)}',
  );
  print('  first 5: ${e.take(5).map((v) => v.toStringAsFixed(6)).toList()}');

  if (File(refPath).existsSync()) {
    print('');
    print('== diff vs reference ==');
    final refBytes = File(refPath).readAsBytesSync();
    final ref = Float32List.view(
      refBytes.buffer,
      refBytes.offsetInBytes,
      refBytes.lengthInBytes ~/ 4,
    );
    double dot = 0.0, absSum = 0.0, absMax = 0.0;
    for (int i = 0; i < ref.length; i++) {
      dot += e[i] * ref[i];
      final d = (e[i] - ref[i]).abs();
      absSum += d;
      if (d > absMax) absMax = d;
    }
    print('  cosine   = ${dot.toStringAsFixed(6)}');
    print('  mean |Δ| = ${(absSum / ref.length).toStringAsFixed(6)}');
    print('  max  |Δ| = ${absMax.toStringAsFixed(6)}');
  }

  swTotal.stop();
  print('');
  print('total wall = ${swTotal.elapsedMilliseconds} ms');
}

double _sqrt(double x) {
  if (x <= 0) return 0.0;
  var y = x;
  var z = (y + 1.0) / 2.0;
  for (int i = 0; i < 30 && z != y; i++) {
    y = z;
    z = (y + x / y) / 2.0;
  }
  return z;
}
