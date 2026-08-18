/// FaceNet InceptionResnetV1 (VGGFace2) smoke test.
///
///   dart run bin/facenet_demo.dart \
///       [--weights PATH] [--input PATH] [--ref PATH]
///
/// Defaults:
///   --weights models/facenet-vggface2/model.safetensors
///   --input   /tmp/facenet_input.raw   (write with scripts/facenet_reference.py)
///   --ref     /tmp/facenet_ref.raw     (optional; compares if present)
///
/// The Python reference lives in [scripts/facenet_reference.py] —
/// it turns a face jpeg into the exact `(x − 127.5)/128` fp32 tensor
/// (`[3, 160, 160]`) that facenet-pytorch expects and writes it to
/// `/tmp/facenet_input.raw`.
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

  print('== build + load ==');
  final swLoad = Stopwatch()..start();
  final model = InceptionResnetV1();
  final report = FaceNetLoader.loadFile(model, weightsPath);
  model.eval();
  swLoad.stop();
  print('  ${swLoad.elapsedMilliseconds} ms  $report');
  if (report.unusedKeys.isNotEmpty) {
    print('  unused (up to 5):');
    for (final k in report.unusedKeys.take(5)) {
      print('    - $k');
    }
  }

  print('');
  print('== read input ==');
  final bytes = File(inputPath).readAsBytesSync();
  const expectN = 3 * 160 * 160;
  if (bytes.lengthInBytes != expectN * 4) {
    stderr.writeln('input must be $expectN fp32 (${expectN * 4} bytes); '
        'got ${bytes.lengthInBytes}');
    exit(2);
  }
  final input = Float32List.view(
    bytes.buffer,
    bytes.offsetInBytes,
    expectN,
  );
  final xT = Tensor.fromFloat32List([1, 3, 160, 160], input);
  print('  loaded  [1, 3, 160, 160]  $inputPath');

  print('');
  print('== forward ==');
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
  print('  norm=${norm.toStringAsFixed(6)}  '
      'min=${mn.toStringAsFixed(4)}  max=${mx.toStringAsFixed(4)}');
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
    if (ref.length != e.length) {
      print('  length mismatch: ref=${ref.length} ours=${e.length}');
    } else {
      // Cosine similarity = dot product (both L2-normalized).
      double dot = 0.0, absSum = 0.0, absMax = 0.0;
      for (int i = 0; i < ref.length; i++) {
        dot += e[i] * ref[i];
        final d = (e[i] - ref[i]).abs();
        absSum += d;
        if (d > absMax) absMax = d;
      }
      print('  cosine     = ${dot.toStringAsFixed(6)}');
      print('  mean |Δ|   = ${(absSum / ref.length).toStringAsFixed(6)}');
      print('  max  |Δ|   = ${absMax.toStringAsFixed(6)}');
      print('  first 5 ref: ${ref.take(5).map((v) => v.toStringAsFixed(6)).toList()}');
    }
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
