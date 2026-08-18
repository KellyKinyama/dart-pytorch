/// FaceNet face-verification demo — loads two (or more) JPEG face
/// crops from `faces_gallery/`, produces 512-d embeddings, and prints
/// pairwise cosine similarities.
///
///   dart run bin/facenet_verify_demo.dart              # default: 3 celebs
///   dart run bin/facenet_verify_demo.dart --gpu        # GPU
///   dart run bin/facenet_verify_demo.dart \
///       "faces_gallery/Brad Pitt/sample_0.jpg" \
///       "faces_gallery/Brad Pitt/sample_1.jpg" \
///       "faces_gallery/Alia Bhatt/sample_0.jpg"
///
/// The gallery images are already face crops, so we just resize to
/// 160×160 (bilinear) and normalize `(x − 127.5) / 128` — the same
/// preprocessing facenet-pytorch's `MTCNN` produces after alignment.
/// For raw wild photos you'd want to run a face detector first
/// (`MTCNN`, `retinaface`, or our future port).
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:dart_pytorch/core/nn/vision/facenet.dart';
import 'package:dart_pytorch/core/nn/vision/facenet_loader.dart';
import 'package:dart_pytorch/core/tensor/tensor.dart';
import 'package:image/image.dart' as img;

const _weightsPathDefault = 'models/facenet-vggface2/model.safetensors';

Future<void> main(List<String> args) async {
  var weightsPath = _weightsPathDefault;
  var useGpu = false;
  final paths = <String>[];
  for (int i = 0; i < args.length; i++) {
    final a = args[i];
    if (a == '--weights' && i + 1 < args.length) {
      weightsPath = args[++i];
    } else if (a == '--gpu') {
      useGpu = true;
    } else {
      paths.add(a);
    }
  }
  if (paths.isEmpty) {
    paths.addAll([
      'faces_gallery/Brad Pitt/sample_0.jpg',
      'faces_gallery/Brad Pitt/sample_1.jpg',
      'faces_gallery/Alia Bhatt/sample_0.jpg',
    ]);
  }
  for (final p in paths) {
    if (!File(p).existsSync()) {
      stderr.writeln('missing: $p');
      exit(2);
    }
  }
  if (!File(weightsPath).existsSync()) {
    stderr.writeln('missing weights: $weightsPath');
    exit(2);
  }

  final device = useGpu ? Device.GPU : Device.CPU;
  final swTotal = Stopwatch()..start();

  print('== build + load ${useGpu ? "(GPU)" : "(CPU)"} ==');
  final swLoad = Stopwatch()..start();
  final model = InceptionResnetV1(device: device);
  final report = FaceNetLoader.loadFile(model, weightsPath);
  model.eval();
  swLoad.stop();
  print('  ${swLoad.elapsedMilliseconds} ms  $report');

  print('');
  print('== embed ${paths.length} faces ==');
  final embeddings = <List<double>>[];
  for (final p in paths) {
    final sw = Stopwatch()..start();
    final x = _decodeAndPrep(p, device);
    final emb = model(x);
    sw.stop();
    embeddings.add(emb.toList());
    print('  ${sw.elapsedMilliseconds} ms  $p');
  }

  print('');
  print('== pairwise cosine similarity ==');
  for (int i = 0; i < paths.length; i++) {
    for (int j = i + 1; j < paths.length; j++) {
      final c = _cosine(embeddings[i], embeddings[j]);
      final tag = c >= 0.4
          ? 'SAME     '
          : c >= 0.25
          ? 'unclear  '
          : 'DIFFERENT';
      print(
        '  $tag  ${c.toStringAsFixed(4)}   '
        '${_lbl(paths[i])}  ↔  ${_lbl(paths[j])}',
      );
    }
  }

  swTotal.stop();
  print('');
  print('total wall = ${swTotal.elapsedMilliseconds} ms');
}

/// Decode a JPEG, resize to 160×160, apply facenet-pytorch's
/// `(x − 127.5) / 128` normalization, and upload to [device].
Tensor _decodeAndPrep(String path, Device device) {
  final bytes = File(path).readAsBytesSync();
  final decoded = img.decodeImage(bytes);
  if (decoded == null) {
    throw StateError('could not decode $path');
  }
  final resized = img.copyResize(
    decoded,
    width: 160,
    height: 160,
    interpolation: img.Interpolation.linear,
  );
  final data = Float32List(3 * 160 * 160);
  const chSize = 160 * 160;
  for (int y = 0; y < 160; y++) {
    for (int x = 0; x < 160; x++) {
      final px = resized.getPixel(x, y);
      final r = px.r.toDouble();
      final g = px.g.toDouble();
      final b = px.b.toDouble();
      final off = y * 160 + x;
      data[off] = (r - 127.5) / 128.0;
      data[chSize + off] = (g - 127.5) / 128.0;
      data[2 * chSize + off] = (b - 127.5) / 128.0;
    }
  }
  return Tensor.fromFloat32List([1, 3, 160, 160], data, device: device);
}

double _cosine(List<double> a, List<double> b) {
  double dot = 0.0;
  for (int i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
  }
  return dot; // both are L2-normalized
}

String _lbl(String p) {
  final parts = p.split(RegExp(r'[/\\]'));
  if (parts.length < 2) return p;
  final person = parts[parts.length - 2];
  final file = parts.last;
  return '$person/$file';
}
