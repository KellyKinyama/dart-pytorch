/// ResNet-50 ImageNet classification demo. Loads torchvision's
/// pretrained weights (converted via
/// `scripts/convert_resnet50_pt_to_safetensors.py`) and runs top-K
/// classification on an input image.
///
///   dart run bin/resnet50_demo.dart --image path/to/photo.jpg
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib \
///     dart run bin/resnet50_demo.dart --gpu --image path/to/photo.jpg
///
/// One-time setup (~100 MB checkpoint):
///   python3 scripts/convert_resnet50_pt_to_safetensors.py \
///       models/resnet50/model.safetensors
///
/// Standard ImageNet preprocessing:
///   1. resize shorter side to 256 (bilinear)
///   2. center-crop 224×224
///   3. divide by 255, subtract mean=[0.485,0.456,0.406],
///      divide by std=[0.229,0.224,0.225]
///
/// The demo prints top-5 predictions with softmax probabilities and
/// class names (from `imagenet_classes.json` dropped alongside the
/// safetensors by the conversion script).
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:image/image.dart' as img;

const _weightsDefault = 'models/resnet50/model.safetensors';
const _labelsDefault = 'models/resnet50/imagenet_classes.json';

// torchvision ImageNet norm.
const _meanR = 0.485;
const _meanG = 0.456;
const _meanB = 0.406;
const _stdR = 0.229;
const _stdG = 0.224;
const _stdB = 0.225;

Future<void> main(List<String> args) async {
  String? imagePath;
  var weightsPath = _weightsDefault;
  var labelsPath = _labelsDefault;
  var useGpu = false;
  var topK = 5;
  for (int i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--image':
        imagePath = args[++i];
        break;
      case '--weights':
        weightsPath = args[++i];
        break;
      case '--labels':
        labelsPath = args[++i];
        break;
      case '--gpu':
        useGpu = true;
        break;
      case '--topk':
        topK = int.parse(args[++i]);
        break;
    }
  }
  if (imagePath == null) {
    stderr.writeln('usage: dart run bin/resnet50_demo.dart --image PATH');
    exit(64);
  }
  for (final p in [imagePath, weightsPath]) {
    if (!File(p).existsSync()) {
      stderr.writeln('missing: $p');
      exit(2);
    }
  }

  final device = useGpu ? Device.GPU : Device.CPU;
  print('Building ResNet-50 (device=${useGpu ? "gpu" : "cpu"})');
  final model = ResNet(ResNetConfig.resnet50(device: device));

  final swLoad = Stopwatch()..start();
  print('Loading safetensors from $weightsPath ...');
  final report = ResNetLoader.loadFile(model, weightsPath);
  model.eval();
  swLoad.stop();
  print('Loaded. $report (${swLoad.elapsedMilliseconds} ms)');

  final labels = File(labelsPath).existsSync()
      ? (jsonDecode(File(labelsPath).readAsStringSync()) as List<dynamic>)
            .cast<String>()
      : null;

  print('');
  print('Preprocessing $imagePath ...');
  final x = _preprocess(imagePath, device: device);

  final swFwd = Stopwatch()..start();
  final logits = model(x);
  final flat = logits.toList();
  swFwd.stop();
  print('Forward: ${swFwd.elapsedMilliseconds} ms');

  final probs = _softmax(flat);
  final top = _topK(probs, topK);
  print('');
  print('== top-$topK ==');
  for (final (idx, p) in top) {
    final name = labels != null && idx < labels.length
        ? labels[idx]
        : 'class_$idx';
    final pct = (p * 100).toStringAsFixed(2).padLeft(6);
    print('  $pct%  $name  (idx=$idx)');
  }
}

Tensor _preprocess(String path, {required Device device}) {
  final bytes = File(path).readAsBytesSync();
  final decoded = img.decodeImage(bytes);
  if (decoded == null) {
    throw StateError('could not decode $path');
  }
  // Resize shorter side to 256 (bilinear).
  final sw = decoded.width;
  final sh = decoded.height;
  final scale = 256 / (sw < sh ? sw : sh);
  final rw = (sw * scale).round();
  final rh = (sh * scale).round();
  final resized = img.copyResize(
    decoded,
    width: rw,
    height: rh,
    interpolation: img.Interpolation.linear,
  );
  // Center-crop 224×224.
  final left = (rw - 224) ~/ 2;
  final top = (rh - 224) ~/ 2;
  final data = Float32List(3 * 224 * 224);
  const chSize = 224 * 224;
  for (int y = 0; y < 224; y++) {
    for (int xi = 0; xi < 224; xi++) {
      final px = resized.getPixel(left + xi, top + y);
      final off = y * 224 + xi;
      data[off] = (px.r.toDouble() / 255.0 - _meanR) / _stdR;
      data[chSize + off] = (px.g.toDouble() / 255.0 - _meanG) / _stdG;
      data[2 * chSize + off] = (px.b.toDouble() / 255.0 - _meanB) / _stdB;
    }
  }
  return Tensor.fromFloat32List([1, 3, 224, 224], data, device: device);
}

List<double> _softmax(List<double> logits) {
  var maxVal = -double.infinity;
  for (final v in logits) {
    if (v > maxVal) maxVal = v;
  }
  final out = List<double>.filled(logits.length, 0);
  double sum = 0;
  for (int i = 0; i < logits.length; i++) {
    final e = math.exp(logits[i] - maxVal);
    out[i] = e;
    sum += e;
  }
  for (int i = 0; i < out.length; i++) {
    out[i] /= sum;
  }
  return out;
}

List<(int, double)> _topK(List<double> probs, int k) {
  final indexed = List<(int, double)>.generate(
    probs.length,
    (i) => (i, probs[i]),
  );
  indexed.sort((a, b) => b.$2.compareTo(a.$2));
  return indexed.sublist(0, k);
}
