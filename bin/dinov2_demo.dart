/// DINOv2 ViT-S/14 image-embedding demo. Loads pretrained
/// `facebook/dinov2-small`, embeds a few images, prints pairwise
/// cosine similarities. SOTA self-supervised vision features — great
/// for image retrieval / clustering / verification without a
/// dedicated model like FaceNet.
///
///   dart run bin/dinov2_demo.dart                    # CPU
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib \
///     dart run bin/dinov2_demo.dart --gpu            # GPU
///   dart run bin/dinov2_demo.dart --image A.jpg B.jpg C.jpg
///
/// One-time weight download:
///   mkdir -p models/dinov2-small
///   for f in model.safetensors config.json preprocessor_config.json; do
///     curl -L -o "models/dinov2-small/$f" \
///       "https://huggingface.co/facebook/dinov2-small/resolve/main/$f"
///   done
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/core/nn/vision/dinov2.dart';
import 'package:dart_pytorch/core/nn/vision/dinov2_loader.dart';
import 'package:dart_pytorch/core/tensor/tensor.dart';
import 'package:image/image.dart' as img;

const _weightsPath = 'models/dinov2-small/model.safetensors';
const _imageSize = 224;
const _patchSize = 14;

Future<void> main(List<String> args) async {
  var weightsPath = _weightsPath;
  var useGpu = false;
  final images = <String>[];
  for (int i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--gpu':
        useGpu = true;
        break;
      case '--weights':
        weightsPath = args[++i];
        break;
      case '--image':
        images.add(args[++i]);
        break;
    }
  }
  if (images.isEmpty) {
    images.addAll([
      'faces_gallery/Brad Pitt/sample_0.jpg',
      'faces_gallery/Brad Pitt/sample_1.jpg',
      'faces_gallery/Alia Bhatt/sample_0.jpg',
      'faces_gallery/Billie Eilish/sample_0.jpg',
    ]);
  }
  for (final p in [weightsPath, ...images]) {
    if (!File(p).existsSync()) {
      stderr.writeln('missing: $p');
      exit(2);
    }
  }

  final device = useGpu ? Device.GPU : Device.CPU;
  final swTotal = Stopwatch()..start();

  print('== build + load ${useGpu ? '(GPU)' : '(CPU)'} ==');
  final swLoad = Stopwatch()..start();
  final model = DinoV2Backbone(
    imageSize: _imageSize,
    patchSize: _patchSize,
    embedDim: 384,
    numLayers: 12,
    numHeads: 6,
    device: device,
  );
  final report = DinoV2Loader.loadFile(model, weightsPath);
  model.eval();
  swLoad.stop();
  print('  ${swLoad.elapsedMilliseconds} ms  $report');
  if (report.unusedKeys.isNotEmpty) {
    print('  unused (first 3):');
    for (final k in report.unusedKeys.take(3)) {
      print('    - $k');
    }
  }

  print('');
  print('== embed ${images.length} images ==');
  final embeds = <List<double>>[];
  for (final path in images) {
    final sw = Stopwatch()..start();
    final tokens = model(_decodeAndPatchify(path, device));
    // Take CLS row (index 0), L2-normalize.
    final all = tokens.toList();
    final d = model.embedDim;
    final cls = List<double>.generate(d, (i) => all[i]);
    _l2Normalize(cls);
    embeds.add(cls);
    sw.stop();
    print('  ${sw.elapsedMilliseconds} ms  $path');
  }

  print('');
  print('== pairwise cosine similarity ==');
  for (int i = 0; i < images.length; i++) {
    for (int j = i + 1; j < images.length; j++) {
      final c = _dot(embeds[i], embeds[j]);
      print(
        '  cos=${c.toStringAsFixed(4)}   '
        '${_lbl(images[i])}  ↔  ${_lbl(images[j])}',
      );
    }
  }

  swTotal.stop();
  print('');
  print('total wall = ${swTotal.elapsedMilliseconds} ms');
}

/// Load JPEG, resize to 224×224, ImageNet-normalize, patchify to
/// `[numPatches, P*P*3]` in DINOv2 (openai-clip-like) channel order.
Tensor _decodeAndPatchify(String path, Device device) {
  final bytes = File(path).readAsBytesSync();
  final decoded = img.decodeImage(bytes)!;
  final resized = img.copyResize(
    decoded,
    width: _imageSize,
    height: _imageSize,
    interpolation: img.Interpolation.linear,
  );
  const meanR = 0.485, meanG = 0.456, meanB = 0.406;
  const stdR = 0.229, stdG = 0.224, stdB = 0.225;

  final numPatchesPerSide = _imageSize ~/ _patchSize;
  final numPatches = numPatchesPerSide * numPatchesPerSide;
  const patchVec = _patchSize * _patchSize * 3;
  final data = Float32List(numPatches * patchVec);
  // Match Conv2d(3, D, k=14) weight layout: flatten each patch as
  // [C, H, W] (channel outermost) so a Linear(patch_pixels -> D) with
  // the reshaped conv weight reproduces the Conv2d dot-product.
  const chStride = _patchSize * _patchSize;
  for (int py = 0; py < numPatchesPerSide; py++) {
    for (int px = 0; px < numPatchesPerSide; px++) {
      final patchIdx = py * numPatchesPerSide + px;
      final base = patchIdx * patchVec;
      for (int y = 0; y < _patchSize; y++) {
        for (int x = 0; x < _patchSize; x++) {
          final srcX = px * _patchSize + x;
          final srcY = py * _patchSize + y;
          final pix = resized.getPixel(srcX, srcY);
          final r = pix.r.toDouble() / 255.0;
          final g = pix.g.toDouble() / 255.0;
          final b = pix.b.toDouble() / 255.0;
          final off = base + y * _patchSize + x;
          data[off + 0 * chStride] = (r - meanR) / stdR;
          data[off + 1 * chStride] = (g - meanG) / stdG;
          data[off + 2 * chStride] = (b - meanB) / stdB;
        }
      }
    }
  }
  return Tensor.fromFloat32List([numPatches, patchVec], data, device: device);
}

void _l2Normalize(List<double> v) {
  double n = 0.0;
  for (final x in v) {
    n += x * x;
  }
  n = math.sqrt(n);
  if (n > 0) {
    for (int i = 0; i < v.length; i++) {
      v[i] /= n;
    }
  }
}

double _dot(List<double> a, List<double> b) {
  double s = 0.0;
  for (int i = 0; i < a.length; i++) {
    s += a[i] * b[i];
  }
  return s;
}

String _lbl(String p) {
  final parts = p.split(RegExp(r'[/\\]'));
  if (parts.length < 2) return p;
  return '${parts[parts.length - 2]}/${parts.last}';
}
