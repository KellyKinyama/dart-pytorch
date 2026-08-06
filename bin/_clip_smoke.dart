/// Smoke-test: load real CLIP-B/32 safetensors, run a forward pass on
/// a real image, and sanity-check the output stats. Not shipped as a
/// user-facing tool — this exists purely to validate the loader.
///
/// Run:
///   dart run bin/_clip_smoke.dart \
///     --weights models/clip-vit-base-patch32/model.safetensors \
///     --image "faces_gallery/Brad Pitt/sample_0.jpg"
library;

import 'dart:io';
import 'dart:math' as math;

import 'package:image/image.dart' as img_pkg;
import 'package:dart_pytorch/dart_pytorch.dart';

void main(List<String> args) {
  String weightsPath = 'models/clip-vit-base-patch32/model.safetensors';
  String imagePath = 'faces_gallery/Brad Pitt/sample_0.jpg';
  for (var i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--weights':
        weightsPath = args[++i];
      case '--image':
        imagePath = args[++i];
    }
  }

  stdout.writeln('=== CLIP-B/32 smoke test ===');
  stdout.writeln('weights: $weightsPath');
  stdout.writeln('image:   $imagePath');

  // Build model with the B/32 preset.
  final cfg = ClipHFLoader.base32Config();
  final model = CLIPVisionModel(cfg);

  final sw = Stopwatch()..start();
  final report = ClipHFLoader.loadFile(model, weightsPath);
  sw.stop();
  stdout.writeln(
    'load ok: prefix="${report.prefix}" consumed=${report.consumedCount} '
    'unused=${report.unusedKeys.length} in ${sw.elapsedMilliseconds} ms',
  );
  if (report.unusedKeys.isNotEmpty) {
    stdout.writeln('first unused: ${report.unusedKeys.take(5).toList()}');
  }

  // Decode + normalise image the way CLIP expects (ImageNet mean/std).
  final patch = _decodeAndPatchify(imagePath, cfg);
  stdout.writeln(
    'patch shape: ${patch.shape}  '
    '(want [${cfg.numPatches}, ${cfg.patchPixels}])',
  );

  final swFwd = Stopwatch()..start();
  final Tensor out = Tensor.noGrad(() => model(patch));
  swFwd.stop();

  final flat = out.toList();
  final n = flat.length;
  final mean = flat.reduce((a, b) => a + b) / n;
  var sq = 0.0;
  var mn = flat[0];
  var mx = flat[0];
  for (final v in flat) {
    sq += (v - mean) * (v - mean);
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  final std = math.sqrt(sq / n);

  stdout.writeln(
    'output shape: ${out.shape} '
    '(want [${cfg.numPatches + 1}, ${cfg.embedDim}])',
  );
  stdout.writeln(
    'output stats: mean=${mean.toStringAsFixed(4)} '
    'std=${std.toStringAsFixed(4)} '
    'min=${mn.toStringAsFixed(4)} max=${mx.toStringAsFixed(4)}',
  );
  stdout.writeln('forward: ${swFwd.elapsedMilliseconds} ms');

  // Sanity checks: post_layernorm means each row should be roughly zero-mean
  // unit-variance (before the affine).
  final rows = out.shape[0];
  final cols = out.shape[1];
  final rowMeans = <double>[];
  final rowStds = <double>[];
  for (var r = 0; r < rows; r++) {
    var m = 0.0;
    for (var c = 0; c < cols; c++) {
      m += flat[r * cols + c];
    }
    m /= cols;
    var s = 0.0;
    for (var c = 0; c < cols; c++) {
      final d = flat[r * cols + c] - m;
      s += d * d;
    }
    rowMeans.add(m);
    rowStds.add(math.sqrt(s / cols));
  }
  final avgRowStd = rowStds.reduce((a, b) => a + b) / rowStds.length;
  stdout.writeln(
    'row std (mean over $rows rows): '
    '${avgRowStd.toStringAsFixed(4)}  '
    '(a random-init model with unit-variance LN affine would give ~1)',
  );

  // Difference between cls token (row 0) and patch tokens should be
  // meaningful, not zero.
  var clsPatchL2 = 0.0;
  for (var c = 0; c < cols; c++) {
    final d = flat[c] - flat[cols + c];
    clsPatchL2 += d * d;
  }
  stdout.writeln(
    'L2(cls − patch[0]): '
    '${math.sqrt(clsPatchL2).toStringAsFixed(4)}',
  );
}

/// Decode `imagePath`, resize to `cfg.imageSize`, apply CLIP ImageNet
/// normalisation, patchify to `[numPatches, patchPixels]` in (row, col,
/// channel) order — same layout as `bin/llama_llava_demo.dart`.
Tensor _decodeAndPatchify(String path, CLIPVisionConfig cfg) {
  final bytes = File(path).readAsBytesSync();
  final decoded = img_pkg.decodeImage(bytes);
  if (decoded == null) {
    throw StateError('could not decode $path');
  }
  final resized = img_pkg.copyResize(
    decoded,
    width: cfg.imageSize,
    height: cfg.imageSize,
    interpolation: img_pkg.Interpolation.linear,
  );

  // CLIP ImageNet mean/std.
  const meanR = 0.48145466, meanG = 0.4578275, meanB = 0.40821073;
  const stdR = 0.26862954, stdG = 0.26130258, stdB = 0.27577711;

  final P = cfg.patchSize;
  final H = cfg.imageSize;
  final W = cfg.imageSize;
  final patchesX = W ~/ P;
  final patchesY = H ~/ P;

  final rowStride = P * P * 3;
  final data = List<double>.filled(cfg.numPatches * rowStride, 0.0);

  for (var py = 0; py < patchesY; py++) {
    for (var px = 0; px < patchesX; px++) {
      final patchIdx = py * patchesX + px;
      final base = patchIdx * rowStride;
      for (var y = 0; y < P; y++) {
        for (var x = 0; x < P; x++) {
          final pix = resized.getPixel(px * P + x, py * P + y);
          final r = pix.rNormalized.toDouble();
          final g = pix.gNormalized.toDouble();
          final b = pix.bNormalized.toDouble();
          final off = base + (y * P + x) * 3;
          data[off + 0] = (r - meanR) / stdR;
          data[off + 1] = (g - meanG) / stdG;
          data[off + 2] = (b - meanB) / stdB;
        }
      }
    }
  }
  return Tensor.fromList([cfg.numPatches, rowStride], data, device: cfg.device);
}
