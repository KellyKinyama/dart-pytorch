/// SAM ViT-B end-to-end demo: image + point-click prompt → segmentation mask.
///
/// Loads a real SAM ViT-B checkpoint (converted from Meta's
/// `sam_vit_b_01ec64.pth` via `scripts/convert_sam_pt_to_safetensors.py`)
/// and runs the full pipeline:
///
///   1. Read an image, resize longest side to 1024, letterbox-pad to
///      1024×1024, normalise with ImageNet mean/std.
///   2. Encode it with [SamImageEncoder] → `[1, 256, 64, 64]`.
///   3. Encode a positive point-click prompt with [SamPromptEncoder].
///   4. Run [SamMaskDecoder] to get 4 mask candidates + IoU scores.
///   5. Pick the highest-IoU mask (or all three "multimask" if
///      `--multimask` is set), upscale from 256×256 to the original
///      image size, and dump a PNG overlay to `--out`.
///
///   dart run bin/sam_demo.dart --image PATH --point X,Y \\
///       [--multimask] [--out mask.png]
///
/// **Memory warning.** SAM ViT-B is ~350 MB fp32 and forward at
/// 1024×1024 loads ~8 GB of activations on CPU (dominated by the
/// per-window attention). This demo is CPU-only; the encoder's
/// windowed attention runs on host currently.
///
/// One-time setup:
///   mkdir -p models/sam-vit-b
///   curl -L -o /tmp/sam_vit_b_01ec64.pth \\
///     https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
///   python3 scripts/convert_sam_pt_to_safetensors.py \\
///     /tmp/sam_vit_b_01ec64.pth models/sam-vit-b/model.safetensors
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart' as img_pkg;
import 'package:dart_pytorch/dart_pytorch.dart';

const _weightsDefault = 'models/sam-vit-b/model.safetensors';

Future<void> main(List<String> args) async {
  String? imagePath;
  String pointCsv = '512,512';
  var weightsPath = _weightsDefault;
  var outPath = 'mask.png';
  var multimask = false;
  for (int i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--image':
        imagePath = args[++i];
        break;
      case '--point':
        pointCsv = args[++i];
        break;
      case '--weights':
        weightsPath = args[++i];
        break;
      case '--out':
        outPath = args[++i];
        break;
      case '--multimask':
        multimask = true;
        break;
    }
  }
  if (imagePath == null) {
    stderr.writeln(
      'usage: dart run bin/sam_demo.dart --image PATH '
      '--point X,Y [--multimask] [--out mask.png]',
    );
    exit(64);
  }
  if (!File(imagePath).existsSync()) {
    stderr.writeln('missing: $imagePath');
    exit(2);
  }
  if (!File(weightsPath).existsSync()) {
    stderr.writeln('missing: $weightsPath');
    stderr.writeln('See setup steps at the top of bin/sam_demo.dart');
    exit(2);
  }
  final pointParts = pointCsv.split(',').map(double.parse).toList();
  if (pointParts.length != 2) {
    stderr.writeln('--point must be "X,Y"; got "$pointCsv"');
    exit(64);
  }
  final rawPx = pointParts[0];
  final rawPy = pointParts[1];

  // Load and preprocess image.
  final swPre = Stopwatch()..start();
  final bytes = File(imagePath).readAsBytesSync();
  final decoded = img_pkg.decodeImage(bytes);
  if (decoded == null) {
    throw StateError('could not decode $imagePath');
  }
  final origH = decoded.height;
  final origW = decoded.width;
  const targetSize = 1024;
  final scale = targetSize / math.max(origH, origW);
  final resizedH = (origH * scale).round();
  final resizedW = (origW * scale).round();
  final resized = img_pkg.copyResize(
    decoded,
    width: resizedW,
    height: resizedH,
    interpolation: img_pkg.Interpolation.linear,
  );
  final imageInput = Float32List(3 * targetSize * targetSize);
  // ImageNet norm (SAM uses [123.675, 116.28, 103.53] / 58.395 etc.).
  const meanR = 123.675, meanG = 116.28, meanB = 103.53;
  const stdR = 58.395, stdG = 57.12, stdB = 57.375;
  for (int y = 0; y < resizedH; y++) {
    for (int x = 0; x < resizedW; x++) {
      final px = resized.getPixel(x, y);
      final off = y * targetSize + x;
      imageInput[off] = (px.r.toDouble() - meanR) / stdR;
      imageInput[targetSize * targetSize + off] =
          (px.g.toDouble() - meanG) / stdG;
      imageInput[2 * targetSize * targetSize + off] =
          (px.b.toDouble() - meanB) / stdB;
    }
  }
  swPre.stop();
  print(
    'preprocess: ${swPre.elapsedMilliseconds} ms  (resized to '
    '$resizedW x $resizedH, letterbox to $targetSize x $targetSize)',
  );

  // Point coordinates in the letterboxed 1024×1024 space.
  final letterX = rawPx * scale;
  final letterY = rawPy * scale;

  // Build model triple.
  print('');
  print('Building SAM ViT-B (image encoder + prompt encoder + mask decoder)');
  final swBuild = Stopwatch()..start();
  final imgEnc = SamImageEncoder(SamImageEncoderConfig.vitB());
  final promptEnc = SamPromptEncoder(
    embedDim: 256,
    imageEmbedH: 64,
    imageEmbedW: 64,
    imageSize: 1024,
    maskInSize: 256,
    maskInputChannels: 16,
  );
  final maskDec = SamMaskDecoder(const SamMaskDecoderConfig());
  swBuild.stop();
  print('  build: ${swBuild.elapsedMilliseconds} ms');

  print('');
  print('Loading weights from $weightsPath ...');
  final swLoad = Stopwatch()..start();
  final report = SamHFLoader.loadFile(
    imageEncoder: imgEnc,
    promptEncoder: promptEnc,
    maskDecoder: maskDec,
    path: weightsPath,
  );
  swLoad.stop();
  print('  $report  (${swLoad.elapsedMilliseconds} ms)');

  print('');
  print('Encoding image ...');
  final swE = Stopwatch()..start();
  final image = Tensor.fromFloat32List([
    1,
    3,
    targetSize,
    targetSize,
  ], imageInput);
  final imageEmb = imgEnc(image);
  swE.stop();
  print('  ${swE.elapsedMilliseconds} ms  → ${imageEmb.shape}');

  print('');
  print(
    'Encoding prompt (single positive point at raw ($rawPx, $rawPy) '
    '= letterboxed (${letterX.toStringAsFixed(1)}, '
    '${letterY.toStringAsFixed(1)}))',
  );
  final swP = Stopwatch()..start();
  final ptTensor = Tensor.fromList([1, 2], [letterX, letterY]);
  final sparse = promptEnc.encodeSparse(
    pointsXY: ptTensor,
    pointLabels: const [SamPointType.positive],
  );
  final dense = promptEnc.encodeDense(); // no mask prompt
  final imagePe = promptEnc.imagePositionEmbedding();
  swP.stop();
  print(
    '  ${swP.elapsedMilliseconds} ms  → sparse ${sparse.shape}, '
    'dense ${dense.shape}',
  );

  print('');
  print('Decoding masks ...');
  final swD = Stopwatch()..start();
  final decOut = maskDec(
    imageEmbedding: imageEmb,
    imagePe: imagePe,
    sparsePrompts: sparse,
    densePrompts: dense,
  );
  swD.stop();
  print(
    '  ${swD.elapsedMilliseconds} ms  → masks ${decOut.masks.shape}, '
    'IoU ${decOut.iouPredictions.shape}',
  );

  final selected = decOut.select(multimask: multimask);
  final chosen = _pickBestByIou(selected.masks, selected.iouPredictions);
  final chosenMask = chosen.mask.toList();
  print('');
  print('== IoU predictions ==');
  final iouVals = selected.iouPredictions.toList();
  for (int i = 0; i < iouVals.length; i++) {
    final marker = i == chosen.index ? ' ← selected' : '';
    print('  mask $i: iou=${iouVals[i].toStringAsFixed(3)}$marker');
  }

  // Upscale the 256×256 selected mask back to original image size and
  // save as PNG (thresholded at 0).
  final maskH = selected.masks.shape[1];
  final maskW = selected.masks.shape[2];
  print('');
  print('Writing overlay PNG to $outPath ...');
  _writeMaskOverlayPng(
    original: decoded,
    logits: chosenMask,
    maskH: maskH,
    maskW: maskW,
    letterboxScale: scale,
    outPath: outPath,
  );
  print('done.');
}

class _Chosen {
  final int index;
  final Tensor mask;
  _Chosen(this.index, this.mask);
}

_Chosen _pickBestByIou(Tensor masks, Tensor iou) {
  final v = iou.toList();
  var best = 0;
  var bestV = v[0];
  for (int i = 1; i < v.length; i++) {
    if (v[i] > bestV) {
      bestV = v[i];
      best = i;
    }
  }
  // Slice out the winning mask ([H, W]).
  final h = masks.shape[1];
  final w = masks.shape[2];
  final data = masks.toList();
  final out = List<double>.filled(h * w, 0);
  for (int i = 0; i < h * w; i++) {
    out[i] = data[best * h * w + i];
  }
  return _Chosen(best, Tensor.fromList([h, w], out));
}

void _writeMaskOverlayPng({
  required img_pkg.Image original,
  required List<double> logits,
  required int maskH,
  required int maskW,
  required double letterboxScale,
  required String outPath,
}) {
  // Upscale mask via nearest-neighbour to the letterboxed 1024×1024
  // space, then crop the used-area, then resize to original H/W.
  const targetSize = 1024;
  final letterMask = List<int>.filled(targetSize * targetSize, 0);
  for (int y = 0; y < targetSize; y++) {
    final sy = (y * maskH ~/ targetSize).clamp(0, maskH - 1);
    for (int x = 0; x < targetSize; x++) {
      final sx = (x * maskW ~/ targetSize).clamp(0, maskW - 1);
      letterMask[y * targetSize + x] = logits[sy * maskW + sx] > 0 ? 1 : 0;
    }
  }
  final resizedH = (original.height * letterboxScale).round();
  final resizedW = (original.width * letterboxScale).round();
  // Crop letterMask to (resizedW, resizedH) top-left.
  final cropped = List<int>.filled(resizedH * resizedW, 0);
  for (int y = 0; y < resizedH; y++) {
    for (int x = 0; x < resizedW; x++) {
      cropped[y * resizedW + x] = letterMask[y * targetSize + x];
    }
  }
  // Draw overlay on top of original.
  final overlay = img_pkg.Image.from(original);
  final origH = original.height;
  final origW = original.width;
  for (int y = 0; y < origH; y++) {
    final sy = (y * resizedH ~/ origH).clamp(0, resizedH - 1);
    for (int x = 0; x < origW; x++) {
      final sx = (x * resizedW ~/ origW).clamp(0, resizedW - 1);
      if (cropped[sy * resizedW + sx] == 1) {
        final px = overlay.getPixel(x, y);
        overlay.setPixel(
          x,
          y,
          img_pkg.ColorRgb8(
            ((px.r + 255) ~/ 2).clamp(0, 255),
            (px.g ~/ 2).toInt().clamp(0, 255),
            (px.b ~/ 2).toInt().clamp(0, 255),
          ),
        );
      }
    }
  }
  File(outPath).writeAsBytesSync(img_pkg.encodePng(overlay));
}
