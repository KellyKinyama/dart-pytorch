/// Real face-recognition training on `ImageFolderDataset`.
///
/// End-to-end training pipeline using the new data loaders instead of
/// synthetic in-memory tensors. Concretely:
///
///  1. Programmatically writes a small "identity gallery" to a temp
///     directory (`/tmp/dp_faces_...`) with 4 identities × 8 PNG
///     images each. Every identity has a distinctive base visual
///     pattern (vertical / horizontal stripes, diagonals, and a
///     concentric-circle "logo") plus per-sample jitter (random
///     phase, hue rotation, Gaussian pixel noise). The result: a
///     folder that decodes and behaves exactly like a real face
///     dataset from disk.
///
///  2. Loads it with [ImageFolderDataset], deterministically split
///     into train / val at 25 %.
///
///  3. Trains a [ViTFaceEmbedding] with triplet loss
///     `relu(‖a-p‖² - ‖a-n‖² + margin)` — anchors and positives are
///     drawn from the same identity, negatives from a different one
///     via `ImageFolderDataset.sampleTriplet()`.
///
///  4. Reports before/after cosine-similarity gap on held-out
///     validation images (same-identity pairs vs different-identity
///     pairs). This is the metric-learning equivalent of "val
///     accuracy" — a well-trained embedding should cluster same-
///     identity images tightly and separate different identities.
///
/// Runs on CPU by default; pass `--gpu` for CUDA.
///
/// Run:
///
///     dart run bin/train_face_folder.dart          # CPU
///     dart run bin/train_face_folder.dart --gpu    # GPU
library;

import 'dart:io';
import 'dart:math' as math;

import 'package:image/image.dart' as img;

import 'package:dart_pytorch/dart_pytorch.dart';

import 'shakespeare_util.dart' show paramScalarCount;

const int _imageSize = 32;
const int _patchSize = 8;
const int _numChannels = 3;
const int _embedDim = 64;
const int _outputDim = 32;
const int _numLayers = 2;
const int _numHeads = 4;

const int _numIdentities = 4;
const int _samplesPerIdentity = 8;
const int _steps = 300;
const int _logEvery = 30;
const double _lr = 1e-3;
const double _margin = 0.4;

/// One image drawn for identity [id], sample [k]. Each identity has
/// a distinctive base "logo" plus per-sample randomization (phase
/// shift, hue rotation, Gaussian pixel noise) so the class has real
/// intra-class variation.
img.Image _drawIdentity(int id, int k, {int size = 32}) {
  final rng = math.Random(id * 1000 + k);
  final img_ = img.Image(width: size, height: size);
  final hueShift = rng.nextDouble() * 40 - 20; // ±20 hue degrees
  final phase = rng.nextInt(size);

  int clamp(num v) => v.clamp(0, 255).toInt();

  for (final p in img_) {
    final x = p.x, y = p.y;
    double r = 0, g = 0, b = 0;

    switch (id) {
      case 0: // Vertical red stripes.
        final s = ((x + phase) ~/ 4) % 2 == 0 ? 220 : 40;
        r = s.toDouble();
        g = 30;
        b = 30;
        break;
      case 1: // Horizontal green stripes.
        final s = ((y + phase) ~/ 4) % 2 == 0 ? 220 : 40;
        r = 30;
        g = s.toDouble();
        b = 30;
        break;
      case 2: // Diagonal blue pattern.
        final s = ((x + y + phase) ~/ 4) % 2 == 0 ? 220 : 40;
        r = 30;
        g = 30;
        b = s.toDouble();
        break;
      case 3: // Purple concentric circles.
        final cx = size / 2 + rng.nextDouble() * 4 - 2;
        final cy = size / 2 + rng.nextDouble() * 4 - 2;
        final d = math.sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
        final s = (d.toInt() ~/ 3) % 2 == 0 ? 220 : 40;
        r = s.toDouble();
        g = 30;
        b = s.toDouble();
        break;
    }

    // Apply hue shift + Gaussian pixel noise.
    r += hueShift;
    g -= hueShift * 0.5;
    b += hueShift * 0.5;
    final noise = 10 * (rng.nextDouble() - 0.5);
    p
      ..r = clamp(r + noise)
      ..g = clamp(g + noise)
      ..b = clamp(b + noise);
  }
  return img_;
}

/// Materialize the identity gallery on disk. Returns the root dir
/// (caller must delete when done).
Directory _buildFaceGallery() {
  final root = Directory.systemTemp.createTempSync('dp_faces_');
  for (int id = 0; id < _numIdentities; id++) {
    final dir = Directory('${root.path}/id_$id')..createSync();
    for (int k = 0; k < _samplesPerIdentity; k++) {
      final png = img.encodePng(_drawIdentity(id, k, size: _imageSize));
      File('${dir.path}/sample_$k.png').writeAsBytesSync(png);
    }
  }
  return root;
}

/// Cosine similarity of two `[1, outputDim]` L2-normalized embeddings.
double _cosine(Tensor a, Tensor b) {
  return Tensor.noGrad(() {
    final av = a.toList();
    final bv = b.toList();
    double dot = 0;
    for (int i = 0; i < av.length; i++) {
      dot += av[i] * bv[i];
    }
    return dot;
  });
}

/// Compute average same-identity and different-identity cosine
/// similarities on the val split. A well-trained embedding gives
/// `sameAvg >> diffAvg`.
({double sameAvg, double diffAvg, int sameN, int diffN}) _evalSeparation(
  ViTFaceEmbedding model,
  ImageFolderDataset val,
) {
  return Tensor.noGrad(() {
    final embeds = <int, List<Tensor>>{};
    for (int i = 0; i < val.length; i++) {
      final s = val[i];
      embeds.putIfAbsent(s.label, () => []).add(model(s.patches));
    }
    double sameSum = 0, diffSum = 0;
    int sameN = 0, diffN = 0;
    for (final entry in embeds.entries) {
      final list = entry.value;
      for (int i = 0; i < list.length; i++) {
        for (int j = i + 1; j < list.length; j++) {
          sameSum += _cosine(list[i], list[j]);
          sameN++;
        }
      }
    }
    final labels = embeds.keys.toList();
    for (int a = 0; a < labels.length; a++) {
      for (int b = a + 1; b < labels.length; b++) {
        for (final ea in embeds[labels[a]]!) {
          for (final eb in embeds[labels[b]]!) {
            diffSum += _cosine(ea, eb);
            diffN++;
          }
        }
      }
    }
    return (
      sameAvg: sameN == 0 ? 0.0 : sameSum / sameN,
      diffAvg: diffN == 0 ? 0.0 : diffSum / diffN,
      sameN: sameN,
      diffN: diffN,
    );
  });
}

void main(List<String> args) {
  final device = args.contains('--gpu') ? Device.GPU : Device.CPU;
  print('=== train_face_folder (${device.name}) ===');

  // 1. Build synthetic-but-real face gallery on disk.
  final root = _buildFaceGallery();
  try {
    print('gallery: ${root.path}');
    print('  $_numIdentities identities × $_samplesPerIdentity PNG samples');

    // 2. Load with ImageFolderDataset (real disk decode).
    final ds = ImageFolderDataset(
      root.path,
      imageSize: _imageSize,
      patchSize: _patchSize,
      valSplit: 0.25,
      device: device,
      seed: 0,
    );
    final val = ds.valSplit();
    print('classes:   ${ds.classes}');
    print('train/val: ${ds.numTrain} / ${ds.numVal}');
    print('per-item:  patches=[${ds.numPatches}, ${ds.patchPixels}]');

    // 3. Build model + optimizer.
    final model = ViTFaceEmbedding(
      imageSize: _imageSize,
      patchSize: _patchSize,
      numChannels: _numChannels,
      embedDim: _embedDim,
      outputDim: _outputDim,
      numLayers: _numLayers,
      numHeads: _numHeads,
      device: device,
      seed: 0,
    );
    final params = model.parameters();
    print('params:    ${paramScalarCount(params)} scalars');

    final opt = Adam(params, lr: _lr);
    final marginT = Tensor.fill([1], _margin, device: device);

    // 4. Baseline val separation.
    final before = _evalSeparation(model, val);
    print('\nBEFORE  same-avg=${before.sameAvg.toStringAsFixed(4)} '
        '(${before.sameN} pairs)  '
        'diff-avg=${before.diffAvg.toStringAsFixed(4)} '
        '(${before.diffN} pairs)  '
        'gap=${(before.sameAvg - before.diffAvg).toStringAsFixed(4)}');

    // 5. Train with triplet loss over triplets sampled from disk.
    print(
        '\ntraining $_steps steps (lr=$_lr, margin=$_margin, triplet loss)...');
    final sw = Stopwatch()..start();
    int trainedSteps = 0;
    double lossSum = 0;
    for (int step = 1; step <= _steps; step++) {
      opt.zeroGrad();
      final t = ds.sampleTriplet();
      final embA = model(t.anchor);
      final embP = model(t.positive);
      final embN = model(t.negative);

      final diffP = embA - embP;
      final diffN = embA - embN;
      final distP = (diffP * diffP).sum();
      final distN = (diffN * diffN).sum();
      final raw = (distP - distN) + marginT;
      final loss = raw.relu();

      final lossVal = loss.toList()[0];
      if (lossVal > 0) {
        loss.backward();
        clipGradNorm(params, 1.0);
        opt.step();
        trainedSteps++;
        lossSum += lossVal;
      }

      if (step == 1 || step % _logEvery == 0 || step == _steps) {
        final ms = sw.elapsedMilliseconds / step;
        final avg = trainedSteps == 0 ? 0.0 : lossSum / trainedSteps;
        print('  step ${step.toString().padLeft(4)}  '
            'triplet=${lossVal.toStringAsFixed(6)}  '
            'trained=${trainedSteps.toString().padLeft(3)}/$step  '
            'avg=${avg.toStringAsFixed(4)}  '
            '(${ms.toStringAsFixed(1)} ms/step)');
      }
    }
    sw.stop();

    // 6. Final val separation.
    model.eval();
    final after = _evalSeparation(model, val);
    print('\nAFTER   same-avg=${after.sameAvg.toStringAsFixed(4)}  '
        'diff-avg=${after.diffAvg.toStringAsFixed(4)}  '
        'gap=${(after.sameAvg - after.diffAvg).toStringAsFixed(4)}');

    final delta =
        (after.sameAvg - after.diffAvg) - (before.sameAvg - before.diffAvg);
    print('\nseparation gap change: '
        '${delta >= 0 ? '+' : ''}${delta.toStringAsFixed(4)}');
    if (delta > 0.05) {
      print('✅ val identities are more separated after training.');
    } else {
      print('⚠️  training did not clearly improve separation on val.');
    }
  } finally {
    root.deleteSync(recursive: true);
  }
}
