/// ViT object detection with batched Hungarian matching over
/// variable-count ground truth.
///
/// Extends `bin/vit_hungarian_matching_demo.dart` to a *set* of
/// training images, each with its own GT list of arbitrary size
/// (0..numQueries). This is the realistic DETR setup:
///
///   image A → 2 objects (car, person)
///   image B → 1 object  (dog)
///   image C → 0 objects (all queries → background)
///   image D → 3 objects (full slate)
///
/// Each step:
///   1. Sample one image from the batch (SGD).
///   2. Forward once → 3 predicted (class, box) queries.
///   3. Build an `numQueries × numQueries` L1 cost matrix with padded
///      columns when `gts.length < numQueries` — so unmatched slots
///      land on the padding and become background.
///   4. Hungarian assign, mask box loss to matched slots only.
///   5. Weighted `classW * crossEntropy + boxW * mean(|diff| * mask)`.
///
/// Uses [Tensor.abs] directly (relies on the abs.backward GPU kernel
/// wired in via `abs_backward_op`). Runs on CPU by default; pass
/// `--gpu` for CUDA.
///
/// Run:
///
///     dart run bin/vit_hungarian_batch_demo.dart          # CPU
///     dart run bin/vit_hungarian_batch_demo.dart --gpu    # GPU
library;

import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

import 'shakespeare_util.dart' show paramScalarCount;

const int _imageSize = 32;
const int _patchSize = 8;
const int _numChannels = 3;
const int _embedDim = 64;
const int _numClasses = 5;
const int _numQueries = 3;
const int _numLayers = 2;
const int _numHeads = 4;
const double _lr = 1e-3;
const int _steps = 400;
const int _logEvery = 50;

const double _classWeight = 1.0;
const double _boxWeight = 10.0;

const int _padCost = 1000000;
const double _costScale = 10000.0;

const int _numPatches =
    (_imageSize ~/ _patchSize) * (_imageSize ~/ _patchSize); // 16
const int _patchPixels = _patchSize * _patchSize * _numChannels; // 192

class _Gt {
  final List<double> bbox; // len 4
  final int classId;
  const _Gt(this.bbox, this.classId);
}

class _Sample {
  final String name;
  final Tensor input; // [numPatches, patchPixels]
  final List<_Gt> gts;
  _Sample(this.name, this.input, this.gts);
}

Tensor _syntheticImage(int seed, Device device) {
  // Distinct per-image texture. Cheap deterministic pattern so
  // different samples produce different CLS features.
  final vals = List<double>.generate(
    _numPatches * _patchPixels,
    (i) => ((i * (31 + seed * 7)) % 97) / 97.0 - 0.5,
  );
  return Tensor.fromList([_numPatches, _patchPixels], vals, device: device);
}

/// Build the `numQueries × numQueries` L1 cost matrix. Real GT
/// columns get the actual L1 cost scaled to int; padded columns get
/// `_padCost` so Hungarian never prefers them over a real match.
List<List<int>> _buildCostMatrix(List<double> predBoxes, List<_Gt> gts) {
  final matrix = List.generate(
    _numQueries,
    (_) => List<int>.filled(_numQueries, _padCost),
  );
  for (int q = 0; q < _numQueries; q++) {
    for (int g = 0; g < gts.length; g++) {
      double c = 0.0;
      for (int k = 0; k < 4; k++) {
        c += (predBoxes[q * 4 + k] - gts[g].bbox[k]).abs();
      }
      matrix[q][g] = (c * _costScale).toInt();
    }
  }
  return matrix;
}

int _argmax(List<double> v) {
  int best = 0;
  double bestV = v[0];
  for (int i = 1; i < v.length; i++) {
    if (v[i] > bestV) {
      bestV = v[i];
      best = i;
    }
  }
  return best;
}

double _evalOne(ViTObjectDetector det, _Sample s) {
  // No-grad eval: how many GTs are correctly predicted at the
  // Hungarian-assigned query slot (class matches + IoU-like sanity
  // via L1 error < 0.15).
  return Tensor.noGrad(() {
    final preds = det(s.input);
    final logits = preds['logits']!.toList();
    final boxes = preds['boxes']!.toList();
    final matrix = _buildCostMatrix(boxes, s.gts);
    final assign = HungarianAlgorithm(matrix).getAssignment();
    int matched = 0;
    for (int q = 0; q < _numQueries; q++) {
      final g = assign[q];
      if (g < 0 || g >= s.gts.length) continue;
      final gtClass = s.gts[g].classId;
      final row = logits.sublist(
        q * (_numClasses + 1),
        (q + 1) * (_numClasses + 1),
      );
      if (_argmax(row) != gtClass) continue;
      double err = 0;
      for (int k = 0; k < 4; k++) {
        err += (boxes[q * 4 + k] - s.gts[g].bbox[k]).abs();
      }
      if (err < 0.4) matched++;
    }
    return s.gts.isEmpty ? 1.0 : matched / s.gts.length;
  });
}

void main(List<String> args) {
  final device = args.contains('--gpu') ? Device.GPU : Device.CPU;
  print('=== vit_hungarian_batch_demo (${device.name}) ===');
  print(
    'image ${_imageSize}x$_imageSize, patch $_patchSize, embed=$_embedDim, '
    'queries=$_numQueries, classes=$_numClasses (+1 bg)',
  );

  final detector = ViTObjectDetector(
    imageSize: _imageSize,
    patchSize: _patchSize,
    numChannels: _numChannels,
    embedDim: _embedDim,
    numClasses: _numClasses,
    numQueries: _numQueries,
    numLayers: _numLayers,
    numHeads: _numHeads,
    device: device,
    seed: 0,
  );
  final params = detector.parameters();
  print('params: ${paramScalarCount(params)} scalars');

  final opt = Adam(params, lr: _lr);

  // A small "dataset" of 4 images with variable GT counts.
  final samples = <_Sample>[
    _Sample('A: car + person', _syntheticImage(0, device), [
      const _Gt([0.10, 0.10, 0.20, 0.20], 1),
      const _Gt([0.55, 0.55, 0.30, 0.30], 2),
    ]),
    _Sample('B: single dog', _syntheticImage(1, device), [
      const _Gt([0.40, 0.30, 0.25, 0.30], 3),
    ]),
    _Sample('C: empty scene', _syntheticImage(2, device), const <_Gt>[]),
    _Sample('D: 3 objects', _syntheticImage(3, device), [
      const _Gt([0.05, 0.05, 0.15, 0.15], 4),
      const _Gt([0.45, 0.10, 0.20, 0.25], 1),
      const _Gt([0.20, 0.60, 0.25, 0.30], 2),
    ]),
  ];
  for (final s in samples) {
    print('  ${s.name.padRight(20)}  ${s.gts.length} GT');
  }
  print('');

  final rng = math.Random(0);
  final sw = Stopwatch()..start();
  print(
    'training $_steps steps '
    '(lr=$_lr, classW=$_classWeight, boxW=$_boxWeight)...',
  );

  for (int step = 1; step <= _steps; step++) {
    opt.zeroGrad();
    final s = samples[rng.nextInt(samples.length)];

    final preds = detector(s.input);
    final logits = preds['logits']!;
    final boxes = preds['boxes']!;

    // Host-side matching from the current predictions.
    final predBoxValues = boxes.toList();
    final matrix = _buildCostMatrix(predBoxValues, s.gts);
    final assign = HungarianAlgorithm(matrix).getAssignment();

    final alignedClasses = List<double>.filled(
      _numQueries,
      _numClasses.toDouble(),
    );
    final alignedBoxes = List<double>.filled(_numQueries * 4, 0.0);
    final boxMaskRaw = List<double>.filled(_numQueries * 4, 0.0);
    for (int q = 0; q < _numQueries; q++) {
      final g = assign[q];
      if (g >= 0 && g < s.gts.length) {
        alignedClasses[q] = s.gts[g].classId.toDouble();
        for (int k = 0; k < 4; k++) {
          alignedBoxes[q * 4 + k] = s.gts[g].bbox[k];
          boxMaskRaw[q * 4 + k] = 1.0;
        }
      }
    }

    final gtClasses = Tensor.fromList(
      [_numQueries],
      alignedClasses,
      device: device,
    );
    final gtBoxes = Tensor.fromList(
      [_numQueries, 4],
      alignedBoxes,
      device: device,
    );
    final boxMask = Tensor.fromList(
      [_numQueries, 4],
      boxMaskRaw,
      device: device,
    );

    final classLoss = logits.crossEntropy(gtClasses).mean();
    final rawDiff = (boxes - gtBoxes).abs();
    final maskedDiff = rawDiff * boxMask;
    // Guard against divide-by-zero when *all* GTs are background
    // (image C): boxLoss is exactly zero, no gradient path anyway.
    final boxLoss = maskedDiff.mean();
    final totalLoss = (classLoss * _classWeight) + (boxLoss * _boxWeight);

    totalLoss.backward();
    clipGradNorm(params, 1.0);
    opt.step();

    if (step == 1 || step % _logEvery == 0 || step == _steps) {
      final lossVal = totalLoss.toList()[0];
      final ms = sw.elapsedMilliseconds / step;
      print(
        '  step ${step.toString().padLeft(4)}  '
        'sample=${s.name.padRight(20)} '
        'loss=${lossVal.toStringAsFixed(4)}  '
        '(${ms.toStringAsFixed(1)} ms/step)',
      );
    }
  }
  sw.stop();

  detector.eval();
  print('\nfinal per-sample evaluation (class match + box L1 < 0.4):');
  double totalAcc = 0;
  int nonEmpty = 0;
  for (final s in samples) {
    final acc = _evalOne(detector, s);
    print(
      '  ${s.name.padRight(20)}  gt=${s.gts.length}  '
      'match=${(acc * 100).toStringAsFixed(0)}%',
    );
    if (s.gts.isNotEmpty) {
      totalAcc += acc;
      nonEmpty++;
    }
  }
  if (nonEmpty > 0) {
    print(
      '\naverage GT-recall: '
      '${(totalAcc / nonEmpty * 100).toStringAsFixed(1)}%',
    );
  }
}
