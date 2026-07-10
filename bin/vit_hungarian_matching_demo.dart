/// DETR-style ViT object detector with Hungarian query→GT matching.
///
/// Port of `dart_cuda/example/bin/example_bipartide_matching3.dart` to
/// dart-pytorch. The classic "predict `numQueries` boxes and match
/// them to a variable set of GT objects" recipe:
///
///   1. Forward: `[logits, boxes]` from [ViTObjectDetector].
///   2. Build an `numQueries × numQueries` L1 cost matrix between
///      predicted and GT boxes. Missing GT columns get a huge padding
///      cost so unmatched queries fall to background.
///   3. [HungarianAlgorithm] returns `assign[q] = gtIndex`.
///   4. Build a *masked* target: matched queries carry the real class
///      + box (`boxMask=1`); unmatched queries carry class
///      `numClasses` (background) and `boxMask=0`.
///   5. Loss = `classWeight * crossEntropy(logits, alignedClasses) +
///              boxWeight   * mean(|pred - gt| * boxMask)`.
///
/// Runs on CPU by default; pass `--gpu` for CUDA.
///
/// Run:
///
///     dart run bin/vit_hungarian_matching_demo.dart          # CPU
///     dart run bin/vit_hungarian_matching_demo.dart --gpu    # GPU
library;

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
const int _epochs = 250;
const int _logEvery = 50;

const double _classWeight = 1.0;
const double _boxWeight = 10.0;

// Big integer sentinel used to pad the cost matrix when there are
// fewer GT objects than queries. Must be much larger than any real
// L1 cost * scale (see _scale) so Hungarian never picks a padded col.
const int _padCost = 1000000;
const double _scale = 10000.0;

const int _numPatches =
    (_imageSize ~/ _patchSize) * (_imageSize ~/ _patchSize); // 16
const int _patchPixels = _patchSize * _patchSize * _numChannels; // 192

class _Gt {
  final List<double> bbox; // len 4
  final int classId;
  const _Gt(this.bbox, this.classId);
}

// L1 cost between predicted and GT boxes, on host. We only need the
// values (no autograd) — the loss uses the raw tensor ops afterwards.
List<List<int>> _buildCostMatrix(
  List<double> predBoxes,
  List<_Gt> gts,
  int numQueries,
) {
  final matrix = List.generate(
    numQueries,
    (_) => List<int>.filled(numQueries, _padCost),
  );
  for (int q = 0; q < numQueries; q++) {
    for (int g = 0; g < gts.length; g++) {
      double c = 0.0;
      for (int k = 0; k < 4; k++) {
        c += (predBoxes[q * 4 + k] - gts[g].bbox[k]).abs();
      }
      matrix[q][g] = (c * _scale).toInt();
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

void _printDetections(
  List<double> pred,
  List<double> target,
  List<int> classes,
  List<double> mask,
) {
  print(
    'Query | Class | Pred BBox                  | Target BBox                | Error',
  );
  print('-' * 90);
  for (int i = 0; i < _numQueries; i++) {
    final p = pred
        .sublist(i * 4, i * 4 + 4)
        .map((v) => v.toStringAsFixed(2))
        .toList();
    final t = target
        .sublist(i * 4, i * 4 + 4)
        .map((v) => v.toStringAsFixed(2))
        .toList();
    double err = 0;
    for (int j = 0; j < 4; j++) {
      err += (pred[i * 4 + j] - target[i * 4 + j]).abs();
    }
    final label = classes[i] == _numClasses ? 'BG   ' : 'ID:${classes[i]} ';
    final maskTag = mask[i] > 0.5 ? '*' : ' ';
    print(
      '  #$i$maskTag | $label | $p | $t | ${err.toStringAsFixed(4)}',
    );
  }
}

void main(List<String> args) {
  final device = args.contains('--gpu') ? Device.GPU : Device.CPU;
  print('=== vit_hungarian_matching_demo (${device.name}) ===');
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

  // Deterministic synthetic patchified image.
  final imgVals = List<double>.generate(
    _numPatches * _patchPixels,
    (i) => ((i * 31) % 97) / 97.0 - 0.5,
  );
  final xInput = Tensor.fromList(
    [_numPatches, _patchPixels],
    imgVals,
    device: device,
  );

  // Only 2 real GT objects; the 3rd query slot should learn "background".
  final gts = <_Gt>[
    _Gt([0.10, 0.10, 0.20, 0.20], 1),
    _Gt([0.50, 0.50, 0.30, 0.30], 2),
  ];

  print('training $_epochs epochs (lr=$_lr, '
      'classW=$_classWeight, boxW=$_boxWeight)...');
  final sw = Stopwatch()..start();

  for (int epoch = 0; epoch <= _epochs; epoch++) {
    opt.zeroGrad();
    final preds = detector(xInput);
    final logits = preds['logits']!; // [numQueries, numClasses+1]
    final boxes = preds['boxes']!; // [numQueries, 4]

    // 1) Hungarian matching on the current predictions (no autograd).
    final predBoxValues = boxes.toList();
    final costMatrix = _buildCostMatrix(predBoxValues, gts, _numQueries);
    final assign = HungarianAlgorithm(costMatrix).getAssignment();

    // 2) Build masked, aligned targets.
    final alignedClasses = List<double>.filled(_numQueries, _numClasses.toDouble());
    final alignedBoxes = List<double>.filled(_numQueries * 4, 0.0);
    final boxMaskRaw = List<double>.filled(_numQueries * 4, 0.0);
    for (int q = 0; q < _numQueries; q++) {
      final g = assign[q];
      if (g >= 0 && g < gts.length) {
        alignedClasses[q] = gts[g].classId.toDouble();
        for (int k = 0; k < 4; k++) {
          alignedBoxes[q * 4 + k] = gts[g].bbox[k];
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

    // 3) Masked + weighted loss.
    final classLoss = logits.crossEntropy(gtClasses).mean();
    final rawDiff = (boxes - gtBoxes).abs();
    final maskedDiff = rawDiff * boxMask;
    final boxLoss = maskedDiff.mean();
    final totalLoss = (classLoss * _classWeight) + (boxLoss * _boxWeight);

    totalLoss.backward();
    clipGradNorm(params, 1.0);
    opt.step();

    if (epoch % _logEvery == 0 || epoch == _epochs) {
      final lossVal = totalLoss.toList()[0];
      final ms = sw.elapsedMilliseconds / (epoch + 1);
      print(
        '\n--- epoch ${epoch.toString().padLeft(4)}  '
        'loss=${lossVal.toStringAsFixed(6)}  '
        '(${ms.toStringAsFixed(1)} ms/step) ---',
      );
      final classIdxs = alignedClasses.map((v) => v.toInt()).toList();
      _printDetections(predBoxValues, alignedBoxes, classIdxs, boxMaskRaw);
    }
  }
  sw.stop();

  detector.eval();
  Tensor.noGrad(() {
    final preds = detector(xInput);
    final logits = preds['logits']!.toList();
    print('\nfinal per-query class predictions:');
    for (int q = 0; q < _numQueries; q++) {
      final row = logits.sublist(
          q * (_numClasses + 1), (q + 1) * (_numClasses + 1));
      final pred = _argmax(row);
      final tag = pred == _numClasses ? 'BG' : 'c$pred';
      print('  q$q -> $tag  '
          '(logits=[${row.map((v) => v.toStringAsFixed(2)).join(', ')}])');
    }
  });

  print('\n✅ Training complete.');
}
