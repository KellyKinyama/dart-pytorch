/// ViT-based multi-object detection (chain-linked variant).
///
/// Minimal port of `dart_cuda/example/bin/example_object_detection.dart`
/// to dart-pytorch. Uses [ViTObjectDetector] with `numQueries=3` slots
/// over a synthetic 32×32 RGB patchified input and trains against a
/// **fixed-order** target — no bipartite matching. The three GT slots
/// carry:
///
///     class 1 @ box (0.10, 0.10, 0.20, 0.20)
///     class 2 @ box (0.50, 0.50, 0.30, 0.30)
///     background  @ zero box
///
/// Loss is classic `crossEntropy + 0.25 * (pred - gt)^2`. Runs on
/// CPU by default; pass `--gpu` for CUDA.
///
/// Run:
///
///     dart run bin/vit_object_detection_demo.dart          # CPU
///     dart run bin/vit_object_detection_demo.dart --gpu    # GPU
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
const int _epochs = 200;
const int _logEvery = 20;

const int _numPatches =
    (_imageSize ~/ _patchSize) * (_imageSize ~/ _patchSize); // 16
const int _patchPixels = _patchSize * _patchSize * _numChannels; // 192

void main(List<String> args) {
  final device = args.contains('--gpu') ? Device.GPU : Device.CPU;
  print('=== vit_object_detection_demo (${device.name}) ===');
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

  // Deterministic "image" input (numPatches, patchPixels). We fill
  // with a smooth ramp so the pattern is fixed across steps.
  final imgVals = List<double>.generate(
    _numPatches * _patchPixels,
    (i) => ((i * 37) % 101) / 101.0 - 0.5,
  );
  final xInput = Tensor.fromList(
    [_numPatches, _patchPixels],
    imgVals,
    device: device,
  );

  // Ground truth (fixed-order).
  final gtClasses = Tensor.fromList(
    [_numQueries],
    [1.0, 2.0, _numClasses.toDouble()], // last slot = background
    device: device,
  );
  final gtBoxes = Tensor.fromList(
    [_numQueries, 4],
    [0.10, 0.10, 0.20, 0.20, 0.50, 0.50, 0.30, 0.30, 0.00, 0.00, 0.00, 0.00],
    device: device,
  );

  final boxScale = Tensor.fill([_numQueries, 4], 0.25, device: device);

  print('training $_epochs epochs (lr=$_lr)...');
  final sw = Stopwatch()..start();
  for (int epoch = 0; epoch <= _epochs; epoch++) {
    opt.zeroGrad();
    final preds = detector(xInput);
    final logits = preds['logits']!; // [numQueries, numClasses+1]
    final boxes = preds['boxes']!; // [numQueries, 4]

    final classLoss = logits.crossEntropy(gtClasses).mean();
    final diff = boxes - gtBoxes;
    final boxLoss = ((diff * diff) * boxScale).mean();
    final totalLoss = classLoss + boxLoss;

    totalLoss.backward();
    clipGradNorm(params, 1.0);
    opt.step();

    if (epoch % _logEvery == 0 || epoch == _epochs) {
      final lossVal = totalLoss.toList()[0];
      final ms = sw.elapsedMilliseconds / (epoch + 1);
      print(
        '  epoch ${epoch.toString().padLeft(4)}  '
        'loss=${lossVal.toStringAsFixed(6)}  '
        '(${ms.toStringAsFixed(1)} ms/step)',
      );
    }
  }
  sw.stop();

  // Final read-out.
  detector.eval();
  Tensor.noGrad(() {
    final preds = detector(xInput);
    final boxes = preds['boxes']!.toList();
    final logits = preds['logits']!.toList();
    print('\nfinal predictions:');
    for (int q = 0; q < _numQueries; q++) {
      final row = logits.sublist(
        q * (_numClasses + 1),
        (q + 1) * (_numClasses + 1),
      );
      final pred = _argmax(row);
      final gt = q < _numQueries ? gtClasses.toList()[q].toInt() : _numClasses;
      final b = boxes.sublist(q * 4, q * 4 + 4);
      final tag = pred == _numClasses ? 'bg' : 'c$pred';
      print(
        '  q$q  gt=c$gt  pred=$tag  '
        'box=[${b.map((v) => v.toStringAsFixed(3)).join(', ')}]',
      );
    }
  });

  print('\n✅ Training complete.');
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
