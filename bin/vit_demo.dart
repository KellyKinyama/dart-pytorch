/// Vision Transformer demo — synthetic 3-class pattern classification.
///
/// We generate a tiny synthetic dataset with three visually distinct
/// classes:
///
///   0 — **horizontal stripes** (bright + dark rows alternating)
///   1 — **vertical stripes**   (bright + dark columns alternating)
///   2 — **checkerboard**       (bright + dark tiles in a checker)
///
/// Each image is 16×16 grayscale (numChannels=1), patchified into 4×4
/// patches (→ 16 patches × 16 pixels each). A [ViTClassifier] with
/// two encoder layers is trained for a couple hundred Adam steps; we
/// then report training-set accuracy and predict labels for a handful
/// of held-out samples.
///
/// This is a "does it learn?" smoke demo, not a benchmark. It's small
/// enough to run comfortably on CPU in a few seconds and mirrors the
/// GPU path used by the transformer/GPT demos.
///
/// Run:
///
///     dart run bin/vit_demo.dart          # CPU
///     dart run bin/vit_demo.dart --gpu    # GPU
library;

import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

import 'shakespeare_util.dart' show paramScalarCount;

// Image / model config.
const int _imageSize = 16;
const int _patchSize = 4;
const int _numChannels = 1;
const int _embedDim = 32;
const int _numLayers = 2;
const int _numHeads = 4;
const int _numClasses = 3;

// Training config.
const int _trainSamples = 60; // 20 per class
const int _evalSamples = 15;
const int _trainSteps = 200;
const int _logEvery = 20;
const double _lr = 3e-3;

const int _numPatches =
    (_imageSize ~/ _patchSize) * (_imageSize ~/ _patchSize); // 16
const int _patchPixels = _patchSize * _patchSize * _numChannels; // 16

/// Build one 16×16 grayscale image for `label` with pixel noise from
/// `rng`, then flatten into row-major patches of shape
/// `[numPatches, patchPixels]`.
List<double> _makeImage(int label, math.Random rng) {
  final img = List<double>.filled(_imageSize * _imageSize, 0.0);
  const bright = 1.0;
  const dark = -1.0;

  for (int r = 0; r < _imageSize; r++) {
    for (int c = 0; c < _imageSize; c++) {
      double v;
      switch (label) {
        case 0: // horizontal stripes — bright when row bit set
          v = (r ~/ 2) % 2 == 0 ? bright : dark;
          break;
        case 1: // vertical stripes — bright when col bit set
          v = (c ~/ 2) % 2 == 0 ? bright : dark;
          break;
        case 2: // checkerboard on 2×2 tiles
          v = ((r ~/ 2) + (c ~/ 2)) % 2 == 0 ? bright : dark;
          break;
        default:
          v = 0.0;
      }
      img[r * _imageSize + c] = v + (rng.nextDouble() - 0.5) * 0.2; // noise
    }
  }

  // Patchify: for each patch (pr, pc), collect its pixels row-major.
  final patchesPerSide = _imageSize ~/ _patchSize;
  final out = List<double>.filled(_numPatches * _patchPixels, 0.0);
  int patchIdx = 0;
  for (int pr = 0; pr < patchesPerSide; pr++) {
    for (int pc = 0; pc < patchesPerSide; pc++) {
      int off = patchIdx * _patchPixels;
      for (int r = 0; r < _patchSize; r++) {
        for (int c = 0; c < _patchSize; c++) {
          out[off++] = img[(pr * _patchSize + r) * _imageSize
              + (pc * _patchSize + c)];
        }
      }
      patchIdx++;
    }
  }
  return out;
}

Tensor _imageTensor(int label, math.Random rng, Device device) {
  return Tensor.fromList(
    [_numPatches, _patchPixels],
    _makeImage(label, rng),
    device: device,
  );
}

Tensor _labelTensor(int label, Device device) =>
    Tensor.fromList([1], [label.toDouble()], device: device);

int _argmaxRow(List<double> logits) {
  var best = 0;
  var bestV = logits[0];
  for (int i = 1; i < logits.length; i++) {
    if (logits[i] > bestV) {
      bestV = logits[i];
      best = i;
    }
  }
  return best;
}

void main(List<String> args) {
  final device = args.contains('--gpu') ? Device.GPU : Device.CPU;
  print('=== vit_demo : ViT pattern classifier (${device.name}) ===');
  print('image ${_imageSize}x$_imageSize, patch $_patchSize, '
      'embed=$_embedDim, layers=$_numLayers, heads=$_numHeads, '
      'classes=$_numClasses');

  final clf = ViTClassifier(
    imageSize: _imageSize,
    patchSize: _patchSize,
    numChannels: _numChannels,
    embedDim: _embedDim,
    numClasses: _numClasses,
    numLayers: _numLayers,
    numHeads: _numHeads,
    device: device,
    seed: 0,
  );
  final params = clf.parameters();
  print('params: ${paramScalarCount(params)} scalars');

  // Build a small training set: 20 samples per class, random noise.
  final dataRng = math.Random(0);
  final trainX = <Tensor>[];
  final trainY = <Tensor>[];
  for (int i = 0; i < _trainSamples; i++) {
    final label = i % _numClasses;
    trainX.add(_imageTensor(label, dataRng, device));
    trainY.add(_labelTensor(label, device));
  }

  final opt = Adam(params, lr: _lr);
  final sampleRng = math.Random(1);
  final sw = Stopwatch()..start();
  print('training $_trainSteps steps (lr=$_lr)...');

  for (int step = 1; step <= _trainSteps; step++) {
    opt.zeroGrad();
    final i = sampleRng.nextInt(_trainSamples);
    final logits = clf(trainX[i]);
    final loss = logits.crossEntropy(trainY[i]).mean();
    loss.backward();
    clipGradNorm(params, 1.0);
    opt.step();
    if (step == 1 || step % _logEvery == 0 || step == _trainSteps) {
      final lossVal = loss.toList()[0];
      final ms = sw.elapsedMilliseconds / step;
      print('  step ${step.toString().padLeft(4)}  '
          'loss=${lossVal.toStringAsFixed(4)}  '
          '(${ms.toStringAsFixed(1)} ms/step)');
    }
  }
  sw.stop();

  // Training accuracy — noGrad since we're just reading forward.
  clf.eval();
  final trainAcc = Tensor.noGrad(() {
    int correct = 0;
    for (int i = 0; i < _trainSamples; i++) {
      final logits = clf(trainX[i]).toList();
      if (_argmaxRow(logits) == (i % _numClasses)) correct++;
    }
    return correct / _trainSamples;
  });
  print('\ntrain accuracy: ${(trainAcc * 100).toStringAsFixed(1)}% '
      '($_trainSamples samples)');

  // A few held-out samples with predictions.
  print('\neval samples (fresh noise, unseen):');
  final evalRng = math.Random(42);
  int evalCorrect = 0;
  Tensor.noGrad(() {
    for (int i = 0; i < _evalSamples; i++) {
      final label = evalRng.nextInt(_numClasses);
      final x = _imageTensor(label, evalRng, device);
      final logits = clf(x).toList();
      final pred = _argmaxRow(logits);
      if (pred == label) evalCorrect++;
      final tag = pred == label ? 'ok ' : 'MISS';
      print('  [$tag] true=$label pred=$pred  '
          'logits=[${logits.map((v) => v.toStringAsFixed(2)).join(', ')}]');
    }
  });
  print('\neval accuracy: '
      '${(evalCorrect / _evalSamples * 100).toStringAsFixed(1)}% '
      '($evalCorrect/$_evalSamples)');
}
