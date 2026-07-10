/// Multimodal-transformer demo — synthetic audio + video classifier.
///
/// A tiny end-to-end example that trains a [MultiModalClassifier] on
/// a synthetic 3-class task where each class has a distinctive audio
/// *and* video pattern, with noise added:
///
///   0 — rising ramps  (audio + video)
///   1 — falling ramps (audio + video)
///   2 — constant mid  (audio + video)
///
/// The classifier fuses per-modality mean-pooled features via
/// concatenation and a Linear head. Trains in a few seconds on CPU
/// and hits 100% accuracy on held-out samples.
///
/// Run:
///
///     dart run bin/multi_modal_demo.dart          # CPU
///     dart run bin/multi_modal_demo.dart --gpu    # GPU
library;

import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

import 'shakespeare_util.dart' show paramScalarCount;

// Data shape.
const int _audioSeqLen = 20;
const int _audioFeatDim = 16;
const int _videoFrames = 12;
const int _videoFeatDim = 20;
const int _numClasses = 3;

// Model shape (same embedDim across modalities to keep the head small).
const int _embedDim = 24;
const int _numLayers = 2;
const int _numHeads = 4;

// Training.
const int _trainSamples = 60;
const int _evalSamples = 15;
const int _trainSteps = 200;
const int _logEvery = 20;
const double _lr = 3e-3;

List<double> _audioClip(int label, math.Random rng) {
  final out = List<double>.filled(_audioSeqLen * _audioFeatDim, 0.0);
  for (int t = 0; t < _audioSeqLen; t++) {
    for (int f = 0; f < _audioFeatDim; f++) {
      final phase = t / _audioSeqLen;
      double v;
      switch (label) {
        case 0:
          v = phase; // rising
          break;
        case 1:
          v = 1.0 - phase; // falling
          break;
        case 2:
          v = 0.5; // constant
          break;
        default:
          v = 0.0;
      }
      out[t * _audioFeatDim + f] = v + (rng.nextDouble() - 0.5) * 0.2;
    }
  }
  return out;
}

List<double> _videoClip(int label, math.Random rng) {
  final out = List<double>.filled(_videoFrames * _videoFeatDim, 0.0);
  for (int t = 0; t < _videoFrames; t++) {
    for (int f = 0; f < _videoFeatDim; f++) {
      final phase = t / _videoFrames;
      double v;
      switch (label) {
        case 0:
          v = phase;
          break;
        case 1:
          v = 1.0 - phase;
          break;
        case 2:
          v = 0.5;
          break;
        default:
          v = 0.0;
      }
      out[t * _videoFeatDim + f] = v + (rng.nextDouble() - 0.5) * 0.2;
    }
  }
  return out;
}

int _argmax(List<double> v) {
  var best = 0;
  var bestV = v[0];
  for (int i = 1; i < v.length; i++) {
    if (v[i] > bestV) {
      bestV = v[i];
      best = i;
    }
  }
  return best;
}

void main(List<String> args) {
  final device = args.contains('--gpu') ? Device.GPU : Device.CPU;
  print('=== multi_modal_demo : audio+video classifier (${device.name}) ===');
  print(
    'audio [$_audioSeqLen, $_audioFeatDim] '
    'video [$_videoFrames, $_videoFeatDim] '
    'classes=$_numClasses  embed=$_embedDim',
  );

  final clf = MultiModalClassifier(
    audio: AudioTransformer(
      featureDim: _audioFeatDim,
      maxSeqLen: _audioSeqLen,
      embedDim: _embedDim,
      numLayers: _numLayers,
      numHeads: _numHeads,
      device: device,
      seed: 0,
    ),
    video: VideoTransformer(
      frameFeatureDim: _videoFeatDim,
      maxFrames: _videoFrames,
      embedDim: _embedDim,
      numLayers: _numLayers,
      numHeads: _numHeads,
      device: device,
      seed: 1,
    ),
    numClasses: _numClasses,
    device: device,
    seed: 2,
  );
  final params = clf.parameters();
  print('params: ${paramScalarCount(params)} scalars');

  // Build a training set (60 samples, balanced across the 3 classes).
  final dataRng = math.Random(0);
  final trainA = <Tensor>[];
  final trainV = <Tensor>[];
  final trainY = <Tensor>[];
  for (int i = 0; i < _trainSamples; i++) {
    final label = i % _numClasses;
    trainA.add(
      Tensor.fromList(
        [_audioSeqLen, _audioFeatDim],
        _audioClip(label, dataRng),
        device: device,
      ),
    );
    trainV.add(
      Tensor.fromList(
        [_videoFrames, _videoFeatDim],
        _videoClip(label, dataRng),
        device: device,
      ),
    );
    trainY.add(Tensor.fromList([1], [label.toDouble()], device: device));
  }

  final opt = Adam(params, lr: _lr);
  final rng = math.Random(1);
  final sw = Stopwatch()..start();
  print('training $_trainSteps steps (lr=$_lr)...');
  for (int step = 1; step <= _trainSteps; step++) {
    opt.zeroGrad();
    final i = rng.nextInt(_trainSamples);
    final logits = clf(trainA[i], trainV[i]);
    final loss = logits.crossEntropy(trainY[i]).mean();
    loss.backward();
    clipGradNorm(params, 1.0);
    opt.step();
    if (step == 1 || step % _logEvery == 0 || step == _trainSteps) {
      final lossVal = loss.toList()[0];
      final ms = sw.elapsedMilliseconds / step;
      print(
        '  step ${step.toString().padLeft(4)}  '
        'loss=${lossVal.toStringAsFixed(4)}  '
        '(${ms.toStringAsFixed(1)} ms/step)',
      );
    }
  }
  sw.stop();

  // Evaluate on held-out samples.
  clf.eval();
  Tensor.noGrad(() {
    int trainCorrect = 0;
    for (int i = 0; i < _trainSamples; i++) {
      final logits = clf(trainA[i], trainV[i]).toList();
      if (_argmax(logits) == i % _numClasses) trainCorrect++;
    }
    print(
      '\ntrain accuracy: '
      '${(trainCorrect / _trainSamples * 100).toStringAsFixed(1)}%',
    );

    print('\neval samples (fresh noise, unseen):');
    final evalRng = math.Random(42);
    int evalCorrect = 0;
    for (int i = 0; i < _evalSamples; i++) {
      final label = evalRng.nextInt(_numClasses);
      final a = Tensor.fromList(
        [_audioSeqLen, _audioFeatDim],
        _audioClip(label, evalRng),
        device: device,
      );
      final v = Tensor.fromList(
        [_videoFrames, _videoFeatDim],
        _videoClip(label, evalRng),
        device: device,
      );
      final logits = clf(a, v).toList();
      final pred = _argmax(logits);
      if (pred == label) evalCorrect++;
      final tag = pred == label ? 'ok ' : 'MISS';
      print(
        '  [$tag] true=$label pred=$pred  '
        'logits=[${logits.map((x) => x.toStringAsFixed(2)).join(', ')}]',
      );
    }
    print(
      '\neval accuracy: '
      '${(evalCorrect / _evalSamples * 100).toStringAsFixed(1)}% '
      '($evalCorrect/$_evalSamples)',
    );
  });
}
