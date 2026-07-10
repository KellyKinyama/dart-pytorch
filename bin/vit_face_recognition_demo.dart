/// ViT face-recognition training with triplet loss.
///
/// Port of `dart_cuda/example/main_face_gpu.dart` with a realistic
/// multi-identity setup so the training signal is actually visible:
///
/// * Four synthetic "identities" — deterministic random patchified
///   images that stand in for base faces.
/// * Each training step samples one identity, adds independent
///   Gaussian jitter to build (anchor, positive), and picks a
///   different identity + jitter as the negative.
/// * `ViTFaceEmbedding` maps `[numPatches, patchPixels]` → an L2-
///   normalized `[1, outputDim]` vector; triplet loss is
///   `relu(||a - p||² - ||a - n||² + margin)`.
///
/// Before and after training we print the full cosine-similarity
/// matrix over the four clean base identities. A well-trained model
/// shows ~1.0 on the diagonal (each identity matches itself) and
/// clearly lower off-diagonal entries (different identities separate).
///
/// Runs on CPU by default; pass `--gpu` for CUDA.
///
/// Run:
///
///     dart run bin/vit_face_recognition_demo.dart          # CPU
///     dart run bin/vit_face_recognition_demo.dart --gpu    # GPU
library;

import 'dart:math' as math;

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
const int _epochs = 150;
const double _lr = 1e-3;
const double _margin = 0.4;
const double _jitter = 0.02;

const int _numPatches =
    (_imageSize ~/ _patchSize) * (_imageSize ~/ _patchSize); // 16
const int _patchPixels = _patchSize * _patchSize * _numChannels; // 192

/// Deterministic patchified "face" image for identity [id].
List<double> _baseFace(int id) {
  final rng = math.Random(1000 + id);
  return List<double>.generate(
    _numPatches * _patchPixels,
    (_) => (rng.nextDouble() - 0.5) * 0.5,
  );
}

/// Adds i.i.d. Gaussian noise with stddev [_jitter] to a base face.
Tensor _jitteredFace(List<double> base, math.Random rng, Device device) {
  final out = List<double>.filled(base.length, 0.0);
  for (int i = 0; i < base.length; i++) {
    // Box–Muller for cheap Gaussian noise.
    final u1 = rng.nextDouble().clamp(1e-9, 1.0);
    final u2 = rng.nextDouble();
    final g = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2);
    out[i] = base[i] + _jitter * g;
  }
  return Tensor.fromList(
    [_numPatches, _patchPixels],
    out,
    device: device,
  );
}

/// Cosine similarity matrix over [_numIdentities] embeddings. Assumes
/// each row is already L2-normalized.
List<List<double>> _simMatrix(
  ViTFaceEmbedding model,
  List<Tensor> cleanFaces,
) {
  return Tensor.noGrad(() {
    final embeds = <List<double>>[];
    for (final face in cleanFaces) {
      embeds.add(model(face).toList());
    }
    return List.generate(_numIdentities, (i) {
      return List.generate(_numIdentities, (j) {
        double dot = 0.0;
        for (int k = 0; k < _outputDim; k++) {
          dot += embeds[i][k] * embeds[j][k];
        }
        return dot;
      });
    });
  });
}

void _printSimMatrix(String label, List<List<double>> m) {
  print('$label cosine-similarity matrix:');
  final header = '        ' +
      List.generate(_numIdentities, (j) => '  id$j  ').join('  ');
  print(header);
  for (int i = 0; i < _numIdentities; i++) {
    final row = m[i]
        .map((v) => v.toStringAsFixed(3).padLeft(6))
        .join('  ');
    print('  id$i  $row');
  }
}

double _tripletMargin(List<List<double>> m) {
  // Average diagonal minus average off-diagonal — a scalar summary of
  // "how well identities separate".
  double diag = 0, off = 0;
  int nOff = 0;
  for (int i = 0; i < _numIdentities; i++) {
    diag += m[i][i];
    for (int j = 0; j < _numIdentities; j++) {
      if (i == j) continue;
      off += m[i][j];
      nOff++;
    }
  }
  return diag / _numIdentities - off / nOff;
}

void main(List<String> args) {
  final device = args.contains('--gpu') ? Device.GPU : Device.CPU;
  print('=== vit_face_recognition_demo (${device.name}) ===');
  print('image ${_imageSize}x$_imageSize, patch $_patchSize, embed=$_embedDim, '
      'outputDim=$_outputDim, identities=$_numIdentities');

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
  print('params: ${paramScalarCount(params)} scalars');

  final opt = Adam(params, lr: _lr);

  // Build the identity gallery once — host-side data, transferred to
  // the target device on demand via `_jitteredFace` / clean `Tensor`.
  final bases = List.generate(_numIdentities, _baseFace);
  final cleanFaces = [
    for (final b in bases)
      Tensor.fromList([_numPatches, _patchPixels], b, device: device),
  ];

  final before = _simMatrix(model, cleanFaces);
  print('');
  _printSimMatrix('BEFORE', before);
  print('  separation (diag - off-diag mean): '
      '${_tripletMargin(before).toStringAsFixed(4)}');

  print('\ntraining $_epochs epochs '
      '(lr=$_lr, margin=$_margin, jitter=$_jitter)...');
  final rng = math.Random(0);
  final sw = Stopwatch()..start();
  final marginT = Tensor.fill([1], _margin, device: device);

  for (int epoch = 0; epoch <= _epochs; epoch++) {
    opt.zeroGrad();

    // Sample identity for anchor + positive; different identity for
    // the negative.
    final iA = rng.nextInt(_numIdentities);
    int iN;
    do {
      iN = rng.nextInt(_numIdentities);
    } while (iN == iA);

    final anchor = _jitteredFace(bases[iA], rng, device);
    final positive = _jitteredFace(bases[iA], rng, device);
    final negative = _jitteredFace(bases[iN], rng, device);

    final embA = model(anchor);
    final embP = model(positive);
    final embN = model(negative);

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
    }

    if (epoch % 15 == 0 || epoch == _epochs) {
      final ms = sw.elapsedMilliseconds / (epoch + 1);
      print('  epoch ${epoch.toString().padLeft(4)}  '
          'triplet=${lossVal.toStringAsFixed(6)}  '
          '(a=$iA, n=$iN, ${ms.toStringAsFixed(1)} ms/step)');
    }
  }
  sw.stop();

  model.eval();
  final after = _simMatrix(model, cleanFaces);
  print('');
  _printSimMatrix('AFTER ', after);
  print('  separation (diag - off-diag mean): '
      '${_tripletMargin(after).toStringAsFixed(4)}');

  final delta = _tripletMargin(after) - _tripletMargin(before);
  print('\nseparation change: '
      '${delta >= 0 ? '+' : ''}${delta.toStringAsFixed(4)}');
  if (delta > 0.05) {
    print('✅ identities are more separated after training.');
  } else {
    print('⚠️  training did not clearly improve separation.');
  }
}
