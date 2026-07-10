/// Real face-recognition training on `ImageFolderDataset`.
///
/// End-to-end training pipeline using the new data loaders instead of
/// synthetic in-memory tensors. Concretely:
///
///  1. Programmatically writes a small "identity gallery" to a
///     stable directory (`./faces_gallery/` by default, relative to
///     the current working directory) with 4 identities × 8 PNG
///     images each. Every identity has a distinctive base visual
///     pattern (vertical / horizontal stripes, diagonals, and a
///     concentric-circle "logo") plus per-sample jitter (random
///     phase, hue rotation, Gaussian pixel noise). The result: a
///     folder that decodes and behaves exactly like a real face
///     dataset from disk — and you can open the PNGs in any image
///     viewer to inspect the samples. Pass `--tmp` to write to a
///     `/tmp/dp_faces_...` directory that is deleted on exit.
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
///     dart run bin/train_face_folder.dart               # CPU, ./faces_gallery/
///     dart run bin/train_face_folder.dart --gpu         # GPU
///     dart run bin/train_face_folder.dart --tmp         # ephemeral /tmp dir
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

/// One 32×32 cartoon face drawn for identity [id], sample [k]. Each
/// identity has a fixed "look" (skin tone, hair color/style, eye
/// color, glasses, mouth style) and per-sample jitter (small head
/// shift, mouth size, skin brightness, subtle pixel noise) so the
/// class has real intra-class variation without breaking recognizable
/// facial structure.
///
///  * id 0 — "Alice":  fair skin, short red hair, blue eyes,  big smile
///  * id 1 — "Bob":    tan skin,  black spiky hair, brown eyes, small smile
///  * id 2 — "Carol":  pale skin, long blonde hair, green eyes, med smile
///  * id 3 — "Dave":   dark skin, curly dark hair,  brown eyes, glasses
img.Image _drawIdentity(int id, int k, {int size = 32}) {
  final rng = math.Random(id * 1000 + k);
  final img_ = img.Image(width: size, height: size);

  int clamp(num v) => v.clamp(0, 255).toInt();

  // Per-identity palette + style knobs.
  late List<int> bg, skin, hair, eye, mouth;
  int hairStyle; // 0=cap, 1=spiky, 2=long, 3=curly
  bool glasses;
  int mouthCurve; // 0=flat, 1=small smile, 2=big smile
  switch (id) {
    case 0: // Alice
      bg = [225, 232, 245];
      skin = [255, 220, 190];
      hair = [180, 60, 40];
      eye = [40, 90, 200];
      mouth = [200, 50, 60];
      hairStyle = 0;
      glasses = false;
      mouthCurve = 2;
      break;
    case 1: // Bob
      bg = [215, 222, 212];
      skin = [220, 175, 130];
      hair = [25, 22, 20];
      eye = [90, 55, 35];
      mouth = [150, 65, 60];
      hairStyle = 1;
      glasses = false;
      mouthCurve = 1;
      break;
    case 2: // Carol
      bg = [235, 220, 240];
      skin = [250, 215, 190];
      hair = [230, 190, 90];
      eye = [70, 160, 100];
      mouth = [220, 90, 105];
      hairStyle = 2;
      glasses = false;
      mouthCurve = 2;
      break;
    case 3: // Dave
      bg = [235, 230, 218];
      skin = [140, 95, 70];
      hair = [45, 32, 28];
      eye = [60, 45, 35];
      mouth = [150, 80, 70];
      hairStyle = 3;
      glasses = true;
      mouthCurve = 1;
      break;
    default:
      throw StateError('unreachable');
  }

  // Per-sample jitter.
  final dx = rng.nextInt(3) - 1;
  final dy = rng.nextInt(3) - 1;
  final skinShift = rng.nextInt(21) - 10;
  final mouthWidth = 4 + rng.nextInt(3); // 4, 5, or 6
  final cx = size ~/ 2 + dx;
  final cy = size ~/ 2 + 1 + dy;
  const headR = 11;

  void setPx(int x, int y, List<int> c) {
    if (x < 0 || y < 0 || x >= size || y >= size) return;
    final p = img_.getPixel(x, y);
    p
      ..r = clamp(c[0])
      ..g = clamp(c[1])
      ..b = clamp(c[2]);
  }

  // 1. Background fill.
  for (int y = 0; y < size; y++) {
    for (int x = 0; x < size; x++) {
      setPx(x, y, bg);
    }
  }

  // 2. Head (filled skin-tone circle).
  final skinLit = [
    skin[0] + skinShift,
    skin[1] + skinShift,
    skin[2] + skinShift,
  ];
  for (int y = 0; y < size; y++) {
    for (int x = 0; x < size; x++) {
      final d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
      if (d2 <= headR * headR) setPx(x, y, skinLit);
    }
  }

  // 3. Hair — style-dependent.
  switch (hairStyle) {
    case 0: // Cap: top ~40 % of head is hair.
      for (int y = cy - headR - 1; y <= cy - 3; y++) {
        for (int x = cx - headR - 1; x <= cx + headR + 1; x++) {
          final d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
          if (d2 <= (headR + 1) * (headR + 1) && d2 >= (headR - 1) * (headR - 1) ||
              (d2 <= headR * headR && y <= cy - 5)) {
            setPx(x, y, hair);
          }
        }
      }
      break;
    case 1: // Spiky short cap + a few spikes above.
      for (int y = cy - headR; y <= cy - 4; y++) {
        for (int x = cx - headR + 1; x <= cx + headR - 1; x++) {
          final d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
          if (d2 <= (headR - 1) * (headR - 1)) setPx(x, y, hair);
        }
      }
      for (int i = -2; i <= 2; i++) {
        final sx = cx + i * 3;
        setPx(sx, cy - headR - 2, hair);
        setPx(sx, cy - headR - 1, hair);
      }
      break;
    case 2: // Long: top cap + side "curtains" down past the head.
      for (int y = cy - headR - 1; y <= cy - 3; y++) {
        for (int x = cx - headR - 1; x <= cx + headR + 1; x++) {
          final d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
          if (d2 <= (headR + 1) * (headR + 1) && y <= cy - 4) {
            setPx(x, y, hair);
          }
        }
      }
      for (int y = cy - 4; y < math.min(size, cy + headR + 3); y++) {
        for (final side in [-1, 1]) {
          setPx(cx + side * (headR + 1), y, hair);
          setPx(cx + side * headR, y, hair);
        }
      }
      break;
    case 3: // Curly: dotted pattern on top half.
      for (int y = cy - headR - 1; y <= cy - 3; y++) {
        for (int x = cx - headR; x <= cx + headR; x++) {
          final d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
          if (d2 <= headR * headR && ((x + y) & 1) == 0) {
            setPx(x, y, hair);
          }
        }
      }
      break;
  }

  // 4. Eyes — 2×2 white sclera + 1px colored pupil.
  final eyeY = cy - 2;
  for (final ex in [cx - 4, cx + 4]) {
    for (int dyE = -1; dyE <= 1; dyE++) {
      for (int dxE = -1; dxE <= 1; dxE++) {
        if (dxE == 0 && dyE == 0) continue;
        setPx(ex + dxE, eyeY + dyE, [250, 250, 250]);
      }
    }
    setPx(ex, eyeY, eye);
  }

  // 5. Optional glasses — round wire frames.
  if (glasses) {
    const frame = [30, 30, 35];
    for (final ex in [cx - 4, cx + 4]) {
      for (int a = 0; a < 24; a++) {
        final theta = a * math.pi / 12;
        final gx = (ex + 3.2 * math.cos(theta)).round();
        final gy = (eyeY + 2.6 * math.sin(theta)).round();
        setPx(gx, gy, frame);
      }
    }
    // Bridge.
    setPx(cx - 1, eyeY, frame);
    setPx(cx, eyeY, frame);
    setPx(cx + 1, eyeY, frame);
  }

  // 6. Nose (single darker skin pixel column).
  final noseTone = [
    clamp(skinLit[0] - 30),
    clamp(skinLit[1] - 30),
    clamp(skinLit[2] - 30),
  ];
  setPx(cx, cy + 1, noseTone);
  setPx(cx, cy + 2, noseTone);

  // 7. Mouth — curved line whose center dips (smile) or is flat.
  final mouthY = cy + 5;
  for (int mx = -mouthWidth; mx <= mouthWidth; mx++) {
    final norm = mx / mouthWidth; // [-1, 1]
    final curve = (mouthCurve * (1 - norm * norm)).round(); // parabolic
    setPx(cx + mx, mouthY + curve, mouth);
  }

  // 8. Subtle pixel noise — realism, and breaks up flat regions so
  //    triplet loss actually has to *learn* invariance.
  for (final p in img_) {
    final n = (rng.nextDouble() - 0.5) * 14;
    p
      ..r = clamp(p.r + n)
      ..g = clamp(p.g + n)
      ..b = clamp(p.b + n);
  }

  return img_;
}

/// Materialize the identity gallery on disk. When [ephemeral] is
/// true, uses a fresh `/tmp/dp_faces_...` directory that the caller
/// is expected to delete. Otherwise uses the stable
/// `./faces_gallery/` directory (cleaned and re-created each run)
/// so the PNGs remain on disk after the script exits and can be
/// opened in any image viewer.
Directory _buildFaceGallery({required bool ephemeral}) {
  final Directory root;
  if (ephemeral) {
    root = Directory.systemTemp.createTempSync('dp_faces_');
  } else {
    root = Directory('faces_gallery');
    if (root.existsSync()) {
      root.deleteSync(recursive: true);
    }
    root.createSync(recursive: true);
  }
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
  final ephemeral = args.contains('--tmp');
  print('=== train_face_folder (${device.name}) ===');

  // 1. Build synthetic-but-real face gallery on disk.
  final root = _buildFaceGallery(ephemeral: ephemeral);
  try {
    print(
      'gallery: ${root.absolute.path}'
      '${ephemeral ? '  (--tmp, deleted on exit)' : '  (kept — open in image viewer)'}',
    );
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
    print(
      '\nBEFORE  same-avg=${before.sameAvg.toStringAsFixed(4)} '
      '(${before.sameN} pairs)  '
      'diff-avg=${before.diffAvg.toStringAsFixed(4)} '
      '(${before.diffN} pairs)  '
      'gap=${(before.sameAvg - before.diffAvg).toStringAsFixed(4)}',
    );

    // 5. Train with triplet loss over triplets sampled from disk.
    print(
      '\ntraining $_steps steps (lr=$_lr, margin=$_margin, triplet loss)...',
    );
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
        print(
          '  step ${step.toString().padLeft(4)}  '
          'triplet=${lossVal.toStringAsFixed(6)}  '
          'trained=${trainedSteps.toString().padLeft(3)}/$step  '
          'avg=${avg.toStringAsFixed(4)}  '
          '(${ms.toStringAsFixed(1)} ms/step)',
        );
      }
    }
    sw.stop();

    // 6. Final val separation.
    model.eval();
    final after = _evalSeparation(model, val);
    print(
      '\nAFTER   same-avg=${after.sameAvg.toStringAsFixed(4)}  '
      'diff-avg=${after.diffAvg.toStringAsFixed(4)}  '
      'gap=${(after.sameAvg - after.diffAvg).toStringAsFixed(4)}',
    );

    final delta =
        (after.sameAvg - after.diffAvg) - (before.sameAvg - before.diffAvg);
    print(
      '\nseparation gap change: '
      '${delta >= 0 ? '+' : ''}${delta.toStringAsFixed(4)}',
    );
    if (delta > 0.05) {
      print('✅ val identities are more separated after training.');
    } else {
      print('⚠️  training did not clearly improve separation on val.');
    }
  } finally {
    if (ephemeral) {
      root.deleteSync(recursive: true);
    } else {
      print('\ngallery preserved at: ${root.absolute.path}');
    }
  }
}
