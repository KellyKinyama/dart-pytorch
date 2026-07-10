/// Real face-recognition training on `ImageFolderDataset` using an
/// actual celebrity-photo dataset.
///
/// End-to-end training pipeline using the new data loaders on real
/// (not synthetic) images. Concretely:
///
///  1. Reads real celebrity face photos from
///     `/mnt/c/Users/kkinyama/dart_cuda/Faces` — a flat directory of
///     `{PersonName}_{index}.jpg` images. The script groups them by
///     name prefix and materializes an `ImageFolder`-style layout at
///     `./faces_gallery/{PersonName}/sample_{k}.jpg`, ready to be
///     opened in any image viewer. Pass `--tmp` to build the gallery
///     in a `/tmp/dp_faces_...` directory that is deleted on exit,
///     or `--synthetic` to fall back to the toy 4-identity cartoon
///     gallery for a quick smoke test.
///
///  2. Loads it with [ImageFolderDataset], deterministically split
///     into train / val at 25 %. Every image is decoded and resized
///     to 32 × 32 inside the dataset.
///
///  3. Trains one of two models depending on the dataset:
///     * **real photos** → [ViTClassifier] with cross-entropy on
///       class indices, reporting top-1 val accuracy. Chosen because
///       triplet loss collapses to a degenerate "all embeddings
///       identical" minimum on a tiny untrained ViT with real 64×64
///       face crops — cross-entropy always has non-zero gradient.
///     * **synthetic cartoons** (`--synthetic`) → [ViTFaceEmbedding]
///       with triplet loss `relu(‖a-p‖² - ‖a-n‖² + margin)`, where
///       anchors and positives come from the same identity via
///       `ImageFolderDataset.sampleTriplet()`. Chosen because the
///       distinctive cartoon palettes give the untrained model a
///       clear starting gap.
///
///  4. Reports before/after task-appropriate val metric — top-1
///     accuracy for the classifier, cosine-similarity gap for the
///     triplet path.
///
/// Runs on CPU by default; pass `--gpu` for CUDA.
///
/// Run:
///
///     dart run bin/train_face_folder.dart               # CPU, real faces
///     dart run bin/train_face_folder.dart --gpu         # GPU
///     dart run bin/train_face_folder.dart --synthetic   # cartoon faces
///     dart run bin/train_face_folder.dart --tmp         # ephemeral /tmp dir
///     dart run bin/train_face_folder.dart --all-classes # use every id (slow)
library;

import 'dart:io';
import 'dart:math' as math;

import 'package:image/image.dart' as img;

import 'package:dart_pytorch/dart_pytorch.dart';

import 'shakespeare_util.dart' show paramScalarCount;

const int _imageSize = 64;
const int _patchSize = 8;
const int _numChannels = 3;
const int _embedDim = 96;
const int _outputDim = 48;
const int _numLayers = 2;
const int _numHeads = 4;

const int _numIdentities = 4;
const int _samplesPerIdentity = 16;
const int _steps = 1500;
const int _logEvery = 150;
const double _lr = 1e-3;
const double _margin = 0.2;

/// Source folder holding flat `{PersonName}_{index}.jpg` files. Copy
/// / rename the constant if your dataset lives elsewhere.
const String _realFacesSource = '/mnt/c/Users/kkinyama/dart_cuda/Faces';

/// Cap the number of identities used by default. `--all-classes`
/// lifts it. Chosen so a full training run stays under ~2 minutes on
/// CPU while still exercising a non-trivial number of classes.
const int _defaultMaxClasses = 8;

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
          if (d2 <= (headR + 1) * (headR + 1) &&
                  d2 >= (headR - 1) * (headR - 1) ||
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
Directory _buildSyntheticGallery({required bool ephemeral}) {
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

/// Scan [_realFacesSource] (flat `{PersonName}_{index}.jpg`), group
/// files by name prefix, and copy the first [samplesPerClass] images
/// of each identity into `./faces_gallery/{PersonName}/sample_{k}.jpg`
/// (or a temp dir when [ephemeral] is true). Only classes with at
/// least [samplesPerClass] source images are kept.
Directory _buildRealGallery({
  required bool ephemeral,
  required int samplesPerClass,
  int? maxClasses,
}) {
  final source = Directory(_realFacesSource);
  if (!source.existsSync()) {
    throw StateError(
      'real-face source folder not found: $_realFacesSource\n'
      'pass --synthetic to skip real faces or fix the _realFacesSource path.',
    );
  }

  // Group source files by name prefix (everything before the last '_').
  final byClass = <String, List<File>>{};
  final imgExts = {'.jpg', '.jpeg', '.png'};
  for (final entry in source.listSync().whereType<File>()) {
    final name = entry.path.split(Platform.pathSeparator).last;
    final dot = name.lastIndexOf('.');
    if (dot < 0 || !imgExts.contains(name.substring(dot).toLowerCase())) {
      continue;
    }
    final stem = name.substring(0, dot);
    final us = stem.lastIndexOf('_');
    if (us <= 0) continue;
    final person = stem.substring(0, us);
    (byClass[person] ??= <File>[]).add(entry);
  }

  // Deterministic ordering: alphabetize classes and, within each,
  // sort files by their trailing integer so we pick sample_0, _1, ...
  final classNames = byClass.keys.toList()..sort();
  final chosen = <String, List<File>>{};
  for (final person in classNames) {
    final files = byClass[person]!
      ..sort((a, b) {
        int idx(File f) {
          final n = f.path.split(Platform.pathSeparator).last;
          final dot = n.lastIndexOf('.');
          final us = n.lastIndexOf('_');
          return int.tryParse(n.substring(us + 1, dot)) ?? 0;
        }

        return idx(a).compareTo(idx(b));
      });
    if (files.length < samplesPerClass) continue;
    chosen[person] = files.sublist(0, samplesPerClass);
    if (maxClasses != null && chosen.length >= maxClasses) break;
  }
  if (chosen.isEmpty) {
    throw StateError(
      'no classes in $_realFacesSource had ≥ $samplesPerClass samples',
    );
  }

  // Materialize the ImageFolder layout.
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
  for (final entry in chosen.entries) {
    // Sanitize class dir name (spaces → _) so path handling stays
    // portable; keep the display name via the folder itself.
    final classDir = Directory('${root.path}/${entry.key}')..createSync();
    for (int k = 0; k < entry.value.length; k++) {
      final src = entry.value[k];
      final ext = src.path
          .substring(src.path.lastIndexOf('.'))
          .toLowerCase();
      final dst = File('${classDir.path}/sample_$k$ext');
      dst.writeAsBytesSync(src.readAsBytesSync());
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

/// Top-1 classification accuracy on the val split — for the
/// classifier path.
({double acc, int correct, int total}) _evalTopK(
  ViTClassifier model,
  ImageFolderDataset val,
) {
  return Tensor.noGrad(() {
    int correct = 0;
    for (int i = 0; i < val.length; i++) {
      final s = val[i];
      final logits = model(s.patches).toList();
      int argmax = 0;
      double best = logits[0];
      for (int c = 1; c < logits.length; c++) {
        if (logits[c] > best) {
          best = logits[c];
          argmax = c;
        }
      }
      if (argmax == s.label) correct++;
    }
    return (
      acc: val.length == 0 ? 0.0 : correct / val.length,
      correct: correct,
      total: val.length,
    );
  });
}

/// Triplet-loss path: `ViTFaceEmbedding` + `triplet` on same/diff-id
/// pairs. Well-suited to the synthetic cartoon gallery, where the
/// initial embedding already separates classes and the margin can
/// push it further.
void _runTriplet(
  ImageFolderDataset ds,
  ImageFolderDataset val,
  Device device,
) {
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
  print('model:     ViTFaceEmbedding, ${paramScalarCount(params)} scalars');

  final opt = Adam(params, lr: _lr);
  final marginT = Tensor.fill([1], _margin, device: device);

  final before = _evalSeparation(model, val);
  print(
    '\nBEFORE  same-avg=${before.sameAvg.toStringAsFixed(4)} '
    '(${before.sameN} pairs)  '
    'diff-avg=${before.diffAvg.toStringAsFixed(4)} '
    '(${before.diffN} pairs)  '
    'gap=${(before.sameAvg - before.diffAvg).toStringAsFixed(4)}',
  );

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
}

/// Classifier path: `ViTClassifier` + cross-entropy on class indices.
/// Well-suited to the real-photo gallery, where triplet loss collapses
/// to a degenerate all-embeddings-identical minimum on the tiny
/// untrained ViT. Cross-entropy always has non-zero gradient, so
/// the model reliably learns to separate the closed set of identities.
void _runClassifier(
  ImageFolderDataset ds,
  ImageFolderDataset val,
  Device device,
) {
  final model = ViTClassifier(
    imageSize: _imageSize,
    patchSize: _patchSize,
    numChannels: _numChannels,
    embedDim: _embedDim,
    numClasses: ds.numClasses,
    numLayers: _numLayers,
    numHeads: _numHeads,
    device: device,
    seed: 0,
  );
  final params = model.parameters();
  print('model:     ViTClassifier, ${paramScalarCount(params)} scalars');

  final opt = Adam(params, lr: _lr);

  final before = _evalTopK(model, val);
  final uniform = 1.0 / ds.numClasses;
  print(
    '\nBEFORE  val top-1 = ${(before.acc * 100).toStringAsFixed(1)}% '
    '(${before.correct}/${before.total}, uniform baseline '
    '${(uniform * 100).toStringAsFixed(1)}%)',
  );

  print(
    '\ntraining $_steps steps (lr=$_lr, cross-entropy, '
    'random train samples)...',
  );
  final sw = Stopwatch()..start();
  double lossSum = 0;
  final rng = math.Random(1);
  for (int step = 1; step <= _steps; step++) {
    opt.zeroGrad();
    final sample = ds[rng.nextInt(ds.length)];
    final logits = model(sample.patches); // [1, numClasses]
    final target = Tensor.fromList(
      [1],
      [sample.label.toDouble()],
      device: device,
    );
    final loss = logits.crossEntropy(target).mean();
    loss.backward();
    clipGradNorm(params, 1.0);
    opt.step();
    lossSum += loss.toList()[0];

    if (step == 1 || step % _logEvery == 0 || step == _steps) {
      final ms = sw.elapsedMilliseconds / step;
      final avg = lossSum / step;
      print(
        '  step ${step.toString().padLeft(4)}  '
        'ce=${loss.toList()[0].toStringAsFixed(4)}  '
        'avg=${avg.toStringAsFixed(4)}  '
        '(${ms.toStringAsFixed(1)} ms/step)',
      );
    }
  }
  sw.stop();

  model.eval();
  final after = _evalTopK(model, val);
  print(
    '\nAFTER   val top-1 = ${(after.acc * 100).toStringAsFixed(1)}% '
    '(${after.correct}/${after.total})',
  );

  final delta = after.acc - before.acc;
  print(
    '\naccuracy change: '
    '${delta >= 0 ? '+' : ''}${(delta * 100).toStringAsFixed(1)}%',
  );
  if (after.acc > uniform * 1.5) {
    print('✅ classifier beats uniform baseline.');
  } else {
    print('⚠️  classifier did not clearly beat uniform baseline.');
  }
}

void main(List<String> args) {
  final device = args.contains('--gpu') ? Device.GPU : Device.CPU;
  final ephemeral = args.contains('--tmp');
  final synthetic = args.contains('--synthetic');
  final allClasses = args.contains('--all-classes');
  print('=== train_face_folder (${device.name}) ===');

  // 1. Build gallery on disk — real photos by default, cartoon
  //    synthetic fallback via --synthetic.
  final Directory root;
  if (synthetic) {
    root = _buildSyntheticGallery(ephemeral: ephemeral);
  } else {
    root = _buildRealGallery(
      ephemeral: ephemeral,
      samplesPerClass: _samplesPerIdentity,
      maxClasses: allClasses ? null : _defaultMaxClasses,
    );
  }
  try {
    final numClasses = root.listSync().whereType<Directory>().length;
    print(
      'gallery: ${root.absolute.path}'
      '${ephemeral ? '  (--tmp, deleted on exit)' : '  (kept — open in image viewer)'}',
    );
    print(
      '  source:  ${synthetic ? 'synthetic cartoons' : _realFacesSource}',
    );
    print('  $numClasses identities × $_samplesPerIdentity samples');

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

    // 3. Train — pick the training strategy that suits the data.
    if (synthetic) {
      _runTriplet(ds, val, device);
    } else {
      _runClassifier(ds, val, device);
    }
  } finally {
    if (ephemeral) {
      root.deleteSync(recursive: true);
    } else {
      print('\ngallery preserved at: ${root.absolute.path}');
    }
  }
}
