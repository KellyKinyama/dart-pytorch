/// Image-to-image similarity search on `faces_gallery/`.
///
/// The same vector-store story off the text axis. Instead of a language
/// model producing sentence embeddings we use a small Vision Transformer
/// producing face embeddings: patchify → `ViTBackbone` → CLS token →
/// L2-normalise → `IndexFlatIP`. Given a query photo, find the k nearest
/// photos in the gallery by cosine similarity.
///
/// The wrinkle: an untrained ViT produces near-random embeddings so
/// nearest-neighbour retrieval is barely above chance. From the
/// `train_face_folder.dart` recipe (see user memory
/// `dart_pytorch_training.md`): triplet loss on a tiny ViT with a few
/// hundred real photos collapses to a degenerate all-embeddings-
/// identical minimum. The fix that works is **cross-entropy
/// classification**, then drop the head and use the backbone's CLS
/// vector as the embedding. That's what this demo does.
///
/// Pipeline:
///
///   1. Load `faces_gallery/` via [ImageFolderDataset] (patchified,
///      80/20 train/val split, deterministic seed).
///   2. Train a small [ViTClassifier] on the train split with Adam +
///      cross-entropy for `_steps` steps (~90 s on CPU for the 8-class
///      default). Grad-clip 1.0.
///   3. Embed **every train image** with `vitClsFeature(backbone(x))`,
///      L2-normalise, add to `IndexFlatIP(embedDim)`.
///   4. For each val image (up to `_maxQueries`), embed it the same
///      way, `index.search(q, k=5)`, print the retrieved gallery
///      names + cosines.
///   5. Report same-identity retrieval rate = fraction of top-k
///      neighbours that share the query's class.
///
/// Run:
///
/// ```sh
///   dart run bin/vector_image_search_demo.dart
/// ```
///
/// Change `--gallery PATH` to point at a different folder-per-class
/// layout (the demo uses the repo's `faces_gallery/`).
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const int _imageSize = 64;
const int _patchSize = 8;
const int _embedDim = 96;
const int _numLayers = 2;
const int _numHeads = 4;

const int _steps = 800;
const int _logEvery = 100;
const double _lr = 1e-3;

const int _topK = 5;
const int _maxQueries = 8;
const int _defaultMaxClasses = 8;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

class _Opts {
  _Opts({required this.gallery, required this.gpu, required this.allClasses});
  final String gallery;
  final bool gpu;
  final bool allClasses;

  Device get device => gpu ? Device.GPU : Device.CPU;
}

_Opts _parseArgs(List<String> args) {
  var gallery = 'faces_gallery';
  var gpu = false;
  var all = false;
  for (var i = 0; i < args.length; i++) {
    final a = args[i];
    String next() {
      if (i + 1 >= args.length) {
        stderr.writeln('missing value for $a');
        exit(64);
      }
      return args[++i];
    }

    switch (a) {
      case '--gallery':
        gallery = next();
        break;
      case '--gpu':
        gpu = true;
        break;
      case '--all-classes':
        all = true;
        break;
      case '-h':
      case '--help':
        stdout.writeln(_help);
        exit(0);
      default:
        stderr.writeln('unknown arg: $a');
        stderr.writeln(_help);
        exit(64);
    }
  }
  return _Opts(gallery: gallery, gpu: gpu, allClasses: all);
}

const String _help =
    '''
Image-to-image similarity search over a folder-per-class gallery.

Usage:
  dart run bin/vector_image_search_demo.dart [flags]

Flags:
  --gallery PATH   root folder with one subdir per class
                   (default: faces_gallery)
  --all-classes    use all subfolders (default: cap at $_defaultMaxClasses)
  --gpu            run on CUDA (default: CPU)
  -h, --help       print this message
''';

// ---------------------------------------------------------------------------

void main(List<String> args) {
  final opts = _parseArgs(args);
  final device = opts.device;

  final root = Directory(opts.gallery);
  if (!root.existsSync()) {
    stderr.writeln('gallery not found: ${opts.gallery}');
    stderr.writeln('expected a folder-per-class layout, e.g.:');
    stderr.writeln('  ${opts.gallery}/Alice/*.jpg');
    stderr.writeln('  ${opts.gallery}/Bob/*.jpg');
    exit(1);
  }

  stdout.writeln('=== vector_image_search (${device.name}) ===');
  stdout.writeln('gallery: ${root.absolute.path}');

  // ---- 1. Load ---------------------------------------------------

  final ds = ImageFolderDataset(
    root.path,
    imageSize: _imageSize,
    patchSize: _patchSize,
    valSplit: 0.20,
    maxClasses: opts.allClasses ? null : _defaultMaxClasses,
    device: device,
    seed: 0,
  );
  final val = ds.valSplit();
  stdout.writeln('classes:   ${ds.classes}');
  stdout.writeln('train/val: ${ds.numTrain} / ${ds.numVal}');
  stdout.writeln('per-item:  patches=[${ds.numPatches}, ${ds.patchPixels}]');

  if (ds.numTrain < ds.numClasses * 2) {
    stderr.writeln(
      'not enough train samples (${ds.numTrain}) for '
      '${ds.numClasses} classes — need at least 2/class',
    );
    exit(1);
  }

  // ---- 2. Train a ViT classifier ---------------------------------

  final model = ViTClassifier(
    imageSize: _imageSize,
    patchSize: _patchSize,
    numChannels: 3,
    embedDim: _embedDim,
    numClasses: ds.numClasses,
    numLayers: _numLayers,
    numHeads: _numHeads,
    device: device,
    seed: 0,
  );
  final params = model.parameters();
  final opt = Adam(params, lr: _lr);

  stdout.writeln('\ntraining $_steps steps (Adam lr=$_lr, cross-entropy)...');
  final rng = math.Random(1);
  final sw = Stopwatch()..start();
  var lossSum = 0.0;
  for (var step = 1; step <= _steps; step++) {
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
      final avg = lossSum / step;
      final ms = sw.elapsedMilliseconds / step;
      stdout.writeln(
        '  step ${step.toString().padLeft(4)}  '
        'ce=${loss.toList()[0].toStringAsFixed(4)}  '
        'avg=${avg.toStringAsFixed(4)}  '
        '(${ms.toStringAsFixed(1)} ms/step)',
      );
    }
  }
  sw.stop();
  model.eval();

  // ---- 3. Embed the train gallery --------------------------------

  stdout.writeln('\nEmbedding ${ds.numTrain} gallery images...');
  final galleryVecs = <Float32List>[];
  final galleryLabels = <int>[];
  for (var i = 0; i < ds.numTrain; i++) {
    final s = ds[i];
    galleryVecs.add(_embed(model, s.patches, _embedDim));
    galleryLabels.add(s.label);
  }
  final index = IndexFlatIP(_embedDim);
  index.add(galleryVecs);
  stdout.writeln('Indexed ${index.ntotal} vectors.');

  // ---- 4. Query with val images ----------------------------------

  final numQueries = math.min(_maxQueries, val.length);
  stdout.writeln(
    '\nQuerying with $numQueries validation images '
    '(top-$_topK cosine neighbours):',
  );

  var totalHits = 0;
  var top1Correct = 0;
  for (var q = 0; q < numQueries; q++) {
    // Pick spaced-out val indices so we cover multiple classes.
    final idx = (q * val.length ~/ numQueries).clamp(0, val.length - 1);
    final vSample = val[idx];
    final qVec = _embed(model, vSample.patches, _embedDim);
    final res = index.search([qVec], _topK);
    final ids = res.ids[0];
    final scores = res.distances[0];

    final qName = ds.classes[vSample.label];
    stdout.writeln('\n---');
    stdout.writeln('Q [val#$idx] "$qName":');
    var hitsThisQ = 0;
    for (var j = 0; j < ids.length; j++) {
      final hitName = ds.classes[galleryLabels[ids[j]]];
      final match = hitName == qName ? '  match' : '';
      stdout.writeln(
        '  ${(j + 1).toString().padLeft(2)}. '
        '${hitName.padRight(24)}  cos=${scores[j].toStringAsFixed(3)}$match',
      );
      if (hitName == qName) hitsThisQ++;
    }
    if (galleryLabels[ids[0]] == vSample.label) top1Correct++;
    totalHits += hitsThisQ;
  }

  final possible = numQueries * _topK;
  stdout.writeln('\n=================================================');
  stdout.writeln(
    'Top-1 accuracy: $top1Correct / $numQueries '
    '= ${(top1Correct / numQueries * 100).toStringAsFixed(1)}%',
  );
  stdout.writeln(
    'Same-identity hits: $totalHits / $possible '
    '= ${(totalHits / possible * 100).toStringAsFixed(1)}%   '
    '(chance = ${(100 / ds.numClasses).toStringAsFixed(0)}%)',
  );
}

/// Compute a single L2-normalised embedding for a patchified image.
Float32List _embed(ViTClassifier model, Tensor patches, int dim) {
  return Tensor.noGrad(() {
    final encoded = model.backbone(patches); // [numPatches+1, embedDim]
    final cls = vitClsFeature(encoded); // [1, embedDim]
    final flat = cls.toList();
    final out = Float32List(dim);
    var sq = 0.0;
    for (var j = 0; j < dim; j++) {
      out[j] = flat[j].toDouble();
      sq += out[j] * out[j];
    }
    final norm = 1.0 / (sq > 1e-24 ? math.sqrt(sq) : 1.0);
    for (var j = 0; j < dim; j++) {
      out[j] *= norm;
    }
    return out;
  });
}
