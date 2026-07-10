/// Image-folder classification / triplet dataset.
///
/// Mirrors the ImageNet-style directory layout:
///
///     <root>/
///       <class_0>/  *.jpg | *.jpeg | *.png
///       <class_1>/  *.jpg | *.jpeg | *.png
///       ...
///
/// Every file is decoded once with `package:image`, resized to
/// `imageSize × imageSize`, normalized to `[0, 1]` (channels-last
/// RGB), and cached in RAM as a `Float32List`.
///
/// Each dataset item is a [FaceSample]:
///
///     patches: Tensor [numPatches, patchSize*patchSize*3]
///     label:   int in [0, numClasses)
///
/// The patchification layout matches [ViTBackbone]: patches iterated
/// row-major over the image, with each patch's `patchSize × patchSize
/// × 3` pixels flattened `(dy, dx, c)`.
///
/// * [sampleTriplet] — draws (anchor, positive, negative) with
///   anchor & positive from the *same* class and negative from a
///   different class, all patchified as [Tensor]s on the requested
///   device. Perfect for face-recognition / metric-learning training.
///
/// * Train / val split is deterministic given a seed.
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart' as img;

import '../tensor/tensor.dart';
import 'dataset.dart';

class _Sample {
  final Float32List flat; // [H*W*3], normalized [0,1], channels-last
  final int label;
  const _Sample(this.flat, this.label);
}

/// One example from an [ImageFolderDataset]: a patchified image plus
/// its integer class label.
class FaceSample {
  final Tensor patches; // [numPatches, patchSize*patchSize*3]
  final int label;
  const FaceSample(this.patches, this.label);
}

/// Bundle of three patchified images used by triplet-loss metric
/// learning (anchor + positive from the same class, negative from a
/// different class).
class TripletSample {
  final Tensor anchor;
  final Tensor positive;
  final Tensor negative;
  final int anchorClass;
  final int negativeClass;
  const TripletSample({
    required this.anchor,
    required this.positive,
    required this.negative,
    required this.anchorClass,
    required this.negativeClass,
  });
}

class ImageFolderDataset extends Dataset<FaceSample> {
  final String rootPath;
  final int imageSize;
  final int patchSize;
  final Device device;
  late final List<String> classes;
  final List<_Sample> _train = [];
  final List<_Sample> _val = [];
  final List<List<int>> _trainByClass = [];
  final math.Random _rng;

  int get numClasses => classes.length;
  int get numTrain => _train.length;
  int get numVal => _val.length;
  int get patchPixels => patchSize * patchSize * 3;
  int get numPatches => (imageSize ~/ patchSize) * (imageSize ~/ patchSize);

  /// Which side of the split this instance indexes into.
  final bool _isVal;

  ImageFolderDataset(
    this.rootPath, {
    required this.imageSize,
    required this.patchSize,
    int maxPerClass = 1 << 30,
    int? maxClasses,
    double valSplit = 0.2,
    this.device = Device.CPU,
    int seed = 7,
    bool val = false,
  })  : _rng = math.Random(seed),
        _isVal = val {
    if (imageSize % patchSize != 0) {
      throw ArgumentError(
        'imageSize ($imageSize) must be divisible by patchSize ($patchSize)',
      );
    }
    final root = Directory(rootPath);
    if (!root.existsSync()) {
      throw ArgumentError('rootPath does not exist: $rootPath');
    }

    final classDirs = root.listSync().whereType<Directory>().toList()
      ..sort((a, b) => a.path.compareTo(b.path));
    if (maxClasses != null && classDirs.length > maxClasses) {
      classDirs.removeRange(maxClasses, classDirs.length);
    }
    final mutableClasses = <String>[];
    final splitRng = math.Random(seed);

    for (final dir in classDirs) {
      final name = dir.path.split(Platform.pathSeparator).last;
      final files = dir
          .listSync()
          .whereType<File>()
          .where((f) {
            final p = f.path.toLowerCase();
            return p.endsWith('.jpg') ||
                p.endsWith('.jpeg') ||
                p.endsWith('.png');
          })
          .toList()
        ..sort((a, b) => a.path.compareTo(b.path));
      if (files.isEmpty) continue;
      final cap = files.length > maxPerClass ? maxPerClass : files.length;
      mutableClasses.add(name);
      final indices = List<int>.generate(cap, (i) => i)..shuffle(splitRng);
      final nVal = (cap * valSplit).round();
      for (int k = 0; k < cap; k++) {
        final f = files[indices[k]];
        final flat = _decode(f);
        if (flat == null) continue;
        final s = _Sample(flat, mutableClasses.length - 1);
        if (k < nVal) {
          _val.add(s);
        } else {
          _train.add(s);
        }
      }
    }
    classes = mutableClasses;

    for (int c = 0; c < classes.length; c++) {
      _trainByClass.add(<int>[]);
    }
    for (int i = 0; i < _train.length; i++) {
      _trainByClass[_train[i].label].add(i);
    }
  }

  Float32List? _decode(File file) {
    try {
      final raw = img.decodeImage(file.readAsBytesSync());
      if (raw == null) return null;
      final resized = img.copyResize(
        raw,
        width: imageSize,
        height: imageSize,
        interpolation: img.Interpolation.linear,
      );
      final flat = Float32List(imageSize * imageSize * 3);
      int i = 0;
      for (final p in resized) {
        flat[i++] = p.r / 255.0;
        flat[i++] = p.g / 255.0;
        flat[i++] = p.b / 255.0;
      }
      return flat;
    } catch (_) {
      return null;
    }
  }

  /// Patchify a flat `H×W×3` (channels-last) image into
  /// `[numPatches, patchPixels]`, row-major over patches. Each
  /// patch's pixels are laid out as `(dy_in_patch, dx_in_patch, c)`.
  Float32List patchify(Float32List flat) {
    final H = imageSize, W = imageSize, P = patchSize;
    final perPatch = P * P * 3;
    final nP = numPatches;
    final out = Float32List(nP * perPatch);
    final patchesPerRow = W ~/ P;
    for (int py = 0; py < H ~/ P; py++) {
      for (int px = 0; px < patchesPerRow; px++) {
        final pIdx = py * patchesPerRow + px;
        final outBase = pIdx * perPatch;
        int w = 0;
        for (int dy = 0; dy < P; dy++) {
          final y = py * P + dy;
          for (int dx = 0; dx < P; dx++) {
            final x = px * P + dx;
            final inBase = (y * W + x) * 3;
            out[outBase + w++] = flat[inBase];
            out[outBase + w++] = flat[inBase + 1];
            out[outBase + w++] = flat[inBase + 2];
          }
        }
      }
    }
    return out;
  }

  Tensor _asTensor(Float32List flat) => Tensor.fromList(
        [numPatches, patchPixels],
        List<double>.from(flat),
        device: device,
      );

  // -------------------------------------------------------------------
  // Dataset<FaceSample> interface
  // -------------------------------------------------------------------

  @override
  int get length => _isVal ? _val.length : _train.length;

  @override
  FaceSample operator [](int index) {
    final s = _isVal ? _val[index] : _train[index];
    return FaceSample(_asTensor(patchify(s.flat)), s.label);
  }

  /// A companion dataset over the validation split of the *same*
  /// underlying files (no re-decoding). Cheap.
  ImageFolderDataset valSplit() {
    if (_isVal) return this;
    final v = ImageFolderDataset._shared(
      this,
      isVal: true,
    );
    return v;
  }

  ImageFolderDataset._shared(ImageFolderDataset other, {required bool isVal})
      : rootPath = other.rootPath,
        imageSize = other.imageSize,
        patchSize = other.patchSize,
        device = other.device,
        _rng = other._rng,
        _isVal = isVal {
    classes = other.classes;
    _train.addAll(other._train);
    _val.addAll(other._val);
    for (int c = 0; c < classes.length; c++) {
      _trainByClass.add(other._trainByClass[c]);
    }
  }

  /// Train classes with `>= 2` samples (needed for triplet sampling).
  List<int> get tripletReadyClasses => [
        for (int c = 0; c < _trainByClass.length; c++)
          if (_trainByClass[c].length >= 2) c,
      ];

  /// Draw one triplet from the train split.
  TripletSample sampleTriplet() {
    final ready = tripletReadyClasses;
    if (ready.length < 2) {
      throw StateError(
        'Need >= 2 classes with >= 2 train samples each for triplets, '
        'got ${ready.length}',
      );
    }
    final aClass = ready[_rng.nextInt(ready.length)];
    int nClass;
    do {
      nClass = ready[_rng.nextInt(ready.length)];
    } while (nClass == aClass);

    final aIdxList = _trainByClass[aClass];
    final i1 = _rng.nextInt(aIdxList.length);
    int i2;
    do {
      i2 = _rng.nextInt(aIdxList.length);
    } while (i2 == i1);
    final nIdxList = _trainByClass[nClass];
    final i3 = _rng.nextInt(nIdxList.length);

    return TripletSample(
      anchor: _asTensor(patchify(_train[aIdxList[i1]].flat)),
      positive: _asTensor(patchify(_train[aIdxList[i2]].flat)),
      negative: _asTensor(patchify(_train[nIdxList[i3]].flat)),
      anchorClass: aClass,
      negativeClass: nClass,
    );
  }
}
