/// FaceNet **unsupervised face clustering**: given a flat directory
/// (or a labeled `faces_gallery/{Id}/*.jpg` tree — labels are ignored
/// and used only to score cluster purity afterwards), embed every
/// image and run single-linkage agglomerative clustering on the 512-d
/// cosine similarity. Prints the discovered clusters + purity vs. the
/// gold labels when they're available.
///
///   dart run bin/facenet/cluster.dart --dir PATH [--gpu]
///
/// Options:
///   --dir PATH         input directory (required). Two layouts work:
///                        • flat:  PATH/*.jpg
///                        • split: PATH/{IdentityName}/*.jpg
///                      In split mode we compare the discovered
///                      clusters against the gold identity labels.
///   --threshold F      default: 0.4  (merge clusters whose closest
///                                     pair has cosine >= this)
///   --per-id N         default: 8    (cap per-identity samples when
///                                     input is in split layout)
///   --gpu              run on GPU
///   --weights PATH     default: models/facenet-vggface2/model.safetensors
///
/// Algorithm:
///   1. Embed every image.
///   2. Build the N×N cosine matrix.
///   3. Single-linkage agglomerative merges — repeatedly find the
///      closest pair of clusters (max-cosine over all cross-cluster
///      pairs) and merge if that similarity >= threshold.
///   4. Report clusters + (if labels available) purity per cluster.
library;

import 'dart:io';

import 'package:dart_pytorch/core/nn/vision/facenet.dart';
import 'package:dart_pytorch/core/nn/vision/facenet_loader.dart';
import 'package:dart_pytorch/core/tensor/tensor.dart';

import '_common.dart';

const _weightsPathDefault = 'models/facenet-vggface2/model.safetensors';

Future<void> main(List<String> args) async {
  String? dir;
  var weightsPath = _weightsPathDefault;
  var threshold = 0.4;
  var perId = 8;
  var useGpu = false;
  for (int i = 0; i < args.length; i++) {
    final a = args[i];
    switch (a) {
      case '--dir':
        dir = args[++i];
        break;
      case '--weights':
        weightsPath = args[++i];
        break;
      case '--threshold':
        threshold = double.parse(args[++i]);
        break;
      case '--per-id':
        perId = int.parse(args[++i]);
        break;
      case '--gpu':
        useGpu = true;
        break;
    }
  }
  if (dir == null || !Directory(dir).existsSync()) {
    stderr.writeln('missing --dir PATH (or not a directory)');
    exit(2);
  }
  if (!File(weightsPath).existsSync()) {
    stderr.writeln('missing weights: $weightsPath');
    exit(2);
  }

  final device = useGpu ? Device.GPU : Device.CPU;
  final swTotal = Stopwatch()..start();

  // ---- gather items (path + optional gold label) ----
  final items = <_Item>[];
  final gallery = scanGallery(dir, perId: perId);
  final splitLayout = gallery.isNotEmpty;
  if (splitLayout) {
    for (final entry in gallery.entries) {
      for (final path in entry.value) {
        items.add(_Item(path: path, gold: entry.key));
      }
    }
  } else {
    for (final e in Directory(dir).listSync()) {
      if (e is File) {
        final low = e.path.toLowerCase();
        if (low.endsWith('.jpg') ||
            low.endsWith('.jpeg') ||
            low.endsWith('.png')) {
          items.add(_Item(path: e.path, gold: null));
        }
      }
    }
    items.sort((a, b) => a.path.compareTo(b.path));
  }
  if (items.length < 2) {
    stderr.writeln('need at least 2 images; got ${items.length}');
    exit(2);
  }

  print('== build + load ${useGpu ? '(GPU)' : '(CPU)'} ==');
  final swLoad = Stopwatch()..start();
  final model = InceptionResnetV1(device: device);
  FaceNetLoader.loadFile(model, weightsPath);
  model.eval();
  swLoad.stop();
  print('  ${swLoad.elapsedMilliseconds} ms  (${items.length} items, '
      '${splitLayout ? gallery.length : '?'} gold identities)');

  // ---- embed all ----
  print('');
  print('== embed ${items.length} faces ==');
  final swE = Stopwatch()..start();
  final embs = <List<double>>[];
  for (final item in items) {
    final e = model(decodeFaceJpeg(item.path, device: device)).toList();
    embs.add(e);
  }
  swE.stop();
  print('  ${swE.elapsedMilliseconds} ms');

  // ---- cosine matrix ----
  final n = items.length;
  final cos = List<List<double>>.generate(n, (_) => List.filled(n, 0.0));
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      final c = cosine(embs[i], embs[j]);
      cos[i][j] = c;
      cos[j][i] = c;
    }
  }

  // ---- single-linkage agglomerative ----
  print('');
  print('== cluster (single-linkage, threshold ${threshold.toStringAsFixed(2)}) ==');
  final clusters = List<Set<int>>.generate(n, (i) => {i});
  bool merged = true;
  while (merged) {
    merged = false;
    double bestSim = -double.infinity;
    int bestA = -1, bestB = -1;
    for (int a = 0; a < clusters.length; a++) {
      for (int b = a + 1; b < clusters.length; b++) {
        // single-linkage: max cross-cluster similarity.
        double s = -double.infinity;
        for (final ai in clusters[a]) {
          for (final bi in clusters[b]) {
            if (cos[ai][bi] > s) s = cos[ai][bi];
          }
        }
        if (s > bestSim) {
          bestSim = s;
          bestA = a;
          bestB = b;
        }
      }
    }
    if (bestSim >= threshold && bestA >= 0) {
      clusters[bestA].addAll(clusters[bestB]);
      clusters.removeAt(bestB);
      merged = true;
    }
  }

  clusters.sort((a, b) => b.length.compareTo(a.length));
  print('  found ${clusters.length} clusters');

  print('');
  print('== clusters ==');
  int purityCorrect = 0;
  for (int c = 0; c < clusters.length; c++) {
    final ids = clusters[c].toList()..sort();
    final labels = <String, int>{};
    for (final i in ids) {
      final g = items[i].gold ?? '(unlabeled)';
      labels[g] = (labels[g] ?? 0) + 1;
    }
    final labelList = labels.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));
    final dominant = labelList.first;
    final purity = ids.isEmpty ? 0.0 : dominant.value / ids.length;
    if (splitLayout) {
      purityCorrect += dominant.value;
    }
    print('  #$c  size=${ids.length}  '
        'dominant=${dominant.key} (${dominant.value}/${ids.length}, '
        'purity ${purity.toStringAsFixed(2)})');
    for (final i in ids.take(3)) {
      print('        ${prettyLabel(items[i].path)}');
    }
    if (ids.length > 3) {
      print('        …and ${ids.length - 3} more');
    }
  }

  if (splitLayout) {
    final overallPurity = purityCorrect / items.length;
    print('');
    print('== quality (labels available) ==');
    print('  overall purity          '
        '${overallPurity.toStringAsFixed(4)}   '
        '($purityCorrect / ${items.length})');
    print('  clusters vs identities  '
        '${clusters.length} / ${gallery.length}');
  }

  swTotal.stop();
  print('');
  print('total wall = ${swTotal.elapsedMilliseconds} ms');
}

class _Item {
  final String path;
  final String? gold;
  _Item({required this.path, required this.gold});
}
