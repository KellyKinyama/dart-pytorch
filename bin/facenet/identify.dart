/// FaceNet **face identification**: given a query photo and a
/// gallery of `faces_gallery/{IdentityName}/*.jpg`, find the closest
/// identity by cosine similarity.
///
///   dart run bin/facenet/identify.dart --query PATH [--gpu]
///   dart run bin/facenet/identify.dart \
///       --query "faces_gallery/Brad Pitt/sample_0.jpg" \
///       --gallery faces_gallery --top 5 --gpu
///
/// Options:
///   --query PATH      query face image (required)
///   --gallery PATH    default: faces_gallery
///   --top N           default: 5   (top-N matches to print)
///   --per-id N        default: 8   (limit samples per identity to
///                                     keep the scan fast)
///   --threshold F     default: 0.4 (SAME / DIFFERENT decision line)
///   --gpu             run on GPU
///   --weights PATH    default: models/facenet-vggface2/model.safetensors
///
/// Algorithm:
///   1. Embed the query once.
///   2. Embed every gallery image once.
///   3. Print the top-N closest images by cosine, then aggregate by
///      identity (average cosine over that identity's samples) and
///      pick the winning identity.
library;

import 'dart:io';

import 'package:dart_pytorch/core/nn/vision/facenet.dart';
import 'package:dart_pytorch/core/nn/vision/facenet_loader.dart';
import 'package:dart_pytorch/core/tensor/tensor.dart';

import '_common.dart';

const _weightsPathDefault = 'models/facenet-vggface2/model.safetensors';

Future<void> main(List<String> args) async {
  String? queryPath;
  var galleryPath = 'faces_gallery';
  var weightsPath = _weightsPathDefault;
  var top = 5;
  var perId = 8;
  var threshold = 0.4;
  var useGpu = false;
  for (int i = 0; i < args.length; i++) {
    final a = args[i];
    switch (a) {
      case '--query':
        queryPath = args[++i];
        break;
      case '--gallery':
        galleryPath = args[++i];
        break;
      case '--weights':
        weightsPath = args[++i];
        break;
      case '--top':
        top = int.parse(args[++i]);
        break;
      case '--per-id':
        perId = int.parse(args[++i]);
        break;
      case '--threshold':
        threshold = double.parse(args[++i]);
        break;
      case '--gpu':
        useGpu = true;
        break;
    }
  }
  if (queryPath == null || !File(queryPath).existsSync()) {
    stderr.writeln('missing --query PATH (or file not found)');
    exit(2);
  }
  if (!File(weightsPath).existsSync()) {
    stderr.writeln('missing weights: $weightsPath');
    exit(2);
  }
  if (!Directory(galleryPath).existsSync()) {
    stderr.writeln('missing gallery: $galleryPath');
    exit(2);
  }

  final device = useGpu ? Device.GPU : Device.CPU;
  final swTotal = Stopwatch()..start();

  print('== build + load ${useGpu ? '(GPU)' : '(CPU)'} ==');
  final swLoad = Stopwatch()..start();
  final model = InceptionResnetV1(device: device);
  FaceNetLoader.loadFile(model, weightsPath);
  model.eval();
  swLoad.stop();
  print('  ${swLoad.elapsedMilliseconds} ms');

  print('');
  print('== embed query ==');
  final swQ = Stopwatch()..start();
  final qEmb = model(decodeFaceJpeg(queryPath, device: device)).toList();
  swQ.stop();
  print('  ${swQ.elapsedMilliseconds} ms  $queryPath');

  print('');
  print('== embed gallery ==');
  final gallery = scanGallery(galleryPath, perId: perId);
  final total = gallery.values.map((v) => v.length).fold<int>(0, (a, b) => a + b);
  final swG = Stopwatch()..start();
  final entries = <_ScoredEntry>[];
  for (final entry in gallery.entries) {
    final id = entry.key;
    for (final path in entry.value) {
      final e = model(decodeFaceJpeg(path, device: device)).toList();
      entries.add(_ScoredEntry(id: id, path: path, score: cosine(qEmb, e)));
    }
  }
  swG.stop();
  print('  ${swG.elapsedMilliseconds} ms  ($total images, '
      '${gallery.length} identities)');

  entries.sort((a, b) => b.score.compareTo(a.score));

  print('');
  print('== top-$top nearest images ==');
  for (int i = 0; i < top && i < entries.length; i++) {
    final e = entries[i];
    final tag = e.score >= threshold ? 'SAME     ' : 'DIFFERENT';
    print('  $tag  cos=${e.score.toStringAsFixed(4)}   '
        '${prettyLabel(e.path)}');
  }

  // Per-identity mean cosine.
  final perIdScore = <String, double>{};
  final perIdCount = <String, int>{};
  for (final e in entries) {
    perIdScore[e.id] = (perIdScore[e.id] ?? 0.0) + e.score;
    perIdCount[e.id] = (perIdCount[e.id] ?? 0) + 1;
  }
  final idScores = perIdScore.entries
      .map((e) => MapEntry(e.key, e.value / perIdCount[e.key]!))
      .toList()
    ..sort((a, b) => b.value.compareTo(a.value));

  print('');
  print('== identity ranking (mean cosine across all samples) ==');
  for (int i = 0; i < idScores.length; i++) {
    final id = idScores[i];
    final marker = i == 0 ? '★' : ' ';
    final tag = id.value >= threshold ? 'SAME     ' : 'DIFFERENT';
    print('  $marker $tag  ${id.value.toStringAsFixed(4)}  ${id.key}');
  }

  final winner = idScores.first;
  print('');
  if (winner.value >= threshold) {
    print('== match: ${winner.key} '
        '(cos=${winner.value.toStringAsFixed(4)}) ==');
  } else {
    print('== no match  (best: ${winner.key} '
        'cos=${winner.value.toStringAsFixed(4)}, threshold=$threshold) ==');
  }

  swTotal.stop();
  print('');
  print('total wall = ${swTotal.elapsedMilliseconds} ms');
}

class _ScoredEntry {
  final String id;
  final String path;
  final double score;
  _ScoredEntry({required this.id, required this.path, required this.score});
}
