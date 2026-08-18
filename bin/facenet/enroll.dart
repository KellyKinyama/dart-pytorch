/// FaceNet **enrollment + lookup**: build a persisted embedding
/// database from a labeled `faces_gallery/{IdentityName}/*.jpg`,
/// then answer identity queries against it — without re-running the
/// backbone on every gallery image every time.
///
/// Two commands:
///
///   dart run bin/facenet/enroll.dart enroll \
///       [--gallery faces_gallery] [--db models/facenet_db.json] \
///       [--per-id 8] [--gpu]
///
///   dart run bin/facenet/enroll.dart query \
///       --image PATH \
///       [--db models/facenet_db.json] [--top 3] [--threshold 0.4] \
///       [--gpu]
///
/// The database (default: `models/facenet_db.json`) stores one
/// per-identity **mean** 512-d embedding (re-normalized to unit norm),
/// plus the number of enrolled samples. Loading + querying an
/// N-identity DB is O(N × 512) per query — pure Dart, no I/O per
/// image.
library;

import 'dart:convert';
import 'dart:io';

import 'package:dart_pytorch/core/nn/vision/facenet.dart';
import 'package:dart_pytorch/core/nn/vision/facenet_loader.dart';
import 'package:dart_pytorch/core/tensor/tensor.dart';

import '_common.dart';

const _weightsPathDefault = 'models/facenet-vggface2/model.safetensors';
const _dbPathDefault = 'models/facenet_db.json';

Future<void> main(List<String> args) async {
  if (args.isEmpty) {
    _printUsage();
    exit(2);
  }
  switch (args.first) {
    case 'enroll':
      await _enroll(args.sublist(1));
      break;
    case 'query':
      await _query(args.sublist(1));
      break;
    default:
      _printUsage();
      exit(2);
  }
}

void _printUsage() {
  stderr.writeln(
    'usage:\n'
    '  dart run bin/facenet/enroll.dart enroll [--gallery PATH] '
    '[--db PATH] [--per-id N] [--gpu] [--weights PATH]\n'
    '  dart run bin/facenet/enroll.dart query --image PATH '
    '[--db PATH] [--top N] [--threshold F] [--gpu] [--weights PATH]',
  );
}

// -------------------------- enroll --------------------------

Future<void> _enroll(List<String> args) async {
  var galleryPath = 'faces_gallery';
  var dbPath = _dbPathDefault;
  var perId = 8;
  var weightsPath = _weightsPathDefault;
  var useGpu = false;
  for (int i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--gallery':
        galleryPath = args[++i];
        break;
      case '--db':
        dbPath = args[++i];
        break;
      case '--per-id':
        perId = int.parse(args[++i]);
        break;
      case '--weights':
        weightsPath = args[++i];
        break;
      case '--gpu':
        useGpu = true;
        break;
    }
  }
  if (!Directory(galleryPath).existsSync()) {
    stderr.writeln('missing gallery: $galleryPath');
    exit(2);
  }

  final device = useGpu ? Device.GPU : Device.CPU;
  print('== build + load ${useGpu ? '(GPU)' : '(CPU)'} ==');
  final swLoad = Stopwatch()..start();
  final model = InceptionResnetV1(device: device);
  FaceNetLoader.loadFile(model, weightsPath);
  model.eval();
  swLoad.stop();
  print('  ${swLoad.elapsedMilliseconds} ms');

  final gallery = scanGallery(galleryPath, perId: perId);
  print('');
  print('== enroll ${gallery.length} identities ==');
  final db = <String, dynamic>{
    'model': 'facenet-vggface2',
    'dim': 512,
    'entries': <Map<String, dynamic>>[],
  };
  final sw = Stopwatch()..start();
  for (final entry in gallery.entries) {
    final id = entry.key;
    final sum = List<double>.filled(512, 0.0);
    var used = 0;
    for (final path in entry.value) {
      final e = model(decodeFaceJpeg(path, device: device)).toList();
      for (int k = 0; k < 512; k++) {
        sum[k] += e[k];
      }
      used++;
    }
    // Mean, then re-normalize.
    var norm = 0.0;
    for (int k = 0; k < 512; k++) {
      sum[k] /= used;
      norm += sum[k] * sum[k];
    }
    norm = _sqrt(norm);
    if (norm > 0) {
      for (int k = 0; k < 512; k++) {
        sum[k] /= norm;
      }
    }
    (db['entries'] as List).add({
      'id': id,
      'samples': used,
      'embedding': sum,
    });
    print('  enrolled  $id  ($used samples)');
  }
  sw.stop();
  print('  ${sw.elapsedMilliseconds} ms');

  final outFile = File(dbPath);
  outFile.parent.createSync(recursive: true);
  outFile.writeAsStringSync(const JsonEncoder.withIndent('  ').convert(db));
  print('');
  print('== wrote $dbPath ==');
  print(
    '  ${gallery.length} identities, '
    '${(db['entries'] as List).map((e) => e['samples']).fold<int>(0, (a, b) => a + (b as int))} '
    'samples, dim=512',
  );
}

// -------------------------- query --------------------------

Future<void> _query(List<String> args) async {
  String? imagePath;
  var dbPath = _dbPathDefault;
  var top = 3;
  var threshold = 0.4;
  var weightsPath = _weightsPathDefault;
  var useGpu = false;
  for (int i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--image':
        imagePath = args[++i];
        break;
      case '--db':
        dbPath = args[++i];
        break;
      case '--top':
        top = int.parse(args[++i]);
        break;
      case '--threshold':
        threshold = double.parse(args[++i]);
        break;
      case '--weights':
        weightsPath = args[++i];
        break;
      case '--gpu':
        useGpu = true;
        break;
    }
  }
  if (imagePath == null || !File(imagePath).existsSync()) {
    stderr.writeln('missing --image PATH (or file not found)');
    exit(2);
  }
  if (!File(dbPath).existsSync()) {
    stderr.writeln('missing $dbPath — run `enroll` first');
    exit(2);
  }

  final device = useGpu ? Device.GPU : Device.CPU;

  print('== load DB ==');
  final db = jsonDecode(File(dbPath).readAsStringSync()) as Map<String, dynamic>;
  final entries = (db['entries'] as List)
      .cast<Map<String, dynamic>>()
      .map((e) => _DbEntry(
            id: e['id'] as String,
            samples: e['samples'] as int,
            embedding: (e['embedding'] as List).cast<num>()
                .map((v) => v.toDouble())
                .toList(),
          ))
      .toList();
  print('  $dbPath  (${entries.length} identities, dim=${db['dim']})');

  print('');
  print('== embed query ${useGpu ? '(GPU)' : '(CPU)'} ==');
  final sw = Stopwatch()..start();
  final model = InceptionResnetV1(device: device);
  FaceNetLoader.loadFile(model, weightsPath);
  model.eval();
  final qEmb = model(decodeFaceJpeg(imagePath, device: device)).toList();
  sw.stop();
  print('  ${sw.elapsedMilliseconds} ms  $imagePath');

  final scored = entries
      .map((e) => MapEntry(e, cosine(qEmb, e.embedding)))
      .toList()
    ..sort((a, b) => b.value.compareTo(a.value));

  print('');
  print('== top-$top matches ==');
  for (int i = 0; i < top && i < scored.length; i++) {
    final e = scored[i];
    final marker = i == 0 ? '★' : ' ';
    final tag = e.value >= threshold ? 'SAME     ' : 'DIFFERENT';
    print(
      '  $marker $tag  cos=${e.value.toStringAsFixed(4)}  '
      '${e.key.id}  (n=${e.key.samples})',
    );
  }

  final winner = scored.first;
  print('');
  if (winner.value >= threshold) {
    print('== match: ${winner.key.id} '
        '(cos=${winner.value.toStringAsFixed(4)}) ==');
  } else {
    print('== no match (best: ${winner.key.id} '
        'cos=${winner.value.toStringAsFixed(4)}, threshold=$threshold) ==');
  }
}

class _DbEntry {
  final String id;
  final int samples;
  final List<double> embedding;
  _DbEntry({required this.id, required this.samples, required this.embedding});
}

double _sqrt(double x) {
  if (x <= 0) return 0.0;
  var y = x;
  var z = (y + 1.0) / 2.0;
  for (int i = 0; i < 30 && z != y; i++) {
    y = z;
    z = (y + x / y) / 2.0;
  }
  return z;
}
