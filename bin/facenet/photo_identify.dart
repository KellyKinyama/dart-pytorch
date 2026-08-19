/// **The full FaceNet pipeline on wild photos.** Detect faces with
/// MTCNN → crop to 160×160 → embed with FaceNet → look up against
/// the enrolled DB (from `bin/facenet/enroll.dart`) or a live
/// gallery.
///
///   dart run bin/facenet/photo_identify.dart \
///       --image PATH \
///       [--db models/facenet_db.json | --gallery faces_gallery] \
///       [--out ANNOTATED.jpg] [--min-face N] [--threshold 0.4] [--gpu]
///
/// This is the demo that closes the story from `commands.md` F1–F7:
/// F7 (enroll) built the DB from cropped gallery images; this app
/// runs the same lookup on any raw photograph — the pipeline
/// works, warts and all.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_pytorch/core/nn/vision/facenet.dart';
import 'package:dart_pytorch/core/nn/vision/facenet_loader.dart';
import 'package:dart_pytorch/core/nn/vision/mtcnn.dart';
import 'package:dart_pytorch/core/nn/vision/mtcnn_detector.dart';
import 'package:dart_pytorch/core/nn/vision/mtcnn_loader.dart';
import 'package:dart_pytorch/core/tensor/tensor.dart';
import 'package:image/image.dart' as img;

import '_common.dart';

Future<void> main(List<String> args) async {
  String? imagePath;
  var dbPath = 'models/facenet_db.json';
  String? galleryPath;
  String? outPath;
  var minFace = 40;
  var threshold = 0.4;
  var useGpu = false;
  var facenetWeights = 'models/facenet-vggface2/model.safetensors';
  var pnetPath = 'models/mtcnn/pnet.safetensors';
  var rnetPath = 'models/mtcnn/rnet.safetensors';
  var onetPath = 'models/mtcnn/onet.safetensors';

  for (int i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--image':
        imagePath = args[++i];
        break;
      case '--db':
        dbPath = args[++i];
        break;
      case '--gallery':
        galleryPath = args[++i];
        break;
      case '--out':
        outPath = args[++i];
        break;
      case '--min-face':
        minFace = int.parse(args[++i]);
        break;
      case '--threshold':
        threshold = double.parse(args[++i]);
        break;
      case '--gpu':
        useGpu = true;
        break;
      case '--facenet-weights':
        facenetWeights = args[++i];
        break;
      case '--pnet':
        pnetPath = args[++i];
        break;
      case '--rnet':
        rnetPath = args[++i];
        break;
      case '--onet':
        onetPath = args[++i];
        break;
    }
  }
  if (imagePath == null || !File(imagePath).existsSync()) {
    stderr.writeln('missing --image PATH');
    exit(2);
  }

  final device = useGpu ? Device.GPU : Device.CPU;
  final swTotal = Stopwatch()..start();

  // ---- load models ----
  print('== load MTCNN + FaceNet ${useGpu ? '(GPU)' : '(CPU)'} ==');
  final sw = Stopwatch()..start();
  final pnet = PNet();
  MTCNNLoader.loadPNet(pnet, pnetPath);
  final rnet = RNet();
  MTCNNLoader.loadRNet(rnet, rnetPath);
  final onet = ONet();
  MTCNNLoader.loadONet(onet, onetPath);
  final detector = MTCNN(
    pnet: pnet,
    rnet: rnet,
    onet: onet,
    minFaceSize: minFace,
  );

  final facenet = InceptionResnetV1(device: device);
  FaceNetLoader.loadFile(facenet, facenetWeights);
  facenet.eval();
  sw.stop();
  print('  ${sw.elapsedMilliseconds} ms');

  // ---- load DB or gallery ----
  print('');
  print('== load identity source ==');
  List<_Enrolled> enrolled;
  if (galleryPath != null && Directory(galleryPath).existsSync()) {
    final gallery = scanGallery(galleryPath, perId: 8);
    print('  gallery:  $galleryPath  (${gallery.length} identities)');
    enrolled = [];
    for (final e in gallery.entries) {
      final sum = List<double>.filled(512, 0.0);
      for (final path in e.value) {
        final emb = facenet(decodeFaceJpeg(path, device: device)).toList();
        for (int k = 0; k < 512; k++) {
          sum[k] += emb[k];
        }
      }
      _l2NormalizeInPlace(sum, e.value.length);
      enrolled.add(_Enrolled(id: e.key, embedding: sum));
    }
  } else if (File(dbPath).existsSync()) {
    final db =
        jsonDecode(File(dbPath).readAsStringSync()) as Map<String, dynamic>;
    enrolled = [
      for (final e in (db['entries'] as List))
        _Enrolled(
          id: e['id'] as String,
          embedding: (e['embedding'] as List)
              .cast<num>()
              .map((v) => v.toDouble())
              .toList(),
        ),
    ];
    print('  db:       $dbPath  (${enrolled.length} identities)');
  } else {
    stderr.writeln('either --db (existing JSON) or --gallery (dir) needed');
    exit(2);
  }

  // ---- detect ----
  print('');
  print('== detect faces ==');
  final image = img.decodeImage(File(imagePath).readAsBytesSync())!;
  final swD = Stopwatch()..start();
  final faces = detector.detect(image);
  swD.stop();
  print(
    '  ${swD.elapsedMilliseconds} ms  → ${faces.length} face(s)  '
    '(${image.width} × ${image.height}, $imagePath)',
  );
  if (faces.isEmpty) {
    print('');
    print('== no faces detected — try --min-face lower ==');
    exit(0);
  }

  // ---- embed each detected face + look up ----
  print('');
  print('== identify each face ==');
  final annotated = outPath != null ? img.Image.from(image) : null;
  for (int i = 0; i < faces.length; i++) {
    final f = faces[i];
    // Crop to 160×160 (facenet-pytorch's extract_face equivalent).
    final crop = _cropTo160(image, f.box);
    final tensor = _imageToTensor(crop, device);
    final emb = facenet(tensor).toList();
    // Rank against enrolled.
    final scored =
        enrolled.map((e) => MapEntry(e.id, _dot(emb, e.embedding))).toList()
          ..sort((a, b) => b.value.compareTo(a.value));
    final winner = scored.first;
    final matched = winner.value >= threshold;
    print(
      '  face #$i  det=${f.prob.toStringAsFixed(3)}  '
      'top=${winner.key} cos=${winner.value.toStringAsFixed(3)}  '
      '${matched ? '→ MATCH' : '→ (below threshold)'}',
    );
    if (scored.length > 1) {
      final r = scored[1];
      print('           runner-up=${r.key} cos=${r.value.toStringAsFixed(3)}');
    }

    if (annotated != null) {
      final labelText = matched
          ? '${winner.key} (${winner.value.toStringAsFixed(2)})'
          : '? (${winner.value.toStringAsFixed(2)})';
      final color = matched
          ? img.ColorRgb8(0, 255, 0)
          : img.ColorRgb8(255, 128, 0);
      img.drawRect(
        annotated,
        x1: f.box[0].round().clamp(0, image.width - 1),
        y1: f.box[1].round().clamp(0, image.height - 1),
        x2: f.box[2].round().clamp(0, image.width - 1),
        y2: f.box[3].round().clamp(0, image.height - 1),
        color: color,
        thickness: 2,
      );
      img.drawString(
        annotated,
        labelText,
        font: img.arial14,
        x: f.box[0].round().clamp(0, image.width - 20),
        y: (f.box[1] - 18).round().clamp(0, image.height - 15),
        color: color,
      );
    }
  }

  if (outPath != null && annotated != null) {
    File(outPath).writeAsBytesSync(img.encodeJpg(annotated));
    print('');
    print('== wrote annotated image ==');
    print('  $outPath');
  }

  swTotal.stop();
  print('');
  print('total wall = ${swTotal.elapsedMilliseconds} ms');
}

// ---------------- helpers ----------------

class _Enrolled {
  final String id;
  final List<double> embedding;
  _Enrolled({required this.id, required this.embedding});
}

void _l2NormalizeInPlace(List<double> v, int n) {
  var norm = 0.0;
  for (int i = 0; i < v.length; i++) {
    v[i] /= n;
    norm += v[i] * v[i];
  }
  norm = norm > 0 ? _sqrt(norm) : 1.0;
  for (int i = 0; i < v.length; i++) {
    v[i] /= norm;
  }
}

double _dot(List<double> a, List<double> b) {
  double s = 0.0;
  for (int i = 0; i < a.length; i++) {
    s += a[i] * b[i];
  }
  return s;
}

double _sqrt(double x) {
  if (x <= 0) return 0.0;
  var y = x;
  var z = (y + 1) / 2;
  for (int i = 0; i < 30 && z != y; i++) {
    y = z;
    z = (y + x / y) / 2;
  }
  return z;
}

img.Image _cropTo160(img.Image src, List<double> box) {
  final x1 = box[0].round().clamp(0, src.width);
  final y1 = box[1].round().clamp(0, src.height);
  final x2 = box[2].round().clamp(0, src.width);
  final y2 = box[3].round().clamp(0, src.height);
  final w = (x2 - x1).clamp(1, src.width);
  final h = (y2 - y1).clamp(1, src.height);
  final crop = img.copyCrop(src, x: x1, y: y1, width: w, height: h);
  return img.copyResize(
    crop,
    width: 160,
    height: 160,
    interpolation: img.Interpolation.linear,
  );
}

Tensor _imageToTensor(img.Image image, Device device) {
  final data = Float32List(3 * 160 * 160);
  const chSize = 160 * 160;
  for (int y = 0; y < 160; y++) {
    for (int x = 0; x < 160; x++) {
      final px = image.getPixel(x, y);
      final off = y * 160 + x;
      data[off] = (px.r.toDouble() - 127.5) / 128.0;
      data[chSize + off] = (px.g.toDouble() - 127.5) / 128.0;
      data[2 * chSize + off] = (px.b.toDouble() - 127.5) / 128.0;
    }
  }
  return Tensor.fromFloat32List([1, 3, 160, 160], data, device: device);
}
