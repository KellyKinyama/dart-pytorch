/// MTCNN face-detection demo. Given an input image, prints detected
/// faces (box, probability, 5 landmarks) and optionally writes an
/// annotated JPEG.
///
///   dart run bin/facenet/detect.dart --image PATH \
///       [--out ANNOTATED.jpg] [--min-face N]
///
/// Uses the safetensors dumped by
/// `scripts/convert_mtcnn_pt_to_safetensors.py` (default location
/// `models/mtcnn/{pnet,rnet,onet}.safetensors`).
library;

import 'dart:io';

import 'package:dart_pytorch/core/nn/vision/mtcnn.dart';
import 'package:dart_pytorch/core/nn/vision/mtcnn_detector.dart';
import 'package:dart_pytorch/core/nn/vision/mtcnn_loader.dart';
import 'package:image/image.dart' as img;

Future<void> main(List<String> args) async {
  String? imagePath;
  String? outPath;
  var minFace = 20;
  var pnetPath = 'models/mtcnn/pnet.safetensors';
  var rnetPath = 'models/mtcnn/rnet.safetensors';
  var onetPath = 'models/mtcnn/onet.safetensors';
  for (int i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--image':
        imagePath = args[++i];
        break;
      case '--out':
        outPath = args[++i];
        break;
      case '--min-face':
        minFace = int.parse(args[++i]);
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
  for (final p in [pnetPath, rnetPath, onetPath]) {
    if (!File(p).existsSync()) {
      stderr.writeln('missing weights: $p');
      exit(2);
    }
  }

  final swTotal = Stopwatch()..start();

  print('== build + load ==');
  final swLoad = Stopwatch()..start();
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
  swLoad.stop();
  print(
    '  ${swLoad.elapsedMilliseconds} ms  '
    '(minFaceSize=$minFace)',
  );

  print('');
  print('== read image ==');
  final image = img.decodeImage(File(imagePath).readAsBytesSync())!;
  print('  ${image.width} × ${image.height}  $imagePath');

  print('');
  print('== detect ==');
  final swD = Stopwatch()..start();
  final faces = detector.detect(image);
  swD.stop();
  print('  ${swD.elapsedMilliseconds} ms  → ${faces.length} face(s)');
  for (int i = 0; i < faces.length; i++) {
    final f = faces[i];
    print(
      '  #$i  prob=${f.prob.toStringAsFixed(3)}   '
      'box=[${f.box.map((v) => v.toStringAsFixed(1)).join(", ")}]',
    );
    for (int p = 0; p < f.landmarks.length; p++) {
      final lm = f.landmarks[p];
      final name = const [
        'leftEye ',
        'rightEye',
        'nose    ',
        'mLeft   ',
        'mRight  ',
      ][p];
      print(
        '        $name  (${lm[0].toStringAsFixed(1)}, '
        '${lm[1].toStringAsFixed(1)})',
      );
    }
  }

  if (outPath != null && faces.isNotEmpty) {
    final annotated = img.Image.from(image);
    for (final f in faces) {
      img.drawRect(
        annotated,
        x1: f.box[0].round().clamp(0, image.width - 1),
        y1: f.box[1].round().clamp(0, image.height - 1),
        x2: f.box[2].round().clamp(0, image.width - 1),
        y2: f.box[3].round().clamp(0, image.height - 1),
        color: img.ColorRgb8(0, 255, 0),
        thickness: 2,
      );
      for (final lm in f.landmarks) {
        img.fillCircle(
          annotated,
          x: lm[0].round().clamp(0, image.width - 1),
          y: lm[1].round().clamp(0, image.height - 1),
          radius: 2,
          color: img.ColorRgb8(255, 0, 0),
        );
      }
    }
    File(outPath).writeAsBytesSync(img.encodeJpg(annotated));
    print('');
    print('== wrote annotated image ==');
    print('  $outPath');
  }

  swTotal.stop();
  print('');
  print('total wall = ${swTotal.elapsedMilliseconds} ms');
}
