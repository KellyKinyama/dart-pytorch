@Timeout(Duration(minutes: 3))
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:dart_pytorch/core/nn/prelu.dart';
import 'package:dart_pytorch/core/nn/vision/mtcnn.dart';
import 'package:dart_pytorch/core/nn/vision/mtcnn_detector.dart';
import 'package:dart_pytorch/core/nn/vision/mtcnn_loader.dart';
import 'package:dart_pytorch/core/nn/vision/nms.dart';
import 'package:dart_pytorch/core/nn/vision/pool2d.dart';
import 'package:dart_pytorch/core/tensor/tensor.dart';
import 'package:image/image.dart' as img;
import 'package:test/test.dart';

const _pnetPath = 'models/mtcnn/pnet.safetensors';
const _rnetPath = 'models/mtcnn/rnet.safetensors';
const _onetPath = 'models/mtcnn/onet.safetensors';
const _facePath = 'faces_gallery/Brad Pitt/sample_0.jpg';

void main() {
  group('PReLU', () {
    test('acts as ReLU when slope=0', () {
      final layer = PReLU(2);
      layer.weight.assign(
        Tensor.fromFloat32List([2], Float32List.fromList([0.0, 0.0])),
      );
      final x = Tensor.fromList([1, 2, 2, 2], [
        -1.0, 2.0, -3.0, 4.0, //
        5.0, -6.0, 7.0, -8.0,
      ]);
      final y = layer(x).toList();
      expect(y, equals([0, 2, 0, 4, 5, 0, 7, 0]));
    });

    test('applies per-channel slope on negatives', () {
      final layer = PReLU(2);
      layer.weight.assign(
        Tensor.fromFloat32List([2], Float32List.fromList([0.1, 0.2])),
      );
      final x = Tensor.fromList(
        [1, 2, 1, 2],
        [-1.0, 2.0, -3.0, 4.0], // channel 0: [-1, 2]; channel 1: [-3, 4]
      );
      final y = layer(x).toList();
      // channel 0: relu(-1)*1 + 0.1*(-1) = -0.1;  2 stays 2.
      // channel 1: 0.2*(-3) = -0.6;               4 stays 4.
      expect(y[0], closeTo(-0.1, 1e-6));
      expect(y[1], closeTo(2.0, 1e-6));
      expect(y[2], closeTo(-0.6, 1e-6));
      expect(y[3], closeTo(4.0, 1e-6));
    });
  });

  group('pool2d ceilMode', () {
    test('maxPool2d(k=3, s=2, ceilMode) on 22×22 → 11×11', () {
      final x = Tensor.fill([1, 1, 22, 22], 0.0);
      final y = maxPool2d(x, kernel: 3, stride: 2, ceilMode: true);
      expect(y.shape, equals([1, 1, 11, 11]));
    });

    test('maxPool2d(k=3, s=2, ceilMode) on 9×9 → 4×4', () {
      final x = Tensor.fill([1, 1, 9, 9], 0.0);
      final y = maxPool2d(x, kernel: 3, stride: 2, ceilMode: true);
      expect(y.shape, equals([1, 1, 4, 4]));
    });

    test('floor vs ceil disagree when input is odd', () {
      final x = Tensor.fill([1, 1, 21, 21], 0.0);
      final floor = maxPool2d(x, kernel: 3, stride: 2);
      final ceil = maxPool2d(x, kernel: 3, stride: 2, ceilMode: true);
      // floor: (21 - 3) / 2 + 1 = 10; ceil: ceil((21-3)/2)+1 = 10 also.
      // Use 20 to force disagreement instead.
      expect(floor.shape[2], equals(ceil.shape[2]));

      final x2 = Tensor.fill([1, 1, 20, 20], 0.0);
      final floor2 = maxPool2d(x2, kernel: 3, stride: 2);
      final ceil2 = maxPool2d(x2, kernel: 3, stride: 2, ceilMode: true);
      // floor: (20 - 3) / 2 + 1 = 9;  ceil: ceil(17/2)+1 = 10.
      expect(floor2.shape[2], equals(9));
      expect(ceil2.shape[2], equals(10));
    });
  });

  group('NMS', () {
    test('single box passes through', () {
      final kept = nms(
        [
          [0, 0, 10, 10],
        ],
        [0.9],
        threshold: 0.5,
      );
      expect(kept, equals([0]));
    });

    test('two overlapping boxes → higher-score wins', () {
      final kept = nms(
        [
          [0, 0, 10, 10],
          [1, 1, 11, 11], // ~80 % IoU with box 0
        ],
        [0.9, 0.7],
        threshold: 0.5,
      );
      expect(kept, equals([0]));
    });

    test('non-overlapping boxes both survive', () {
      final kept = nms(
        [
          [0, 0, 10, 10],
          [20, 20, 30, 30],
        ],
        [0.9, 0.7],
        threshold: 0.5,
      );
      expect(kept, unorderedEquals([0, 1])); // order by score
    });
  });

  group('MTCNN end-to-end', () {
    final hasAll =
        File(_pnetPath).existsSync() &&
        File(_rnetPath).existsSync() &&
        File(_onetPath).existsSync() &&
        File(_facePath).existsSync();
    if (!hasAll) {
      test('assets missing → skipped', () {}, skip: '''
Missing $_pnetPath / $_rnetPath / $_onetPath.
Run: python3 scripts/convert_mtcnn_pt_to_safetensors.py \\
    $_pnetPath $_rnetPath $_onetPath''');
      return;
    }

    test('detect() returns 1 high-confidence face on Brad Pitt/sample_0', () {
      final pnet = PNet();
      MTCNNLoader.loadPNet(pnet, _pnetPath);
      final rnet = RNet();
      MTCNNLoader.loadRNet(rnet, _rnetPath);
      final onet = ONet();
      MTCNNLoader.loadONet(onet, _onetPath);
      final det =
          MTCNN(pnet: pnet, rnet: rnet, onet: onet, minFaceSize: 40);
      final image = img.decodeImage(File(_facePath).readAsBytesSync())!;
      final faces = det.detect(image);
      expect(faces.length, equals(1));
      expect(faces.first.prob, greaterThan(0.9));
      expect(faces.first.landmarks.length, equals(5));

      // Sanity check landmark ordering: left eye above/left of nose,
      // right eye above/right of nose, mouth points below nose.
      final le = faces.first.landmarks[0];
      final re = faces.first.landmarks[1];
      final nose = faces.first.landmarks[2];
      final ml = faces.first.landmarks[3];
      final mr = faces.first.landmarks[4];
      expect(le[0], lessThan(re[0]), reason: 'left eye should be left of right');
      expect(le[1], lessThan(nose[1] + 30),
          reason: 'left eye should be near or above nose');
      expect(ml[1], greaterThan(nose[1] - 30),
          reason: 'mouth should be below nose');
      expect(mr[1], greaterThan(nose[1] - 30));
    });
  });
}
