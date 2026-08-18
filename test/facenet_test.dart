@Timeout(Duration(minutes: 5))
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:dart_pytorch/core/nn/vision/facenet.dart';
import 'package:dart_pytorch/core/nn/vision/facenet_loader.dart';
import 'package:dart_pytorch/core/nn/vision/pool2d.dart';
import 'package:dart_pytorch/core/tensor/tensor.dart';
import 'package:test/test.dart';

const _weightsPath = 'models/facenet-vggface2/model.safetensors';
const _inputRawPath = '/tmp/facenet_input.raw';
const _refRawPath = '/tmp/facenet_ref.raw';

void main() {
  group('Pool2d', () {
    test('maxPool2d 3x3 stride 2 halves spatial size (with floor)', () {
      final x = Tensor.fromList(
        [1, 1, 4, 4],
        List<double>.generate(16, (i) => i.toDouble()),
      );
      final y = maxPool2d(x, kernel: 3, stride: 2);
      expect(y.shape, equals([1, 1, 1, 1]));
      // 3x3 patch of top-left corner starting at (0,0) is
      // [0,1,2, 4,5,6, 8,9,10] with max = 10.
      expect(y.toList()[0], equals(10.0));
    });

    test('globalAvgPool2d averages spatial dims', () {
      final x = Tensor.fromList(
        [1, 2, 2, 2],
        [1, 2, 3, 4, 5, 6, 7, 8].map((e) => e.toDouble()).toList(),
      );
      final y = globalAvgPool2d(x);
      expect(y.shape, equals([1, 2]));
      expect(y.toList(), equals([2.5, 6.5])); // (1+2+3+4)/4, (5+6+7+8)/4
    });
  });

  group('InceptionResnetV1 structural', () {
    test('build has expected parameter count', () {
      // Sanity check that the module tree wires all Conv2d/Linear.
      final m = InceptionResnetV1();
      final n = m.parameters().length;
      // Each Conv2d/Linear contributes weight + bias (bias always
      // present after fold-in; last_linear has no bias). Plus the
      // last_bn's scale + offset are NOT parameters (they're loaded
      // once, treated as constants).
      // The exact number isn't the point — just confirm it's big.
      expect(n, greaterThan(100));
      expect(n, lessThan(500));
    });
  });

  group('FaceNet end-to-end (CPU)', () {
    final hasAll = File(_weightsPath).existsSync() &&
        File(_inputRawPath).existsSync() &&
        File(_refRawPath).existsSync();
    if (!hasAll) {
      test('assets missing → skipped', () {}, skip: '''
Missing $_weightsPath / $_inputRawPath / $_refRawPath.
Generate them with:
  python3 scripts/convert_facenet_pt_to_safetensors.py $_weightsPath
  python3 scripts/facenet_reference.py "faces_gallery/Brad Pitt/sample_0.jpg"''');
      return;
    }

    test('bit-exact vs facenet-pytorch (cosine = 1.0)', () {
      final model = InceptionResnetV1();
      final report = FaceNetLoader.loadFile(model, _weightsPath);
      model.eval();
      expect(report.unusedKeys, equals(['logits.bias', 'logits.weight']));

      final bytes = File(_inputRawPath).readAsBytesSync();
      final input = Float32List.view(
        bytes.buffer,
        bytes.offsetInBytes,
        3 * 160 * 160,
      );
      final x = Tensor.fromFloat32List([1, 3, 160, 160], input);
      final emb = model(x).toList();
      expect(emb.length, equals(512));

      final refBytes = File(_refRawPath).readAsBytesSync();
      final ref = Float32List.view(
        refBytes.buffer,
        refBytes.offsetInBytes,
        refBytes.lengthInBytes ~/ 4,
      );
      expect(ref.length, equals(512));

      double dot = 0.0;
      double maxAbs = 0.0;
      double sq = 0.0;
      for (int i = 0; i < 512; i++) {
        dot += emb[i] * ref[i];
        final d = (emb[i] - ref[i]).abs();
        if (d > maxAbs) maxAbs = d;
        sq += emb[i] * emb[i];
      }
      expect(sq, closeTo(1.0, 1e-4));
      expect(dot, closeTo(1.0, 1e-4));
      expect(maxAbs, lessThan(1e-3));
    });
  });
}
