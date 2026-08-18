@Timeout(Duration(minutes: 5))
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:dart_pytorch/core/nn/vision/facenet.dart';
import 'package:dart_pytorch/core/nn/vision/facenet_loader.dart';
import 'package:dart_pytorch/core/tensor/tensor.dart';
import 'package:test/test.dart';

const _weightsPath = 'models/facenet-vggface2/model.safetensors';
const _inputRawPath = '/tmp/facenet_input.raw';
const _refRawPath = '/tmp/facenet_ref.raw';

void main() {
  group('FaceNet end-to-end (GPU)', () {
    final hasAll = File(_weightsPath).existsSync() &&
        File(_inputRawPath).existsSync() &&
        File(_refRawPath).existsSync();
    if (!hasAll) {
      test('assets missing → skipped', () {}, skip: 'run facenet CPU test first');
      return;
    }

    test('GPU forward matches Python reference (cosine = 1.0)', () {
      final model = InceptionResnetV1(device: Device.GPU);
      FaceNetLoader.loadFile(model, _weightsPath);
      model.eval();

      final bytes = File(_inputRawPath).readAsBytesSync();
      final input = Float32List.view(
        bytes.buffer,
        bytes.offsetInBytes,
        3 * 160 * 160,
      );
      final x = Tensor.fromFloat32List(
        [1, 3, 160, 160],
        input,
        device: Device.GPU,
      );
      final emb = model(x).toList();
      expect(emb.length, equals(512));

      final refBytes = File(_refRawPath).readAsBytesSync();
      final ref = Float32List.view(
        refBytes.buffer,
        refBytes.offsetInBytes,
        refBytes.lengthInBytes ~/ 4,
      );

      double dot = 0.0;
      double sq = 0.0;
      for (int i = 0; i < 512; i++) {
        dot += emb[i] * ref[i];
        sq += emb[i] * emb[i];
      }
      expect(sq, closeTo(1.0, 1e-4));
      expect(dot, greaterThan(0.999));
    });
  });
}
