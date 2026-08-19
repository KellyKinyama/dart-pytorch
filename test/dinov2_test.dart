@Timeout(Duration(minutes: 3))
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:dart_pytorch/core/nn/vision/dinov2.dart';
import 'package:dart_pytorch/core/nn/vision/dinov2_loader.dart';
import 'package:dart_pytorch/core/tensor/tensor.dart';
import 'package:test/test.dart';

const _weightsPath = 'models/dinov2-small/model.safetensors';

void main() {
  group('DinoV2Backbone', () {
    test('parameter list is non-empty and grows with layers', () {
      final small = DinoV2Backbone(
        imageSize: 224,
        patchSize: 14,
        embedDim: 64,
        numLayers: 2,
        numHeads: 4,
      );
      final medium = DinoV2Backbone(
        imageSize: 224,
        patchSize: 14,
        embedDim: 64,
        numLayers: 4,
        numHeads: 4,
      );
      expect(small.parameters().length, greaterThan(0));
      expect(
        medium.parameters().length,
        greaterThan(small.parameters().length),
      );
    });

    test('call produces [numPatches + 1, embedDim]', () {
      final m = DinoV2Backbone(
        imageSize: 28,
        patchSize: 14,
        embedDim: 32,
        numLayers: 1,
        numHeads: 4,
      );
      final input = Tensor.fill([4, 14 * 14 * 3], 0.0);
      final out = m(input);
      expect(out.shape, equals([5, 32])); // 4 patches + 1 CLS
    });
  });

  group('DINOv2 loader (facebook/dinov2-small)', () {
    if (!File(_weightsPath).existsSync()) {
      test(
        'weights missing → skipped',
        () {},
        skip:
            '''
Missing $_weightsPath.
Fetch: mkdir -p models/dinov2-small
       curl -L -o models/dinov2-small/model.safetensors \\
         https://huggingface.co/facebook/dinov2-small/resolve/main/model.safetensors''',
      );
      return;
    }

    test('load consumes every tensor (mask_token deliberately unused)', () {
      final model = DinoV2Backbone(
        imageSize: 224,
        patchSize: 14,
        embedDim: 384,
        numLayers: 12,
        numHeads: 6,
      );
      final report = DinoV2Loader.loadFile(model, _weightsPath);
      expect(report.consumedCount, equals(222));
      expect(report.unusedKeys, isEmpty);
    });

    test('forward on zero-image produces non-trivial CLS features', () {
      final model = DinoV2Backbone(
        imageSize: 224,
        patchSize: 14,
        embedDim: 384,
        numLayers: 12,
        numHeads: 6,
        device: Device.CPU,
      );
      DinoV2Loader.loadFile(model, _weightsPath);
      model.eval();
      final zeros = Tensor.fromFloat32List(
        [256, 14 * 14 * 3],
        Float32List(256 * 14 * 14 * 3),
        device: Device.CPU,
      );
      final feats = model(zeros).toList();
      // CLS row (first 384 elements) should not all be zero.
      double sumAbs = 0.0;
      for (int i = 0; i < 384; i++) {
        sumAbs += feats[i].abs();
      }
      expect(
        sumAbs,
        greaterThan(1.0),
        reason: 'CLS features are all near zero — loader likely broken',
      );
    });
  });
}
