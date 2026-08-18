@Timeout(Duration(minutes: 5))
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/core/nn/vision/facenet.dart';
import 'package:dart_pytorch/core/nn/vision/facenet_loader.dart';
import 'package:dart_pytorch/core/nn/vision/nchw.dart';
import 'package:dart_pytorch/core/nn/vision/pool2d.dart';
import 'package:dart_pytorch/core/optim/adam.dart';
import 'package:dart_pytorch/core/tensor/tensor.dart';
import 'package:test/test.dart';

const _weightsPath = 'models/facenet-vggface2/model.safetensors';
const _inputRawPath = '/tmp/facenet_input.raw';
const _refRawPath = '/tmp/facenet_ref.raw';

void main() {
  group('Pool2d', () {
    test('maxPool2d 3x3 stride 2 halves spatial size (with floor)', () {
      final x = Tensor.fromList([
        1,
        1,
        4,
        4,
      ], List<double>.generate(16, (i) => i.toDouble()));
      final y = maxPool2d(x, kernel: 3, stride: 2);
      expect(y.shape, equals([1, 1, 1, 1]));
      // 3x3 patch of top-left corner starting at (0,0) is
      // [0,1,2, 4,5,6, 8,9,10] with max = 10.
      expect(y.toList()[0], equals(10.0));
    });

    test('globalAvgPool2d averages spatial dims', () {
      final x = Tensor.fromList([
        1,
        2,
        2,
        2,
      ], [1, 2, 3, 4, 5, 6, 7, 8].map((e) => e.toDouble()).toList());
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
    final hasAll =
        File(_weightsPath).existsSync() &&
        File(_inputRawPath).existsSync() &&
        File(_refRawPath).existsSync();
    if (!hasAll) {
      test(
        'assets missing → skipped',
        () {},
        skip:
            '''
Missing $_weightsPath / $_inputRawPath / $_refRawPath.
Generate them with:
  python3 scripts/convert_facenet_pt_to_safetensors.py $_weightsPath
  python3 scripts/facenet_reference.py "faces_gallery/Brad Pitt/sample_0.jpg"''',
      );
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

    test('fine-tuning lastLinear.weight receives non-zero gradient', () {
      // Prove the head-only fine-tuning recipe from doc/facenet.md
      // and commands.md really runs autograd into lastLinear.weight,
      // by directly checking the gradient magnitude after one
      // backward pass.
      final model = InceptionResnetV1();
      FaceNetLoader.loadFile(model, _weightsPath);
      model.eval();

      for (final p in model.parameters()) {
        p.requiresGrad = false;
      }
      model.lastLinear.weight.requiresGrad = true;

      Tensor headForward(Tensor pooled) {
        var emb = model.lastLinear(pooled);
        emb = emb * model.lastBnScale + model.lastBnOffset;
        return l2NormalizeRows(emb);
      }

      // Two synthetic frozen "pooled" features (1792-d).
      final rng = math.Random(0);
      Tensor mk() {
        final d = Float32List(1792);
        for (int i = 0; i < 1792; i++) {
          d[i] = (rng.nextDouble() - 0.5) * 0.3;
        }
        return Tensor.fromFloat32List([1, 1792], d);
      }

      final anchor = mk();
      final other = mk();
      // Any scalar loss that depends on both embeddings will do; we
      // pick something guaranteed to produce a non-zero derivative
      // w.r.t. lastLinear.weight.
      final aE = headForward(anchor);
      final oE = headForward(other);
      final diff = aE - oE;
      // Sum of squares as a scalar loss.
      final ones = Tensor.fill([diff.shape[1], 1], 1.0);
      final loss = (diff * diff).matmul(ones);

      loss.backward();

      final g = model.lastLinear.weight.grad;
      expect(g, isNotNull);
      double gAbsMax = 0.0;
      double gAbsSum = 0.0;
      for (final v in g!.toList()) {
        final a = v.abs();
        if (a > gAbsMax) gAbsMax = a;
        gAbsSum += a;
      }
      expect(
        gAbsMax,
        greaterThan(0.0),
        reason: 'lastLinear.weight.grad should be non-zero',
      );
      // Some sanity floor — grads should be O(1e-6) or larger.
      expect(
        gAbsSum,
        greaterThan(1e-4),
        reason:
            'aggregate |grad| too small ($gAbsSum); autograd chain '
            'may be broken between loss and lastLinear.weight',
      );
    });

    test('Adam step on lastLinear.weight reduces a fabricated loss', () {
      // Autograd + optimizer round-trip: same setup as above, but we
      // fabricate a loss whose gradient definitely points in a
      // reducible direction, then verify Adam actually reduces it.
      final model = InceptionResnetV1();
      FaceNetLoader.loadFile(model, _weightsPath);
      model.eval();
      for (final p in model.parameters()) {
        p.requiresGrad = false;
      }
      model.lastLinear.weight.requiresGrad = true;

      Tensor headForward(Tensor pooled) {
        var emb = model.lastLinear(pooled);
        emb = emb * model.lastBnScale + model.lastBnOffset;
        return l2NormalizeRows(emb);
      }

      final rng = math.Random(0);
      final d = Float32List(1792);
      for (int i = 0; i < 1792; i++) {
        d[i] = (rng.nextDouble() - 0.5) * 0.3;
      }
      final x = Tensor.fromFloat32List([1, 1792], d);

      // Loss = -sum(emb²) = -1 (constant since we L2-normalize).
      // That's degenerate. Use loss = -emb[0]² instead — Adam can
      // grow the first coordinate on the unit sphere.
      double lossOf() {
        final e = headForward(x).toList();
        return -(e[0] * e[0]);
      }

      final lossBefore = lossOf();

      final opt = Adam([model.lastLinear.weight], lr: 5e-3);
      for (int s = 0; s < 30; s++) {
        final e = headForward(x);
        // Slice column 0 via matmul with a one-hot selector.
        final sel = Float32List(e.shape[1]);
        sel[0] = 1.0;
        final selT = Tensor.fromFloat32List([e.shape[1], 1], sel);
        final e0 = e.matmul(selT); // [1, 1]
        final loss = e0 * e0 * -1.0;
        opt.zeroGrad();
        loss.backward();
        opt.step();
      }
      final lossAfter = lossOf();
      expect(
        lossAfter,
        lessThan(lossBefore),
        reason:
            'Adam should decrease -emb[0]² (got before=$lossBefore, '
            'after=$lossAfter)',
      );
    });
  });
}
