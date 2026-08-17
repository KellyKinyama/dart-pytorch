import 'dart:io';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

const _weightsPath = 'models/lc0/744706.pb.gz';

void main() {
  final available = File(_weightsPath).existsSync();

  group('Lc0Reader on 744706.pb.gz (skipped if weights absent)', () {
    late Lc0Weights w;

    setUpAll(() {
      if (!available) return;
      w = Lc0Reader.readFile(_weightsPath);
    });

    test('recognises a classical 128x10 net', () {
      if (!available) return;
      expect(w.filters, 128);
      expect(w.numBlocks, 10);
    });

    test('input conv weights shape is [128, 112, 3, 3]', () {
      if (!available) return;
      expect(w.input.weights.shape, [128, 112, 3, 3]);
    });

    test('each residual block has two [128, 128, 3, 3] convs', () {
      if (!available) return;
      for (int i = 0; i < w.residual.length; i++) {
        expect(w.residual[i].conv1.weights.shape, [128, 128, 3, 3]);
        expect(w.residual[i].conv2.weights.shape, [128, 128, 3, 3]);
      }
    });

    test('policy head is (128 3x3 -> policyOutputPlanes 3x3)', () {
      if (!available) return;
      expect(w.policy1.weights.shape, [128, 128, 3, 3]);
      expect(w.policyOut.weights.shape[1], 128);
      expect(w.policyOut.weights.shape[2], 3);
      expect(w.policyOut.weights.shape[3], 3);
      expect(w.policyOutputPlanes, greaterThanOrEqualTo(64));
    });

    test('value head: 1x1 conv to Vf filters, then FC -> FC -> WDL', () {
      if (!available) return;
      expect(w.valueConv.weights.shape[1], 128);
      expect(w.valueConv.weights.shape[2], 1);
      expect(w.valueConv.weights.shape[3], 1);

      expect(w.ip1ValW.shape[0], w.valueFCUnits);
      expect(w.ip1ValW.shape[1], w.valueFilters * 64);
      expect(w.ip1ValB.shape, [w.valueFCUnits]);

      expect(w.wdl, anyOf(equals(1), equals(3)));
      expect(w.ip2ValW.shape, [w.wdl, w.valueFCUnits]);
      expect(w.ip2ValB.shape, [w.wdl]);
    });

    test('BN vectors have per-channel shape', () {
      if (!available) return;
      expect(w.input.bnMeans.shape, [128]);
      expect(w.input.bnStddivs.shape, [128]);
      expect(w.input.bnGammas.shape, [128]);
      expect(w.input.bnBetas.shape, [128]);
    });

    test('dequantized values are finite', () {
      if (!available) return;
      for (final v in w.input.weights.toList()) {
        expect(v.isFinite, isTrue);
      }
      for (final v in w.residual.last.conv2.weights.toList()) {
        expect(v.isFinite, isTrue);
      }
    });
  });
}
