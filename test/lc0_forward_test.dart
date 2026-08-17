import 'dart:io';
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

const _weightsPath = 'models/lc0/744706.pb.gz';

Tensor _zeros112() => Tensor.fromList(
  [1, 112, 8, 8],
  Float32List(1 * 112 * 8 * 8).toList(),
  device: Device.CPU,
);

void main() {
  final available = File(_weightsPath).existsSync();

  group('Lc0Net forward (skipped if weights absent)', () {
    late Lc0Weights w;
    late Lc0Net net;

    setUpAll(() {
      if (!available) return;
      w = Lc0Reader.readFile(_weightsPath);
      net = Lc0Net(w);
    });

    test('runs forward on an all-zero [1,112,8,8] input', () {
      if (!available) return;
      final sw = Stopwatch()..start();
      final out = net(_zeros112());
      stdout.writeln('  forward pass: ${sw.elapsedMilliseconds} ms');

      expect(out.policyLogits.shape, [1, w.policyOutputPlanes, 8, 8]);
      expect(out.value.shape, [1, w.wdl]);

      for (final v in out.policyLogits.toList()) {
        expect(v.isFinite, isTrue);
      }
      for (final v in out.value.toList()) {
        expect(v.isFinite, isTrue);
      }
    });

    test('WDL head sums to ~1 (softmax output)', () {
      if (!available) return;
      final out = net(_zeros112());
      if (w.wdl == 3) {
        final v = out.value.toList();
        final s = v.fold<double>(0.0, (a, b) => a + b);
        expect(s, closeTo(1.0, 1e-4));
        for (final p in v) {
          expect(p, greaterThanOrEqualTo(0.0));
          expect(p, lessThanOrEqualTo(1.0));
        }
      }
    });
  });
}
