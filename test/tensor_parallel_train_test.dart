import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

/// Tensor-parallel layers are GPU-only; skip on CPU-only / no-GPU hosts.
bool _gpuReady() {
  try {
    return Tensor.gpuCount >= 1;
  } catch (_) {
    return false;
  }
}

List<double> _rand(math.Random r, int n, double s) =>
    List<double>.generate(n, (_) => (r.nextDouble() * 2 - 1) * s);

void main() {
  final gpu = _gpuReady();

  group('TensorParallelMLP training (distributed backward)', () {
    test(
      'gradients reach every sharded weight and loss decreases',
      () {
        const model = 32;
        const hidden = 128;
        const tokens = 4;
        const steps = 40;
        final rng = math.Random(0);

        final upW = Tensor.fromList(
          [hidden, model],
          _rand(rng, hidden * model, 0.1),
        );
        final downW = Tensor.fromList(
          [model, hidden],
          _rand(rng, model * hidden, 0.1),
        );

        final mlp = TensorParallelMLP.fromWeights(
          upWeight: upW,
          downWeight: downW,
          trainable: true,
        );

        // At least the two weight shards must be exposed as parameters.
        expect(mlp.parameters().length, greaterThanOrEqualTo(2));

        final outDev = mlp.down.outputDevice;
        final x = Tensor.fromList(
          [tokens, model],
          _rand(rng, tokens * model, 0.5),
          device: Device.GPU,
        ).toGpu(outDev);
        final target = Tensor.fromList(
          [tokens, model],
          _rand(rng, tokens * model, 0.5),
          device: Device.GPU,
        ).toGpu(outDev);

        final opt = SGD(mlp.parameters(), lr: 0.1, momentum: 0.9);

        double? first;
        var last = double.infinity;
        for (var s = 0; s < steps; s++) {
          final out = mlp(x);
          final diff = out - target;
          final loss = (diff * diff).mean();
          final v = loss.toList().first;
          first ??= v;
          last = v;

          opt.zeroGrad();
          loss.backward();

          // Every trainable shard received a gradient on its own card.
          for (final p in mlp.parameters()) {
            expect(p.grad, isNotNull);
          }
          opt.step();
        }

        expect(last < first! * 0.5, isTrue,
            reason: 'loss $first -> $last did not halve');
      },
      skip: gpu ? false : 'no CUDA GPU available',
    );
  });
}
