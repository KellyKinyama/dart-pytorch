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

  group('TensorParallelTransformerBlock', () {
    test(
      'matches single-GPU TransformerBlock (forward, with mask)',
      () {
        const embedDim = 64;
        const numHeads = 8;
        const n = 6;
        final rng = math.Random(0);

        // Reference block on a single GPU (eval mode, no dropout).
        final block = TransformerBlock(
          embedDim,
          numHeads,
          attnBias: true,
          device: Device.GPU,
          seed: 5,
        )..eval();

        final x = Tensor.fromList(
          [n, embedDim],
          _rand(rng, n * embedDim, 0.5),
          device: Device.GPU,
        );
        final maskVals = List<double>.filled(n * n, 0);
        for (var i = 0; i < n; i++) {
          for (var j = i + 1; j < n; j++) {
            maskVals[i * n + j] = -1e9;
          }
        }
        final mask = Tensor.fromList([n, n], maskVals, device: Device.GPU);

        final ref = block(x, mask: mask).to(Device.CPU).toFloat32List();

        final tp = TensorParallelTransformerBlock.fromBlock(block);
        final got = tp(x, mask: mask).to(Device.CPU).toFloat32List();

        expect(got.length, ref.length);
        var maxDiff = 0.0;
        for (var i = 0; i < ref.length; i++) {
          final d = (got[i] - ref[i]).abs();
          if (d > maxDiff) maxDiff = d;
        }
        expect(maxDiff < 1e-3, isTrue, reason: 'max abs diff $maxDiff');
      },
      skip: gpu ? false : 'no CUDA GPU available',
    );
  });

  group('TensorParallelTransformerStack', () {
    test(
      'matches a sequence of single-GPU TransformerBlocks',
      () {
        const embedDim = 48;
        const numHeads = 6;
        const n = 5;
        const depth = 3;
        final rng = math.Random(1);

        final blocks = [
          for (var i = 0; i < depth; i++)
            TransformerBlock(
              embedDim,
              numHeads,
              attnBias: true,
              device: Device.GPU,
              seed: 100 + i,
            )..eval(),
        ];

        final x = Tensor.fromList(
          [n, embedDim],
          _rand(rng, n * embedDim, 0.5),
          device: Device.GPU,
        );
        final maskVals = List<double>.filled(n * n, 0);
        for (var i = 0; i < n; i++) {
          for (var j = i + 1; j < n; j++) {
            maskVals[i * n + j] = -1e9;
          }
        }
        final mask = Tensor.fromList([n, n], maskVals, device: Device.GPU);

        // Reference: apply the blocks sequentially on one GPU.
        var ref = x;
        for (final b in blocks) {
          ref = b(ref, mask: mask);
        }
        final refData = ref.to(Device.CPU).toFloat32List();

        final stack = TensorParallelTransformerStack.fromBlocks(blocks);
        final got = stack(x, mask: mask).to(Device.CPU).toFloat32List();

        expect(got.length, refData.length);
        var maxDiff = 0.0;
        for (var i = 0; i < refData.length; i++) {
          final d = (got[i] - refData[i]).abs();
          if (d > maxDiff) maxDiff = d;
        }
        expect(maxDiff < 1e-3, isTrue, reason: 'max abs diff $maxDiff');
      },
      skip: gpu ? false : 'no CUDA GPU available',
    );

    test(
      'pipeline split across device groups matches the reference',
      () {
        const embedDim = 48;
        const numHeads = 6;
        const n = 4;
        const depth = 4;
        final rng = math.Random(2);

        final blocks = [
          for (var i = 0; i < depth; i++)
            TransformerBlock(
              embedDim,
              numHeads,
              device: Device.GPU,
              seed: 200 + i,
            )..eval(),
        ];

        final x = Tensor.fromList(
          [n, embedDim],
          _rand(rng, n * embedDim, 0.5),
          device: Device.GPU,
        );

        var ref = x;
        for (final b in blocks) {
          ref = b(ref);
        }
        final refData = ref.to(Device.CPU).toFloat32List();

        // Two pipeline stages (clamped to the device count, so this also
        // exercises the single-GPU degenerate case safely).
        final stack =
            TensorParallelTransformerStack.fromBlocks(blocks, pipelineStages: 2);
        final got = stack(x).to(Device.CPU).toFloat32List();

        var maxDiff = 0.0;
        for (var i = 0; i < refData.length; i++) {
          final d = (got[i] - refData[i]).abs();
          if (d > maxDiff) maxDiff = d;
        }
        expect(maxDiff < 1e-3, isTrue, reason: 'max abs diff $maxDiff');
      },
      skip: gpu ? false : 'no CUDA GPU available',
    );
  });
}
