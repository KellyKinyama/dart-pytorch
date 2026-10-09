import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

/// Whether a working CUDA build is present. Tensor-parallel layers are
/// GPU-only, so these tests skip on CPU-only / no-GPU machines.
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

  group('TensorParallelMultiHeadAttention', () {
    test(
      'matches single-GPU MultiHeadAttention (plain MHA)',
      () {
        const embedDim = 64;
        const numHeads = 8;
        const n = 6;
        final rng = math.Random(0);

        final mha = MultiHeadAttention(
          embedDim,
          numHeads,
          bias: true,
          device: Device.GPU,
          seed: 7,
        );
        final x = Tensor.fromList(
          [n, embedDim],
          _rand(rng, n * embedDim, 0.5),
          device: Device.GPU,
        );

        final ref = mha(x).to(Device.CPU).toFloat32List();

        final tp = TensorParallelMultiHeadAttention.fromAttention(mha);
        final got = tp(x).to(Device.CPU).toFloat32List();

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

    test(
      'matches single-GPU MultiHeadAttention (GQA)',
      () {
        const embedDim = 64;
        const numHeads = 8;
        const numKvHeads = 2;
        const n = 5;
        final rng = math.Random(1);

        final mha = MultiHeadAttention(
          embedDim,
          numHeads,
          numKvHeads: numKvHeads,
          bias: false,
          device: Device.GPU,
          seed: 3,
        );
        final x = Tensor.fromList(
          [n, embedDim],
          _rand(rng, n * embedDim, 0.5),
          device: Device.GPU,
        );

        final ref = mha(x).to(Device.CPU).toFloat32List();
        final tp = TensorParallelMultiHeadAttention.fromAttention(mha);
        final got = tp(x).to(Device.CPU).toFloat32List();

        var maxDiff = 0.0;
        for (var i = 0; i < ref.length; i++) {
          final d = (got[i] - ref[i]).abs();
          if (d > maxDiff) maxDiff = d;
        }
        expect(maxDiff < 1e-3, isTrue, reason: 'max abs diff $maxDiff');
      },
      skip: gpu ? false : 'no CUDA GPU available',
    );

    test(
      'matches single-GPU MHA with an additive causal mask',
      () {
        const embedDim = 48;
        const numHeads = 6;
        const n = 5;
        final rng = math.Random(2);

        final mha = MultiHeadAttention(
          embedDim,
          numHeads,
          bias: true,
          device: Device.GPU,
          seed: 11,
        );
        final x = Tensor.fromList(
          [n, embedDim],
          _rand(rng, n * embedDim, 0.5),
          device: Device.GPU,
        );
        // Lower-triangular (causal) additive mask.
        final maskVals = List<double>.filled(n * n, 0);
        for (var i = 0; i < n; i++) {
          for (var j = 0; j < n; j++) {
            if (j > i) maskVals[i * n + j] = -1e9;
          }
        }
        final mask = Tensor.fromList([n, n], maskVals, device: Device.GPU);

        final ref = mha(x, mask: mask).to(Device.CPU).toFloat32List();
        final tp = TensorParallelMultiHeadAttention.fromAttention(mha);
        final got = tp(x, mask: mask).to(Device.CPU).toFloat32List();

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
}
