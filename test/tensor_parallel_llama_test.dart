import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

bool _gpuReady() {
  try {
    return Tensor.gpuCount >= 1;
  } catch (_) {
    return false;
  }
}

List<double> _rand(math.Random r, int n, double s) =>
    List<double>.generate(n, (_) => (r.nextDouble() * 2 - 1) * s);

double _maxDiff(List<double> a, List<double> b) {
  var m = 0.0;
  for (var i = 0; i < a.length; i++) {
    final d = (a[i] - b[i]).abs();
    if (d > m) m = d;
  }
  return m;
}

void main() {
  final gpu = _gpuReady();

  group('TensorParallelSwiGlu', () {
    test(
      'matches the single-GPU SwiGluFfn',
      () {
        const dim = 48;
        const hidden = 128;
        const n = 6;
        final rng = math.Random(0);

        final ffn = SwiGluFfn(dim, hidden, device: Device.GPU, seed: 3);
        final x = Tensor.fromList(
          [n, dim],
          _rand(rng, n * dim, 0.5),
          device: Device.GPU,
        );

        final ref = ffn(x).to(Device.CPU).toFloat32List();
        final tp = TensorParallelSwiGlu.fromSwiGlu(ffn);
        final got = tp(x).to(Device.CPU).toFloat32List();

        expect(got.length, ref.length);
        expect(_maxDiff(got, ref) < 1e-3, isTrue);
      },
      skip: gpu ? false : 'no CUDA GPU available',
    );
  });

  group('TensorParallelLlamaBlock', () {
    test(
      'matches the single-GPU LlamaBlock (GQA + RoPE + causal mask)',
      () {
        const embedDim = 48;
        const numHeads = 6;
        const numKvHeads = 2;
        const ffnDim = 128;
        const n = 5;
        final rng = math.Random(1);

        final rope = RopeCache(
          maxCtx: n,
          headDim: embedDim ~/ numHeads,
          device: Device.GPU,
        );
        final block = LlamaBlock(
          embedDim,
          numHeads,
          numKvHeads: numKvHeads,
          ffnDim: ffnDim,
          rope: rope,
          device: Device.GPU,
          seed: 5,
        );

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
        final tp = TensorParallelLlamaBlock.fromBlock(block);
        final got = tp(x, mask: mask).to(Device.CPU).toFloat32List();

        expect(_maxDiff(got, ref) < 1e-3, isTrue);
      },
      skip: gpu ? false : 'no CUDA GPU available',
    );
  });

  group('TensorParallelLlamaStack KV cache', () {
    test(
      'cached incremental decoding matches the full causal forward',
      () {
        const embedDim = 48;
        const numHeads = 6;
        const numKvHeads = 2;
        const ffnDim = 128;
        const t = 4;
        const depth = 2;
        final rng = math.Random(2);

        final rope = RopeCache(
          maxCtx: t,
          headDim: embedDim ~/ numHeads,
          device: Device.GPU,
        );
        final blocks = [
          for (var i = 0; i < depth; i++)
            LlamaBlock(
              embedDim,
              numHeads,
              numKvHeads: numKvHeads,
              ffnDim: ffnDim,
              rope: rope,
              device: Device.GPU,
              seed: 100 + i,
            ),
        ];
        final stack = TensorParallelLlamaStack.fromBlocks(blocks);

        final xData = _rand(rng, t * embedDim, 0.5);
        final xFull =
            Tensor.fromList([t, embedDim], xData, device: Device.GPU);
        final maskVals = List<double>.filled(t * t, 0);
        for (var i = 0; i < t; i++) {
          for (var j = i + 1; j < t; j++) {
            maskVals[i * t + j] = -1e9;
          }
        }
        final mask = Tensor.fromList([t, t], maskVals, device: Device.GPU);
        final full = stack(xFull, mask: mask).to(Device.CPU).toFloat32List();

        final caches = stack.newCache();
        var maxDiff = 0.0;
        for (var step = 0; step < t; step++) {
          final row = xData.sublist(step * embedDim, (step + 1) * embedDim);
          final xt = Tensor.fromList([1, embedDim], row, device: Device.GPU);
          final outT =
              stack(xt, caches: caches).to(Device.CPU).toFloat32List();
          for (var j = 0; j < embedDim; j++) {
            final d = (outT[j] - full[step * embedDim + j]).abs();
            if (d > maxDiff) maxDiff = d;
          }
        }
        expect(maxDiff < 1e-3, isTrue, reason: 'max abs diff $maxDiff');
      },
      skip: gpu ? false : 'no CUDA GPU available',
    );
  });
}
