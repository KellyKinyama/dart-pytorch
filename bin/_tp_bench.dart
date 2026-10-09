/// Tensor-parallel scaling benchmark.
///
/// Times a tensor-parallel transformer stack's forward pass across every
/// GPU count from 1..N and both gather modes (GPU-native peer copy vs the
/// portable host-staged path), reporting tokens/s, relative speedup, and
/// the per-GPU weight footprint. Warmup iterations are excluded and each
/// timed iteration reads the output to force CUDA to synchronise.
///
///   dart run bin/_tp_bench.dart
///   dart run bin/_tp_bench.dart --tokens 256 --depth 8 --embed 1024
///
/// Requires the multi-GPU native lib (rebuild libmat_mul from
/// lib/native/src/engine.cu). With one visible GPU it still reports a
/// single-device baseline.
library;

import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

int _argInt(List<String> a, String name, int fallback) {
  final i = a.indexOf(name);
  if (i >= 0 && i + 1 < a.length) return int.tryParse(a[i + 1]) ?? fallback;
  return fallback;
}

Future<void> main(List<String> args) async {
  await ensureNativeLib();

  final embedDim = _argInt(args, '--embed', 1024);
  final numHeads = _argInt(args, '--heads', 16);
  final tokens = _argInt(args, '--tokens', 128);
  final depth = _argInt(args, '--depth', 4);
  final iters = _argInt(args, '--iters', 20);
  final warmup = _argInt(args, '--warmup', 5);

  final gpus = Tensor.gpuCount;
  print('=== tensor-parallel scaling benchmark ===');
  print('visible GPUs: $gpus | embedDim: $embedDim, heads: $numHeads, '
      'tokens: $tokens, depth: $depth, iters: $iters (warmup $warmup)');

  final rng = math.Random(0);
  final blocks = [
    for (var i = 0; i < depth; i++)
      TransformerBlock(
        embedDim,
        numHeads,
        attnBias: true,
        device: Device.GPU,
        seed: 10 + i,
      )..eval(),
  ];

  // Per-block weight bytes (fp32): attention qkv+o (4·D²) + FFN (2·D·4D).
  final perBlockParams = 4 * embedDim * embedDim + 2 * embedDim * (4 * embedDim);
  final totalBytes = perBlockParams * depth * 4;
  print('model weight footprint: ${_mb(totalBytes)} MB total');

  final x = Tensor.fromList(
    [tokens, embedDim],
    List<double>.generate(
      tokens * embedDim,
      (_) => (rng.nextDouble() * 2 - 1) * 0.5,
    ),
    device: Device.GPU,
  );

  double timeForward(TensorParallelTransformerStack stack) {
    for (var w = 0; w < warmup; w++) {
      stack(x).toList(); // force sync
    }
    final sw = Stopwatch()..start();
    for (var i = 0; i < iters; i++) {
      stack(x).toList(); // D2H read synchronises the device
    }
    sw.stop();
    return sw.elapsedMicroseconds / 1e6;
  }

  print('');
  print('GPUs  gather   tokens/s   speedup   PCIe/iter   per-GPU weights');
  double? baseline;
  for (var g = 1; g <= gpus; g++) {
    final devs = List<int>.generate(g, (i) => i);
    final perGpu = _mb((totalBytes / g).round());
    for (final native in [false, true]) {
      Tensor.useGpuCollectives = native;
      final stack =
          TensorParallelTransformerStack.fromBlocks(blocks, devices: devs);
      // Actual resident memory on each shard's card (built, pre-forward).
      final mem = [for (final d in devs) _mb(Tensor.gpuMemUsed(d))];
      Tensor.resetPcieBytes();
      final secs = timeForward(stack);
      final tps = tokens * iters / secs;
      final pcie = _mb((Tensor.pcieBytesMoved / iters).round());
      baseline ??= tps;
      final speedup = tps / baseline!;
      print('${g.toString().padLeft(4)}  '
          '${(native ? 'peer' : 'host').padRight(6)}  '
          '${tps.toStringAsFixed(1).padLeft(9)}  '
          '${speedup.toStringAsFixed(2).padLeft(6)}x  '
          '${pcie.padLeft(7)} MB  '
          '~$perGpu MB');
      if (g == 1 && native) {
        // Single GPU: report measured residency once.
        print('       measured GPU memory used: ${mem.join(', ')} MB');
      }
      // Single GPU: the two gather modes are identical, skip the dup row.
      if (g == 1) break;
    }
  }
  Tensor.useGpuCollectives = true; // restore default
  print('');
  print('note: per-GPU weights ~ total/GPUs confirms tensor sharding; '
      'PCIe/iter + peer-vs-host tokens/s show the collective cost.');
}

String _mb(int bytes) => (bytes / (1024 * 1024)).toStringAsFixed(1);
