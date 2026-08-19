/// DDPM primitives demo — showcases the noise schedule and forward
/// diffusion from [lib/core/nn/diffusion.dart] on a synthetic 32×32
/// grayscale "checkerboard" image.
///
///   dart run bin/ddpm_demo.dart
///
/// Prints the per-timestep signal / noise ratio (`sqrt(αb)` vs
/// `sqrt(1-αb)`) at t ∈ {0, 100, 500, 999} for the default T=1000
/// linear schedule, then runs `forwardDiffuse` at each of those
/// timesteps and reports the resulting sample statistics
/// (mean, variance, min, max). Also runs a forward pass through
/// [TinyUNet] to sanity-check the U-Net wiring (the U-Net is
/// untrained here — this only demonstrates the machinery).
///
/// This demo is inference-only; DDPM training-from-scratch requires
/// Conv2d input-gradient support (currently `Conv2d` only backprops
/// into its own weights, not into its input).
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

void main(List<String> args) {
  // Optional --seed override for reproducible output.
  var seed = 42;
  for (int i = 0; i < args.length; i++) {
    if (args[i] == '--seed') seed = int.parse(args[++i]);
  }

  print('== DDPM noise schedule (linear, T=1000, β∈[1e-4, 2e-2]) ==');
  final sch = NoiseSchedule.linear();
  const ts = [0, 100, 500, 999];
  print(
    '  ${'t'.padLeft(4)}  ${'β_t'.padLeft(9)}  ${'α̅_t'.padLeft(10)}  '
    '${'√α̅_t'.padLeft(10)}  ${'√(1-α̅_t)'.padLeft(12)}',
  );
  for (final t in ts) {
    print(
      '  ${t.toString().padLeft(4)}  '
      '${sch.betas[t].toStringAsExponential(2).padLeft(9)}  '
      '${sch.alphaBars[t].toStringAsFixed(6).padLeft(10)}  '
      '${sch.sqrtAlphaBars[t].toStringAsFixed(6).padLeft(10)}  '
      '${sch.sqrtOneMinusAlphaBars[t].toStringAsFixed(6).padLeft(12)}',
    );
  }

  print('');
  print('== forward diffusion on a 32×32 synthetic checkerboard ==');
  const size = 32;
  final buf = Float32List(size * size);
  for (int y = 0; y < size; y++) {
    for (int x = 0; x < size; x++) {
      buf[y * size + x] = ((x ~/ 4 + y ~/ 4) % 2 == 0) ? 1.0 : -1.0;
    }
  }
  final x0 = Tensor.fromFloat32List([1, 1, size, size], buf);
  print('  x0 stats: ${_stats(x0.toList())}');
  for (final t in ts) {
    final r = sch.forwardDiffuse(x0, t, seed: seed + t);
    print(
      '  t=${t.toString().padLeft(4)}  '
      'x_t stats: ${_stats(r.xT.toList())}',
    );
  }

  print('');
  print('== TinyUNet forward smoke test (untrained) ==');
  final unet = TinyUNet(hidden: 16, totalTimesteps: 1000);
  final swU = Stopwatch()..start();
  final noise = unet(x0, 500);
  swU.stop();
  print(
    '  input ${x0.shape}  ->  output ${noise.shape}  '
    '(${swU.elapsedMilliseconds} ms)',
  );
  print('  predicted-noise stats: ${_stats(noise.toList())}');
  print('');
  print(
    '  Note: the U-Net is randomly initialised in this demo; to '
    'actually denoise\n  you\'d train it on a dataset with '
    '`ε̂ ↔ ε` MSE regression, which needs Conv2d\n  input-grad '
    '(currently missing — inference only in this repo).',
  );
}

String _stats(List<double> vals) {
  var min = double.infinity;
  var max = -double.infinity;
  double sum = 0;
  double sqSum = 0;
  for (final v in vals) {
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
    sqSum += v * v;
  }
  final mean = sum / vals.length;
  final variance = sqSum / vals.length - mean * mean;
  final std = math.sqrt(math.max(0, variance));
  return 'mean=${mean.toStringAsFixed(3)}  std=${std.toStringAsFixed(3)}  '
      'range=[${min.toStringAsFixed(2)}, ${max.toStringAsFixed(2)}]';
}
