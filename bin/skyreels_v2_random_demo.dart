/// Aspirational SkyReels-V2 DiT random-weight validator.
///
/// Runs a single forward through a random-initialised
/// [SkyReelsV2Model] to prove the port compiles and produces finite
/// outputs — not to render actual videos. See
/// [doc/skyreels_v2_port.md](../doc/skyreels_v2_port.md) for what
/// this port doesn't cover.
///
///   dart run bin/skyreels_v2_random_demo.dart               # 1.3B, tiny video
///   dart run bin/skyreels_v2_random_demo.dart --preset 14b  # 14B config
///   dart run bin/skyreels_v2_random_demo.dart --tokens 64 --text-len 8
library;

import 'dart:io';
import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

Future<void> main(List<String> args) async {
  var preset = '1.3b';
  var tokens = 32;
  var textLen = 8;
  var step = 500.0;
  var seed = 0;
  for (int i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--preset':
        preset = args[++i];
        break;
      case '--tokens':
        tokens = int.parse(args[++i]);
        break;
      case '--text-len':
        textLen = int.parse(args[++i]);
        break;
      case '--step':
        step = double.parse(args[++i]);
        break;
      case '--seed':
        seed = int.parse(args[++i]);
        break;
      case '-h':
      case '--help':
        stdout.writeln(
          'usage: skyreels_v2_random_demo [--preset 1.3b|14b] '
          '[--tokens N] [--text-len N] [--step F] [--seed N]',
        );
        return;
    }
  }

  final cfg = switch (preset.toLowerCase()) {
    '1.3b' || 'df1.3b' => SkyReelsV2Config.df1_3B(seed: seed),
    '14b' || 'df14b' => SkyReelsV2Config.df14B(seed: seed),
    _ => throw ArgumentError('unknown preset "$preset" (use 1.3b or 14b)'),
  };

  print('== SkyReels-V2 DiT (aspirational port) ==');
  print('  preset  : $preset');
  print('  dim     : ${cfg.dim}');
  print('  layers  : ${cfg.numLayers}');
  print('  heads   : ${cfg.numHeads} (headDim ${cfg.headDim})');
  print('  ffn     : ${cfg.ffnDim}');
  print('  textDim : ${cfg.textDim}');
  print('  outDim  : ${cfg.outDim}');
  print('  patch   : ${cfg.patchSize} (product ${cfg.patchProduct})');
  print('  tokens  : $tokens (video patches)');
  print('  textLen : $textLen (text tokens)');

  final swInit = Stopwatch()..start();
  final model = SkyReelsV2Model(cfg);
  swInit.stop();
  print(
    '  init    : ${swInit.elapsedMilliseconds} ms '
    '(fp32 random weights)',
  );

  final rng = math.Random(seed);
  final patchWidth = cfg.inDim * cfg.patchProduct;
  final patchesData = List<double>.generate(
    tokens * patchWidth,
    (_) => (rng.nextDouble() - 0.5) * 0.1,
  );
  final patches = Tensor.fromList(
    [tokens, patchWidth],
    patchesData,
    device: Device.CPU,
  );
  final textData = List<double>.generate(
    textLen * cfg.textDim,
    (_) => (rng.nextDouble() - 0.5) * 0.1,
  );
  final text = Tensor.fromList(
    [textLen, cfg.textDim],
    textData,
    device: Device.CPU,
  );

  print('');
  print('== forward ==');
  final swF = Stopwatch()..start();
  final y = Tensor.noGrad(() => model(patches, text, step));
  swF.stop();
  print('  wall    : ${swF.elapsedMilliseconds} ms');
  print(
    '  output  : ${y.shape} '
    '(expected [$tokens, ${cfg.patchProduct * cfg.outDim}])',
  );

  final row = y.toList();
  var mn = double.infinity;
  var mx = double.negativeInfinity;
  var sum = 0.0;
  var nan = 0;
  for (final v in row) {
    if (v.isNaN) {
      nan++;
      continue;
    }
    if (v < mn) mn = v;
    if (v > mx) mx = v;
    sum += v;
  }
  final finite = row.length - nan;
  print(
    '  stats   : min=${mn.toStringAsFixed(3)} '
    'max=${mx.toStringAsFixed(3)} '
    'mean=${(sum / (finite == 0 ? 1 : finite)).toStringAsFixed(5)} '
    'nan=$nan/${row.length}',
  );
  if (nan > 0) {
    stderr.writeln('WARNING: forward produced NaN(s).');
  } else {
    print(
      '  status  : OK — SkyReels-V2 DiT skeleton compiles and '
      'produces finite outputs.',
    );
  }
  print('');
  print(
    'This does NOT render video. See doc/skyreels_v2_port.md for '
    'what a real port would need on top of this skeleton (3D VAE, '
    'T5, UniPC scheduler, 3D RoPE, patch Conv3D, i2v cross-attn, '
    'diffusion forcing).',
  );
}
