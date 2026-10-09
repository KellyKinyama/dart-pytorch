/// Tensor-parallel transformer block across several GPUs — one full
/// pre-LN encoder block (attention + MLP) sharded over every visible GPU
/// in a single process, checked against the single-device block.
///
///   dart run bin/tensor_parallel_transformer_demo.dart
///
/// Requires the multi-GPU native lib (rebuild libmat_mul from
/// lib/native/src/engine.cu). Runs on a single GPU too.
library;

import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

const int _embedDim = 256;
const int _numHeads = 8;
const int _tokens = 16;

Future<void> main() async {
  await ensureNativeLib();

  print('=== tensor-parallel transformer block demo ===');
  print('visible GPUs: ${Tensor.gpuCount} | embedDim: $_embedDim, '
      'heads: $_numHeads, tokens: $_tokens');

  final rng = math.Random(0);
  final block = TransformerBlock(
    _embedDim,
    _numHeads,
    attnBias: true,
    device: Device.GPU,
    seed: 7,
  )..eval();

  final x = Tensor.fromList(
    [_tokens, _embedDim],
    List<double>.generate(_tokens * _embedDim, (_) => (rng.nextDouble() * 2 - 1) * 0.5),
    device: Device.GPU,
  );
  final maskVals = List<double>.filled(_tokens * _tokens, 0);
  for (var i = 0; i < _tokens; i++) {
    for (var j = i + 1; j < _tokens; j++) {
      maskVals[i * _tokens + j] = -1e9;
    }
  }
  final mask = Tensor.fromList([_tokens, _tokens], maskVals, device: Device.GPU);

  final ref = block(x, mask: mask).to(Device.CPU).toFloat32List();

  final tp = TensorParallelTransformerBlock.fromBlock(block);
  print('attention heads across GPUs: '
      '${tp.mha.shards.map((s) => '${s.qCount}@gpu${s.device}').toList()}');
  print('MLP up/down across GPUs: ${tp.mlp.up.devices} / ${tp.mlp.down.devices}');
  final got = tp(x, mask: mask).to(Device.CPU).toFloat32List();

  var maxDiff = 0.0;
  for (var i = 0; i < ref.length; i++) {
    final d = (got[i] - ref[i]).abs();
    if (d > maxDiff) maxDiff = d;
  }
  print('output shape: [$_tokens, $_embedDim]');
  print('max abs diff vs single-GPU reference: ${maxDiff.toStringAsExponential(3)}');
  print(maxDiff < 1e-3
      ? 'OK — tensor-parallel transformer block matches the reference.'
      : 'WARNING — mismatch exceeds tolerance.');
}
