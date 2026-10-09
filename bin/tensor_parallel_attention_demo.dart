/// Tensor-parallel multi-head attention across several GPUs — one layer,
/// one process. Shards a [MultiHeadAttention]'s heads across every
/// visible GPU and checks the output against the single-device layer.
///
///   dart run bin/tensor_parallel_attention_demo.dart
///
/// Requires the multi-GPU native lib (rebuild libmat_mul from
/// lib/native/src/engine.cu). On a single visible GPU it still runs,
/// degenerating to ordinary attention on device 0.
library;

import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

const int _embedDim = 256;
const int _numHeads = 8;
const int _numKvHeads = 8; // set < _numHeads for GQA
const int _tokens = 16;

Future<void> main() async {
  await ensureNativeLib();

  final gpus = Tensor.gpuCount;
  print('=== tensor-parallel attention demo ===');
  print('visible GPUs: $gpus | embedDim: $_embedDim, heads: $_numHeads, '
      'kvHeads: $_numKvHeads, tokens: $_tokens');

  final rng = math.Random(0);
  final mha = MultiHeadAttention(
    _embedDim,
    _numHeads,
    numKvHeads: _numKvHeads,
    bias: true,
    device: Device.GPU,
    seed: 42,
  );

  final x = Tensor.fromList(
    [_tokens, _embedDim],
    List<double>.generate(_tokens * _embedDim, (_) => (rng.nextDouble() * 2 - 1) * 0.5),
    device: Device.GPU,
  );

  // Causal mask so the demo exercises the masked path too.
  final maskVals = List<double>.filled(_tokens * _tokens, 0);
  for (var i = 0; i < _tokens; i++) {
    for (var j = i + 1; j < _tokens; j++) {
      maskVals[i * _tokens + j] = -1e9;
    }
  }
  final mask = Tensor.fromList([_tokens, _tokens], maskVals, device: Device.GPU);

  final ref = mha(x, mask: mask).to(Device.CPU).toFloat32List();

  final tp = TensorParallelMultiHeadAttention.fromAttention(mha);
  print('heads split across GPUs: '
      '${tp.shards.map((s) => '${s.qCount}@gpu${s.device}').toList()}');
  final got = tp(x, mask: mask).to(Device.CPU).toFloat32List();

  var maxDiff = 0.0;
  for (var i = 0; i < ref.length; i++) {
    final d = (got[i] - ref[i]).abs();
    if (d > maxDiff) maxDiff = d;
  }
  print('output shape: [$_tokens, $_embedDim]');
  print('max abs diff vs single-GPU reference: ${maxDiff.toStringAsExponential(3)}');
  print(maxDiff < 1e-3
      ? 'OK — tensor-parallel attention matches the reference.'
      : 'WARNING — mismatch exceeds tolerance.');
}
