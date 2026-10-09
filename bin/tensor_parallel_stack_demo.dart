/// Multi-block tensor-parallel transformer across several GPUs — a full
/// encoder stack in one process. Each block is tensor-parallel; with
/// `--pipeline N` the blocks are also split into N pipeline stages across
/// disjoint device groups. Checked against the single-device reference.
///
///   dart run bin/tensor_parallel_stack_demo.dart            # TP only
///   dart run bin/tensor_parallel_stack_demo.dart --pipeline 2
///
/// Requires the multi-GPU native lib (rebuild libmat_mul from
/// lib/native/src/engine.cu). Runs on a single GPU too (pipeline stages
/// clamp to the device count).
library;

import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

const int _embedDim = 256;
const int _numHeads = 8;
const int _tokens = 16;
const int _depth = 6;

Future<void> main(List<String> args) async {
  await ensureNativeLib();

  var pipeline = 1;
  final pi = args.indexOf('--pipeline');
  if (pi >= 0 && pi + 1 < args.length) {
    pipeline = int.tryParse(args[pi + 1]) ?? 1;
  }

  print('=== tensor-parallel transformer stack demo ===');
  print('visible GPUs: ${Tensor.gpuCount} | depth: $_depth, embedDim: '
      '$_embedDim, heads: $_numHeads, tokens: $_tokens, pipelineStages: $pipeline');

  final rng = math.Random(0);
  final blocks = [
    for (var i = 0; i < _depth; i++)
      TransformerBlock(
        _embedDim,
        _numHeads,
        attnBias: true,
        device: Device.GPU,
        seed: 10 + i,
      )..eval(),
  ];

  final x = Tensor.fromList(
    [_tokens, _embedDim],
    List<double>.generate(_tokens * _embedDim, (_) => (rng.nextDouble() * 2 - 1) * 0.5),
    device: Device.GPU,
  );

  var ref = x;
  for (final b in blocks) {
    ref = b(ref);
  }
  final refData = ref.to(Device.CPU).toFloat32List();

  final stack =
      TensorParallelTransformerStack.fromBlocks(blocks, pipelineStages: pipeline);
  print('block output devices (pipeline hand-offs): ${stack.blockOutputDevices}');
  final got = stack(x).to(Device.CPU).toFloat32List();

  var maxDiff = 0.0;
  for (var i = 0; i < refData.length; i++) {
    final d = (got[i] - refData[i]).abs();
    if (d > maxDiff) maxDiff = d;
  }
  print('output shape: [$_tokens, $_embedDim]');
  print('max abs diff vs single-GPU reference: ${maxDiff.toStringAsExponential(3)}');
  print(maxDiff < 1e-3
      ? 'OK — $_depth-block tensor-parallel stack matches the reference.'
      : 'WARNING — mismatch exceeds tolerance.');
}
