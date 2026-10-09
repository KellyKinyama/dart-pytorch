/// Shard a real Llama-family checkpoint (SmolLM2 by default) across GPUs
/// and match its logits — exercising the GQA + RoPE + SwiGLU + RMSNorm
/// tensor-parallel path end to end.
///
///   dart run bin/tensor_parallel_llama_demo.dart --model path/to/model.safetensors
///   dart run bin/tensor_parallel_llama_demo.dart --model ... --config smollm2-360m
///
/// Loads the model onto GPU 0, builds a tensor-parallel Llama stack from
/// its decoder blocks across every visible GPU, then runs
/// embedding -> sharded blocks -> final RMSNorm -> tied lm_head and
/// compares the logits + next-token argmax against the single-GPU forward.
///
/// Requires the multi-GPU native lib (rebuild libmat_mul from
/// lib/native/src/engine.cu) and a local Llama/SmolLM2 safetensors file.
library;

import 'package:dart_pytorch/dart_pytorch.dart';

String? _arg(List<String> a, String name) {
  final i = a.indexOf(name);
  return (i >= 0 && i + 1 < a.length) ? a[i + 1] : null;
}

int _argMax(List<double> row) {
  var best = 0;
  for (var i = 1; i < row.length; i++) {
    if (row[i] > row[best]) best = i;
  }
  return best;
}

Future<void> main(List<String> args) async {
  await ensureNativeLib();

  final modelPath = _arg(args, '--model');
  if (modelPath == null) {
    print('usage: dart run bin/tensor_parallel_llama_demo.dart '
        '--model <model.safetensors> [--config smollm2-135m|smollm2-360m]');
    return;
  }
  final configName = _arg(args, '--config') ?? 'smollm2-135m';
  // Keep the RoPE table small for a short prompt (one copy per shard).
  const promptLen = 4;

  final cfg = switch (configName) {
    'smollm2-135m' =>
      LlamaHFLoader.smollm2_135mConfig(device: Device.GPU, maxCtx: 64),
    'smollm2-360m' =>
      LlamaHFLoader.smollm2_360mConfig(device: Device.GPU, maxCtx: 64),
    _ => throw ArgumentError('unknown --config $configName'),
  };

  print('=== tensor-parallel Llama demo ===');
  print('visible GPUs: ${Tensor.gpuCount} | config: $configName '
      '(D=${cfg.embedDim}, heads=${cfg.numHeads}, kvHeads=${cfg.numKvHeads}, '
      'layers=${cfg.numLayers}, ffn=${cfg.ffnDim})');

  final model = Llama(cfg);
  print('loading weights from $modelPath ...');
  LlamaHFLoader.loadFile(model, modelPath);
  model.eval();

  final ids = <double>[1, 2, 3, 4].sublist(0, promptLen);
  final n = ids.length;
  final tokens = Tensor.fromList([n], ids, device: Device.GPU);

  // Reference logits: the stock single-GPU forward.
  final ref = Tensor.noGrad(() => model(tokens)).to(Device.CPU).toFloat32List();

  // Sharded forward: reuse embedding, final norm, and tied head; replace
  // the decoder stack with a tensor-parallel one.
  final tp = TensorParallelLlamaStack.fromBlocks(model.blocks);
  print('blocks sharded across GPUs (output devices): ${tp.blockOutputDevices}');
  print('attention KV-heads per shard (block 0): '
      '${tp.blocks.first.attn.shards.map((s) => 'q${s.qCount}@gpu${s.device}').toList()}');

  final got = Tensor.noGrad(() {
    var h = model.embedIn(tokens);
    final maskVals = List<double>.filled(n * n, 0);
    for (var i = 0; i < n; i++) {
      for (var j = i + 1; j < n; j++) {
        maskVals[i * n + j] = -1e9;
      }
    }
    final mask = Tensor.fromList([n, n], maskVals, device: Device.GPU);
    h = tp(h, mask: mask);
    h = model.finalNorm(h);
    return cfg.tieWeights
        ? h.matmul(model.embedIn.weight.transpose())
        : model.untiedHead!(h);
  }).to(Device.CPU).toFloat32List();

  final vocab = cfg.vocabSize;
  final maxDiff = () {
    var m = 0.0;
    for (var i = 0; i < ref.length; i++) {
      final d = (got[i] - ref[i]).abs();
      if (d > m) m = d;
    }
    return m;
  }();
  final refTop = _argMax(ref.sublist((n - 1) * vocab, n * vocab));
  final gotTop = _argMax(got.sublist((n - 1) * vocab, n * vocab));

  print('logits shape: [$n, $vocab]');
  print('max abs diff vs single-GPU reference: ${maxDiff.toStringAsExponential(3)}');
  print('next-token argmax: reference=$refTop  sharded=$gotTop');
  print(refTop == gotTop && maxDiff < 1.0
      ? 'OK — sharded Llama reproduces the reference prediction.'
      : 'WARNING — sharded output diverges from the reference.');
}
