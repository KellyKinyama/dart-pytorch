/// Shard a real GPT-2 checkpoint across GPUs and match its logits.
///
///   dart run bin/tensor_parallel_gpt2_demo.dart --model path/to/model.safetensors
///   dart run bin/tensor_parallel_gpt2_demo.dart --model ... --config distilgpt2
///
/// Loads GPT-2 onto GPU 0, builds a tensor-parallel stack from its
/// transformer blocks across every visible GPU, then runs
/// embeddings -> sharded blocks -> final LayerNorm -> tied lm_head and
/// compares the logits against the single-GPU reference forward. GPT-2's
/// fused QKV is already split into per-head Linears by the loader, and
/// its tanh-GELU FFN is now supported by the tensor-parallel MLP.
///
/// Requires the multi-GPU native lib (rebuild libmat_mul from
/// lib/native/src/engine.cu) and a local GPT-2 safetensors file.
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
    print('usage: dart run bin/tensor_parallel_gpt2_demo.dart '
        '--model <model.safetensors> [--config gpt2|distilgpt2|gpt2-medium|gpt2-large]');
    return;
  }
  final configName = _arg(args, '--config') ?? 'gpt2';

  final cfg = switch (configName) {
    'distilgpt2' => GPT2HFLoader.distilGpt2Config(device: Device.GPU),
    'gpt2' || 'gpt2-small' => GPT2HFLoader.gpt2SmallConfig(device: Device.GPU),
    'gpt2-medium' => GPT2HFLoader.gpt2MediumConfig(device: Device.GPU),
    'gpt2-large' => GPT2HFLoader.gpt2LargeConfig(device: Device.GPU),
    _ => throw ArgumentError('unknown --config $configName'),
  };

  print('=== tensor-parallel GPT-2 demo ===');
  print('visible GPUs: ${Tensor.gpuCount} | config: $configName '
      '(D=${cfg.embedDim}, heads=${cfg.numHeads}, layers=${cfg.numLayers})');

  final gpt = GPT(cfg);
  print('loading weights from $modelPath ...');
  GPT2HFLoader.loadFile(gpt, modelPath);
  gpt.eval();

  // Fixed prompt: GPT-2 BPE tokens for "The world is".
  final ids = <double>[464, 995, 318];
  final n = ids.length;
  final tokens = Tensor.fromList([n], ids, device: Device.GPU);

  // Reference logits: the stock single-GPU forward.
  final ref = Tensor.noGrad(() => gpt(tokens)).to(Device.CPU).toFloat32List();

  // Sharded forward: reuse the model's embeddings, final norm, and tied
  // head; replace the block stack with a tensor-parallel one.
  final tp = TensorParallelTransformerStack.fromBlocks(gpt.encoder.blocks);
  print('blocks sharded across GPUs (output devices): ${tp.blockOutputDevices}');
  print('attention heads per shard (block 0): '
      '${tp.blocks.first.mha.shards.map((s) => '${s.qCount}@gpu${s.device}').toList()}');

  final got = Tensor.noGrad(() {
    var h = gpt.tokenEmb(tokens);
    h = gpt.posEmb(h);
    h = gpt.embedDrop(h); // no-op in eval
    final maskVals = List<double>.filled(n * n, 0);
    for (var i = 0; i < n; i++) {
      for (var j = i + 1; j < n; j++) {
        maskVals[i * n + j] = -1e9;
      }
    }
    final mask = Tensor.fromList([n, n], maskVals, device: Device.GPU);
    h = tp(h, mask: mask);
    final fn = gpt.encoder.finalNorm;
    if (fn != null) h = fn(h);
    return h.matmul(gpt.tokenEmb.weight.transpose());
  }).to(Device.CPU).toFloat32List();

  final vocab = cfg.vocabSize;
  var maxDiff = 0.0;
  for (var i = 0; i < ref.length; i++) {
    final d = (got[i] - ref[i]).abs();
    if (d > maxDiff) maxDiff = d;
  }
  // Next-token prediction from the last position's logits.
  final refLast = ref.sublist((n - 1) * vocab, n * vocab);
  final gotLast = got.sublist((n - 1) * vocab, n * vocab);
  final refTop = _argMax(refLast);
  final gotTop = _argMax(gotLast);

  print('logits shape: [$n, $vocab]');
  print('max abs diff vs single-GPU reference: ${maxDiff.toStringAsExponential(3)}');
  print('next-token argmax: reference=$refTop  sharded=$gotTop');
  print(refTop == gotTop && maxDiff < 1.0
      ? 'OK — sharded GPT-2 reproduces the reference prediction.'
      : 'WARNING — sharded output diverges from the reference.');
}
