/// DeepSeek-R1-Distill-Qwen-1.5B demo — chain-of-thought reasoning
/// LM (1.78 B params). Architecturally a Qwen2.5-1.5B distillation
/// of DeepSeek-R1, so the model tends to emit `<think>...</think>`
/// reasoning traces followed by an answer.
///
///   dart run bin/deepseek_r1_qwen_1_5b_demo.dart              # CPU
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib \
///     dart run bin/deepseek_r1_qwen_1_5b_demo.dart --gpu      # GPU
///   dart run bin/deepseek_r1_qwen_1_5b_demo.dart \
///     --prompt "How many rs are in strawberry?"
///
/// **VRAM warning.** Fp32 weights are ~7 GB (won't fit on a 6 GB
/// card). Prefer `--keep-fp16` and a Q2/Q4 checkpoint if available.
/// Fp32 CPU works fine (slow — a few tokens/second).
///
/// One-time weight download:
///   mkdir -p models/deepseek-r1-distill-qwen-1.5b
///   for f in model.safetensors tokenizer.json; do
///     curl -L -o "models/deepseek-r1-distill-qwen-1.5b/$f" \
///       "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/resolve/main/$f"
///   done
library;

import 'dart:io';

import '_llama_encoder.dart';

const _weightsDefault =
    'models/deepseek-r1-distill-qwen-1.5b/model.safetensors';
const _tokenizerDefault = 'models/deepseek-r1-distill-qwen-1.5b/tokenizer.json';

Future<void> main(List<String> args) async {
  var weightsPath = _weightsDefault;
  var tokenizerPath = _tokenizerDefault;
  var prompt = 'Solve step by step: what is 47 * 13?';
  var maxNew = 120;
  var useGpu = false;
  var keepFp16 = false;
  for (int i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--gpu':
        useGpu = true;
        break;
      case '--keep-fp16':
        keepFp16 = true;
        break;
      case '--prompt':
        prompt = args[++i];
        break;
      case '--max-new':
        maxNew = int.parse(args[++i]);
        break;
      case '--weights':
        weightsPath = args[++i];
        break;
      case '--tokenizer':
        tokenizerPath = args[++i];
        break;
    }
  }
  for (final p in [weightsPath, tokenizerPath]) {
    if (!File(p).existsSync()) {
      stderr.writeln('missing: $p');
      exit(2);
    }
  }

  final sw = Stopwatch()..start();
  final bundle = loadLlamaEncoder(
    path: weightsPath,
    vocabPath: tokenizerPath,
    preset: 'deepseek-r1-distill-qwen-1.5b',
    gpu: useGpu,
    keepFp16: keepFp16,
  );
  final model = bundle.model;
  final tokenizer = bundle.tokenizer;
  sw.stop();
  print('load wall: ${sw.elapsedMilliseconds} ms');

  print('');
  print('== prompt ==');
  print('  "$prompt"');

  final ids = tokenizer.encode(prompt);
  final promptList = ids.map((i) => i.toDouble()).toList();

  print('');
  print('== generate ($maxNew tokens, greedy) ==');
  final swG = Stopwatch()..start();
  final generated = model.generate(
    promptList,
    maxNewTokens: maxNew,
    temperature: 0.0,
  );
  swG.stop();
  final full = generated.map((v) => v.toInt()).toList();
  final newTokens = full.sublist(ids.length);
  final text = tokenizer.decode(newTokens);
  print(
    '  ${swG.elapsedMilliseconds} ms  (${newTokens.length} tokens, '
    '${(newTokens.length * 1000.0 / swG.elapsedMilliseconds).toStringAsFixed(1)} tok/s)',
  );

  print('');
  print('== completion ==');
  print(prompt + text);
}
