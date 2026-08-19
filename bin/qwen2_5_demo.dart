/// Qwen2.5-0.5B (base / -Instruct) completion demo. Alibaba's modern
/// small LM on the Llama architecture (24 layers, hidden=896, GQA
/// 14→2), plus Qwen2's one quirk vs. Llama-3: **biases on Q/K/V** —
/// handled transparently by `LlamaConfig.attentionBias = true`.
///
///   dart run bin/qwen2_5_demo.dart              # CPU, greedy
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib \
///     dart run bin/qwen2_5_demo.dart --gpu      # GPU
///   dart run bin/qwen2_5_demo.dart --prompt "The capital of France is"
///
/// One-time weight download (~1 GB fp32 / 500 MB fp16):
///   mkdir -p models/qwen2.5-0.5b
///   for f in model.safetensors config.json tokenizer.json; do
///     curl -L -o "models/qwen2.5-0.5b/$f" \
///       "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/$f"
///   done
///
/// Swap `--preset qwen2.5-1.5b` (1.5B) or `--preset qwen2.5-3b`
/// (3B) for the bigger sizes; adjust `--weights` / `--tokenizer` to
/// match. Instruct variants share the same architecture — the only
/// difference is fine-tuning weights and the chat template, which
/// this demo does not apply (use the base prompt path).
library;

import 'dart:io';

import '_llama_encoder.dart';

const _weightsDefault = 'models/qwen2.5-0.5b/model.safetensors';
const _tokenizerDefault = 'models/qwen2.5-0.5b/tokenizer.json';

Future<void> main(List<String> args) async {
  var weightsPath = _weightsDefault;
  var tokenizerPath = _tokenizerDefault;
  var preset = 'qwen2.5-0.5b';
  var prompt = 'The capital of France is';
  var maxNew = 40;
  var useGpu = false;
  for (int i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--gpu':
        useGpu = true;
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
      case '--preset':
        preset = args[++i];
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
    preset: preset,
    gpu: useGpu,
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
