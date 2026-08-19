/// DeepSeek-Coder 1.3B completion demo — dense Llama-family code LM
/// (1.35 B params, full attention with 16 heads, no GQA, tied word
/// embeddings, 32 256-token BPE vocab with code-oriented merges).
///
///   dart run bin/deepseek_coder_1_3b_demo.dart              # CPU
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib \
///     dart run bin/deepseek_coder_1_3b_demo.dart --gpu      # GPU
///   dart run bin/deepseek_coder_1_3b_demo.dart \
///     --prompt "def fibonacci(n):\n"
///
/// Fp32 weights are ~5.4 GB (tight on 6 GB VRAM — prefer CPU or
/// `--keep-fp16` if the checkpoint has fp16 tensors).
///
/// One-time weight download:
///   mkdir -p models/deepseek-coder-1.3b-instruct
///   for f in model.safetensors tokenizer.json; do
///     curl -L -o "models/deepseek-coder-1.3b-instruct/$f" \
///       "https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct/resolve/main/$f"
///   done
library;

import 'dart:io';

import '_llama_encoder.dart';

const _weightsDefault =
    'models/deepseek-coder-1.3b-instruct/model.safetensors';
const _tokenizerDefault =
    'models/deepseek-coder-1.3b-instruct/tokenizer.json';

Future<void> main(List<String> args) async {
  var weightsPath = _weightsDefault;
  var tokenizerPath = _tokenizerDefault;
  var prompt = 'def quicksort(arr):\n';
  var maxNew = 100;
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
    preset: 'deepseek-coder-1.3b',
    gpu: useGpu,
    keepFp16: keepFp16,
  );
  final model = bundle.model;
  final tokenizer = bundle.tokenizer;
  sw.stop();
  print('load wall: ${sw.elapsedMilliseconds} ms');

  print('');
  print('== prompt ==');
  print(prompt);

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
