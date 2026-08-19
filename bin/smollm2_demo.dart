/// SmolLM2-135M-Instruct chat / completion demo. Modern SOTA small LM
/// on the Llama architecture (30 layers, hidden=576, GQA 9→3).
///
///   dart run bin/smollm2_demo.dart              # CPU, greedy
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib \
///     dart run bin/smollm2_demo.dart --gpu      # GPU
///   dart run bin/smollm2_demo.dart --prompt "The capital of France is"
///
/// One-time weight download:
///   mkdir -p models/smollm2-135m
///   for f in model.safetensors config.json tokenizer.json; do
///     curl -L -o "models/smollm2-135m/$f" \
///       "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/$f"
///   done
library;

import 'dart:io';

import '_llama_encoder.dart';

const _weightsDefault = 'models/smollm2-135m/model.safetensors';
const _tokenizerDefault = 'models/smollm2-135m/tokenizer.json';

Future<void> main(List<String> args) async {
  var weightsPath = _weightsDefault;
  var tokenizerPath = _tokenizerDefault;
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
    preset: 'smollm2-135m',
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
