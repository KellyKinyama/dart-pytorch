/// AirLLM-style layer-streaming Llama demo.
///
/// Instead of loading all L transformer layers into memory, this
/// demo keeps only the token embedding, final norm, lm_head, and
/// one resident block — layer weights stream off disk per forward
/// pass. Peak resident RAM ≈ embed table + one layer + activations.
///
///   dart run bin/llama_streaming_demo.dart \
///     --weights models/smollm2-135m/model.safetensors \
///     --tokenizer models/smollm2-135m/tokenizer.json \
///     --preset smollm2-135m \
///     --prompt "The capital of France is" \
///     --max-new 20
///
/// Also accepts `--weights model.safetensors.index.json` for HF
/// sharded checkpoints (Llama-3.1-8B, etc.), though on WSL you'll
/// hit disk-bandwidth limits well before RAM limits.
library;

import 'dart:io';

import 'package:dart_pytorch/dart_pytorch.dart';

import '_llama_encoder.dart';

Future<void> main(List<String> args) async {
  var weightsPath = 'models/smollm2-135m/model.safetensors';
  var tokenizerPath = 'models/smollm2-135m/tokenizer.json';
  var preset = 'smollm2-135m';
  var prompt = 'The capital of France is';
  var maxNew = 20;
  var profile = false;
  for (int i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--weights':
        weightsPath = args[++i];
        break;
      case '--tokenizer':
        tokenizerPath = args[++i];
        break;
      case '--preset':
        preset = args[++i];
        break;
      case '--prompt':
        prompt = args[++i];
        break;
      case '--max-new':
        maxNew = int.parse(args[++i]);
        break;
      case '--profile':
        profile = true;
        break;
      case '-h':
      case '--help':
        stdout.writeln(
          'usage: llama_streaming_demo [--weights P] [--tokenizer P] '
          '[--preset NAME] [--prompt S] [--max-new N] [--profile]',
        );
        return;
    }
  }
  for (final p in [weightsPath, tokenizerPath]) {
    if (!File(p).existsSync()) {
      stderr.writeln('missing: $p');
      exit(2);
    }
  }

  final cfg = configForLlamaPreset(preset, Device.CPU);
  final tokenizer = HFBpeTokenizer.loadFile(tokenizerPath);

  print('== streaming Llama runner ==');
  print('  preset : $preset');
  print('  layers : ${cfg.numLayers}');
  print('  D / H  : ${cfg.embedDim} / ${cfg.numHeads} (kv=${cfg.numKvHeads})');
  print('  vocab  : ${cfg.vocabSize}');
  print('  weights: $weightsPath');

  final swOpen = Stopwatch()..start();
  final reader = ShardedSafeTensorsReader.open(weightsPath);
  swOpen.stop();
  print('  header : ${swOpen.elapsedMilliseconds} ms');

  final swInit = Stopwatch()..start();
  final runner = LlamaStreamingRunner(cfg, reader, profile: profile);
  swInit.stop();
  print(
    '  init   : ${swInit.elapsedMilliseconds} ms '
    '(persistent weights loaded)',
  );

  final layerBytes = estimateLayerBytes(reader, cfg);
  print(
    '  layer  : ${(layerBytes / (1024 * 1024)).toStringAsFixed(1)} MB '
    'on disk per layer (× ${cfg.numLayers} layers)',
  );

  final ids = tokenizer.encode(prompt);
  final promptF = ids.map((i) => i.toDouble()).toList();
  print('');
  print('== prompt ==');
  print('  "$prompt"');
  print('  ids: $ids');

  print('');
  print('== generate ($maxNew tokens, greedy) ==');
  final swG = Stopwatch()..start();
  final out = runner.generate(promptF, maxNewTokens: maxNew);
  swG.stop();
  final newIds = out.skip(ids.length).map((v) => v.toInt()).toList();
  final text = tokenizer.decode(newIds);
  print(
    '  wall   : ${swG.elapsedMilliseconds} ms '
    '(${(newIds.length * 1000.0 / swG.elapsedMilliseconds).toStringAsFixed(2)} tok/s)',
  );
  print('');
  print('== completion ==');
  print(prompt + text);

  runner.close();
}
