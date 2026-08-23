/// AirLLM-style layer-streaming GPT-J-6B demo.
///
/// GPT-J-6B is ~12 GB fp16 / ~24 GB fp32 — will not fit on a 6 GB
/// GPU, and even loading the full checkpoint into a fresh CPU model
/// peaks at ~40 GB RAM. This runner streams one of the 28 layers
/// from disk per forward pass; peak resident RAM ≈ embed table +
/// lm_head + one layer + activations (~1.5–2 GB fp16).
///
/// It's *slow* — every generated token re-reads all 28 layers off
/// disk. Use it to verify a huge model works on modest hardware,
/// not for interactive chat.
///
///   dart run bin/gptj_streaming_demo.dart \
///     --weights models/gpt-j-6b/model.safetensors \
///     --tokenizer models/gpt-j-6b/tokenizer.json \
///     --prompt "Once upon a time," --max-new 5
library;

import 'dart:io';

import 'package:dart_pytorch/dart_pytorch.dart';

Future<void> main(List<String> args) async {
  var weightsPath = 'models/gpt-j-6b/model.safetensors';
  var tokenizerPath = 'models/gpt-j-6b/tokenizer.json';
  var prompt = 'Once upon a time,';
  var maxNew = 5;
  var profile = false;
  for (int i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--weights':
        weightsPath = args[++i];
        break;
      case '--tokenizer':
        tokenizerPath = args[++i];
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
          'usage: gptj_streaming_demo [--weights P] [--tokenizer P] '
          '[--prompt S] [--max-new N] [--profile]',
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

  final cfg = GPTJHFLoader.gptJ6bConfig(device: Device.CPU);
  final tokenizer = HFBpeTokenizer.loadFile(tokenizerPath);

  print('== streaming GPT-J runner ==');
  print('  layers  : ${cfg.numLayers}');
  print('  D / H   : ${cfg.embedDim} / ${cfg.numHeads} (headDim '
      '${cfg.embedDim ~/ cfg.numHeads}, rot=${cfg.rotaryDim})');
  print('  vocab   : ${cfg.vocabSize}');
  print('  weights : $weightsPath');

  final swOpen = Stopwatch()..start();
  final reader = ShardedSafeTensorsReader.open(weightsPath);
  swOpen.stop();
  print('  header  : ${swOpen.elapsedMilliseconds} ms');

  final swInit = Stopwatch()..start();
  final runner = GPTJStreamingRunner(cfg, reader, profile: profile);
  swInit.stop();
  print('  init    : ${swInit.elapsedMilliseconds} ms '
      '(wte + ln_f + lm_head loaded)');

  final layerBytes = estimateGPTJLayerBytes(reader, cfg);
  print('  layer   : ${(layerBytes / (1024 * 1024)).toStringAsFixed(1)} MB '
      'on disk per layer (× ${cfg.numLayers} layers)');

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
  print('  wall    : ${swG.elapsedMilliseconds} ms '
      '(${(newIds.length * 1000.0 / swG.elapsedMilliseconds).toStringAsFixed(3)} tok/s)');
  print('');
  print('== completion ==');
  print(prompt + text);

  runner.close();
}
