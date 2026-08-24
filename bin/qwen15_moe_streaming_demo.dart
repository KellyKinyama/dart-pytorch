/// AirLLM-style layer + per-expert streaming for real
/// `Qwen/Qwen1.5-MoE-A2.7B-Chat` weights.
///
/// Wraps [Qwen15MoEStreamingRunner] with a tokenizer + argparse
/// harness matching the pattern of [bin/smollm2_demo.dart]. Works
/// once the full 8-shard checkpoint has been downloaded (~28.6 GB
/// bf16).
///
///   dart run bin/qwen15_moe_streaming_demo.dart \
///     --index ~/models/qwen1.5-moe-a2.7b-chat/model.safetensors.index.json \
///     --tokenizer ~/models/qwen1.5-moe-a2.7b-chat/tokenizer.json \
///     --prompt "The capital of France is" --max-new 10
///
/// Peak resident RAM ≈ 1.5 GB (embed + lm_head + one resident block
/// + streaming scratch). Disk I/O per token ≈ per-layer non-expert
/// weights + K expert triplets (~150 MB per layer × 24 layers ≈
/// 3.6 GB streamed off disk per generated token).
library;

import 'dart:io';

import 'package:dart_pytorch/dart_pytorch.dart';

Future<void> main(List<String> args) async {
  var indexPath =
      '${_defaultHome()}/models/qwen1.5-moe-a2.7b-chat/'
      'model.safetensors.index.json';
  var tokenizerPath =
      '${_defaultHome()}/models/qwen1.5-moe-a2.7b-chat/'
      'tokenizer.json';
  var prompt = 'The capital of France is';
  var maxNew = 10;
  var profile = false;
  for (int i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--index':
        indexPath = args[++i];
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
          'usage: qwen15_moe_streaming_demo [--index P] [--tokenizer P] '
          '[--prompt S] [--max-new N] [--profile]',
        );
        return;
    }
  }

  for (final p in [indexPath, tokenizerPath]) {
    if (!File(p).existsSync()) {
      stderr.writeln('missing: $p');
      stderr.writeln('download with:');
      stderr.writeln(
        '  hf download Qwen/Qwen1.5-MoE-A2.7B-Chat '
        '--local-dir ~/models/qwen1.5-moe-a2.7b-chat',
      );
      exit(2);
    }
  }

  final cfg = const Qwen15MoEConfig();
  final tokenizer = HFBpeTokenizer.loadFile(tokenizerPath);

  print('== Qwen1.5-MoE-A2.7B streaming demo (real weights) ==');
  print('  index        : $indexPath');
  print('  layers       : ${cfg.numLayers}');
  print('  D            : ${cfg.dim}');
  print('  E / K        : ${cfg.numExperts} / ${cfg.topK}');

  final swOpen = Stopwatch()..start();
  final reader = ShardedSafeTensorsReader.open(indexPath);
  swOpen.stop();
  print('  header parse : ${swOpen.elapsedMilliseconds} ms');

  final swInit = Stopwatch()..start();
  final runner = Qwen15MoEStreamingRunner(cfg, reader, profile: profile);
  swInit.stop();
  print(
    '  runner init  : ${swInit.elapsedMilliseconds} ms '
    '(persistent embed + head + rope + resident block)',
  );

  final ids = tokenizer.encode(prompt);
  final promptF = ids.map((i) => i.toDouble()).toList();
  print('');
  print('== prompt ==');
  print('  "$prompt"');
  print('  ids: $ids');

  print('');
  print('== generate ($maxNew tokens, greedy w/ KV cache) ==');
  final swG = Stopwatch()..start();
  final out = runner.generate(promptF, maxNewTokens: maxNew);
  swG.stop();
  final newIds = out.skip(ids.length).map((v) => v.toInt()).toList();
  final text = tokenizer.decode(newIds);
  print(
    '  wall         : ${swG.elapsedMilliseconds} ms '
    '(${(newIds.length * 1000.0 / swG.elapsedMilliseconds).toStringAsFixed(3)} tok/s)',
  );
  print('');
  print('== completion ==');
  print(prompt + text);

  runner.close();
}

String _defaultHome() =>
    Platform.environment['HOME'] ?? Platform.environment['USERPROFILE'] ?? '.';
