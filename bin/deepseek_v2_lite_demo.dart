/// DeepSeek-V2-Lite (~16 B total, 2.4 B active per token) inference
/// demo. Loads the HF `deepseek-ai/DeepSeek-V2-Lite` safetensors via
/// [DeepSeekV2HFLoader] and runs a greedy autoregressive completion.
///
/// **Memory warning.** DeepSeek-V2-Lite is ~32 GB in fp32 (16 GB in
/// fp16). This demo defaults to CPU and requires ≥ 32 GB of system
/// RAM. It will **not** fit in a 6 GB GPU. Even the ~2.4 B **active**
/// parameter count during forward is misleading — MLA + MoE experts
/// are all resident in memory even when routed away from.
///
/// Usage:
///   dart run bin/deepseek_v2_lite_demo.dart --prompt "Machine learning is"
///
/// One-time weight download (~30 GB; sharded safetensors):
///   mkdir -p models/deepseek-v2-lite
///   for f in $(curl -s https://huggingface.co/api/models/deepseek-ai/DeepSeek-V2-Lite \
///       | grep '"rfilename":' | grep '.safetensors' | sed 's/.*"rfilename":"\\([^"]*\\)".*/\\1/'); do
///     curl -L -o "models/deepseek-v2-lite/$f" \
///       "https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/resolve/main/$f"
///   done
///   # Also fetch: tokenizer.json, config.json, model.safetensors.index.json
///
/// The naive no-cache greedy loop means per-token cost grows linearly
/// with prompt length. At Lite's 27-layer / 64-expert MoE scale on
/// CPU, expect roughly 30–60 s per token — this is a plumbing demo,
/// not a chat interface.
library;

import 'dart:io';

import 'package:dart_pytorch/dart_pytorch.dart';

const _weightsDefault = 'models/deepseek-v2-lite/model.safetensors.index.json';
const _tokenizerDefault = 'models/deepseek-v2-lite/tokenizer.json';

Future<void> main(List<String> args) async {
  var weightsPath = _weightsDefault;
  var tokenizerPath = _tokenizerDefault;
  var prompt = 'Machine learning is';
  var maxNew = 8;
  for (int i = 0; i < args.length; i++) {
    switch (args[i]) {
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
      stderr.writeln(
        'See the docstring at the top of '
        'bin/deepseek_v2_lite_demo.dart for setup steps.',
      );
      exit(2);
    }
  }

  final swBuild = Stopwatch()..start();
  print(
    'Building DeepSeekV2-Lite (27 layers, hidden 2048, 16 heads, '
    'MLA + 64 routed + 2 shared experts)',
  );
  final model = DeepSeekV2Model(DeepSeekV2Config.lite());
  swBuild.stop();
  print('  build: ${swBuild.elapsedMilliseconds} ms');

  final swLoad = Stopwatch()..start();
  print('Loading sharded safetensors from $weightsPath ...');
  final report = weightsPath.endsWith('.index.json')
      ? DeepSeekV2HFLoader.loadSharded(model, weightsPath)
      : DeepSeekV2HFLoader.loadFile(model, weightsPath);
  swLoad.stop();
  print('  load: ${swLoad.elapsedMilliseconds} ms  → $report');

  print('');
  print('Loading tokenizer from $tokenizerPath');
  final tokenizer = HFBpeTokenizer.loadFile(tokenizerPath);

  print('');
  print('== prompt ==');
  print('  "$prompt"');
  final ids = tokenizer.encode(prompt);
  print('  ${ids.length} tokens');

  print('');
  print('== greedy generate ($maxNew tokens, temperature=0) ==');
  print('  (naive no-cache — expect 30–60 s / token on CPU)');
  final swG = Stopwatch()..start();
  final gen = model.generate(
    ids.map((i) => i.toDouble()).toList(),
    maxNewTokens: maxNew,
    temperature: 0.0,
  );
  swG.stop();

  final full = gen.map((v) => v.toInt()).toList();
  final newTokens = full.sublist(ids.length);
  final text = tokenizer.decode(newTokens);
  final tokPerSec = newTokens.length * 1000.0 / swG.elapsedMilliseconds;
  print(
    '  ${swG.elapsedMilliseconds} ms  '
    '(${newTokens.length} new tokens, ${tokPerSec.toStringAsFixed(3)} tok/s)',
  );

  print('');
  print('== completion ==');
  print(prompt + text);
}
