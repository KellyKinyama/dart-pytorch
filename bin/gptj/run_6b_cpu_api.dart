/// End-to-end **CPU** inference runner for **GPT-J-6B** (EleutherAI,
/// 28 layers, 4096 embed, 16 heads, rotary_dim=64).
///
/// GPT-J-6B is ~24 GB in fp32 — it will *not* fit on a 6 GB GPU as
/// float32. This runner defaults to CPU; you need ~28 GB RAM free
/// during load (safetensors mmap + Dart-side Float32 copy).
///
/// Weights (~24 GB fp32 or ~12 GB fp16) can be fetched with:
///
/// ```
/// mkdir -p models/gpt-j-6b
/// curl -L -o models/gpt-j-6b/model.safetensors \
///   https://huggingface.co/EleutherAI/gpt-j-6B/resolve/main/model.safetensors
/// curl -L -o models/gpt-j-6b/tokenizer.json \
///   https://huggingface.co/EleutherAI/gpt-j-6B/resolve/main/tokenizer.json
/// ```
///
/// Example:
///
/// ```
/// dart run bin/gptj/run_6b_cpu_api.dart \
///     --text "Once upon a time," --max-tokens 20
/// ```
///
/// See `bin/_gptj_hf_api_common.dart` for the shared flag / HTTP
/// server implementation.
library;

import 'dart:io';

import 'package:dart_pytorch/dart_pytorch.dart';

import '../_gptj_hf_api_common.dart';

Future<void> main(List<String> args) async {
  await runGPTJApi(
    modelName: 'gpt-j-6b',
    defaultPath: 'models/gpt-j-6b/model.safetensors',
    configFactory: ({required Device device}) =>
        GPTJHFLoader.gptJ6bConfig(device: device),
    args: args,
  );
  exit(0);
}
