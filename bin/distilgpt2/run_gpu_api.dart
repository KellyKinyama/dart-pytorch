/// Run **distilgpt2** (82 M params) with the full CLI + HTTP API on
/// the GPU. See `bin/_gpt2_hf_api_common.dart` for the shared runner.
///
/// ## Quickstart
///
/// ```sh
///   # Weights: one-off download (~330 MB).
///   mkdir -p models/distilgpt2
///   curl -L --progress-bar \
///     -o models/distilgpt2/model.safetensors \
///     https://huggingface.co/distilgpt2/resolve/main/model.safetensors
///   curl -L -o models/distilgpt2/vocab.json \
///     https://huggingface.co/distilgpt2/resolve/main/vocab.json
///
///   # One-shot:
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib \
///     dart run bin/distilgpt2/run_gpu_api.dart \
///       --prompt 464,995,318 --max-tokens 20 --temperature 0.8 \
///       --top-k 40 --seed 42
///
///   # HTTP server:
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib \
///     dart run bin/distilgpt2/run_gpu_api.dart --serve --port 8080
///
///   # From another shell:
///   curl -s -X POST http://127.0.0.1:8080/generate \
///     -H 'content-type: application/json' \
///     -d '{"tokens":[464,995,318],"maxNewTokens":20,"temperature":0.8,"topK":40,"seed":42}'
/// ```
///
/// distilgpt2 is a 6-layer distillation of gpt2 with the same tokenizer
/// and same weight layout — the standard `GPT2HFLoader` handles it with
/// no changes; only the config differs (6 layers instead of 12).
///
/// Weights (~330 MB) easily fit on a 6 GB GPU.
library;

import 'package:dart_pytorch/dart_pytorch.dart';

import '../_gpt2_hf_api_common.dart';

Future<void> main(List<String> args) => runGpt2Api(
  modelName: 'distilgpt2',
  defaultPath: 'models/distilgpt2/model.safetensors',
  configFactory: ({required device}) =>
      GPT2HFLoader.distilGpt2Config(device: device),
  args: args,
);
