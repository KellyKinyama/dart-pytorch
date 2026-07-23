/// Run **gpt2-medium** (345 M params) with the full CLI + HTTP API on
/// the GPU. See `bin/_gpt2_hf_api_common.dart` for the shared runner.
///
/// ## Quickstart
///
/// ```sh
///   # Weights (~1.5 GB, one-off):
///   mkdir -p models/gpt2-medium
///   curl -L --progress-bar \
///     -o models/gpt2-medium/model.safetensors \
///     https://huggingface.co/gpt2-medium/resolve/main/model.safetensors
///   curl -L -o models/gpt2-medium/vocab.json \
///     https://huggingface.co/gpt2-medium/resolve/main/vocab.json
///
///   # One-shot:
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib \
///     dart run bin/gpt2/medium/run_gpu_api.dart \
///       --prompt 464,995,318 --max-tokens 40 --temperature 0.8 \
///       --top-k 40 --seed 42
///
///   # HTTP server:
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib \
///     dart run bin/gpt2/medium/run_gpu_api.dart --serve --port 8080
///
///   # From another shell:
///   curl -s -X POST http://127.0.0.1:8080/generate \
///     -H 'content-type: application/json' \
///     -d '{"tokens":[464,995,318],"maxNewTokens":20,"temperature":0.8,"topK":40,"seed":42}'
/// ```
///
/// Weights ~1.5 GB + activations for short contexts fit in a 6 GB GPU.
library;

import 'package:dart_pytorch/dart_pytorch.dart';

import '../../_gpt2_hf_api_common.dart';

Future<void> main(List<String> args) => runGpt2Api(
  modelName: 'gpt2-medium',
  defaultPath: 'models/gpt2-medium/model.safetensors',
  configFactory: ({required device}) =>
      GPT2HFLoader.gpt2MediumConfig(device: device),
  args: args,
);
