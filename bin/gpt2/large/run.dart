/// Run the HF GPT-2 **large** demo (774 M params, ~3 GB fp32).
///
/// Requires substantial RAM — a full fp32 forward on this preset uses
/// several GB of activations on top of the 3 GB of weights.
///
/// ```sh
///   cd /mnt/c/Users/kkinyama/dart-pytorch
///   mkdir -p models/gpt2-large
///   curl -L --progress-bar \
///     -o models/gpt2-large/model.safetensors \
///     https://huggingface.co/gpt2-large/resolve/main/model.safetensors
///   curl -L -o models/gpt2-large/vocab.json \
///     https://huggingface.co/gpt2-large/resolve/main/vocab.json
///   dart run bin/gpt2/large/run.dart
/// ```
///
/// Optional: pass a custom safetensors path as the first argument,
/// otherwise defaults to `models/gpt2-large/model.safetensors`.
library;

import 'package:dart_pytorch/dart_pytorch.dart';

import '../../_gpt2_hf_demo_common.dart';

void main(List<String> args) {
  final path = args.isNotEmpty
      ? args.first
      : 'models/gpt2-large/model.safetensors';
  runGpt2Demo(
    path: path,
    cfg: GPT2HFLoader.gpt2LargeConfig(),
    sizeLabel: 'large',
  );
}
