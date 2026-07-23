/// Run the HF GPT-2 **small** demo (117 M params, ~500 MB fp32).
///
/// ```sh
///   cd /mnt/c/Users/kkinyama/dart-pytorch
///   mkdir -p models/gpt2
///   curl -L --progress-bar \
///     -o models/gpt2/model.safetensors \
///     https://huggingface.co/gpt2/resolve/main/model.safetensors
///   curl -L -o models/gpt2/vocab.json \
///     https://huggingface.co/gpt2/resolve/main/vocab.json
///   dart run bin/gpt2/small/run.dart
/// ```
///
/// Optional: pass a custom safetensors path as the first argument,
/// otherwise defaults to `models/gpt2/model.safetensors`.
library;

import 'package:dart_pytorch/dart_pytorch.dart';

import '../../_gpt2_hf_demo_common.dart';

void main(List<String> args) {
  final path = args.isNotEmpty ? args.first : 'models/gpt2/model.safetensors';
  runGpt2Demo(
    path: path,
    cfg: GPT2HFLoader.gpt2SmallConfig(),
    sizeLabel: 'small',
  );
}
