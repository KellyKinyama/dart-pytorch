/// Run the HF GPT-2 **xl** demo (1.5 B params, ~6 GB fp32).
///
/// This preset is memory-heavy: 6 GB just for weights and several
/// more for activations. On a 6 GB GPU it will not fit in fp32 —
/// use CPU with plenty of RAM (16 GB+ recommended).
///
/// ```sh
///   cd /mnt/c/Users/kkinyama/dart-pytorch
///   mkdir -p models/gpt2-xl
///   curl -L --progress-bar \
///     -o models/gpt2-xl/model.safetensors \
///     https://huggingface.co/gpt2-xl/resolve/main/model.safetensors
///   curl -L -o models/gpt2-xl/vocab.json \
///     https://huggingface.co/gpt2-xl/resolve/main/vocab.json
///   dart run bin/gpt2/xl/run.dart
/// ```
///
/// Optional: pass a custom safetensors path as the first argument,
/// otherwise defaults to `models/gpt2-xl/model.safetensors`.
library;

import 'package:dart_pytorch/dart_pytorch.dart';

import '../../_gpt2_hf_demo_common.dart';

void main(List<String> args) {
  final path = args.isNotEmpty ? args.first : 'models/gpt2-xl/model.safetensors';
  runGpt2Demo(
    path: path,
    cfg: GPT2HFLoader.gpt2XLConfig(),
    sizeLabel: 'xl',
  );
}
