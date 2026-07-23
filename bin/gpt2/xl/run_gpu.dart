/// Run the HF GPT-2 **xl** demo on the **GPU** (1.5 B params).
///
/// Same as `run.dart`, but builds the `GPT` with `device: Device.GPU`.
///
/// **Will not fit in a 6 GB GPU** at fp32 — weights alone are ~6 GB,
/// activations push you well past 10 GB. Use CPU with plenty of RAM
/// for this preset, or wait for fp16/int8 support.
///
/// ```sh
///   cd /mnt/c/Users/kkinyama/dart-pytorch
///   mkdir -p models/gpt2-xl
///   curl -L --progress-bar \
///     -o models/gpt2-xl/model.safetensors \
///     https://huggingface.co/gpt2-xl/resolve/main/model.safetensors
///   curl -L -o models/gpt2-xl/vocab.json \
///     https://huggingface.co/gpt2-xl/resolve/main/vocab.json
///   dart run bin/gpt2/xl/run_gpu.dart
/// ```
///
/// Optional: pass a custom safetensors path as the first argument,
/// otherwise defaults to `models/gpt2-xl/model.safetensors`.
library;

import 'package:dart_pytorch/dart_pytorch.dart';

import '../../_gpt2_hf_demo_common.dart';

void main(List<String> args) {
  final path = args.isNotEmpty
      ? args.first
      : 'models/gpt2-xl/model.safetensors';
  runGpt2Demo(
    path: path,
    cfg: GPT2HFLoader.gpt2XLConfig(device: Device.GPU),
    sizeLabel: 'xl (gpu)',
  );
}
