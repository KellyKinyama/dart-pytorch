/// Run the HF GPT-2 **medium** demo on the **GPU** (345 M params).
///
/// Same as `run.dart`, but builds the `GPT` with `device: Device.GPU`.
///
/// ```sh
///   cd /mnt/c/Users/kkinyama/dart-pytorch
///   mkdir -p models/gpt2-medium
///   curl -L --progress-bar \
///     -o models/gpt2-medium/model.safetensors \
///     https://huggingface.co/gpt2-medium/resolve/main/model.safetensors
///   curl -L -o models/gpt2-medium/vocab.json \
///     https://huggingface.co/gpt2-medium/resolve/main/vocab.json
///   dart run bin/gpt2/medium/run_gpu.dart
/// ```
///
/// Memory footprint (rough, fp32): weights ~1.5 GB, activations
/// ~2-3 GB. Fits in a 6 GB GPU.
///
/// Optional: pass a custom safetensors path as the first argument,
/// otherwise defaults to `models/gpt2-medium/model.safetensors`.
library;

import 'package:dart_pytorch/dart_pytorch.dart';

import '../../_gpt2_hf_demo_common.dart';

void main(List<String> args) {
  final path = args.isNotEmpty
      ? args.first
      : 'models/gpt2-medium/model.safetensors';
  runGpt2Demo(
    path: path,
    cfg: GPT2HFLoader.gpt2MediumConfig(device: Device.GPU),
    sizeLabel: 'medium (gpu)',
  );
}
