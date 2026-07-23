/// Run the HF GPT-2 **large** demo on the **GPU** (774 M params).
///
/// Same as `run.dart`, but builds the `GPT` with `device: Device.GPU`.
///
/// ```sh
///   cd /mnt/c/Users/kkinyama/dart-pytorch
///   mkdir -p models/gpt2-large
///   curl -L --progress-bar \
///     -o models/gpt2-large/model.safetensors \
///     https://huggingface.co/gpt2-large/resolve/main/model.safetensors
///   curl -L -o models/gpt2-large/vocab.json \
///     https://huggingface.co/gpt2-large/resolve/main/vocab.json
///   dart run bin/gpt2/large/run_gpu.dart
/// ```
///
/// Memory footprint (rough, fp32): weights ~3 GB, activations
/// ~3-4 GB. **Tight** on a 6 GB GPU — likely OOM once activations
/// and the KV cache add up. Consider CPU for this preset unless
/// you have a >=10 GB GPU.
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
    cfg: GPT2HFLoader.gpt2LargeConfig(device: Device.GPU),
    sizeLabel: 'large (gpu)',
  );
}
