/// Run the HF GPT-2 **small** demo on the **GPU** (117 M params).
///
/// Same as `run.dart`, but builds the `GPT` with `device: Device.GPU`
/// so weights and activations live on the GPU via the CUDA FFI
/// backend.
///
/// ```sh
///   cd /mnt/c/Users/kkinyama/dart-pytorch
///   # 1. One-off: make sure the CUDA FFI native lib is built.
///   #    (See README for the cmake+make steps.)
///   # 2. Download weights (same as CPU version):
///   mkdir -p models/gpt2
///   curl -L --progress-bar \
///     -o models/gpt2/model.safetensors \
///     https://huggingface.co/gpt2/resolve/main/model.safetensors
///   curl -L -o models/gpt2/vocab.json \
///     https://huggingface.co/gpt2/resolve/main/vocab.json
///   # 3. Run on GPU. If cudaGetDeviceCount reports 0 devices even
///   #    though nvidia-smi works, force the WSL stub search path:
///   #    LD_LIBRARY_PATH=/usr/lib/wsl/lib dart run bin/gpt2/small/run_gpu.dart
///   dart run bin/gpt2/small/run_gpu.dart
/// ```
///
/// Memory footprint (rough, fp32): weights ~500 MB, activations
/// ~1-2 GB for the 3-token prompt + 8-token generation. Fits in a
/// 6 GB GPU.
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
    cfg: GPT2HFLoader.gpt2SmallConfig(device: Device.GPU),
    sizeLabel: 'small (gpu)',
  );
}
