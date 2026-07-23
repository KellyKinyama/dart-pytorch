/// End-to-end GPU inference runner for **Pythia-410m** (24 layers,
/// 1024 embed, 16 heads, ~810 MB in fp32). Fits comfortably on a
/// 6 GB GPU alongside a few thousand cached KV tokens.
///
/// See `bin/_pythia_hf_api_common.dart` for the shared flag / HTTP
/// server implementation.
///
/// Example:
///
/// ```
/// LD_LIBRARY_PATH=/usr/lib/wsl/lib \
///   dart run bin/pythia/run_410m_gpu_api.dart \
///     --prompt 510,1533,310 --max-tokens 20
/// ```
library;

import 'dart:io';

import 'package:dart_pytorch/dart_pytorch.dart';

import '../_pythia_hf_api_common.dart';

Future<void> main(List<String> args) async {
  await runPythiaApi(
    modelName: 'pythia-410m',
    defaultPath: 'models/pythia-410m/model.safetensors',
    configFactory: ({required Device device}) =>
        PythiaHFLoader.pythia410mConfig(device: device),
    args: args,
  );
  exit(0);
}
