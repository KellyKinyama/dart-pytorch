/// End-to-end GPU inference runner for **Pythia-1b** (16 layers,
/// 2048 embed, 8 heads, ~3.3 GB in fp32). Fits on a 6 GB GPU with
/// room for ~1000 KV-cached tokens.
///
/// See `bin/_pythia_hf_api_common.dart` for the shared flag / HTTP
/// server implementation.
///
/// Example:
///
/// ```
/// LD_LIBRARY_PATH=/usr/lib/wsl/lib \
///   dart run bin/pythia/run_1b_gpu_api.dart \
///     --text "Once upon a time," --max-tokens 30
/// ```
library;

import 'dart:io';

import 'package:dart_pytorch/dart_pytorch.dart';

import '../_pythia_hf_api_common.dart';

Future<void> main(List<String> args) async {
  await runPythiaApi(
    modelName: 'pythia-1b',
    defaultPath: 'models/pythia-1b/model.safetensors',
    configFactory: ({required Device device}) =>
        PythiaHFLoader.pythia1bConfig(device: device),
    args: args,
  );
  exit(0);
}
