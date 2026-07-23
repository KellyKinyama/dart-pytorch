/// End-to-end GPU inference runner for **Pythia-160m**.
///
/// See `bin/_pythia_hf_api_common.dart` for the shared flag / HTTP
/// server implementation. This shim just wires model-specific defaults.
///
/// Example:
///
/// ```
/// LD_LIBRARY_PATH=/usr/lib/wsl/lib \
///   dart run bin/pythia/run_gpu_api.dart \
///     --prompt 510,973,310 --max-tokens 20
/// ```
library;

import 'dart:io';

import 'package:dart_pytorch/dart_pytorch.dart';

import '../_pythia_hf_api_common.dart';

Future<void> main(List<String> args) async {
  await runPythiaApi(
    modelName: 'pythia-160m',
    defaultPath: 'models/pythia-160m/model.safetensors',
    configFactory: ({required Device device}) =>
        PythiaHFLoader.pythia160mConfig(device: device),
    args: args,
  );
  exit(0);
}
