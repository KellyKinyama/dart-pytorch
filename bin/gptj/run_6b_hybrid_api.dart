/// Hybrid CPU/GPU inference runner for **GPT-J-6B**.
///
/// Places the first `--gpu-layers N` transformer blocks on the GPU
/// (giving them the CUDA matmul), and the remaining `28 - N` blocks
/// on CPU. Token embedding (`wte`) and output stack (`ln_f` +
/// `lm_head`) stay on CPU — each is ~826 MB in fp32 and would
/// dominate a 6 GB card's memory.
///
/// Per-block fp32 VRAM cost is roughly
/// `12 * embedDim^2 * 4 bytes ≈ 805 MB`. Budget ~5 blocks max on a
/// 6 GB GPU. Activations move across PCIe at each device boundary
/// (`[N, D]` — ~480 KB for a 30-token prompt; negligible).
///
/// Weights (~24 GB fp32) can be fetched with:
///
/// ```
/// mkdir -p models/gpt-j-6b
/// curl -L -o models/gpt-j-6b/model.safetensors \
///   https://huggingface.co/EleutherAI/gpt-j-6B/resolve/main/model.safetensors
/// curl -L -o models/gpt-j-6b/tokenizer.json \
///   https://huggingface.co/EleutherAI/gpt-j-6B/resolve/main/tokenizer.json
/// ```
///
/// Example (put blocks 0..4 on GPU, rest on CPU):
///
/// ```
/// LD_LIBRARY_PATH=/usr/lib/wsl/lib \
///   dart run bin/gptj/run_6b_hybrid_api.dart \
///     --gpu-layers 5 --text "Once upon a time," --max-tokens 20
/// ```
///
/// Requires ~28 GB free host RAM during load (~22 GB stays on CPU,
/// ~4 GB uploads to VRAM).
///
/// See `bin/_gptj_hf_api_common.dart` for shared flag / HTTP server.
library;

import 'dart:io';

import 'package:dart_pytorch/dart_pytorch.dart';

import '../_gptj_hf_api_common.dart';

Future<void> main(List<String> args) async {
  await runGPTJApi(
    modelName: 'gpt-j-6b-hybrid',
    defaultPath: 'models/gpt-j-6b/model.safetensors',
    configFactory: ({required Device device, int? gpuLayers}) {
      // `device` from --cpu/--gpu is ignored in hybrid mode — the
      // config factory sets each block's placement explicitly.
      final n = gpuLayers ?? 4; // sensible default for a 6 GB GPU
      return GPTJHFLoader.gptJ6bHybridConfig(gpuLayers: n);
    },
    args: args,
  );
  exit(0);
}
