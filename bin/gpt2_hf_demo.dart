/// End-to-end demo: load HuggingFace GPT-2 (`.safetensors`) weights
/// into our `GPT` module and run a forward pass.
///
/// ## Quickstart (copy-paste)
///
/// ```sh
///   # From the repo root:
///   cd /mnt/c/Users/kkinyama/dart-pytorch
///
///   # 1. Download the weights (523 MB) and vocab (1 MB). Public, no auth.
///   #    Skip either step if the file already exists locally.
///   mkdir -p models/gpt2
///   curl -L --progress-bar \
///     -o models/gpt2/model.safetensors \
///     https://huggingface.co/gpt2/resolve/main/model.safetensors
///   curl -L \
///     -o models/gpt2/vocab.json \
///     https://huggingface.co/gpt2/resolve/main/vocab.json
///
///   # 2. Run this demo end-to-end (CPU; takes ~40 s on the first run
///   #    while the 500 MB safetensors is parsed).
///   dart run bin/gpt2_hf_demo.dart models/gpt2/model.safetensors
/// ```
///
/// Expected tail of the output:
///
/// ```text
///   Top-5 next-token ids for "The world is":
///        257    logit=-110.827  " a"
///       5609    logit=-111.145  " changing"
///        407    logit=-111.372  " not"
///       1016    logit=-111.593  " going"
///       1336    logit=-111.637  " full"
///
///   Greedy 8-token continuation:
///     text: "The world is a better place if you're a good"
/// ```
///
/// ## Larger presets
///
/// ```sh
///   # gpt2-medium (~1.5 GB fp32), gpt2-large (~3 GB), gpt2-xl (~6 GB).
///   mkdir -p models/gpt2-medium
///   curl -L --progress-bar \
///     -o models/gpt2-medium/model.safetensors \
///     https://huggingface.co/gpt2-medium/resolve/main/model.safetensors
///   curl -L -o models/gpt2-medium/vocab.json \
///     https://huggingface.co/gpt2-medium/resolve/main/vocab.json
///   dart run bin/gpt2_hf_demo.dart \
///     models/gpt2-medium/model.safetensors --size medium
/// ```
///
/// ## What the demo does
///
///   * Builds a `GPT` with `GPT2HFLoader.gpt2SmallConfig()` (or one of
///     the larger presets when passed `--size medium|large|xl`).
///   * Loads the safetensors file, transposes/splits the fused HF
///     Conv1D weights, and copies them into every module tensor.
///   * Runs a forward pass on a small fixed sequence of BPE token ids
///     (default: `[464, 995, 318]` — the tokens for `"The world is"`
///     in the GPT-2 tokenizer). Prints the top-5 next-token ids and
///     their raw logits.
///   * If `<model_dir>/vocab.json` is present alongside the weights,
///     each top-5 id is decoded to its BPE token string.
///   * Runs an 8-token greedy continuation to exercise the full
///     encoder + KV-cache `generate()` path with imported weights.
///
/// Note: this demo does not do full BPE **encoding** of arbitrary
/// input text — that requires `merges.txt` and the byte-level BPE
/// merge algorithm, which is out of scope here. The prompt is fixed
/// to a hard-coded token-id sequence.
library;

import 'dart:io';

import 'package:dart_pytorch/dart_pytorch.dart';

import '_gpt2_hf_demo_common.dart';

void main(List<String> args) {
  final positional = <String>[];
  var size = 'small';
  for (int i = 0; i < args.length; i++) {
    final a = args[i];
    if (a == '--size' && i + 1 < args.length) {
      size = args[++i];
    } else if (a.startsWith('--size=')) {
      size = a.substring('--size='.length);
    } else if (a.startsWith('--')) {
      stderr.writeln('Unknown flag: $a');
      exit(64);
    } else {
      positional.add(a);
    }
  }
  if (positional.isEmpty) {
    stderr.writeln(
      'usage: gpt2_hf_demo <path/to/model.safetensors> '
      '[--size small|medium|large|xl]\n'
      '\n'
      'Quickstart:\n'
      '  mkdir -p models/gpt2\n'
      '  curl -L --progress-bar \\\n'
      '    -o models/gpt2/model.safetensors \\\n'
      '    https://huggingface.co/gpt2/resolve/main/model.safetensors\n'
      '  curl -L \\\n'
      '    -o models/gpt2/vocab.json \\\n'
      '    https://huggingface.co/gpt2/resolve/main/vocab.json\n'
      '  dart run bin/gpt2_hf_demo.dart models/gpt2/model.safetensors\n'
      '\n'
      'Or use one of the size-specific runners under bin/gpt2/:\n'
      '  dart run bin/gpt2/small/run.dart\n'
      '  dart run bin/gpt2/medium/run.dart\n'
      '  dart run bin/gpt2/large/run.dart\n'
      '  dart run bin/gpt2/xl/run.dart',
    );
    exit(64);
  }

  final cfg = switch (size) {
    'small' => GPT2HFLoader.gpt2SmallConfig(),
    'medium' => GPT2HFLoader.gpt2MediumConfig(),
    'large' => GPT2HFLoader.gpt2LargeConfig(),
    'xl' => GPT2HFLoader.gpt2XLConfig(),
    _ => throw ArgumentError('unknown --size "$size"'),
  };

  runGpt2Demo(path: positional.first, cfg: cfg, sizeLabel: size);
}
