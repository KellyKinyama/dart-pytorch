/// End-to-end demo: load HuggingFace GPT-2 (`.safetensors`) weights
/// into our `GPT` module and run a forward pass.
///
/// Usage:
///
/// ```sh
///   # 1. On any machine with Python + `pip install transformers safetensors`:
///   python - <<'PY'
///   from transformers import GPT2LMHeadModel
///   m = GPT2LMHeadModel.from_pretrained("gpt2")   # or gpt2-medium, ...
///   m.save_pretrained("gpt2", safe_serialization=True)
///   PY
///
///   # 2. Run this demo pointing at the safetensors file:
///   dart run bin/gpt2_hf_demo.dart gpt2/model.safetensors
/// ```
///
/// The demo:
///
///   * Builds a `GPT` with `GPT2HFLoader.gpt2SmallConfig()` (or one of
///     the larger presets when passed `--size medium|large|xl`).
///   * Loads the safetensors file, transposes/splits the fused HF
///     Conv1D weights, and copies them into every module tensor.
///   * Runs a forward pass on a small fixed sequence of BPE token ids
///     (default: `[464, 995, 318]` — the tokens for `"The world is"`
///     in the GPT-2 tokenizer). Prints the top-5 next-token ids and
///     their raw logits.
///
/// Note: producing readable text would additionally require loading
/// the GPT-2 BPE (`vocab.json` + `merges.txt`), which uses a different
/// on-disk format than our built-in [`BpeTokenizer`]. This demo stays
/// at the token-id level to keep the surface area small and to focus
/// on the weight-import path itself.
library;

import 'dart:io';
import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

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
      '[--size small|medium|large|xl]',
    );
    exit(64);
  }
  final path = positional.first;
  final f = File(path);
  if (!f.existsSync()) {
    stderr.writeln('gpt2_hf_demo: file not found: $path');
    exit(66);
  }

  final cfg = switch (size) {
    'small' => GPT2HFLoader.gpt2SmallConfig(),
    'medium' => GPT2HFLoader.gpt2MediumConfig(),
    'large' => GPT2HFLoader.gpt2LargeConfig(),
    'xl' => GPT2HFLoader.gpt2XLConfig(),
    _ => throw ArgumentError('unknown --size "$size"'),
  };

  stdout.writeln(
    'Building GPT (size=$size, embed=${cfg.embedDim}, '
    'layers=${cfg.numLayers}, heads=${cfg.numHeads}, vocab=${cfg.vocabSize})...',
  );
  final gpt = GPT(cfg);

  stdout.writeln('Loading safetensors from $path ...');
  final t0 = DateTime.now();
  final report = GPT2HFLoader.loadFile(gpt, path);
  final dt = DateTime.now().difference(t0);
  stdout.writeln('Loaded in ${dt.inMilliseconds} ms. $report');
  if (report.unusedKeys.isNotEmpty) {
    stdout.writeln(
      '  ignored ${report.unusedKeys.length} HF keys '
      '(e.g. ${report.unusedKeys.take(3).join(", ")})',
    );
  }

  // Fixed prompt: the GPT-2 BPE ids for "The world is".
  //   464   = "The"
  //   995   = " world"
  //   318   = " is"
  final tokens = <double>[464, 995, 318];
  stdout.writeln('Running forward on tokens $tokens ...');
  final input = Tensor.fromList([tokens.length], tokens, device: cfg.device);
  final logits = Tensor.noGrad(() => gpt(input));
  final flat = logits.toList(); // [N, V] flattened row-major
  final lastBase = (tokens.length - 1) * cfg.vocabSize;

  // Grab the top-5 ids from the last-row logits.
  final row = List<double>.generate(cfg.vocabSize, (i) => flat[lastBase + i]);
  final idxs = List<int>.generate(cfg.vocabSize, (i) => i)
    ..sort((a, b) => row[b].compareTo(row[a]));
  stdout.writeln('Top-5 next-token ids:');
  for (int i = 0; i < 5; i++) {
    final id = idxs[i];
    stdout.writeln(
      '  ${id.toString().padLeft(6)}    logit=${row[id].toStringAsFixed(3)}',
    );
  }
  // For a rough sanity check, print sum(logits.abs()) — should be a
  // stable finite number, not NaN/Inf.
  var s = 0.0;
  for (final v in row) {
    s += v.abs();
  }
  stdout.writeln(
    'sum|logits| on last row = ${s.toStringAsFixed(3)} '
    '(max |logit| = ${row.map((x) => x.abs()).reduce(math.max).toStringAsFixed(3)})',
  );
}
