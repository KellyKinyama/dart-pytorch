/// End-to-end demo: load HuggingFace GPT-2 (`.safetensors`) weights
/// into our `GPT` module and run a forward pass.
///
/// Usage:
///
/// ```sh
///   # 1. Download the weights (public, no auth):
///   mkdir -p models/gpt2
///   curl -L -o models/gpt2/model.safetensors \
///     https://huggingface.co/gpt2/resolve/main/model.safetensors
///
///   # 2. (Optional) also download vocab.json to get token strings
///   #    printed next to the ids:
///   curl -L -o models/gpt2/vocab.json \
///     https://huggingface.co/gpt2/resolve/main/vocab.json
///
///   # 3. Run this demo:
///   dart run bin/gpt2_hf_demo.dart models/gpt2/model.safetensors
///   # or larger:
///   dart run bin/gpt2_hf_demo.dart <path> --size medium|large|xl
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
///   * If `<model_dir>/vocab.json` is present alongside the weights,
///     each top-5 id is decoded to its BPE token string.
///
/// Note: this demo does not do full BPE **encoding** of arbitrary
/// input text — that requires `merges.txt` and the byte-level BPE
/// merge algorithm, which is out of scope here. The prompt is fixed
/// to a hard-coded token-id sequence.
library;

import 'dart:convert';
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

  // Optional token-string lookup from vocab.json living next to the
  // safetensors file. HF's vocab.json is `{token_string: id}`; the
  // byte-BPE ' ' encoding uses 'Ġ' (U+0120), newline uses 'Ċ'
  // (U+010A) — we only pretty-print those two so human eyes can
  // parse the output; a full byte-BPE decoder is not needed just
  // for inspection.
  final vocabPath = '${File(path).parent.path}/vocab.json';
  Map<int, String>? idToTok;
  if (File(vocabPath).existsSync()) {
    stdout.writeln('Decoding token ids using $vocabPath');
    final raw =
        jsonDecode(File(vocabPath).readAsStringSync()) as Map<String, dynamic>;
    idToTok = <int, String>{
      for (final e in raw.entries) (e.value as num).toInt(): e.key.toString(),
    };
  }
  String pretty(int id) {
    final s = idToTok?[id];
    if (s == null) return '';
    return s.replaceAll('\u0120', ' ').replaceAll('\u010A', r'\n');
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
  stdout.writeln('Top-5 next-token ids for "The world is":');
  for (int i = 0; i < 5; i++) {
    final id = idxs[i];
    final tok = idToTok == null ? '' : '  "${pretty(id)}"';
    stdout.writeln(
      '  ${id.toString().padLeft(6)}    '
      'logit=${row[id].toStringAsFixed(3)}$tok',
    );
  }
  var s = 0.0;
  for (final v in row) {
    s += v.abs();
  }
  stdout.writeln(
    'sum|logits| on last row = ${s.toStringAsFixed(3)} '
    '(max |logit| = ${row.map((x) => x.abs()).reduce(math.max).toStringAsFixed(3)})',
  );

  // Greedy continuation for a few tokens, just to demonstrate the
  // whole pipeline (encoder + KV-cache `generate()`) works with
  // imported weights.
  stdout.writeln('\nGreedy 8-token continuation:');
  final gen = gpt.generate(tokens, maxNewTokens: 8, temperature: 0.0);
  final genIds = gen.map((d) => d.toInt()).toList();
  stdout.writeln('  ids: $genIds');
  if (idToTok != null) {
    final decoded = genIds.map(pretty).join();
    stdout.writeln('  text: "$decoded"');
  }
}
