/// Shared demo body used by `bin/gpt2_hf_demo.dart` and by the
/// per-size runners in `bin/gpt2/{small,medium,large,xl}/run.dart`.
///
/// Exposes a single entry point, [runGpt2Demo], that:
///   * builds a `GPT` from the given config,
///   * loads a HuggingFace safetensors file into it,
///   * runs a forward pass on the fixed prompt "The world is",
///   * prints top-5 next-token ids (with pretty-printed BPE strings
///     when `<model_dir>/vocab.json` exists), and
///   * greedy-generates 8 continuation tokens.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

/// Runs the full "load HF gpt2 weights and generate" demo.
///
/// [path] is the path to a `model.safetensors` file on disk.
/// [cfg]  is a matching `GPTConfig` (typically one of the
///        `GPT2HFLoader.gpt2*Config()` presets).
/// [sizeLabel] is a short name (e.g. `"small"`, `"medium"`)
///        printed in the header for context.
void runGpt2Demo({
  required String path,
  required GPTConfig cfg,
  required String sizeLabel,
}) {
  final f = File(path);
  if (!f.existsSync()) {
    stderr.writeln('gpt2_hf_demo: file not found: $path');
    exit(66);
  }

  stdout.writeln(
    'Building GPT (size=$sizeLabel, embed=${cfg.embedDim}, '
    'layers=${cfg.numLayers}, heads=${cfg.numHeads}, '
    'vocab=${cfg.vocabSize})...',
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
    '(max |logit| = '
    '${row.map((x) => x.abs()).reduce(math.max).toStringAsFixed(3)})',
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
