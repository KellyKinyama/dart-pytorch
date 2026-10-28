/// Run a Needle 3 model from a `.cact` file: load the weights, build a
/// tool-calling prompt, greedily decode the assistant turn, and print it.
///
/// This is the end-to-end driver of the Dart port of Cactus Needle 3. Get a
/// `.cact` file first (weights are on Hugging Face, Apache-2.0 tooling):
///
///   pip install cactus-needle
///   needle build --platform windows-x86_64   # writes needle3.cact
///
/// Then:
///
///   dart run bin/needle_run.dart --model needle3.cact \
///     --user "turn on the kitchen lights" \
///     --tools "[{\"name\":\"set_lights\",\"parameters\":{...}}]"
///
/// Pure CPU; no KV cache (recomputes the sequence each step), so keep
/// --max small. Greedy decoding only — the grammar/confidence gates are
/// not ported.
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:dart_pytorch/core/nn/needle/cact_format.dart';
import 'package:dart_pytorch/core/nn/needle/needle_generate.dart';
import 'package:dart_pytorch/core/nn/needle/needle_loader.dart';
import 'package:dart_pytorch/core/nn/needle/needle_tokenizer.dart';

String _arg(List<String> a, String name, String fallback) {
  final i = a.indexOf(name);
  return (i >= 0 && i + 1 < a.length) ? a[i + 1] : fallback;
}

int _argInt(List<String> a, String name, int fallback) {
  final i = a.indexOf(name);
  if (i >= 0 && i + 1 < a.length) return int.tryParse(a[i + 1]) ?? fallback;
  return fallback;
}

void main(List<String> args) {
  final modelPath = _arg(args, '--model', 'needle3.cact');
  final user = _arg(args, '--user', 'turn on the kitchen lights');
  final tools = _arg(args, '--tools', '');
  final date = _arg(args, '--date', '');
  final maxNew = _argInt(args, '--max', 48);

  final f = File(modelPath);
  if (!f.existsSync()) {
    stderr.writeln('no such file: $modelPath');
    stderr.writeln('get one with: pip install cactus-needle && '
        'needle build --platform <your-platform>  (writes needle3.cact)');
    exitCode = 1;
    return;
  }

  stdout.writeln('loading $modelPath ...');
  final cact = CactFile.parse(Uint8List.fromList(f.readAsBytesSync()));
  final model = loadNeedleModel(cact);
  final tokBlob = cact.tokenizerBlob;
  if (tokBlob == null) {
    stderr.writeln('archive carries no tokenizer');
    exitCode = 1;
    return;
  }
  final tok = NeedleTokenizer.fromBlob(cact.blob(tokBlob));
  print('model: ${model.config}');

  final prompt = buildToolPrompt(
    tok,
    toolsJson: tools,
    user: user,
    date: date.isEmpty ? null : date,
  );
  print('prompt: ${prompt.length} tokens');

  // Engine forces <think> first; stop at <|im_end|> or </s>.
  final gen = generate(
    model,
    promptIds: prompt,
    maxNewTokens: maxNew,
    forceFirst: idThinkStart,
    stopIds: {idImEnd, idEos},
  );

  print('');
  print('--- assistant (${gen.length} tokens) ---');
  print(tok.decode(gen));
}
