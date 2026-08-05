/// Interactive Llama-3 chat CLI.
///
/// Loads a Llama-3 instruct checkpoint (default: Llama-3.2-1B-Instruct),
/// then runs a stdin REPL that speaks the official Llama-3 chat template:
///
///   <|begin_of_text|>
///   <|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>
///   <|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>
///   <|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>
///   ...
///
/// The tokenizer already recognises those special-token literals in
/// `encode()`, so we just build the template as a string and reuse
/// `Llama.generate` — stopping decoding at the first `<|eot_id|>`.
///
/// Commands inside the REPL:
///   :quit    Exit.
///   :reset   Wipe conversation history (system prompt is kept).
///   :sys X   Replace the system prompt with X and reset.
///
/// Usage:
///   dart run bin/llama_chat.dart [flags]
///
/// Common flags:
///   --path PATH        safetensors weights   (default: models/llama-3.2-1b-instruct/model.safetensors)
///   --vocab PATH       tokenizer.json         (default: models/llama-3.2-1b-instruct/tokenizer.json)
///   --preset NAME      llama-3.2-1b | llama-3.2-3b | llama-3.1-8b  (default: llama-3.2-1b)
///   --gpu              run on CUDA (default: CPU)
///   --system "..."     initial system prompt
///                       (default: "You are a helpful, concise assistant.")
///   --max-new N        max tokens per reply   (default: 256)
///   --temperature F    sampling temperature   (default: 0.7)
///   --top-k K          top-K sampling         (default: 40, 0 disables)
///
/// Examples:
///   # Llama-3.2-1B on CPU, defaults
///   dart run bin/llama_chat.dart
///
///   # Llama-3.2-1B on GPU (WSL2 needs LD_LIBRARY_PATH for the CUDA driver stub)
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib dart run bin/llama_chat.dart --gpu
///
///   # Custom system prompt, greedy decoding
///   dart run bin/llama_chat.dart --system "You are a Dart expert." --temperature 0.0
library;

import 'dart:io';

import '_llama_encoder.dart';

// ---------------------------------------------------------------------------
// CLI parsing
// ---------------------------------------------------------------------------

class _Opts {
  _Opts({
    required this.path,
    required this.vocabPath,
    required this.preset,
    required this.gpu,
    required this.system,
    required this.maxNew,
    required this.temperature,
    required this.topK,
  });

  final String path;
  final String vocabPath;
  final String preset;
  final bool gpu;
  final String system;
  final int maxNew;
  final double temperature;
  final int topK;
}

_Opts _parseArgs(List<String> args) {
  var path = 'models/llama-3.2-1b-instruct/model.safetensors';
  var vocab = 'models/llama-3.2-1b-instruct/tokenizer.json';
  var preset = 'llama-3.2-1b';
  var gpu = false;
  var system = 'You are a helpful, concise assistant.';
  var maxNew = 256;
  var temperature = 0.7;
  var topK = 40;

  for (var i = 0; i < args.length; i++) {
    final a = args[i];
    switch (a) {
      case '--path':
        path = args[++i];
      case '--vocab':
        vocab = args[++i];
      case '--preset':
        preset = args[++i];
      case '--gpu':
        gpu = true;
      case '--system':
        system = args[++i];
      case '--max-new':
        maxNew = int.parse(args[++i]);
      case '--temperature':
        temperature = double.parse(args[++i]);
      case '--top-k':
        topK = int.parse(args[++i]);
      case '-h' || '--help':
        stdout.writeln(_help);
        exit(0);
      default:
        stderr.writeln('unknown flag "$a" (see --help)');
        exit(64);
    }
  }
  return _Opts(
    path: path,
    vocabPath: vocab,
    preset: preset,
    gpu: gpu,
    system: system,
    maxNew: maxNew,
    temperature: temperature,
    topK: topK,
  );
}

const String _help = '''
Interactive Llama-3 chat CLI.

Usage:
  dart run bin/llama_chat.dart [flags]

Model:
  --path PATH        safetensors weights (default: models/llama-3.2-1b-instruct/model.safetensors)
  --vocab PATH       tokenizer.json      (default: models/llama-3.2-1b-instruct/tokenizer.json)
  --preset NAME      llama-3.2-1b | llama-3.2-3b | llama-3.1-8b  (default: llama-3.2-1b)
  --gpu              run on CUDA (default: CPU)

Sampling:
  --system "..."     initial system prompt
  --max-new N        max tokens per reply   (default: 256)
  --temperature F    sampling temperature   (default: 0.7; use 0.0 for greedy)
  --top-k K          top-K sampling         (default: 40, 0 disables)

REPL commands:
  :quit              exit
  :reset             wipe conversation history (system prompt kept)
  :sys <text>        replace system prompt and reset

Examples:
  # Llama-3.2-1B on CPU, defaults
  dart run bin/llama_chat.dart

  # Llama-3.2-1B on GPU (WSL2 needs LD_LIBRARY_PATH for the CUDA driver stub)
  LD_LIBRARY_PATH=/usr/lib/wsl/lib dart run bin/llama_chat.dart --gpu

  # Custom system prompt, greedy
  dart run bin/llama_chat.dart --system "You are a Dart expert." --temperature 0.0
''';

// ---------------------------------------------------------------------------
// Chat template helpers
// ---------------------------------------------------------------------------

String _systemPrefix(String system) {
  final s = system.trim();
  if (s.isEmpty) return '<|begin_of_text|>';
  return '<|begin_of_text|>'
      '<|start_header_id|>system<|end_header_id|>\n\n$s<|eot_id|>';
}

String _userTurn(String msg) =>
    '<|start_header_id|>user<|end_header_id|>\n\n$msg<|eot_id|>'
    '<|start_header_id|>assistant<|end_header_id|>\n\n';

// ---------------------------------------------------------------------------
// Main REPL
// ---------------------------------------------------------------------------

void main(List<String> args) {
  final opts = _parseArgs(args);
  final loaded = loadLlamaEncoder(
    path: opts.path,
    vocabPath: opts.vocabPath,
    preset: opts.preset,
    gpu: opts.gpu,
  );
  final model = loaded.model;
  final tok = loaded.tokenizer;
  final cfg = loaded.config;

  final eot = tok.llamaEotId;
  if (eot == null) {
    stderr.writeln(
      'warning: tokenizer has no <|eot_id|> — using <|endoftext|> fallback',
    );
  }
  final stopId = eot ?? tok.endOfTextId;

  var system = opts.system;
  var chatText = _systemPrefix(system);

  stdout.writeln(
    '\nReady. Type a message, or :quit / :reset / :sys <text>. '
    'Ctrl+D to exit.\n',
  );

  while (true) {
    stdout.write('you> ');
    final line = stdin.readLineSync();
    if (line == null) {
      stdout.writeln();
      break;
    }
    final trimmed = line.trim();
    if (trimmed.isEmpty) continue;
    if (trimmed == ':quit') break;
    if (trimmed == ':reset') {
      chatText = _systemPrefix(system);
      stdout.writeln('[history cleared]');
      continue;
    }
    if (trimmed.startsWith(':sys ')) {
      system = trimmed.substring(5).trim();
      chatText = _systemPrefix(system);
      stdout.writeln('[system prompt updated; history cleared]');
      continue;
    }

    final prompt = chatText + _userTurn(trimmed);
    var promptIds = tok.encode(prompt);

    // If the conversation has grown past maxCtx, wipe history and retry
    // with just the current turn. Simpler than rebuilding turn structure.
    if (promptIds.length + opts.maxNew > cfg.maxCtx) {
      stdout.writeln(
        '[history exceeds context (${promptIds.length} + ${opts.maxNew} > '
        '${cfg.maxCtx}); resetting]',
      );
      chatText = _systemPrefix(system);
      promptIds = tok.encode(chatText + _userTurn(trimmed));
      if (promptIds.length + opts.maxNew > cfg.maxCtx) {
        stdout.writeln(
          '[single turn still exceeds context — skipping]\n',
        );
        continue;
      }
    }

    final full = model.generate(
      promptIds.map((i) => i.toDouble()).toList(),
      maxNewTokens: opts.maxNew,
      temperature: opts.temperature,
      topK: opts.topK == 0 ? null : opts.topK,
    );
    final newIds = full
        .sublist(promptIds.length)
        .map((d) => d.toInt())
        .toList();
    List<int> answerIds = newIds;
    if (stopId != null) {
      final idx = newIds.indexOf(stopId);
      if (idx >= 0) answerIds = newIds.sublist(0, idx);
    }
    final answer = tok.decode(answerIds).trim();
    stdout.writeln('\nbot> $answer\n');

    // Persist this turn — assistant reply plus trailing <|eot_id|>.
    chatText = '$prompt$answer<|eot_id|>';
  }
}
