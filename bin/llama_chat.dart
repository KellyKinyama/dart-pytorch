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
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

import '_llama_encoder.dart';
import '_lm_encoder.dart' show centerAndNormalize, meanVector;

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
    required this.docs,
    required this.chunkTokens,
    required this.chunkStride,
    required this.topKDocs,
  });

  final String path;
  final String vocabPath;
  final String preset;
  final bool gpu;
  final String system;
  final int maxNew;
  final double temperature;
  final int topK;
  final List<String> docs;
  final int chunkTokens;
  final int chunkStride;
  final int topKDocs;
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
  final docs = <String>[];
  var chunkTokens = 200;
  var chunkStride = 100;
  var topKDocs = 4;

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
      case '--docs':
        docs.add(args[++i]);
      case '--chunk-tokens':
        chunkTokens = int.parse(args[++i]);
      case '--chunk-stride':
        chunkStride = int.parse(args[++i]);
      case '--top-k-docs':
        topKDocs = int.parse(args[++i]);
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
    docs: docs,
    chunkTokens: chunkTokens,
    chunkStride: chunkStride,
    topKDocs: topKDocs,
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

RAG (optional):
  --docs PATH        (repeatable) file or directory to index for retrieval
                     augmented answers. Dirs are walked for *.txt / *.md.
  --chunk-tokens N   chunk length in tokens (default: 200)
  --chunk-stride N   sliding stride         (default: 100 = 50% overlap)
  --top-k-docs K     chunks retrieved per turn (default: 4)

REPL commands:
  :quit              exit
  :reset             wipe conversation history (system prompt kept)
  :sys <text>        replace system prompt and reset
  :sources           print the chunks retrieved for the last answer (RAG only)

Examples:
  # Llama-3.2-1B on CPU, defaults
  dart run bin/llama_chat.dart

  # Llama-3.2-1B on GPU (WSL2 needs LD_LIBRARY_PATH for the CUDA driver stub)
  LD_LIBRARY_PATH=/usr/lib/wsl/lib dart run bin/llama_chat.dart --gpu

  # Custom system prompt, greedy
  dart run bin/llama_chat.dart --system "You are a Dart expert." --temperature 0.0

  # RAG over a folder of notes
  dart run bin/llama_chat.dart --docs notes/ --docs README.md
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
// RAG: file ingest + chunk embedding + IndexFlatIP retrieval
// ---------------------------------------------------------------------------

class _RagChunk {
  _RagChunk({
    required this.docTitle,
    required this.startTok,
    required this.endTok,
    required this.text,
  });
  final String docTitle;
  final int startTok;
  final int endTok;
  final String text;
}

/// Expand `--docs` args to a concrete list of files. If a path is a
/// directory, walk it for *.txt / *.md (non-recursive — simple by design).
List<File> _resolveDocFiles(List<String> paths) {
  final files = <File>[];
  for (final p in paths) {
    final type = FileSystemEntity.typeSync(p);
    if (type == FileSystemEntityType.notFound) {
      stderr.writeln('warning: --docs path not found: $p');
      continue;
    }
    if (type == FileSystemEntityType.directory) {
      final dir = Directory(p);
      for (final e in dir.listSync()) {
        if (e is! File) continue;
        final ext = e.path.toLowerCase();
        if (ext.endsWith('.txt') || ext.endsWith('.md')) {
          files.add(e);
        }
      }
    } else {
      files.add(File(p));
    }
  }
  files.sort((a, b) => a.path.compareTo(b.path));
  return files;
}

/// Ingest every doc file: tokenize, split into `chunkTokens`-sized
/// windows with `chunkStride`, embed each chunk via the Llama encoder,
/// then compute the corpus mean, centre + L2-normalise each vector,
/// and pack them into an [IndexFlatIP]. Returns null if there's
/// nothing to index.
({IndexFlat index, Float32List mean, List<_RagChunk> chunks})? _buildRagIndex({
  required List<File> files,
  required Llama model,
  required HFBpeTokenizer tokenizer,
  required int chunkTokens,
  required int chunkStride,
}) {
  final chunks = <_RagChunk>[];
  final rawVecs = <Float32List>[];
  final d = model.config.embedDim;

  for (final f in files) {
    final title = f.uri.pathSegments.last;
    final content = f.readAsStringSync();
    final tokens = tokenizer.encode(content);
    if (tokens.isEmpty) continue;

    void addChunk(int start, int end, List<int> ids) {
      final text = tokenizer.decode(ids);
      final raw = lastTokenHiddenLlama(model, ids);
      chunks.add(
        _RagChunk(docTitle: title, startTok: start, endTok: end, text: text),
      );
      rawVecs.add(raw);
    }

    if (tokens.length <= chunkTokens) {
      addChunk(0, tokens.length, tokens);
    } else {
      for (var start = 0; start < tokens.length; start += chunkStride) {
        final end = start + chunkTokens < tokens.length
            ? start + chunkTokens
            : tokens.length;
        addChunk(start, end, tokens.sublist(start, end));
        if (end == tokens.length) break;
      }
    }
    stdout.writeln(
      '[rag] $title -> ${tokens.length} tok, chunks so far: ${chunks.length}',
    );
  }

  if (chunks.isEmpty) return null;

  final mean = meanVector(rawVecs, d);
  final index = IndexFlatIP(d);
  for (final v in rawVecs) {
    index.add([centerAndNormalize(v, mean)]);
  }
  return (index: index, mean: mean, chunks: chunks);
}

/// Retrieve `k` chunks most similar to `query` (cosine — index holds
/// centre + L2-normalised vectors, so an inner-product search *is*
/// cosine similarity).
List<({_RagChunk chunk, double score})> _retrieve({
  required String query,
  required Llama model,
  required HFBpeTokenizer tokenizer,
  required IndexFlat index,
  required Float32List mean,
  required List<_RagChunk> chunks,
  required int k,
}) {
  final ids = tokenizer.encode(query);
  if (ids.isEmpty || chunks.isEmpty) return const [];
  final qRaw = lastTokenHiddenLlama(model, ids);
  final qVec = centerAndNormalize(qRaw, mean);
  final kEff = k < chunks.length ? k : chunks.length;
  final res = index.search([qVec], kEff);
  final out = <({_RagChunk chunk, double score})>[];
  for (var i = 0; i < res.ids[0].length; i++) {
    out.add((chunk: chunks[res.ids[0][i]], score: res.distances[0][i]));
  }
  return out;
}

/// Prepend the retrieved context to the user's message using a
/// simple "excerpts + question" template. The whole block goes
/// inside a single Llama-3 user turn.
String _augmentWithContext(
  String message,
  List<({_RagChunk chunk, double score})> hits,
) {
  if (hits.isEmpty) return message;
  final buf = StringBuffer();
  buf.writeln(
    'Use the following excerpts from the knowledge base to help answer. '
    'If they are not relevant, ignore them.',
  );
  buf.writeln();
  for (var i = 0; i < hits.length; i++) {
    final c = hits[i].chunk;
    buf.writeln('[${i + 1}] (${c.docTitle}, tokens ${c.startTok}-${c.endTok})');
    buf.writeln(c.text.trim());
    buf.writeln();
  }
  buf.writeln('Question: $message');
  return buf.toString();
}

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

  // ---- Optional RAG index over --docs files ------------------------
  IndexFlat? ragIndex;
  Float32List? ragMean;
  List<_RagChunk> ragChunks = const [];
  if (opts.docs.isNotEmpty) {
    final files = _resolveDocFiles(opts.docs);
    if (files.isEmpty) {
      stderr.writeln(
        '[rag] no readable files under --docs — retrieval disabled',
      );
    } else {
      stdout.writeln('[rag] embedding ${files.length} file(s)...');
      final built = _buildRagIndex(
        files: files,
        model: model,
        tokenizer: tok,
        chunkTokens: opts.chunkTokens,
        chunkStride: opts.chunkStride,
      );
      if (built != null) {
        ragIndex = built.index;
        ragMean = built.mean;
        ragChunks = built.chunks;
        stdout.writeln(
          '[rag] indexed ${ragChunks.length} chunks from ${files.length} file(s)',
        );
      }
    }
  }

  var system = opts.system;
  var chatText = _systemPrefix(system);
  var lastHits = const <({_RagChunk chunk, double score})>[];

  final ragOn = ragIndex != null;
  stdout.writeln(
    '\nReady. Type a message, or :quit / :reset / :sys <text>'
    '${ragOn ? " / :sources" : ""}. Ctrl+D to exit.\n',
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
    if (trimmed == ':sources') {
      if (!ragOn) {
        stdout.writeln('[no --docs indexed]');
      } else if (lastHits.isEmpty) {
        stdout.writeln('[no retrieval has run yet]');
      } else {
        for (var i = 0; i < lastHits.length; i++) {
          final h = lastHits[i];
          final preview = h.chunk.text.length > 140
              ? '${h.chunk.text.substring(0, 140).replaceAll('\n', ' ')}...'
              : h.chunk.text.replaceAll('\n', ' ');
          stdout.writeln(
            '  [${i + 1}] ${h.chunk.docTitle} '
            '(tok ${h.chunk.startTok}-${h.chunk.endTok}) '
            'score=${h.score.toStringAsFixed(3)}',
          );
          stdout.writeln('      $preview');
        }
      }
      continue;
    }

    // Retrieval — before we tokenise the user turn, so the augmented
    // message goes into the prompt.
    var userMsg = trimmed;
    if (ragOn) {
      lastHits = _retrieve(
        query: trimmed,
        model: model,
        tokenizer: tok,
        index: ragIndex,
        mean: ragMean!,
        chunks: ragChunks,
        k: opts.topKDocs,
      );
      userMsg = _augmentWithContext(trimmed, lastHits);
    }

    final prompt = chatText + _userTurn(userMsg);
    var promptIds = tok.encode(prompt);

    // If the conversation has grown past maxCtx, wipe history and retry
    // with just the current turn. Simpler than rebuilding turn structure.
    if (promptIds.length + opts.maxNew > cfg.maxCtx) {
      stdout.writeln(
        '[history exceeds context (${promptIds.length} + ${opts.maxNew} > '
        '${cfg.maxCtx}); resetting]',
      );
      chatText = _systemPrefix(system);
      promptIds = tok.encode(chatText + _userTurn(userMsg));
      if (promptIds.length + opts.maxNew > cfg.maxCtx) {
        stdout.writeln('[single turn still exceeds context — skipping]\n');
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
    // NOTE: we store the *augmented* user turn in history so future
    // turns can reference the retrieved excerpts. Turn on ":reset"
    // if you want a clean slate.
    chatText = '$prompt$answer<|eot_id|>';
  }
}
