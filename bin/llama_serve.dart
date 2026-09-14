/// HTTP server for Llama-3 models — same wire protocol as the GPT-2
/// `bin/*_api.dart` runners, so the RAG chat server ([R15]) picks it up
/// with just a URL change.
///
/// Two shells:
///
///   # shell A — LLM (instruction-tuned Llama-3.2-1B by default)
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib \
///     dart run bin/llama_serve.dart --gpu --port 8080
///
///   # shell B — RAG chat + browser UI (unchanged)
///   dart run bin/db_rag_chat_server.dart \
///       --llm http://127.0.0.1:8080 --port 8090 \
///       --corpus data/support_faq.txt
///
///   # open http://127.0.0.1:8090/ and chat.
///
/// Endpoints (matching bin/_gpt2_hf_api_common.dart so drop-in works):
///
///   GET  /health   { status, model, device, weights, chat_template }
///   GET  /info     { model, embedDim, numLayers, numHeads, vocabSize, maxCtx, eot_id }
///   POST /generate JSON {
///     "text":         `"<prompt>"`,        // required (or "messages")
///     "messages":     `[ {role, content}, ... ]`,  // optional structured form
///     "system":       "You are ...",       // optional; wraps chat template
///     "maxNewTokens": 128,                 // default 128
///     "temperature":  0.7,                 // default 0.7 (Llama sweet spot)
///     "topK":         40,                  // default 40; 0 disables
///     "seed":         42                   // optional deterministic sampling
///   } → { text: prompt+reply, newTokens: [...], elapsedMs }
///
/// Flags:
///
///   --path PATH        weights safetensors
///                        (default: models/llama-3.2-1b-instruct/model.safetensors)
///   --vocab PATH       tokenizer.json
///                        (default: models/llama-3.2-1b-instruct/tokenizer.json)
///   --preset NAME      llama-3.2-1b | llama-3.2-3b | llama-3.1-8b
///                        (default: llama-3.2-1b)
///   --gpu              run on CUDA (default: CPU)
///   --host H           HTTP bind host (default 127.0.0.1)
///   --port P           HTTP port (default 8080)
///   --system "..."     default system prompt inserted into the chat template
///   --raw              disable chat-template wrapping (send `text` verbatim
///                        to model.generate). Use this if your caller already
///                        formats Llama chat markers into `text`.
///
/// Chat-template wrapping (default ON — this is what makes Llama-instruct
/// actually follow instructions instead of pattern-matching the input):
///
///   <|begin_of_text|>
///   <|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>
///   <|start_header_id|>user<|end_header_id|>\n\n{client_text}<|eot_id|>
///   <|start_header_id|>assistant<|end_header_id|>\n\n
///
/// Generation stops at the first `<|eot_id|>` in the assistant reply so
/// the model doesn't drift into a synthetic user turn.
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

import '_llama_encoder.dart';

Future<void> main(List<String> args) async {
  final o = _parseArgs(args);

  if (!File(o.path).existsSync()) {
    stderr.writeln('llama_serve: weights not found: ${o.path}');
    exit(66);
  }
  if (!File(o.vocabPath).existsSync()) {
    stderr.writeln('llama_serve: tokenizer.json not found: ${o.vocabPath}');
    exit(66);
  }

  final loaded = loadLlamaEncoder(
    path: o.path,
    vocabPath: o.vocabPath,
    preset: o.preset,
    gpu: o.gpu,
  );
  final model = loaded.model;
  final tok = loaded.tokenizer;
  final cfg = loaded.config;

  final eot = tok.llamaEotId ?? tok.endOfTextId;
  if (eot == null) {
    stderr.writeln(
      'llama_serve: tokenizer defines neither <|eot_id|> nor <|endoftext|> '
      '— generation will not stop early on chat boundaries',
    );
  }
  final deviceLabel = o.gpu ? 'gpu' : 'cpu';

  if (o.oneShot != null) {
    _oneShot(
      model: model,
      tok: tok,
      cfg: cfg,
      eot: eot,
      prompt: o.oneShot!,
      system: o.system,
      chatMode: !o.raw,
      maxNew: o.maxNew,
      temperature: o.temperature,
      topK: o.topK,
      seed: o.seed,
    );
    return;
  }

  final serverInfo = _ServerInfo(
    modelName: o.preset,
    modelPath: o.path,
    deviceLabel: deviceLabel,
    cfg: cfg,
    eot: eot,
    tok: tok,
    model: model,
    defaultSystem: o.system,
    chatModeDefault: !o.raw,
  );

  final server = await HttpServer.bind(o.host, o.port);
  stdout.writeln('llama_serve: listening on http://${o.host}:${o.port}');
  stdout.writeln(
    '  GET  /health   |  GET  /info   |  POST /generate  '
    '(chat_template=${!o.raw ? "on" : "off"})',
  );

  ProcessSignal.sigint.watch().listen((_) async {
    stdout.writeln('\nllama_serve: shutting down');
    await server.close(force: true);
    exit(0);
  });

  await for (final req in server) {
    _handle(req, serverInfo).catchError((Object e, StackTrace st) {
      stderr.writeln(
        'llama_serve: handler error on ${req.method} ${req.uri.path}: $e\n$st',
      );
      try {
        req.response
          ..statusCode = 500
          ..headers.contentType = ContentType.json
          ..write(jsonEncode({'error': e.toString()}));
        req.response.close();
      } catch (_) {}
    });
  }
}

// ---------------------------------------------------------------------------
// HTTP routing.
// ---------------------------------------------------------------------------

class _ServerInfo {
  final String modelName;
  final String modelPath;
  final String deviceLabel;
  final LlamaConfig cfg;
  final int? eot;
  final HFBpeTokenizer tok;
  final Llama model;
  final String defaultSystem;
  final bool chatModeDefault;

  _ServerInfo({
    required this.modelName,
    required this.modelPath,
    required this.deviceLabel,
    required this.cfg,
    required this.eot,
    required this.tok,
    required this.model,
    required this.defaultSystem,
    required this.chatModeDefault,
  });
}

Future<void> _handle(HttpRequest req, _ServerInfo info) async {
  final method = req.method;
  final path = req.uri.path;
  final startedAt = DateTime.now();
  final remote =
      '${req.connectionInfo?.remoteAddress.address ?? '?'}:${req.connectionInfo?.remotePort ?? 0}';
  int status = 200;
  int? bytesIn;
  String? extra;

  void writeJson(int s, Object body) {
    status = s;
    req.response
      ..statusCode = s
      ..headers.contentType = ContentType.json
      ..headers.set('access-control-allow-origin', '*')
      ..write(jsonEncode(body));
  }

  try {
    if (method == 'GET' && path == '/health') {
      writeJson(200, {
        'status': 'ok',
        'model': info.modelName,
        'device': info.deviceLabel,
        'weights': info.modelPath,
        'chat_template': info.chatModeDefault,
      });
    } else if (method == 'GET' && path == '/info') {
      writeJson(200, {
        'model': info.modelName,
        'device': info.deviceLabel,
        'embedDim': info.cfg.embedDim,
        'numLayers': info.cfg.numLayers,
        'numHeads': info.cfg.numHeads,
        'numKvHeads': info.cfg.numKvHeads,
        'vocabSize': info.cfg.vocabSize,
        'maxCtx': info.cfg.maxCtx,
        'eot_id': info.eot,
      });
    } else if (method == 'POST' && path == '/generate') {
      final body = await utf8.decoder.bind(req).join();
      bytesIn = body.length;
      final json = (jsonDecode(body) as Map).cast<String, dynamic>();
      // Print inputs BEFORE generation so the user sees activity while
      // the (blocking) forward pass runs.
      _logGenerateRequest(json);
      final result = await _runGeneration(info, json);
      extra =
          'prompt=${result['promptTokens']} new=${result['newTokensCount']} '
          'gen_ms=${result['elapsedMs']}';
      writeJson(200, result);
    } else {
      writeJson(404, {'error': 'not found: $method $path'});
    }
  } catch (e) {
    writeJson(400, {'error': e.toString()});
    extra = 'err="${e.toString()}"';
  }
  await req.response.close();
  final ms = DateTime.now().difference(startedAt).inMilliseconds;
  final ts = startedAt.toIso8601String().substring(11, 23);
  final inPart = bytesIn == null ? '' : ' in=${bytesIn}B';
  final tail = extra == null ? '' : ' :: $extra';
  stdout.writeln(
    '[$ts] $remote  $method $path  $status$inPart  ${ms}ms$tail',
  );
}

// Show what came in on /generate: prompt preview, sampling knobs.
void _logGenerateRequest(Map<String, dynamic> json) {
  final text = (json['text'] as String?) ?? '';
  final messages = json['messages'] as List?;
  final preview = messages != null
      ? 'messages(${messages.length})'
      : '"${_ellipsize(text.replaceAll('\n', ' '), 80)}"';
  final maxNew = (json['maxNewTokens'] as num?)?.toInt() ?? 128;
  final temp = (json['temperature'] as num?)?.toDouble() ?? 0.7;
  final topK = (json['topK'] as num?)?.toInt() ?? 40;
  stdout.writeln(
    '  → prompt: $preview  '
    '(maxNew=$maxNew temp=$temp topK=$topK)',
  );
}

String _ellipsize(String s, int max) =>
    s.length <= max ? s : '${s.substring(0, max - 1)}\u2026';

// ---------------------------------------------------------------------------
// Chat-template wrapping + generation.
// ---------------------------------------------------------------------------

Future<Map<String, Object?>> _runGeneration(
  _ServerInfo info,
  Map<String, dynamic> json,
) async {
  final clientText = (json['text'] as String?) ?? '';
  final messages = json['messages'] as List?;
  final rawText = messages == null && (json['raw'] == true);
  final chatMode = !rawText && info.chatModeDefault;
  final system = (json['system'] as String?) ?? info.defaultSystem;
  final maxNew = (json['maxNewTokens'] as num?)?.toInt() ?? 128;
  final temperature = (json['temperature'] as num?)?.toDouble() ?? 0.7;
  final topK = (json['topK'] as num?)?.toInt() ?? 40;
  final seedRaw = json['seed'];
  final seed = seedRaw is num ? seedRaw.toInt() : null;

  if (clientText.isEmpty && messages == null) {
    throw ArgumentError('either "text" or "messages" is required');
  }

  // Build the model-side prompt.
  final String modelPrompt;
  if (messages != null) {
    modelPrompt = _renderMessages(system, messages);
  } else if (chatMode) {
    modelPrompt = _renderMessages(system, [
      {'role': 'user', 'content': clientText},
    ]);
  } else {
    modelPrompt = clientText;
  }

  final promptIds = info.tok.encode(modelPrompt);
  if (promptIds.length + maxNew > info.cfg.maxCtx) {
    throw StateError(
      'prompt (${promptIds.length} tok) + maxNewTokens ($maxNew) exceeds '
      'model context ${info.cfg.maxCtx}',
    );
  }

  final t0 = DateTime.now();
  final rng = seed != null ? math.Random(seed) : null;
  final full = info.model.generate(
    promptIds.map((i) => i.toDouble()).toList(),
    maxNewTokens: maxNew,
    temperature: temperature,
    topK: topK <= 0 ? null : topK,
    rng: rng,
  );
  final elapsed = DateTime.now().difference(t0).inMilliseconds;

  final newIds = full.sublist(promptIds.length).map((d) => d.toInt()).toList();
  var truncated = newIds;
  if (info.eot != null) {
    final i = newIds.indexOf(info.eot!);
    if (i >= 0) truncated = newIds.sublist(0, i);
  }
  final assistantReply = info.tok.decode(truncated).trim();

  // Wire-compat with bin/*_api.dart: return { text: clientPrompt + reply, ... }
  // so callers that strip the prefix (like db_rag_chat_server.dart) see only
  // the assistant's actual reply.
  final replyForClient = _replyForClient(clientText, messages, assistantReply);
  return {
    'model': info.modelName,
    'text': replyForClient,
    'newTokens': truncated,
    'elapsedMs': elapsed,
    'promptTokens': promptIds.length,
    'newTokensCount': truncated.length,
  };
}

// Assemble the returned `text` field so the client's `startsWith(prompt)`
// stripping trick still works. For text-mode clients, prepend the raw
// prompt; for messages-mode clients, just return the assistant reply.
String _replyForClient(
  String clientText,
  List? messages,
  String assistantReply,
) {
  if (messages != null) return assistantReply;
  return clientText + assistantReply;
}

/// Render the Llama-3 chat template. `messages` is a list of
/// `{role: "system"|"user"|"assistant", content: "..."}` maps. A `system`
/// argument (if non-empty) is prepended when the messages don't already
/// start with a system role.
String _renderMessages(String system, List messages) {
  final sb = StringBuffer('<|begin_of_text|>');
  final normalized = <_ChatMessage>[];
  final hasLeadingSystem =
      messages.isNotEmpty &&
      (messages.first as Map)['role']?.toString().toLowerCase() == 'system';
  if (!hasLeadingSystem && system.trim().isNotEmpty) {
    normalized.add(_ChatMessage('system', system.trim()));
  }
  for (final m in messages) {
    final map = (m as Map).cast<String, Object?>();
    final role = (map['role'] as String?)?.toLowerCase() ?? 'user';
    final content = (map['content'] as String?) ?? '';
    normalized.add(_ChatMessage(role, content));
  }
  for (final m in normalized) {
    sb.write(
      '<|start_header_id|>${m.role}<|end_header_id|>\n\n${m.content}<|eot_id|>',
    );
  }
  sb.write('<|start_header_id|>assistant<|end_header_id|>\n\n');
  return sb.toString();
}

class _ChatMessage {
  final String role;
  final String content;
  _ChatMessage(this.role, this.content);
}

// ---------------------------------------------------------------------------
// One-shot smoke test — bypasses the HTTP path.
// ---------------------------------------------------------------------------

void _oneShot({
  required Llama model,
  required HFBpeTokenizer tok,
  required LlamaConfig cfg,
  required int? eot,
  required String prompt,
  required String system,
  required bool chatMode,
  required int maxNew,
  required double temperature,
  required int topK,
  required int? seed,
}) {
  final modelPrompt = chatMode
      ? _renderMessages(system, [
          {'role': 'user', 'content': prompt},
        ])
      : prompt;
  final promptIds = tok.encode(modelPrompt);
  stdout.writeln(
    '\n== one-shot generation (${promptIds.length} prompt tok) ==',
  );
  final t0 = DateTime.now();
  final full = model.generate(
    promptIds.map((i) => i.toDouble()).toList(),
    maxNewTokens: maxNew,
    temperature: temperature,
    topK: topK <= 0 ? null : topK,
    rng: seed != null ? math.Random(seed) : null,
  );
  final elapsed = DateTime.now().difference(t0).inMilliseconds;
  final newIds = full.sublist(promptIds.length).map((d) => d.toInt()).toList();
  var truncated = newIds;
  if (eot != null) {
    final i = newIds.indexOf(eot);
    if (i >= 0) truncated = newIds.sublist(0, i);
  }
  final reply = tok.decode(truncated).trim();
  stdout.writeln('user> $prompt');
  stdout.writeln('bot > $reply');
  stdout.writeln(
    '(${truncated.length} new tok in $elapsed ms  '
    '= ${(truncated.length * 1000 / elapsed).toStringAsFixed(1)} tok/s)',
  );
}

// ---------------------------------------------------------------------------
// CLI parsing.
// ---------------------------------------------------------------------------

class _Opts {
  String path;
  String vocabPath;
  String preset;
  bool gpu;
  bool raw;
  String system;
  String host;
  int port;
  int maxNew;
  double temperature;
  int topK;
  int? seed;
  String? oneShot;
  _Opts({
    required this.path,
    required this.vocabPath,
    required this.preset,
    required this.gpu,
    required this.raw,
    required this.system,
    required this.host,
    required this.port,
    required this.maxNew,
    required this.temperature,
    required this.topK,
    required this.seed,
    required this.oneShot,
  });
}

_Opts _parseArgs(List<String> args) {
  final o = _Opts(
    path: 'models/llama-3.2-1b-instruct/model.safetensors',
    vocabPath: 'models/llama-3.2-1b-instruct/tokenizer.json',
    preset: 'llama-3.2-1b',
    gpu: false,
    raw: false,
    system:
        'You are a helpful, concise assistant. Answer using only the '
        'context the user provides. If the context does not contain the '
        "answer, say \"I don't know based on the provided context.\"",
    host: '127.0.0.1',
    port: 8080,
    maxNew: 128,
    temperature: 0.7,
    topK: 40,
    seed: null,
    oneShot: null,
  );
  for (var i = 0; i < args.length; i++) {
    final a = args[i];
    String need() {
      if (i + 1 >= args.length) {
        stderr.writeln('llama_serve: missing value for $a');
        exit(64);
      }
      return args[++i];
    }

    switch (a) {
      case '--path':
        o.path = need();
      case '--vocab':
        o.vocabPath = need();
      case '--preset':
        o.preset = need();
      case '--gpu':
        o.gpu = true;
      case '--raw':
        o.raw = true;
      case '--system':
        o.system = need();
      case '--host':
        o.host = need();
      case '--port':
        o.port = int.parse(need());
      case '--max-new':
        o.maxNew = int.parse(need());
      case '--temperature':
        o.temperature = double.parse(need());
      case '--top-k':
        o.topK = int.parse(need());
      case '--seed':
        o.seed = int.parse(need());
      case '--text':
        // One-shot smoke test — generate one reply, print, exit.
        o.oneShot = need();
      case '--serve':
        // No-op: this binary is a server by default, but --serve is
        // accepted for CLI parity with the GPT-2 runners.
        break;
      case '-h' || '--help':
        stdout.writeln(_help);
        exit(0);
      default:
        stderr.writeln('llama_serve: unknown flag "$a" (see --help)');
        exit(64);
    }
  }
  return o;
}

const _help = '''
Llama-3 HTTP inference server.

Usage:
  dart run bin/llama_serve.dart [flags]

Model:
  --path PATH        safetensors (default: models/llama-3.2-1b-instruct/model.safetensors)
  --vocab PATH       tokenizer.json (default: models/llama-3.2-1b-instruct/tokenizer.json)
  --preset NAME      llama-3.2-1b | llama-3.2-3b | llama-3.1-8b (default: llama-3.2-1b)
  --gpu              run on CUDA

Server:
  --host H           bind host (default 127.0.0.1)
  --port P           bind port (default 8080)
  --system "..."     default system prompt for the chat template
  --raw              disable chat-template wrapping (send `text` verbatim)

Sampling:
  --max-new N        (default 128)
  --temperature F    (default 0.7)
  --top-k K          (default 40; 0 disables)
  --seed S           deterministic sampling

One-shot smoke test:
  --text "..."       generate once, print, exit (bypasses HTTP)

Examples:
  # Serve Llama-3.2-1B-Instruct on the default port
  LD_LIBRARY_PATH=/usr/lib/wsl/lib \\
    dart run bin/llama_serve.dart --gpu

  # Wire it into the RAG chat server (unchanged)
  dart run bin/db_rag_chat_server.dart \\
      --llm http://127.0.0.1:8080 --port 8090 \\
      --corpus data/support_faq.txt

  # Client call (matches bin/*_api.dart /generate)
  curl -sS -X POST http://127.0.0.1:8080/generate \\
    -H 'content-type: application/json' \\
    -d '{"text":"Explain BM25 in one sentence."}'
''';
