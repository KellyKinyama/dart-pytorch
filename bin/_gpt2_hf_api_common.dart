/// Shared CLI + HTTP API runner for any HuggingFace GPT-2-family
/// model that our `GPT2HFLoader` can load (regular gpt2, gpt2-medium,
/// gpt2-large, gpt2-xl, distilgpt2, ...).
///
/// Callers just supply:
///   * a short model name (used in log lines and JSON responses),
///   * a default safetensors path,
///   * a `GPTConfig` factory that returns the matching config for a
///     chosen `Device` (CPU or GPU).
///
/// See `bin/gpt2/medium/run_gpu_api.dart` or
/// `bin/distilgpt2/run_gpu_api.dart` for concrete uses.
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

/// Entry point — parses CLI args, loads weights, then either runs a
/// single generation or starts the HTTP server (with `--serve`).
Future<void> runGpt2Api({
  required String modelName,
  required String defaultPath,
  required GPTConfig Function({required Device device}) configFactory,
  required List<String> args,
}) async {
  final o = _parseArgs(args, modelName: modelName, defaultPath: defaultPath);
  final m = _buildAndLoad(o, modelName, configFactory);
  if (o.serve) {
    await _runServer(m, o, modelName);
  } else {
    _runOnce(m, o);
  }
}

// ---------------------------------------------------------------------------
// CLI parsing
// ---------------------------------------------------------------------------

class _Options {
  _Options(this.defaultPath) : path = defaultPath;
  final String defaultPath;
  String path;
  String? vocabPath;
  String? promptText;
  List<double> prompt = <double>[464, 995, 318];
  int maxNewTokens = 20;
  double temperature = 0.0;
  int topK = 0;
  int? seed;
  bool useCache = true;
  bool cpu = false;
  bool serve = false;
  String host = '127.0.0.1';
  int port = 8080;
}

_Options _parseArgs(
  List<String> args, {
  required String modelName,
  required String defaultPath,
}) {
  final o = _Options(defaultPath);
  String need(int i, String flag) {
    if (i + 1 >= args.length) {
      stderr.writeln('missing value for $flag');
      exit(64);
    }
    return args[i + 1];
  }

  for (int i = 0; i < args.length; i++) {
    final a = args[i];
    switch (a) {
      case '--path':
        o.path = need(i, a);
        i++;
      case '--vocab':
        o.vocabPath = need(i, a);
        i++;
      case '--prompt':
        o.prompt = need(i, a)
            .split(',')
            .map((s) => s.trim())
            .where((s) => s.isNotEmpty)
            .map(double.parse)
            .toList();
        i++;
      case '--text':
        o.promptText = need(i, a);
        i++;
      case '--max-tokens':
        o.maxNewTokens = int.parse(need(i, a));
        i++;
      case '--temperature':
        o.temperature = double.parse(need(i, a));
        i++;
      case '--top-k':
        o.topK = int.parse(need(i, a));
        i++;
      case '--seed':
        o.seed = int.parse(need(i, a));
        i++;
      case '--no-cache':
        o.useCache = false;
      case '--cpu':
        o.cpu = true;
      case '--serve':
        o.serve = true;
      case '--host':
        o.host = need(i, a);
        i++;
      case '--port':
        o.port = int.parse(need(i, a));
        i++;
      case '-h':
      case '--help':
        _printUsage(modelName, defaultPath, stdout);
        exit(0);
      default:
        stderr.writeln('unknown flag: $a');
        _printUsage(modelName, defaultPath, stderr);
        exit(64);
    }
  }
  return o;
}

void _printUsage(String modelName, String defaultPath, IOSink out) {
  out.writeln('usage: dart run <this-file> [flags]');
  out.writeln('');
  out.writeln('Flags:');
  out.writeln('  --path PATH          safetensors file (default $defaultPath)');
  out.writeln(
    '  --vocab PATH         tokenizer.json or vocab.json (encode + decode)',
  );
  out.writeln('  --prompt IDS         comma-separated BPE token ids');
  out.writeln(
    '  --text  STR          prompt as text (BPE-encoded, needs tokenizer.json)',
  );
  out.writeln('  --max-tokens N       new tokens to generate (default 20)');
  out.writeln('  --temperature T      0 = greedy (default 0.0)');
  out.writeln('  --top-k K            0 = disabled (default 0)');
  out.writeln('  --seed S             RNG seed (default none)');
  out.writeln('  --no-cache           disable KV-cache fast path');
  out.writeln('  --cpu                build the model on CPU (default: GPU)');
  out.writeln('  --serve              run HTTP server');
  out.writeln('  --host H             HTTP bind host (default 127.0.0.1)');
  out.writeln('  --port P             HTTP port    (default 8080)');
}

// ---------------------------------------------------------------------------
// Model + generation
// ---------------------------------------------------------------------------

class _Loaded {
  _Loaded(
    this.gpt,
    this.cfg,
    this.tok,
    this.idToTok,
    this.modelPath,
    this.deviceLabel,
  );
  final GPT gpt;
  final GPTConfig cfg;

  /// Full BPE tokenizer (encode + decode) if `tokenizer.json` was
  /// found; else `null` and only [idToTok] is populated for decode.
  final HFBpeTokenizer? tok;
  final Map<int, String>? idToTok;
  final String modelPath;
  final String deviceLabel;
}

_Loaded _buildAndLoad(
  _Options o,
  String modelName,
  GPTConfig Function({required Device device}) configFactory,
) {
  final f = File(o.path);
  if (!f.existsSync()) {
    stderr.writeln('$modelName: file not found: ${o.path}');
    exit(66);
  }
  final device = o.cpu ? Device.CPU : Device.GPU;
  final deviceLabel = o.cpu ? 'cpu' : 'gpu';
  final cfg = configFactory(device: device);
  stdout.writeln(
    'Building GPT ($modelName, $deviceLabel, embed=${cfg.embedDim}, '
    'layers=${cfg.numLayers}, heads=${cfg.numHeads})...',
  );
  final gpt = GPT(cfg);

  stdout.writeln('Loading safetensors from ${o.path} ...');
  final t0 = DateTime.now();
  final report = GPT2HFLoader.loadFile(gpt, o.path);
  final dt = DateTime.now().difference(t0);
  stdout.writeln('Loaded in ${dt.inMilliseconds} ms. $report');

  // Prefer tokenizer.json (full encode + decode); fall back to
  // vocab.json (decode only, id → token-string map).
  final modelDir = File(o.path).parent.path;
  final tokJsonPath =
      o.vocabPath != null && o.vocabPath!.endsWith('tokenizer.json')
      ? o.vocabPath!
      : '$modelDir/tokenizer.json';
  final vocabJsonPath =
      o.vocabPath != null && !o.vocabPath!.endsWith('tokenizer.json')
      ? o.vocabPath!
      : '$modelDir/vocab.json';

  HFBpeTokenizer? tok;
  Map<int, String>? idToTok;
  if (File(tokJsonPath).existsSync()) {
    stdout.writeln('Loading BPE tokenizer from $tokJsonPath');
    try {
      tok = HFBpeTokenizer.loadFile(tokJsonPath);
    } catch (e) {
      stderr.writeln('tokenizer.json parse failed: $e');
    }
  }
  if (tok == null && File(vocabJsonPath).existsSync()) {
    stdout.writeln('Decoding token ids using $vocabJsonPath');
    final raw =
        jsonDecode(File(vocabJsonPath).readAsStringSync())
            as Map<String, dynamic>;
    idToTok = <int, String>{
      for (final e in raw.entries) (e.value as num).toInt(): e.key.toString(),
    };
  }
  if (tok == null && idToTok == null) {
    stdout.writeln(
      '(no tokenizer.json / vocab.json — text encode/decode disabled)',
    );
  }
  return _Loaded(gpt, cfg, tok, idToTok, o.path, deviceLabel);
}

String _decode(_Loaded m, List<int> ids) {
  if (m.tok != null) return m.tok!.decode(ids);
  final idToTok = m.idToTok;
  if (idToTok == null) return '';
  final sb = StringBuffer();
  for (final id in ids) {
    final s = idToTok[id];
    if (s == null) continue;
    sb.write(s.replaceAll('\u0120', ' ').replaceAll('\u010A', '\n'));
  }
  return sb.toString();
}

class _GenRequest {
  _GenRequest({
    required this.tokens,
    this.maxNewTokens = 20,
    this.temperature = 0.0,
    this.topK = 0,
    this.seed,
    this.useCache = true,
  });
  final List<double> tokens;
  final int maxNewTokens;
  final double temperature;
  final int topK;
  final int? seed;
  final bool useCache;
}

class _GenResult {
  _GenResult(this.tokens, this.newTokens, this.text, this.elapsedMs);
  final List<int> tokens;
  final List<int> newTokens;
  final String text;
  final int elapsedMs;

  Map<String, dynamic> toJson() => <String, dynamic>{
    'tokens': tokens,
    'newTokens': newTokens,
    'text': text,
    'elapsedMs': elapsedMs,
  };
}

_GenResult _runGeneration(_Loaded m, _GenRequest req) {
  if (req.tokens.isEmpty) {
    throw ArgumentError('tokens must be non-empty');
  }
  final t0 = DateTime.now();
  final rng = req.seed != null ? math.Random(req.seed!) : null;
  final gen = m.gpt.generate(
    req.tokens,
    maxNewTokens: req.maxNewTokens,
    temperature: req.temperature,
    topK: req.topK <= 0 ? null : req.topK,
    rng: rng,
    useCache: req.useCache,
  );
  final ids = gen.map((d) => d.toInt()).toList();
  final promptLen = req.tokens.length;
  final newIds = ids.sublist(promptLen);
  final text = _decode(m, ids);
  return _GenResult(
    ids,
    newIds,
    text,
    DateTime.now().difference(t0).inMilliseconds,
  );
}

// ---------------------------------------------------------------------------
// One-shot mode
// ---------------------------------------------------------------------------

void _runOnce(_Loaded m, _Options o) {
  final prompt = _resolvePrompt(m, o);
  stdout.writeln('');
  stdout.writeln(
    'Generating: prompt=${prompt.map((d) => d.toInt()).toList()} '
    'max=${o.maxNewTokens} temp=${o.temperature} topK=${o.topK} '
    'seed=${o.seed ?? "none"} cache=${o.useCache}',
  );
  final res = _runGeneration(
    m,
    _GenRequest(
      tokens: prompt,
      maxNewTokens: o.maxNewTokens,
      temperature: o.temperature,
      topK: o.topK,
      seed: o.seed,
      useCache: o.useCache,
    ),
  );
  stdout.writeln('  ids : ${res.tokens}');
  if (m.tok != null || m.idToTok != null) {
    stdout.writeln('  text: "${res.text}"');
  }
  stdout.writeln('  ${res.elapsedMs} ms for ${res.newTokens.length} tokens');
}

List<double> _resolvePrompt(_Loaded m, _Options o) {
  if (o.promptText != null) {
    if (m.tok == null) {
      stderr.writeln(
        '--text supplied but no tokenizer.json loaded (vocab.json is decode-only)',
      );
      exit(65);
    }
    final ids = m.tok!.encode(o.promptText!);
    if (ids.isEmpty) {
      stderr.writeln('--text tokenized to zero ids');
      exit(65);
    }
    return ids.map((i) => i.toDouble()).toList();
  }
  return o.prompt;
}

// ---------------------------------------------------------------------------
// HTTP server mode
// ---------------------------------------------------------------------------

Future<void> _runServer(_Loaded m, _Options o, String modelName) async {
  final server = await HttpServer.bind(o.host, o.port);
  stdout.writeln(
    '$modelName (${m.deviceLabel}) listening on '
    'http://${o.host}:${o.port}',
  );
  stdout.writeln('  GET  /health');
  stdout.writeln('  GET  /info');
  stdout.writeln('  POST /generate  (JSON body — see file header)');

  await for (final req in server) {
    try {
      await _handle(m, req, modelName);
    } catch (e, st) {
      stderr.writeln('handler error: $e\n$st');
      try {
        req.response
          ..statusCode = 500
          ..headers.contentType = ContentType.json
          ..write(jsonEncode({'error': e.toString()}));
        await req.response.close();
      } catch (_) {}
    }
  }
}

Future<void> _handle(_Loaded m, HttpRequest req, String modelName) async {
  final method = req.method;
  final path = req.uri.path;

  void writeJson(int status, Object body) {
    req.response
      ..statusCode = status
      ..headers.contentType = ContentType.json
      ..write(jsonEncode(body));
  }

  if (method == 'GET' && path == '/health') {
    writeJson(200, {
      'status': 'ok',
      'model': modelName,
      'device': m.deviceLabel,
      'weights': m.modelPath,
    });
  } else if (method == 'GET' && path == '/info') {
    writeJson(200, {
      'model': modelName,
      'device': m.deviceLabel,
      'embedDim': m.cfg.embedDim,
      'numLayers': m.cfg.numLayers,
      'numHeads': m.cfg.numHeads,
      'vocabSize': m.cfg.vocabSize,
      'maxCtx': m.cfg.maxCtx,
    });
  } else if (method == 'POST' && path == '/generate') {
    final body = await utf8.decoder.bind(req).join();
    final Map<String, dynamic> json = (jsonDecode(body) as Map)
        .cast<String, dynamic>();
    final rawTokens = (json['tokens'] as List?) ?? const [];
    final textField = json['text'] as String?;
    List<double> tokens;
    if (textField != null && textField.isNotEmpty) {
      if (m.tok == null) {
        writeJson(400, {
          'error':
              'field "text" supplied but no tokenizer.json loaded '
              '(vocab.json is decode-only)',
        });
        await req.response.close();
        return;
      }
      tokens = m.tok!.encode(textField).map((i) => i.toDouble()).toList();
    } else if (rawTokens.isNotEmpty) {
      tokens = rawTokens.map((e) => (e as num).toDouble()).toList();
    } else {
      writeJson(400, {
        'error': 'either "text" or non-empty "tokens" is required',
      });
      await req.response.close();
      return;
    }
    final gr = _GenRequest(
      tokens: tokens,
      maxNewTokens: (json['maxNewTokens'] as num?)?.toInt() ?? 20,
      temperature: (json['temperature'] as num?)?.toDouble() ?? 0.0,
      topK: (json['topK'] as num?)?.toInt() ?? 0,
      seed: (json['seed'] as num?)?.toInt(),
      useCache: (json['useCache'] as bool?) ?? true,
    );
    final res = _runGeneration(m, gr);
    writeJson(200, res.toJson());
  } else {
    writeJson(404, {'error': 'not found: $method $path'});
  }
  await req.response.close();
}
