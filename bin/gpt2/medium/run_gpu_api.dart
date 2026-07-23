/// Full-featured HF GPT-2 **medium** runner on the **GPU**, exposing
/// configurable sampling parameters and an optional HTTP API.
///
/// This is the "API" companion to `run_gpu.dart` — same underlying
/// model, but with everything the plain demo hardcodes turned into a
/// CLI flag (or an HTTP request body).
///
/// ## Quickstart
///
/// ```sh
///   # 1. Make sure the weights are on disk (only needed once):
///   mkdir -p models/gpt2-medium
///   curl -L --progress-bar \
///     -o models/gpt2-medium/model.safetensors \
///     https://huggingface.co/gpt2-medium/resolve/main/model.safetensors
///   curl -L -o models/gpt2-medium/vocab.json \
///     https://huggingface.co/gpt2-medium/resolve/main/vocab.json
///
///   # 2. Single-shot generation with custom parameters:
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib \
///     dart run bin/gpt2/medium/run_gpu_api.dart \
///       --prompt 464,995,318 \
///       --max-tokens 40 \
///       --temperature 0.8 \
///       --top-k 40 \
///       --seed 42
///
///   # 3. Or run it as a local HTTP server:
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib \
///     dart run bin/gpt2/medium/run_gpu_api.dart --serve --port 8080
///
///   # Then from another shell:
///   curl -s http://127.0.0.1:8080/health
///   curl -s -X POST http://127.0.0.1:8080/generate \
///     -H 'content-type: application/json' \
///     -d '{"tokens":[464,995,318],"maxNewTokens":20,"temperature":0.8,"topK":40,"seed":42}'
/// ```
///
/// ## CLI flags
///
///   --path PATH          safetensors file (default: models/gpt2-medium/model.safetensors)
///   --vocab PATH         vocab.json for pretty-printing (default: sibling of --path)
///   --prompt IDS         comma-separated GPT-2 BPE token ids (default: 464,995,318 = "The world is")
///   --max-tokens N       new tokens to generate (default: 20)
///   --temperature T      sampling temperature; 0 = greedy (default: 0.0)
///   --top-k K            top-k sampling; 0 = disabled (default: 0)
///   --seed S             RNG seed for sampling (default: none)
///   --no-cache           disable the KV-cache fast path (slower, for parity checks)
///   --serve              run as HTTP server instead of one-shot
///   --host H             HTTP bind host (default: 127.0.0.1)
///   --port P             HTTP port    (default: 8080)
///
/// ## HTTP API
///
///   GET  /health   -> `{"status":"ok", ...model info}`
///   GET  /info     -> model config summary
///   POST /generate -> JSON body `{tokens:[int], maxNewTokens:int?,
///                                 temperature:double?, topK:int?,
///                                 seed:int?, useCache:bool?}`
///                     returns `{tokens:[int], newTokens:[int],
///                              text:string, elapsedMs:int}`
///
/// Fits in a 6 GB GPU (weights ~1.5 GB + activations for short
/// contexts).
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

// ---------------------------------------------------------------------------
// CLI parsing
// ---------------------------------------------------------------------------

class _Options {
  String path = 'models/gpt2-medium/model.safetensors';
  String? vocabPath;
  List<double> prompt = <double>[464, 995, 318];
  int maxNewTokens = 20;
  double temperature = 0.0;
  int topK = 0;
  int? seed;
  bool useCache = true;
  bool serve = false;
  String host = '127.0.0.1';
  int port = 8080;
}

_Options _parseArgs(List<String> args) {
  final o = _Options();
  String? need(int i, String flag) {
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
        o.path = need(i, a)!;
        i++;
      case '--vocab':
        o.vocabPath = need(i, a)!;
        i++;
      case '--prompt':
        final raw = need(i, a)!;
        o.prompt = raw
            .split(',')
            .map((s) => s.trim())
            .where((s) => s.isNotEmpty)
            .map((s) => double.parse(s))
            .toList();
        i++;
      case '--max-tokens':
        o.maxNewTokens = int.parse(need(i, a)!);
        i++;
      case '--temperature':
        o.temperature = double.parse(need(i, a)!);
        i++;
      case '--top-k':
        o.topK = int.parse(need(i, a)!);
        i++;
      case '--seed':
        o.seed = int.parse(need(i, a)!);
        i++;
      case '--no-cache':
        o.useCache = false;
      case '--serve':
        o.serve = true;
      case '--host':
        o.host = need(i, a)!;
        i++;
      case '--port':
        o.port = int.parse(need(i, a)!);
        i++;
      case '-h':
      case '--help':
        _printUsageAndExit(0);
      default:
        stderr.writeln('unknown flag: $a');
        _printUsageAndExit(64);
    }
  }
  return o;
}

Never _printUsageAndExit(int code) {
  final out = code == 0 ? stdout : stderr;
  out.writeln('usage: dart run bin/gpt2/medium/run_gpu_api.dart [flags]');
  out.writeln('');
  out.writeln('Flags:');
  out.writeln('  --path PATH          safetensors file');
  out.writeln('  --vocab PATH         vocab.json for pretty-printing');
  out.writeln('  --prompt IDS         comma-separated GPT-2 BPE token ids');
  out.writeln('  --max-tokens N       new tokens to generate (default 20)');
  out.writeln('  --temperature T      0 = greedy (default 0.0)');
  out.writeln('  --top-k K            0 = disabled (default 0)');
  out.writeln('  --seed S             RNG seed (default none)');
  out.writeln('  --no-cache           disable KV-cache fast path');
  out.writeln('  --serve              run HTTP server');
  out.writeln('  --host H             HTTP bind host (default 127.0.0.1)');
  out.writeln('  --port P             HTTP port    (default 8080)');
  exit(code);
}

// ---------------------------------------------------------------------------
// Model + generation
// ---------------------------------------------------------------------------

class _Loaded {
  _Loaded(this.gpt, this.cfg, this.idToTok, this.modelPath);
  final GPT gpt;
  final GPTConfig cfg;
  final Map<int, String>? idToTok;
  final String modelPath;
}

_Loaded _buildAndLoad(_Options o) {
  final f = File(o.path);
  if (!f.existsSync()) {
    stderr.writeln('gpt2 medium (gpu): file not found: ${o.path}');
    exit(66);
  }
  final cfg = GPT2HFLoader.gpt2MediumConfig(device: Device.GPU);
  stdout.writeln(
    'Building GPT (gpt2-medium, gpu, embed=${cfg.embedDim}, '
    'layers=${cfg.numLayers}, heads=${cfg.numHeads})...',
  );
  final gpt = GPT(cfg);

  stdout.writeln('Loading safetensors from ${o.path} ...');
  final t0 = DateTime.now();
  final report = GPT2HFLoader.loadFile(gpt, o.path);
  final dt = DateTime.now().difference(t0);
  stdout.writeln('Loaded in ${dt.inMilliseconds} ms. $report');

  final vocabPath = o.vocabPath ?? '${File(o.path).parent.path}/vocab.json';
  Map<int, String>? idToTok;
  if (File(vocabPath).existsSync()) {
    stdout.writeln('Decoding token ids using $vocabPath');
    final raw =
        jsonDecode(File(vocabPath).readAsStringSync()) as Map<String, dynamic>;
    idToTok = <int, String>{
      for (final e in raw.entries) (e.value as num).toInt(): e.key.toString(),
    };
  } else {
    stdout.writeln('(no vocab.json at $vocabPath — token strings unavailable)');
  }
  return _Loaded(gpt, cfg, idToTok, o.path);
}

String _decode(Map<int, String>? idToTok, List<int> ids) {
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
  final text = _decode(m.idToTok, ids);
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
  stdout.writeln('');
  stdout.writeln(
    'Generating: prompt=${o.prompt.map((d) => d.toInt()).toList()} '
    'max=${o.maxNewTokens} temp=${o.temperature} topK=${o.topK} '
    'seed=${o.seed ?? "none"} cache=${o.useCache}',
  );
  final res = _runGeneration(
    m,
    _GenRequest(
      tokens: o.prompt,
      maxNewTokens: o.maxNewTokens,
      temperature: o.temperature,
      topK: o.topK,
      seed: o.seed,
      useCache: o.useCache,
    ),
  );
  stdout.writeln('  ids : ${res.tokens}');
  if (m.idToTok != null) {
    stdout.writeln('  text: "${res.text}"');
  }
  stdout.writeln('  ${res.elapsedMs} ms for ${res.newTokens.length} tokens');
}

// ---------------------------------------------------------------------------
// HTTP server mode
// ---------------------------------------------------------------------------

Future<void> _runServer(_Loaded m, _Options o) async {
  final server = await HttpServer.bind(o.host, o.port);
  stdout.writeln('gpt2-medium (gpu) listening on http://${o.host}:${o.port}');
  stdout.writeln('  GET  /health');
  stdout.writeln('  GET  /info');
  stdout.writeln('  POST /generate  (JSON body — see file header)');

  await for (final req in server) {
    try {
      await _handle(m, req);
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

Future<void> _handle(_Loaded m, HttpRequest req) async {
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
      'model': 'gpt2-medium',
      'device': 'gpu',
      'weights': m.modelPath,
    });
  } else if (method == 'GET' && path == '/info') {
    writeJson(200, {
      'model': 'gpt2-medium',
      'device': 'gpu',
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
    if (rawTokens.isEmpty) {
      writeJson(400, {'error': 'field "tokens" is required and non-empty'});
    } else {
      final gr = _GenRequest(
        tokens: rawTokens.map((e) => (e as num).toDouble()).toList(),
        maxNewTokens: (json['maxNewTokens'] as num?)?.toInt() ?? 20,
        temperature: (json['temperature'] as num?)?.toDouble() ?? 0.0,
        topK: (json['topK'] as num?)?.toInt() ?? 0,
        seed: (json['seed'] as num?)?.toInt(),
        useCache: (json['useCache'] as bool?) ?? true,
      );
      final res = _runGeneration(m, gr);
      writeJson(200, res.toJson());
    }
  } else {
    writeJson(404, {'error': 'not found: $method $path'});
  }
  await req.response.close();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

Future<void> main(List<String> args) async {
  final o = _parseArgs(args);
  final m = _buildAndLoad(o);
  if (o.serve) {
    await _runServer(m, o);
  } else {
    _runOnce(m, o);
  }
}
