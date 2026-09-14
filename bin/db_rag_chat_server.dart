/// Browser chat UI over the R14 pipeline. Same split-of-responsibilities
/// as [bin/db_rag_http_demo.dart] — MiniLM in-process for embeddings,
/// dart-db-server in-process for retrieval, external HTTP LLM for
/// generation — but wrapped in an HTTP server that serves its own
/// single-file chat page. Open the URL, type in the browser, get
/// grounded answers.
///
/// Two shells:
///
///   # shell A — start any bin/*_api.dart runner as the LLM
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib \
///     dart run bin/distilgpt2/run_gpu_api.dart --serve --port 8080
///
///   # shell B — start the chat server + browser UI
///   dart run bin/db_rag_chat_server.dart \
///       --llm http://127.0.0.1:8080 \
///       --port 8090 \
///       --corpus data/support_faq.txt
///
/// Then open http://127.0.0.1:8090/ in a browser.
///
/// Endpoints:
///
///   GET  /            single-file chat HTML
///   GET  /health      { status, model_url, db_path, chunks }
///   GET  /status      { chunks, sources: [...], history_len }
///   POST /chat        { message } → { reply, retrieved, ms }
///   POST /upload      body: text/plain, header: X-Filename
///                     → { ok, filename, chunks_added, chunks_total }
///   POST /reset       drop the table + clear conversation history
///
/// Prerequisites: MiniLM weights under `models/minilm/` and a
/// dart_db_server sibling checkout wired up via pubspec (see R13).
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_db_server/dart_db_server.dart';
import 'package:dart_pytorch/dart_pytorch.dart';

const _weightsPath = 'models/minilm/model.safetensors';
const _vocabPath = 'models/minilm/vocab.txt';

const _embedDim = 384;
const _table = 'chunks';

// Max user-message length accepted from the browser (bytes).
const _maxChatBytes = 8 * 1024;
// Max single upload body (bytes).
const _maxUploadBytes = 5 * 1024 * 1024;
// Corpus cap — protects the free-tier laptop from an oops "upload /var/log".
const _maxTotalRows = 5000;

Future<void> main(List<String> args) async {
  final opts = _parseArgs(args);
  final llmBase = (opts['llm'] ?? 'http://127.0.0.1:8080').replaceAll(
    RegExp(r'/$'),
    '',
  );
  final port = int.tryParse(opts['port'] ?? '8090') ?? 8090;
  final dbPath = opts['db-file'] ?? 'data/rag_chat.json';
  final corpusPath = opts['corpus'];
  final topK = int.tryParse(opts['k'] ?? '3') ?? 3;
  final maxNew = int.tryParse(opts['max-new'] ?? '80') ?? 80;
  final temperature = double.tryParse(opts['temperature'] ?? '0.7') ?? 0.7;
  final historyTurns = int.tryParse(opts['history-turns'] ?? '3') ?? 3;
  final embedGpu = opts.containsKey('embed-gpu');

  for (final p in [_weightsPath, _vocabPath]) {
    if (!File(p).existsSync()) {
      stderr.writeln('db_rag_chat_server: missing $p');
      exit(64);
    }
  }
  if (corpusPath != null && !File(corpusPath).existsSync()) {
    stderr.writeln('db_rag_chat_server: --corpus $corpusPath not found');
    exit(64);
  }

  stdout.writeln('== load MiniLM-L6-v2 (${embedGpu ? "GPU" : "CPU"}) ==');
  final sw = Stopwatch()..start();
  final encoder = _Encoder.load(useGpu: embedGpu);
  stdout.writeln('  ${sw.elapsedMilliseconds} ms (dim=$_embedDim)');

  final llm = _LlmClient(llmBase);
  try {
    await llm.ping();
    stdout.writeln('== LLM at $llmBase reachable ==');
  } catch (e) {
    stderr.writeln(
      'db_rag_chat_server: LLM at $llmBase unreachable: $e\n'
      '  start one first, e.g.:\n'
      '    dart run bin/distilgpt2/run_gpu_api.dart --serve --port 8080',
    );
    llm.close();
    exit(69);
  }

  stdout.writeln('== open dart-db-server at $dbPath ==');
  await Directory(_dirOf(dbPath)).create(recursive: true);
  final db = await Database.open(dbPath);
  final store = _ChunkStore(db, encoder);
  await store.ensureSchema();

  if (corpusPath != null) {
    final added = await store.ingestFile(corpusPath);
    stdout.writeln('== ingested $added rows from $corpusPath ==');
  }
  await db.warmVectorIndexes();
  stdout.writeln('== chunks in store: ${await store.count()} ==');

  final history = _History(historyTurns);
  final chat = _ChatEngine(
    store: store,
    llm: llm,
    history: history,
    topK: topK,
    maxNewTokens: maxNew,
    temperature: temperature,
  );

  final server = await HttpServer.bind(InternetAddress.anyIPv4, port);
  stdout.writeln('== http://${_localAddr()}:$port ==  (Ctrl-C to stop)');

  ProcessSignal.sigint.watch().listen((_) async {
    stdout.writeln('\nshutting down…');
    await server.close(force: true);
    await db.close();
    llm.close();
    exit(0);
  });

  await for (final req in server) {
    _handle(req, chat).catchError((Object e, StackTrace st) {
      stderr.writeln('handler error on ${req.method} ${req.uri.path}: $e\n$st');
    });
  }
}

// ---------------------------------------------------------------------------
// HTTP routing.
// ---------------------------------------------------------------------------

Future<void> _handle(HttpRequest req, _ChatEngine chat) async {
  final path = req.uri.path;
  final method = req.method;
  try {
    if (method == 'GET' && path == '/') {
      req.response
        ..statusCode = 200
        ..headers.contentType = ContentType.html
        ..write(_chatHtml);
    } else if (method == 'GET' && path == '/health') {
      _writeJson(req, 200, {
        'status': 'ok',
        'llm': chat.llm.base,
        'db': chat.store.db.path,
        'chunks': await chat.store.count(),
      });
    } else if (method == 'GET' && path == '/status') {
      _writeJson(req, 200, {
        'chunks': await chat.store.count(),
        'sources': await chat.store.listSources(),
        'history_len': chat.history.length,
      });
    } else if (method == 'POST' && path == '/chat') {
      final body = await _readBody(req, _maxChatBytes);
      final msg = ((jsonDecode(body) as Map)['message'] ?? '').toString();
      if (msg.trim().isEmpty) {
        _writeJson(req, 400, {'error': 'message is required'});
      } else {
        _writeJson(req, 200, await chat.reply(msg.trim()));
      }
    } else if (method == 'POST' && path == '/upload') {
      final filename = req.headers.value('x-filename') ?? 'upload.txt';
      final body = await _readBody(req, _maxUploadBytes);
      final added = await chat.store.ingestText(filename, body);
      _writeJson(req, 200, {
        'ok': true,
        'filename': filename,
        'chunks_added': added,
        'chunks_total': await chat.store.count(),
      });
    } else if (method == 'POST' && path == '/reset') {
      await chat.store.reset();
      chat.history.clear();
      _writeJson(req, 200, {'ok': true});
    } else {
      _writeJson(req, 404, {'error': 'not found: $method $path'});
    }
  } catch (e) {
    _writeJson(req, 500, {'error': e.toString()});
  }
  await req.response.close();
}

Future<String> _readBody(HttpRequest req, int max) async {
  final buf = <int>[];
  await for (final chunk in req) {
    buf.addAll(chunk);
    if (buf.length > max) {
      throw StateError('body exceeds $max bytes');
    }
  }
  return utf8.decode(buf);
}

void _writeJson(HttpRequest req, int status, Object body) {
  req.response
    ..statusCode = status
    ..headers.contentType = ContentType.json
    ..headers.set('access-control-allow-origin', '*')
    ..write(jsonEncode(body));
}

// ---------------------------------------------------------------------------
// Chat engine — retrieve → prompt → LLM → strip prefix.
// ---------------------------------------------------------------------------

class _ChatEngine {
  final _ChunkStore store;
  final _LlmClient llm;
  final _History history;
  final int topK;
  final int maxNewTokens;
  final double temperature;

  _ChatEngine({
    required this.store,
    required this.llm,
    required this.history,
    required this.topK,
    required this.maxNewTokens,
    required this.temperature,
  });

  Future<Map<String, Object?>> reply(String userMessage) async {
    final t0 = DateTime.now();
    final hits = await store.retrieve(userMessage, k: topK);
    // Only ground the LLM on passages that are actually close to the query.
    final relevant = hits
        .where((h) => h.distance <= _relevanceDistanceCutoff)
        .toList(growable: false);
    final prompt = _buildPrompt(userMessage, relevant, history.snapshot());
    final generation = await llm.generate(
      prompt: prompt,
      maxNewTokens: maxNewTokens,
      temperature: temperature,
    );
    final answer = _cleanAnswer(generation);
    history.push(userMessage, answer);
    return {
      'reply': answer,
      'retrieved': [
        for (final h in hits)
          {
            'topic': h.topic,
            'source': h.source,
            'text': h.text,
            'rrf': h.rrf,
            'distance': h.distance,
            'used': h.distance <= _relevanceDistanceCutoff,
          },
      ],
      'ms': DateTime.now().difference(t0).inMilliseconds,
    };
  }
}

String _buildPrompt(String question, List<_Hit> hits, List<_Turn> history) {
  // Flush-left turn markers only — indentation makes small base LMs
  // reproduce the indented markers, which then escape our transcript trim.
  final sb = StringBuffer();
  sb.writeln('Answer the question using only the context below.');
  sb.writeln('If the context does not contain the answer, reply exactly:');
  sb.writeln('  "I don\'t know based on the provided context."');
  sb.writeln();
  sb.writeln('Context:');
  if (hits.isEmpty) {
    sb.writeln('(no relevant passages found in the store)');
  } else {
    for (var i = 0; i < hits.length; i++) {
      sb.writeln('- ${hits[i].text}');
    }
  }
  sb.writeln();
  for (final t in history) {
    sb.writeln('User: ${t.user}');
    sb.writeln('Assistant: ${t.assistant}');
  }
  sb.writeln('User: $question');
  sb.write('Assistant:');
  return sb.toString();
}

String _cleanAnswer(String s) {
  var out = s;
  // Small base LMs keep the transcript going with User:/Assistant: markers,
  // sometimes indented. Cut at the first such marker whatever the indent.
  final markers = RegExp(
    r'\n\s*(User|Assistant|Question|Answer|Context|Passages)\s*:',
    caseSensitive: false,
  );
  final m = markers.firstMatch(out);
  if (m != null) out = out.substring(0, m.start);
  // Strip a leading "Assistant:" the model sometimes echoes.
  out = out.trim();
  if (out.toLowerCase().startsWith('assistant:')) {
    out = out.substring('assistant:'.length).trim();
  }
  return out;
}

// ---------------------------------------------------------------------------
// Conversation history (bounded ring).
// ---------------------------------------------------------------------------

class _Turn {
  final String user;
  final String assistant;
  _Turn(this.user, this.assistant);
}

class _History {
  final int maxPairs;
  final List<_Turn> _turns = [];
  _History(this.maxPairs);

  int get length => _turns.length;
  List<_Turn> snapshot() => List.unmodifiable(_turns);
  void push(String user, String assistant) {
    _turns.add(_Turn(user, assistant));
    while (_turns.length > maxPairs) {
      _turns.removeAt(0);
    }
  }

  void clear() => _turns.clear();
}

// ---------------------------------------------------------------------------
// Chunk store — dart-db-server side of the pipeline.
// ---------------------------------------------------------------------------

class _Hit {
  final String source;
  final String topic;
  final String text;
  final double rrf;
  final double distance;
  _Hit(this.source, this.topic, this.text, this.rrf, this.distance);
}

// Cosine distance above this = query is essentially orthogonal to every
// corpus vector, so the top-k hits are noise. We tell the LLM the corpus
// has no relevant context instead of grounding on unrelated passages.
const double _relevanceDistanceCutoff = 0.75;

class _ChunkStore {
  final Database db;
  final _Encoder encoder;
  _ChunkStore(this.db, this.encoder);

  Future<void> ensureSchema() async {
    await db.execute(
      'CREATE TABLE IF NOT EXISTS $_table ('
      'id INTEGER PRIMARY KEY AUTOINCREMENT, '
      'source TEXT NOT NULL, '
      'topic TEXT NOT NULL, '
      'chunk_text TEXT NOT NULL, '
      'embedding BLOB VECTOR('
      'dim=$_embedDim, '
      'kind=hnsw, '
      'metric=cosine, '
      'm=16, '
      'ef_construction=64, '
      "filter_cols='topic'"
      ')'
      ')',
    );
  }

  Future<int> count() async {
    final r = await db.execute('SELECT COUNT(*) FROM $_table');
    return r.rows.isEmpty ? 0 : (r.rows.first[0] as num).toInt();
  }

  Future<List<String>> listSources() async {
    final r = await db.execute(
      'SELECT DISTINCT source FROM $_table ORDER BY source',
    );
    return [for (final row in r.rows) row[0] as String];
  }

  Future<int> ingestFile(String path) async {
    return ingestText(path, File(path).readAsStringSync());
  }

  Future<int> ingestText(String source, String raw) async {
    final lines = raw
        .split(RegExp(r'\r?\n'))
        .map((l) => l.trim())
        .where((l) => l.isNotEmpty)
        .toList();
    final existing = await count();
    final room = _maxTotalRows - existing;
    if (room <= 0) return 0;
    final take = lines.length > room ? lines.sublist(0, room) : lines;
    for (final line in take) {
      final vec = encoder.embed(line);
      final vecJson = _vecJson(vec);
      final topic = _classify(line);
      await db.execute(
        'INSERT INTO $_table (source, topic, chunk_text, embedding) VALUES '
        '(${_sqlEscape(source)}, ${_sqlEscape(topic)}, '
        "${_sqlEscape(line)}, VEC(${_sqlEscape(vecJson)}))",
      );
    }
    await db.warmVectorIndexes();
    return take.length;
  }

  Future<void> reset() async {
    await db.execute('DROP TABLE IF EXISTS $_table');
    await ensureSchema();
  }

  Future<List<_Hit>> retrieve(String query, {required int k}) async {
    if (await count() == 0) return const [];
    final qv = _vecJson(encoder.embed(query));
    final bm25 = _fts5Sanitize(query);
    final r = await db.execute(
      'SELECT c.source, c.topic, c.chunk_text, s.rrf_score, s.distance '
      'FROM vec_hybrid_search('
      "'$_table', 'embedding', 'chunk_text', "
      'VEC(${_sqlEscape(qv)}), ${_sqlEscape(bm25)}, $k, 60'
      ') AS s '
      'JOIN $_table c ON c.id = s.rowid '
      'ORDER BY s.rrf_score DESC',
    );
    return [
      for (final row in r.rows)
        _Hit(
          row[0] as String,
          row[1] as String,
          row[2] as String,
          (row[3] as num).toDouble(),
          (row[4] as num).toDouble(),
        ),
    ];
  }
}

// ---------------------------------------------------------------------------
// LLM client — POST /generate on any bin/*_api.dart runner.
// ---------------------------------------------------------------------------

class _LlmClient {
  final String base;
  final HttpClient _http = HttpClient();
  _LlmClient(this.base);

  void close() => _http.close(force: true);

  Future<void> ping() async {
    final req = await _http.getUrl(Uri.parse('$base/health'));
    final res = await req.close();
    final body = await utf8.decoder.bind(res).join();
    if (res.statusCode != 200) {
      throw StateError('/health returned ${res.statusCode}: $body');
    }
  }

  Future<String> generate({
    required String prompt,
    required int maxNewTokens,
    required double temperature,
  }) async {
    final req = await _http.postUrl(Uri.parse('$base/generate'));
    req.headers.contentType = ContentType.json;
    req.write(
      jsonEncode({
        'text': prompt,
        'maxNewTokens': maxNewTokens,
        'temperature': temperature,
        'topK': 40,
      }),
    );
    final res = await req.close();
    final body = await utf8.decoder.bind(res).join();
    if (res.statusCode != 200) {
      throw StateError('LLM /generate ${res.statusCode}: $body');
    }
    final json = jsonDecode(body) as Map<String, Object?>;
    final full = (json['text'] as String?) ?? '';
    return full.startsWith(prompt) ? full.substring(prompt.length) : full;
  }
}

// ---------------------------------------------------------------------------
// MiniLM encoder wrapper.
// ---------------------------------------------------------------------------

class _Encoder {
  final WordPieceTokenizer tok;
  final SentenceEncoder enc;
  final Device device;
  _Encoder(this.tok, this.enc, this.device);

  static _Encoder load({bool useGpu = false}) {
    final device = useGpu ? Device.GPU : Device.CPU;
    final tok = WordPieceTokenizer.fromVocabFile(_vocabPath);
    final bert = BertModel(BertHFLoader.miniLmL6V2Config(device: device));
    BertHFLoader.loadFile(bert, _weightsPath);
    final enc = SentenceEncoder.wrap(bert);
    enc.eval();
    return _Encoder(tok, enc, device);
  }

  Float32List embed(String text, {int maxLength = 256}) {
    final ids = tok.encode(text, maxLength: maxLength);
    final cpuT = Tensor.fromList(
      [ids.length],
      [for (final i in ids) i.toDouble()],
    );
    final t = device == Device.CPU ? cpuT : cpuT.to(device);
    return Tensor.noGrad(() => Float32List.fromList(enc(t).toList()));
  }
}

// ---------------------------------------------------------------------------
// Keyword topic tagger — good enough for the filter_cols demo.
// ---------------------------------------------------------------------------

const _topicRules = <String, List<String>>{
  'security': [
    'password',
    'mfa',
    'multi-factor',
    'authenticator',
    'sso',
    'encrypted',
    'encryption',
    'tls',
    'webauthn',
    'soc 2',
    'gdpr',
    'hipaa',
    'ccpa',
  ],
  'billing': [
    'subscription',
    'billing',
    'cancel',
    'refund',
    'charge',
    'trial',
    'paypal',
    'credit',
    'mastercard',
    'visa',
    'apple pay',
    'discount',
    'non-profit',
    'cryptocurrency',
  ],
  'api': ['api', 'rate', 'developer console', 'api key'],
  'plans': [
    'plan',
    'seats',
    'team',
    'quota',
    'upload',
    'backup',
    'enterprise',
    'pro plan',
  ],
  'mobile': ['mobile', 'app store', 'google play', 'ios', 'android'],
  'support': [
    'support',
    'ticket',
    'phone',
    'status page',
    'incident',
    'help widget',
    'holidays',
    'hours',
  ],
};

String _classify(String text) {
  final t = text.toLowerCase();
  for (final entry in _topicRules.entries) {
    for (final kw in entry.value) {
      if (t.contains(kw)) return entry.key;
    }
  }
  return 'general';
}

// ---------------------------------------------------------------------------
// Formatting + FTS5 helpers (shared with bin/db_rag_demo.dart).
// ---------------------------------------------------------------------------

String _vecJson(Float32List v) {
  final sb = StringBuffer('[');
  for (int i = 0; i < v.length; i++) {
    if (i > 0) sb.write(',');
    sb.write(v[i].toStringAsFixed(6));
  }
  sb.write(']');
  return sb.toString();
}

String _sqlEscape(String s) => "'${s.replaceAll("'", "''")}'";

const _fts5Stopwords = <String>{
  'a',
  'an',
  'the',
  'is',
  'are',
  'was',
  'were',
  'be',
  'been',
  'am',
  'do',
  'does',
  'did',
  'of',
  'to',
  'in',
  'on',
  'for',
  'and',
  'or',
  'not',
  'but',
  'i',
  'you',
  'we',
  'they',
  'it',
  'he',
  'she',
  'my',
  'your',
  'our',
  'how',
  'what',
  'when',
  'where',
  'why',
  'who',
  'which',
  'can',
  'could',
  'would',
  'should',
  'will',
  'may',
  'might',
  'this',
  'that',
  'these',
  'those',
  'there',
  'here',
};

String _fts5Sanitize(String query) {
  final terms = query
      .toLowerCase()
      .replaceAll(RegExp(r'[^a-z0-9\s]'), ' ')
      .split(RegExp(r'\s+'))
      .where((t) => t.isNotEmpty && !_fts5Stopwords.contains(t))
      .toList();
  return terms.isEmpty ? 'a' : terms.join(' OR ');
}

String _dirOf(String path) {
  final i = path.lastIndexOf(Platform.pathSeparator);
  final j = path.lastIndexOf('/');
  final k = i > j ? i : j;
  return k <= 0 ? '.' : path.substring(0, k);
}

String _localAddr() {
  // Prefer 127.0.0.1 in the log line — anyIPv4 is what we actually bind.
  return '127.0.0.1';
}

Map<String, String> _parseArgs(List<String> args) {
  final out = <String, String>{};
  for (int i = 0; i < args.length; i++) {
    final a = args[i];
    if (!a.startsWith('--')) continue;
    final key = a.substring(2);
    if (i + 1 < args.length && !args[i + 1].startsWith('--')) {
      out[key] = args[i + 1];
      i++;
    } else {
      out[key] = 'true';
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Single-file chat UI. Vanilla HTML/CSS/JS — no build step.
// ---------------------------------------------------------------------------

const String _chatHtml = r'''
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>db_rag_chat — MiniLM + dart-db-server + HTTP LLM</title>
  <style>
    :root { color-scheme: light dark; }
    body {
      font: 14px/1.45 -apple-system, "Segoe UI", Roboto, sans-serif;
      max-width: 780px; margin: 24px auto; padding: 0 16px;
    }
    header { display: flex; justify-content: space-between; align-items: baseline; }
    h1 { font-size: 18px; margin: 0; }
    .meta { color: #888; font-size: 12px; }
    #log {
      border: 1px solid #ccc4; border-radius: 8px; padding: 12px;
      height: 62vh; overflow-y: auto; margin: 12px 0;
      background: canvas;
    }
    .turn { margin: 10px 0; }
    .turn .role { font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: #888; }
    .turn.user .role { color: #367; }
    .turn.assistant .role { color: #632; }
    .turn .text { white-space: pre-wrap; margin-top: 2px; }
    .hits { margin-top: 6px; font-size: 12px; color: #888; }
    .hits summary { cursor: pointer; }
    .hit { padding: 4px 0; }
    .hit.skipped { opacity: .55; }
    .hit .tag { display: inline-block; padding: 0 6px; border-radius: 10px; background: #8884; margin-right: 6px; }
    form { display: flex; gap: 8px; }
    input[type=text] { flex: 1; padding: 8px 10px; border: 1px solid #ccc4; border-radius: 6px; font: inherit; background: canvas; color: canvastext; }
    button { padding: 8px 14px; border: 0; border-radius: 6px; background: #367; color: white; cursor: pointer; }
    button.secondary { background: transparent; color: canvastext; border: 1px solid #ccc4; }
    button:disabled { opacity: .5; }
    #upload { display: flex; gap: 8px; align-items: center; margin-top: 8px; font-size: 12px; color: #888; }
    .err { color: #b33; }
  </style>
</head>
<body>
  <header>
    <h1>db_rag_chat</h1>
    <div class="meta" id="status">…</div>
  </header>

  <div id="log"></div>

  <form id="chat">
    <input id="msg" type="text" placeholder="Ask a question about the corpus…" autofocus>
    <button type="submit">Send</button>
    <button type="button" class="secondary" id="reset">Reset</button>
  </form>

  <div id="upload">
    <label>Upload .txt / .md: <input type="file" id="file" accept=".txt,.md,text/plain,text/markdown"></label>
    <span id="uplog"></span>
  </div>

<script>
const log = document.getElementById('log');
const status = document.getElementById('status');
const form = document.getElementById('chat');
const msg = document.getElementById('msg');
const resetBtn = document.getElementById('reset');
const file = document.getElementById('file');
const uplog = document.getElementById('uplog');

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function addTurn(role, text, retrieved) {
  const t = el('div', 'turn ' + role);
  t.appendChild(el('div', 'role', role));
  t.appendChild(el('div', 'text', text));
  if (retrieved && retrieved.length) {
    const used = retrieved.filter(h => h.used !== false).length;
    const det = el('details', 'hits');
    det.appendChild(el('summary', null,
      'retrieved ' + retrieved.length + ' passage(s), ' + used + ' used'));
    for (const h of retrieved) {
      const line = el('div', 'hit' + (h.used === false ? ' skipped' : ''));
      line.appendChild(el('span', 'tag', h.topic));
      const dist = (h.distance !== undefined)
          ? '  dist=' + h.distance.toFixed(3) : '';
      const flag = (h.used === false) ? '  [off-topic, dropped]' : '';
      line.appendChild(document.createTextNode(
        'rrf=' + h.rrf.toFixed(4) + dist + flag + '  ' + h.text));
      det.appendChild(line);
    }
    t.appendChild(det);
  }
  log.appendChild(t);
  log.scrollTop = log.scrollHeight;
}

async function refreshStatus() {
  try {
    const r = await fetch('/status');
    const j = await r.json();
    status.textContent =
      j.chunks + ' chunks, ' + j.sources.length + ' source(s), ' +
      j.history_len + ' turn(s)';
  } catch (e) { status.textContent = ''; }
}

form.addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const text = msg.value.trim();
  if (!text) return;
  msg.value = '';
  addTurn('user', text);
  const button = form.querySelector('button[type=submit]');
  button.disabled = true;
  try {
    const r = await fetch('/chat', {
      method: 'POST',
      headers: {'content-type': 'application/json'},
      body: JSON.stringify({message: text}),
    });
    const j = await r.json();
    if (j.error) addTurn('assistant', '[error] ' + j.error);
    else addTurn('assistant', j.reply || '(empty)', j.retrieved);
  } catch (e) {
    addTurn('assistant', '[network error] ' + e);
  } finally {
    button.disabled = false;
    msg.focus();
    refreshStatus();
  }
});

resetBtn.addEventListener('click', async () => {
  await fetch('/reset', {method: 'POST'});
  log.innerHTML = '';
  uplog.textContent = '';
  refreshStatus();
});

file.addEventListener('change', async () => {
  const f = file.files[0];
  if (!f) return;
  uplog.textContent = 'uploading ' + f.name + '…';
  try {
    const body = await f.text();
    const r = await fetch('/upload', {
      method: 'POST',
      headers: {'x-filename': f.name, 'content-type': 'text/plain'},
      body,
    });
    const j = await r.json();
    if (j.ok) {
      uplog.textContent =
        f.name + ' → ' + j.chunks_added + ' new chunk(s), ' +
        j.chunks_total + ' total';
    } else {
      uplog.innerHTML = '<span class=err>' + (j.error || 'upload failed') + '</span>';
    }
  } catch (e) {
    uplog.innerHTML = '<span class=err>' + e + '</span>';
  } finally {
    file.value = '';
    refreshStatus();
  }
});

refreshStatus();
</script>
</body>
</html>
''';
