/// Retrieve-then-generate RAG against `dart-db-server` **plus** an
/// HTTP-served LLM. Companion to [bin/db_rag_demo.dart] — that demo
/// stops at retrieval, this one closes the loop.
///
/// Split of responsibilities:
///
///   * `dart_pytorch` (in-process) — MiniLM computes the query and
///     corpus embeddings.
///   * `dart_db_server` (in-process, `Database.open`) — stores the
///     chunks, runs `vec_hybrid_search` (vector + BM25 via RRF).
///   * `bin/*_api.dart` (separate process, HTTP) — the LLM generator.
///     Any of the runners listed under "Common overrides" in
///     [commands.md] works: distilgpt2, gpt2-medium, pythia-*, gpt-j-6b.
///
/// Wire it up in two shells:
///
///   # shell A — start any *_api.dart runner with --serve
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib \
///     dart run bin/distilgpt2/run_gpu_api.dart --serve --port 8080
///
///   # shell B — retrieve + generate
///   dart run bin/db_rag_http_demo.dart \
///       --llm http://127.0.0.1:8080 \
///       --query "how do I turn on 2FA?"
///
/// Prerequisites: MiniLM weights under `models/minilm/` (see
/// [commands.md § E2](../commands.md)) and the LLM weights + tokenizer
/// for whichever `*_api.dart` runner you launch in shell A.
///
/// Prints for each query: the retrieved passages (topic + score), the
/// prompt that was sent to the LLM, and the generated answer.
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_db_server/dart_db_server.dart';
import 'package:dart_pytorch/dart_pytorch.dart';

const _weightsPath = 'models/minilm/model.safetensors';
const _vocabPath = 'models/minilm/vocab.txt';
const _corpusPath = 'data/support_faq.txt';

// MiniLM-L6-v2 dim; must match `dim=` in the CREATE TABLE.
const _embedDim = 384;
const _table = 'chunks';

const List<String> _defaultQueries = [
  'how do I enable multi-factor authentication?',
  'what payment methods do you support?',
  'is the service available on Android?',
];

Future<void> main(List<String> args) async {
  final opts = _parseArgs(args);
  final llmBase = (opts['llm'] ?? 'http://127.0.0.1:8080').replaceAll(
    RegExp(r'/$'),
    '',
  );
  final topK = int.tryParse(opts['k'] ?? '3') ?? 3;
  final maxNewTokens = int.tryParse(opts['max-new'] ?? '60') ?? 60;
  final temperature = double.tryParse(opts['temperature'] ?? '0.7') ?? 0.7;
  final seed = int.tryParse(opts['seed'] ?? '42') ?? 42;
  final queries = <String>[
    if (opts['query'] != null) opts['query']!,
    if (opts['query'] == null) ..._defaultQueries,
  ];

  for (final p in [_weightsPath, _vocabPath, _corpusPath]) {
    if (!File(p).existsSync()) {
      stderr.writeln('db_rag_http_demo: missing $p');
      stderr.writeln('  see the header comment of bin/db_rag_http_demo.dart.');
      exit(64);
    }
  }

  final http = HttpClient();
  try {
    await _pingLlm(http, llmBase);
  } catch (e) {
    stderr.writeln('db_rag_http_demo: LLM at $llmBase is unreachable: $e');
    stderr.writeln(
      '  start one first, e.g.:\n'
      '    dart run bin/distilgpt2/run_gpu_api.dart --serve --port 8080',
    );
    http.close(force: true);
    exit(69);
  }

  _banner(
    'load MiniLM-L6-v2 (${opts.containsKey('embed-gpu') ? "GPU" : "CPU"})',
  );
  final sw = Stopwatch()..start();
  final encoder = _Encoder.load(useGpu: opts.containsKey('embed-gpu'));
  sw.stop();
  stdout.writeln('  ${sw.elapsedMilliseconds} ms  (dim=$_embedDim)');

  final dbPath =
      '${Directory.systemTemp.path}${Platform.pathSeparator}'
      'db_rag_http_${DateTime.now().microsecondsSinceEpoch}.json';
  try {
    File(dbPath).deleteSync();
  } catch (_) {}

  _banner('open dart-db-server (in-process) at $dbPath');
  final db = await Database.open(dbPath);
  try {
    await _createSchema(db);
    final passages = _loadPassages(_corpusPath);
    await _ingest(db, encoder, passages);
    await db.warmVectorIndexes();

    for (final q in queries) {
      final hits = await _retrieve(db, encoder, q, k: topK);
      final prompt = _buildPrompt(q, hits);
      final answer = await _callLlm(
        http,
        llmBase,
        prompt,
        maxNewTokens: maxNewTokens,
        temperature: temperature,
        seed: seed,
      );
      _renderAnswer(q, hits, prompt, answer);
    }
  } finally {
    await db.close();
    try {
      File(dbPath).deleteSync();
    } catch (_) {}
    http.close(force: true);
  }
}

// ---------------------------------------------------------------------------
// Retrieval — hybrid vector + BM25 (RRF) via dart-db-server.
// ---------------------------------------------------------------------------

class _Hit {
  final String topic;
  final String text;
  final double rrf;
  _Hit(this.topic, this.text, this.rrf);
}

Future<List<_Hit>> _retrieve(
  Database db,
  _Encoder encoder,
  String query, {
  required int k,
}) async {
  final qv = _vecJson(encoder.embed(query));
  final bm25 = _fts5Sanitize(query);
  final r = await db.execute(
    'SELECT c.topic, c.chunk_text, s.rrf_score '
    'FROM vec_hybrid_search('
    "'$_table', 'embedding', 'chunk_text', "
    'VEC(${_sqlEscape(qv)}), ${_sqlEscape(bm25)}, $k, 60'
    ') AS s '
    'JOIN $_table c ON c.id = s.rowid '
    'ORDER BY s.rrf_score DESC',
  );
  return [
    for (final row in r.rows)
      _Hit(row[0] as String, row[1] as String, (row[2] as num).toDouble()),
  ];
}

// ---------------------------------------------------------------------------
// Prompt assembly + HTTP call to the LLM's /generate endpoint.
// ---------------------------------------------------------------------------

String _buildPrompt(String question, List<_Hit> hits) {
  final sb = StringBuffer();
  sb.writeln('Answer the question using only the passages below.');
  sb.writeln('If the passages do not contain the answer, say so.');
  sb.writeln();
  sb.writeln('Passages:');
  for (var i = 0; i < hits.length; i++) {
    sb.writeln('  ${i + 1}. ${hits[i].text}');
  }
  sb.writeln();
  sb.writeln('Question: $question');
  sb.write('Answer:');
  return sb.toString();
}

Future<String> _callLlm(
  HttpClient http,
  String base,
  String prompt, {
  required int maxNewTokens,
  required double temperature,
  required int seed,
}) async {
  final req = await http.postUrl(Uri.parse('$base/generate'));
  req.headers.contentType = ContentType.json;
  req.write(
    jsonEncode({
      'text': prompt,
      'maxNewTokens': maxNewTokens,
      'temperature': temperature,
      'topK': 40,
      'seed': seed,
    }),
  );
  final res = await req.close();
  final body = await utf8.decoder.bind(res).join();
  if (res.statusCode != 200) {
    throw StateError('LLM /generate returned ${res.statusCode}: $body');
  }
  final json = jsonDecode(body) as Map<String, Object?>;
  final full = (json['text'] as String?) ?? '';
  // The server returns the decoded prompt + generation. Strip the prompt
  // so we only surface what the model added.
  return full.startsWith(prompt) ? full.substring(prompt.length) : full;
}

Future<void> _pingLlm(HttpClient http, String base) async {
  final req = await http.getUrl(Uri.parse('$base/health'));
  final res = await req.close();
  final body = await utf8.decoder.bind(res).join();
  if (res.statusCode != 200) {
    throw StateError('/health returned ${res.statusCode}: $body');
  }
}

// ---------------------------------------------------------------------------
// Rendering.
// ---------------------------------------------------------------------------

void _renderAnswer(
  String query,
  List<_Hit> hits,
  String prompt,
  String answer,
) {
  _banner('Q: $query');
  stdout.writeln('  retrieved (top ${hits.length}):');
  for (var i = 0; i < hits.length; i++) {
    final h = hits[i];
    final trimmed = h.text.length > 80
        ? '${h.text.substring(0, 77)}...'
        : h.text;
    stdout.writeln(
      '    ${i == 0 ? '★' : ' '} [${h.topic}] rrf=${h.rrf.toStringAsFixed(4)}\n'
      '        $trimmed',
    );
  }
  stdout.writeln('  prompt (${prompt.length} chars) sent to LLM /generate');
  stdout.writeln('  answer:');
  for (final line in answer.trim().split('\n')) {
    stdout.writeln('    $line');
  }
}

// ---------------------------------------------------------------------------
// Schema + ingest — same DDL as bin/db_rag_demo.dart.
// ---------------------------------------------------------------------------

Future<void> _createSchema(Database db) async {
  _banner('create schema (drops any previous $_table)');
  await db.execute('DROP TABLE IF EXISTS $_table');
  await db.execute(
    'CREATE TABLE $_table ('
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

Future<void> _ingest(
  Database db,
  _Encoder encoder,
  List<_Passage> passages,
) async {
  _banner('ingest ${passages.length} passages');
  final sw = Stopwatch()..start();
  for (final p in passages) {
    final vec = encoder.embed(p.text);
    final vecJson = _vecJson(vec);
    await db.execute(
      'INSERT INTO $_table (source, topic, chunk_text, embedding) VALUES '
      '(${_sqlEscape(p.source)}, ${_sqlEscape(p.topic)}, '
      "${_sqlEscape(p.text)}, VEC(${_sqlEscape(vecJson)}))",
    );
  }
  sw.stop();
  stdout.writeln('  ${sw.elapsedMilliseconds} ms');
}

// ---------------------------------------------------------------------------
// MiniLM sentence-encoder wrapper.
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
// Corpus loading + topic tagger.
// ---------------------------------------------------------------------------

class _Passage {
  final String source;
  final String topic;
  final String text;
  _Passage(this.source, this.topic, this.text);
}

List<_Passage> _loadPassages(String path) {
  final raw = File(path).readAsStringSync();
  final lines = raw
      .split(RegExp(r'\r?\n'))
      .map((l) => l.trim())
      .where((l) => l.isNotEmpty)
      .toList();
  return [for (final l in lines) _Passage(path, _classify(l), l)];
}

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
// Formatting + FTS5 helpers.
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

void _banner(String s) {
  stdout.writeln('');
  stdout.writeln('─── $s');
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
