/// End-to-end RAG walkthrough against `dart-db-server`'s vector store,
/// **in-process** — no separate server, no network hop. Mirrors the recipe at
///   `dart-db-server/doc/rag-semantic-search.md`
/// using this repo's `all-MiniLM-L6-v2` sentence encoder to compute the
/// 384-dim embeddings. Every retrieval mode the DB exposes is exercised
/// against the same corpus so the differences are obvious:
///
///   1. Native inline vector index         → `CREATE ... BLOB VECTOR(...)`
///   2. Plain semantic k-NN                → `vec_search`
///   3. Payload-filtered k-NN              → `vec_search_filtered`
///   4. Hybrid vector + BM25 (RRF)         → `vec_hybrid_search`
///   5. Range / near-duplicate detection   → `vec_range_search`
///   6. Admin surface                      → `PRAGMA vector_index_*`
///
/// Prerequisites (one-time MiniLM download, ~87 MB — same as `bin/qa.dart`):
///
///   mkdir -p models/minilm && cd models/minilm
///   for f in config.json tokenizer_config.json vocab.txt \
///            model.safetensors; do
///     curl -sSL -O \
///       "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/$f"
///   done
///
/// Run:
///
///   dart run bin/db_rag_demo.dart
///   dart run bin/db_rag_demo.dart --query "how do I turn on 2FA?"
///   dart run bin/db_rag_demo.dart --db-file /tmp/rag.json    # persist
///
/// With no `--db-file`, the demo runs entirely in a temp file that is
/// deleted on exit — safe to re-run repeatedly.
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

// MiniLM-L6-v2 output dim; must match `dim=` in the CREATE TABLE.
const _embedDim = 384;
const _table = 'chunks';

const List<String> _defaultQueries = [
  'how do I enable multi-factor authentication?',
  'what payment methods do you support?',
  'can I get a refund after two weeks?',
  'is the service available on Android?',
];

Future<void> main(List<String> args) async {
  final opts = _parseArgs(args);
  final queries = <String>[
    if (opts['query'] != null) opts['query']!,
    if (opts['query'] == null) ..._defaultQueries,
  ];

  for (final p in [_weightsPath, _vocabPath, _corpusPath]) {
    if (!File(p).existsSync()) {
      stderr.writeln('db_rag_demo: missing $p');
      stderr.writeln('  see the header comment of bin/db_rag_demo.dart.');
      exit(64);
    }
  }

  _banner('load MiniLM-L6-v2');
  final swLoad = Stopwatch()..start();
  final encoder = _Encoder.load();
  swLoad.stop();
  stdout.writeln('  ${swLoad.elapsedMilliseconds} ms  (dim=$_embedDim)');

  // Ephemeral store unless the caller passed --db-file.
  final userPath = opts['db-file'];
  final dbPath =
      userPath ??
      '${Directory.systemTemp.path}${Platform.pathSeparator}'
          'db_rag_demo_${DateTime.now().microsecondsSinceEpoch}.json';
  if (userPath == null) {
    try {
      File(dbPath).deleteSync();
    } catch (_) {}
  }

  _banner('open dart-db-server (in-process) at $dbPath');
  final db = await Database.open(dbPath);
  try {
    await _createSchema(db);
    final passages = _loadPassages(_corpusPath);
    await _ingest(db, encoder, passages);
    await _adminPragmas(db);

    for (final q in queries) {
      await _plainKnn(db, encoder, q);
      await _filteredKnn(db, encoder, q);
      await _hybridSearch(db, encoder, q);
    }
    await _rangeSearchDemo(db, encoder);
  } finally {
    await db.close();
    if (userPath == null) {
      try {
        File(dbPath).deleteSync();
      } catch (_) {}
    }
  }
}

// ---------------------------------------------------------------------------
// 1. Schema — inline vector index, filter columns, HNSW.
// ---------------------------------------------------------------------------

Future<void> _createSchema(Database db) async {
  _banner('1. create schema (drops any previous $_table)');
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
  stdout.writeln(
    '  BLOB VECTOR(dim=$_embedDim, kind=hnsw, metric=cosine, '
    "filter_cols='topic')",
  );
}

// ---------------------------------------------------------------------------
// 2. Ingest — encode → INSERT ... VEC('[...]')
// ---------------------------------------------------------------------------

Future<void> _ingest(
  Database db,
  _Encoder encoder,
  List<_Passage> passages,
) async {
  _banner('2. ingest ${passages.length} passages');
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
  stdout.writeln(
    '  ${sw.elapsedMilliseconds} ms  '
    '(${(sw.elapsedMilliseconds / passages.length).toStringAsFixed(1)} '
    'ms/passage, MiniLM encode + INSERT)',
  );

  final r = await db.execute('SELECT COUNT(*) FROM $_table');
  stdout.writeln('  row count: ${r.rows.first[0]}');

  // HNSW is built lazily on first query. Warm eagerly so the admin dump
  // below reflects a built index and later queries pay no build cost.
  await db.warmVectorIndexes();
}

// ---------------------------------------------------------------------------
// 3. Admin — PRAGMA vector_index_list / _stats
// ---------------------------------------------------------------------------

Future<void> _adminPragmas(Database db) async {
  _banner('3. admin: PRAGMA vector_index_list / _stats');
  _dump(await db.execute('PRAGMA vector_index_list'));
  _dump(await db.execute("PRAGMA vector_index_stats('$_table.embedding')"));
}

// ---------------------------------------------------------------------------
// 4. Plain semantic k-NN — `vec_search`
// ---------------------------------------------------------------------------

Future<void> _plainKnn(Database db, _Encoder encoder, String query) async {
  _banner('4. vec_search — plain semantic top-3 for:\n     "$query"');
  final qv = _vecJson(encoder.embed(query));
  final r = await db.execute(
    'SELECT c.topic, c.chunk_text, s.distance '
    'FROM vec_search('
    "'$_table', 'embedding', VEC(${_sqlEscape(qv)}), 3"
    ') AS s '
    'JOIN $_table c ON c.id = s.rowid '
    'ORDER BY s.distance',
  );
  _dumpRanked(r);
}

// ---------------------------------------------------------------------------
// 5. Filtered k-NN — `vec_search_filtered` restricted to a topic.
// ---------------------------------------------------------------------------

Future<void> _filteredKnn(Database db, _Encoder encoder, String query) async {
  final topic = _pickTopic(query);
  _banner(
    "5. vec_search_filtered — top-3 within topic='$topic' for:\n"
    '     "$query"',
  );
  final qv = _vecJson(encoder.embed(query));
  final filterJson = jsonEncode({'topic': topic});
  final r = await db.execute(
    'SELECT c.topic, c.chunk_text, s.distance '
    'FROM vec_search_filtered('
    "'$_table', 'embedding', VEC(${_sqlEscape(qv)}), 3, "
    '${_sqlEscape(filterJson)}'
    ') AS s '
    'JOIN $_table c ON c.id = s.rowid '
    'ORDER BY s.distance',
  );
  _dumpRanked(r);
}

// ---------------------------------------------------------------------------
// 6. Hybrid vector + BM25 — the marquee RAG retrieval mode.
// ---------------------------------------------------------------------------

Future<void> _hybridSearch(Database db, _Encoder encoder, String query) async {
  _banner(
    '6. vec_hybrid_search — vector + BM25 (RRF) top-3 for:\n     "$query"',
  );
  final qv = _vecJson(encoder.embed(query));
  final bm25Query = _fts5Sanitize(query);
  final r = await db.execute(
    'SELECT c.topic, c.chunk_text, s.distance, s.bm25, s.rrf_score '
    'FROM vec_hybrid_search('
    "'$_table', 'embedding', 'chunk_text', "
    'VEC(${_sqlEscape(qv)}), ${_sqlEscape(bm25Query)}, 3, 60'
    ') AS s '
    'JOIN $_table c ON c.id = s.rowid '
    'ORDER BY s.rrf_score DESC',
  );
  _dumpRanked(r);
}

// ---------------------------------------------------------------------------
// 7. Range search — every passage within a cosine-distance threshold of a
//    known probe, useful for near-duplicate / clustering pipelines.
// ---------------------------------------------------------------------------

Future<void> _rangeSearchDemo(Database db, _Encoder encoder) async {
  const probe =
      'To reset your password, click the "Forgot password?" link on '
      'the sign-in page and follow the emailed instructions.';
  const threshold = 0.35;
  _banner(
    '7. vec_range_search — every passage within cosine distance '
    '$threshold of:\n     "${probe.substring(0, 60)}..."',
  );
  final qv = _vecJson(encoder.embed(probe));
  final r = await db.execute(
    'SELECT c.topic, c.chunk_text, s.distance '
    'FROM vec_range_search('
    "'$_table', 'embedding', VEC(${_sqlEscape(qv)}), $threshold"
    ') AS s '
    'JOIN $_table c ON c.id = s.rowid '
    'ORDER BY s.distance',
  );
  _dumpRanked(r);
}

// ---------------------------------------------------------------------------
// MiniLM sentence encoder wrapper — mirrors bin/qa.dart.
// ---------------------------------------------------------------------------

class _Encoder {
  final WordPieceTokenizer tok;
  final SentenceEncoder enc;
  _Encoder(this.tok, this.enc);

  static _Encoder load() {
    final tok = WordPieceTokenizer.fromVocabFile(_vocabPath);
    final bert = BertModel(BertHFLoader.miniLmL6V2Config());
    BertHFLoader.loadFile(bert, _weightsPath);
    final enc = SentenceEncoder.wrap(bert);
    enc.eval();
    return _Encoder(tok, enc);
  }

  Float32List embed(String text, {int maxLength = 256}) {
    final ids = tok.encode(text, maxLength: maxLength);
    final t = Tensor.fromList(
      [ids.length],
      [for (final i in ids) i.toDouble()],
    );
    return Tensor.noGrad(() => Float32List.fromList(enc(t).toList()));
  }
}

// ---------------------------------------------------------------------------
// Corpus loading + tiny keyword-based topic tagger. The tag lets us
// demonstrate `filter_cols` / `vec_search_filtered` without pulling in a
// full classifier — the point of the demo is the DB surface, not NLP.
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

String _pickTopic(String query) {
  final tag = _classify(query);
  return tag == 'general' ? 'support' : tag;
}

// ---------------------------------------------------------------------------
// Formatting helpers.
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

// FTS5's query grammar accepts bareword tokens joined by AND (implicit)
// or OR. Punctuation like `?`, `!`, `-` inside a raw user question makes
// it throw, and AND semantics on a natural-language question would drop
// to zero BM25 whenever any query word is absent from the doc. Strip to
// alphanumeric terms, drop stopwords, then OR the rest so partial
// keyword overlap still contributes to the BM25 side.
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

void _dump(QueryResult r) {
  if (r.columns.isNotEmpty) {
    stdout.writeln('  ${r.columns.join(' | ')}');
  }
  for (final row in r.rows) {
    stdout.writeln('  ${row.join(' | ')}');
  }
  if (r.rows.isEmpty) {
    stdout.writeln('  (no rows)');
  }
}

void _dumpRanked(QueryResult r) {
  if (r.rows.isEmpty) {
    stdout.writeln('  (no rows)');
    return;
  }
  for (var i = 0; i < r.rows.length; i++) {
    final row = r.rows[i];
    final topic = row[0];
    final text = row[1] as String;
    final trimmed = text.length > 80 ? '${text.substring(0, 77)}...' : text;
    final metrics = <String>[];
    for (var c = 2; c < row.length; c++) {
      final v = row[c];
      final s = v is num ? v.toStringAsFixed(4) : '$v';
      metrics.add('${r.columns.length > c ? r.columns[c] : "col$c"}=$s');
    }
    stdout.writeln(
      '  ${i == 0 ? '★' : ' '} [$topic]  ${metrics.join('  ')}\n'
      '        $trimmed',
    );
  }
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
