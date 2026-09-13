# RAG on a SQL vector database — `dart_pytorch` × `dart_db_server`

Two Dart packages, zero Python, zero external services. `dart_pytorch`
computes the embeddings; `dart_db_server` stores them, indexes them,
filters them, and fuses them with BM25 — all in-process, all SQL.
Together they are a self-contained production-shape RAG stack that
runs anywhere a single Dart binary runs.

> Reference implementation: [bin/db_rag_demo.dart](../bin/db_rag_demo.dart).
> User-facing walkthrough: [commands.md § R13](../commands.md).
> Upstream vector docs: [dart-db-server/doc/rag-semantic-search.md](https://github.com/KellyKinyama/dart-db-server/blob/main/doc/rag-semantic-search.md).

## Why this pairing matters

RAG in most stacks looks like this:

```
Python embedder  ──►  FAISS in memory  ──►  Python glue  ──►  LLM
       │                    │                    │
       └─ heavyweight        └─ no filters,      └─ no persistence,
          runtime               no BM25,             no SQL joins,
                                no SQL               no admin surface
```

RAG in this stack looks like this:

```
dart_pytorch  ──►  dart_db_server  ──►  LLM
    │                    │
    │                    ├─ HNSW / Flat / IVFPQ / LSH / PQ / IVFFlat
    │                    ├─ Reciprocal-Rank-Fusion hybrid vector + BM25
    │                    ├─ Payload filter columns (multi-tenant free)
    │                    ├─ Range search (near-dup, clustering)
    │                    ├─ SQL JOINs against your other tables
    │                    ├─ Persistence, admin PRAGMAs, health checks
    │                    └─ All in one Dart process
    │
    ├─ MiniLM (E2) — 23 M, 384-d, the workhorse
    ├─ BGE-small-en-v1.5 (E1) — 33 M, 384-d, SOTA-quality
    ├─ CLIP-ViT-B/32 (E4) — 151 M, 512-d, image + text
    ├─ word2vec (E3) — from-scratch, tiny
    └─ any BERT-family via `SentenceEncoder.wrap(backbone)`
```

**One Dart binary. One `pub get`. One `Database.open`.** No FAISS, no
sqlite native lib, no PyTorch, no HTTP fan-out, no separate embedding
service. That's the asset.

## Split of responsibilities

| Concern | Package | Where |
| --- | --- | --- |
| Tokenize text | `dart_pytorch` | [WordPieceTokenizer](../lib/core/data/wordpiece_tokenizer.dart) |
| Load pretrained BERT weights | `dart_pytorch` | [BertHFLoader.miniLmL6V2Config](../lib/core/nn/bert_hf_loader.dart) |
| Forward pass → `[1, 384]` embedding | `dart_pytorch` | [SentenceEncoder](../lib/core/nn/sentence/sentence_encoder.dart) |
| Store row + vector | `dart_db_server` | `INSERT ... VEC('[...]')` |
| Build & maintain vector index | `dart_db_server` | `BLOB VECTOR(kind=hnsw, ...)` DDL |
| Build & maintain FTS5 corpus | `dart_db_server` | auto, on any `TEXT` column named in `vec_hybrid_search` |
| Filter, k-NN, RRF fusion, range query | `dart_db_server` | `vec_*` table-valued functions |
| Admin, warmup, verify, rebuild | `dart_db_server` | `PRAGMA vector_index_*` |
| Persistence | `dart_db_server` | JSON file passed to `Database.open` |

The demo is 400 lines. Roughly a quarter of it is the MiniLM wrapper,
half is SQL string-building, the rest is CLI + formatting.

## The five roles `dart_db_server` plays in `bin/db_rag_demo.dart`

### 1. Runtime — a whole SQL engine in one call

```dart
import 'package:dart_db_server/dart_db_server.dart';

final db = await Database.open('data/rag.json');
try {
  // ... everything below ...
} finally {
  await db.close();
}
```

That's the entire "install & run a vector database" step. Persisted
to a JSON file; pass a fresh path for an ephemeral store.

### 2. Schema — the vector index is declared *in the DDL*

```sql
CREATE TABLE chunks (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  source       TEXT    NOT NULL,
  topic        TEXT    NOT NULL,
  chunk_text   TEXT    NOT NULL,
  embedding    BLOB VECTOR(
    dim=384,               -- must match your embedder
    kind=hnsw,             -- flat | hnsw | ivfflat | lsh | pq | ivfpq
    metric=cosine,         -- cosine | l2 | l2sq | ip
    m=16,
    ef_construction=64,
    filter_cols='topic'    -- opts into O(1) payload-filter pruning
  )
);
```

One `CREATE TABLE` provisions:

- an HNSW cosine index on `embedding`,
- an inverse index on `topic` for filtered search,
- a lazily-built FTS5 corpus over `chunk_text` (used by hybrid search).

No separate `CREATE INDEX`, no vector-extension setup, no FTS5 virtual
table. It's one column type: `BLOB VECTOR(...)`.

### 3. Ingest — clean split

`dart_pytorch` produces the vector; `dart_db_server` stores it:

```dart
final vec     = encoder.embed(chunkText);        // dart_pytorch: Float32List(384)
final vecJson = '[${vec.join(",")}]';
await db.execute(
  "INSERT INTO chunks (source, topic, chunk_text, embedding) "
  "VALUES ('faq', 'security', 'MFA is available via TOTP...', "
  "        VEC('$vecJson'))",                    // dart_db_server: VEC() cast
);
```

`VEC('[...]')` is `dart_db_server`'s SQL cast from a JSON array literal
to a typed vector blob. `dart_pytorch` never sees the DB; `dart_db_server`
never sees the model.

Bulk load bypasses per-row SQL parsing entirely:

```sql
SELECT * FROM vec_batch_insert(
  'chunks', 'id', 'embedding',
  '[{"id":1,"vec":[...]}, {"id":2,"vec":[...]}, ...]'
);
```

Then warm the index once so HNSW isn't built lazily on the first user
query:

```dart
await db.warmVectorIndexes();
```

### 4. Retrieval — four table-valued functions cover every RAG mode

Every retrieval mode is a **table-valued function** that composes with
plain SQL `JOIN`s. That means no app-side scoring, no app-side merging
— you pull whatever columns you want back with the hits.

| Mode | SQL | When to use |
| --- | --- | --- |
| Plain semantic k-NN | `vec_search(t, col, VEC(q), k)` | Baseline RAG retrieval |
| Payload-filtered k-NN | `vec_search_filtered(t, col, VEC(q), k, '{"tenant":42}')` | Multi-tenant SaaS, source-scoped search |
| Hybrid vector + BM25 (RRF) | `vec_hybrid_search(t, vec_col, text_col, VEC(q), 'terms', k, rrf_k)` | **Default choice for real RAG** |
| Range / near-duplicate | `vec_range_search(t, col, VEC(q), threshold)` | Deduplication, clustering, plagiarism |

Example — the hybrid mode is the marquee one:

```sql
SELECT c.topic, c.chunk_text,
       s.distance, s.bm25, s.rrf_score
FROM vec_hybrid_search(
       'chunks', 'embedding', 'chunk_text',
       VEC('[0.11, -0.03, ...]'),
       'multi factor OR authentication OR 2fa',
       8, 60
     ) AS s
JOIN chunks c ON c.id = s.rowid
ORDER BY s.rrf_score DESC;
```

Reciprocal Rank Fusion means a passage that scores well on *both* the
semantic side (paraphrase recall) and the BM25 side (exact-keyword
recall) wins — nearly always strictly better than either signal alone.

Two FTS5 gotchas the demo's `_fts5Sanitize` handles for you:

- FTS5's parser only accepts bareword tokens — punctuation like `?`
  or `-` in a raw user question makes it throw.
- FTS5 defaults to **AND**. On a natural-language question this
  zeroes the BM25 side whenever any word is missing from a doc.
  Drop stopwords and OR the rest.

### 5. Admin surface — the operational half of a real database

```sql
PRAGMA vector_index_list;                                -- every vector column
PRAGMA vector_index_stats('chunks.embedding');           -- n / live / tombstones / bytes
PRAGMA vector_index_verify('chunks.embedding');          -- integrity check
PRAGMA vector_index_warm('chunks.embedding');            -- build now
PRAGMA vector_index_rebuild('chunks.embedding');         -- force clean rebuild
PRAGMA vector_analyze('chunks.embedding');               -- measure recall of chosen kind
```

Bake `vector_verify_all` into a health check and `vector_warm_all`
into your app's startup path — first-query latency then includes
zero build cost.

## End-to-end retriever

Straight from [bin/db_rag_demo.dart](../bin/db_rag_demo.dart), the
production-shape RAG retriever is about twenty lines:

```dart
Future<List<Passage>> retrieve(
  Database db,
  Encoder encoder,
  String question, {
  int k = 8,
  String? topicFilter,
}) async {
  final qv     = encoder.embed(question);
  final vecStr = '[${qv.join(",")}]';
  final bm25   = fts5Sanitize(question);  // strip punct + stopwords, OR

  final r = await db.execute('''
    SELECT c.id, c.source, c.topic, c.chunk_text, s.rrf_score
    FROM vec_hybrid_search(
           'chunks', 'embedding', 'chunk_text',
           VEC('$vecStr'), '$bm25', $k, 60
         ) AS s
    JOIN chunks c ON c.id = s.rowid
    ${topicFilter == null ? '' : "WHERE c.topic = '$topicFilter'"}
    ORDER BY s.rrf_score DESC
  ''');

  return [
    for (final row in r.rows)
      Passage(
        id:     row[0] as int,
        source: row[1] as String,
        topic:  row[2] as String,
        text:   row[3] as String,
        rrf:    (row[4] as num).toDouble(),
      ),
  ];
}
```

Feed the resulting `Passage.text` list into your LLM's prompt as
grounding context and you have a working RAG loop.

## Choosing an embedder

Any 384-d, 512-d, 768-d, 1024-d, 1536-d, or 3072-d model works — just
match `dim=` in the DDL. Cheat sheet:

| Provider | Model | Dim | Notes |
| --- | --- | --- | --- |
| **`dart_pytorch` local** | `all-MiniLM-L6-v2` | 384 | Workhorse, ~87 MB. Used by the demo. |
| **`dart_pytorch` local** | `BAAI/bge-small-en-v1.5` | 384 | SOTA in the ~30 M range, drop-in. |
| **`dart_pytorch` local** | CLIP-ViT-B/32 | 512 | Joint image + text — swap embedder, keep the DB. |
| OpenAI | `text-embedding-3-small` | 1536 | Cheap, strong. |
| OpenAI | `text-embedding-3-large` | 3072 | Higher quality, ~5× cost. |
| Cohere | `embed-english-v3.0` | 1024 | Strong on retrieval benchmarks. |

`dart_db_server` does not care where the vector came from — the same
schema works for any provider. This is the whole point of the
separation: swap MiniLM for CLIP and the SQL stays identical, only
`dim=384` becomes `dim=512`.

## Choosing an index kind

| Corpus size | `kind=` | Rationale |
| --- | --- | --- |
| < 100 k       | `flat`   | 100 % recall, no build cost |
| 100 k – 1 M   | `hnsw`   | Default RAG choice; tune `m`, `ef_construction` |
| > 1 M         | `ivfpq`  | Add `filter_cols=` for tenant / category pre-pruning |
| Corpus > RAM  | `USING paged` table + `PRAGMA vector_index_warm_all` at startup |

For a runnable comparison on synthetic data see
[bin/vector_index_benchmark_demo.dart](../bin/vector_index_benchmark_demo.dart)
(R7 in [commands.md](../commands.md)).

## Running the demo

```sh
# One-time
mkdir -p models/minilm && cd models/minilm
for f in config.json tokenizer_config.json vocab.txt model.safetensors; do
  curl -sSL -O \
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/$f"
done
cd -

# Every run
dart run bin/db_rag_demo.dart
dart run bin/db_rag_demo.dart --query "how do I turn on 2FA?"
dart run bin/db_rag_demo.dart --db-file /tmp/rag.json    # persist
```

Prints seven numbered sections — DDL, ingest, admin, plain k-NN,
filtered k-NN, hybrid RRF, range search — over the tiny
[data/support_faq.txt](../data/support_faq.txt) corpus. Exit code 0
means all four vector functions and the admin surface work end-to-end.

## Where to go next

- **Full command reference**: [commands.md § R13](../commands.md) (12 subsections, every SQL statement documented).
- **Upstream recipe series**: `dart-db-server/doc/rag-semantic-search.md`, `hybrid-search.md`, `multi-tenant-search.md`, `recommendations.md`, `duplicate-detection.md`, `index-selection.md`, `operations.md`.
- **In-repo vector toolkit deep dive** (FAISS-format, IndexFlat, IVFPQ, HNSW from first principles): [doc/vectors/README.md](vectors/README.md).
- **RAG recipes on top of this stack**: R1–R12 in [commands.md](../commands.md).
