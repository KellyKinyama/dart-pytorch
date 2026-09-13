# The pitch: production-shape RAG in one Dart binary

Two Dart packages: **[`dart_pytorch`](https://github.com/KellyKinyama/dart-pytorch)**
computes embeddings, **[`dart_db_server`](https://github.com/KellyKinyama/dart-db-server)**
stores and searches them. Together they are a **self-contained,
offline-capable, no-C-dependency RAG runtime** that runs anywhere
Dart runs — server, desktop, CLI, embedded, Flutter mobile.

There is no other stack in the Dart ecosystem that closes this loop.
There are very few stacks in *any* ecosystem that close it in a single
process with no native dependencies.

That is the pitch.

---

## The one-line claim

> **The only vector database in the Dart ecosystem, and one of a very
> small number anywhere, that ships SQL + native vector index + FTS5 BM25
> + hybrid RRF + payload filters + MySQL wire in a single binary with
> no C dependencies — paired with a pure-Dart embedding model runtime
> that turns it into a complete end-to-end RAG stack.**

Everything below defends that sentence.

---

## The 30-second demo

```sh
# One-time
mkdir -p models/minilm && cd models/minilm
for f in config.json tokenizer_config.json vocab.txt model.safetensors; do
  curl -sSL -O \
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/$f"
done
cd -

# The show
dart run bin/db_rag_demo.dart --query "how do I turn on 2FA?"
```

That's it. One command. No Docker, no Python, no FAISS install, no
Postgres, no embedding API keys, no separate index service. The demo
runs through seven RAG modes — DDL, ingest, admin, plain k-NN,
filtered k-NN, hybrid vector+BM25 with RRF, range search — over a
support-FAQ corpus, in-process, in a few seconds.

Source: [bin/db_rag_demo.dart](bin/db_rag_demo.dart) · Architecture:
[doc/db_vector_rag.md](doc/db_vector_rag.md) · Full command
reference: [commands.md § R13](commands.md).

---

## Feature parity with the incumbents

This is a peer-level vector database. Compared to what a team would
otherwise reach for:

| Feature | **This stack** | pgvector | Qdrant | Weaviate | FAISS |
| --- | --- | --- | --- | --- | --- |
| SQL surface | ✅ | ✅ | ❌ (REST) | ❌ (GraphQL) | ❌ |
| HNSW index | ✅ | ✅ | ✅ | ✅ | ✅ |
| IVF / PQ / IVFPQ | ✅ | ⚠️ (ivfflat only) | ✅ | ❌ | ✅ |
| LSH | ✅ | ❌ | ❌ | ❌ | ✅ |
| Hybrid vec + BM25 (RRF) | ✅ **built-in** | ⚠️ (tsvector glue) | ✅ | ✅ | ❌ |
| Payload filter with O(1) pruning | ✅ | ✅ | ✅ | ✅ | ❌ |
| Range / near-duplicate search | ✅ | ✅ | ✅ | ⚠️ | ✅ |
| Batch queries | ✅ | ❌ | ✅ | ⚠️ | ✅ |
| Row-to-row k-NN join | ✅ | ⚠️ (self-join) | ❌ | ❌ | ❌ |
| Wire protocol interop | ✅ **MySQL** | ✅ Postgres | REST | GraphQL | in-proc |
| Deploys as | **one Dart binary** | Postgres server | Rust binary | Go binary | library |
| Bundled pure-Dart embedder | ✅ | ❌ | ❌ | ⚠️ (module) | ❌ |
| Zero native dependencies | ✅ | ❌ | ❌ | ❌ | ❌ |

The last two rows are the moat.

---

## What ships in `dart_db_server`

Concrete, verified in code — not roadmap.

### Six index kinds

`flat` · `hnsw` · `ivfflat` · `lsh` · `pq` · `ivfpq`

Every kind that FAISS ships, plus LSH, plus a graph-based ANN — with
one column-type declaration:

```sql
embedding BLOB VECTOR(dim=384, kind=hnsw, metric=cosine,
                     m=16, ef_construction=64, filter_cols='tenant')
```

### Four distance metrics

`cosine` · `l2` · `l2sq` · `ip`

### Nine table-valued query functions

Every retrieval mode you would build a real RAG system out of:

| Function | What it does |
| --- | --- |
| `vec_search` | Plain semantic k-NN |
| `vec_search_filtered` | k-NN with pre-filter via payload index |
| `vec_search_batch` | Multi-query k-NN in one call |
| `vec_search_filtered_batch` | Multi-query filtered k-NN |
| `vec_range_search` | Every row within a distance threshold |
| `vec_hybrid_search` | **Vector + BM25 via Reciprocal Rank Fusion** — the marquee RAG mode |
| `vec_hybrid_search_batch` | Multi-query hybrid RRF |
| `vec_search_join` | Row-to-row k-NN join (recommendations) |
| `vec_search_join_filtered` | Same, with payload filter |
| `vec_batch_insert` | Bulk ingest bypassing per-row SQL parse |

All are **table-valued functions** — they compose with plain SQL
`JOIN`s. No app-side scoring, no app-side merging. It's just SQL.

### Built-in FTS5 BM25

Not an extension. Any `TEXT` column named in `vec_hybrid_search`
gets a lazily-built BM25 corpus automatically. This is why hybrid
retrieval "just works" without a separate search engine.

### O(1) payload filtering for multi-tenant RAG

```sql
filter_cols='tenant,kind'
```

The engine builds an inverse index (`col=value → row-position set`)
so filtered searches intersect candidate rows **before** touching the
vector index. Multi-tenant SaaS RAG is a one-line schema change.

### Full admin / operational surface

```sql
PRAGMA vector_index_list;
PRAGMA vector_index_stats('t.col');
PRAGMA vector_index_verify[_all];
PRAGMA vector_index_rebuild[_all];
PRAGMA vector_index_warm[_all];
PRAGMA vector_analyze('t.col');   -- measure actual recall
```

Everything you need for a health check, a warmup hook, a
maintenance job, or a recall SLA.

### Out-of-core storage

`CREATE TABLE ... USING paged` + `PRAGMA vector_index_warm_all`
handles corpora bigger than RAM. Same query surface, same SQL,
persistent index.

### Automatic index maintenance

HNSW tombstones on `DELETE`/`UPDATE`; auto-rebuild triggers at 30 %
tombstone ratio on the next query. No `REINDEX` chores.

### Three access modes, same engine

1. **In-process** — `Database.open('rag.json')`.
2. **TCP JSON-line server** — `dart run bin/dart_db_server.dart --port 4555`.
3. **MySQL wire compatibility** — point *any* MySQL driver (Node, PHP,
   Python, Go, Rust, another Dart process) at it.

### Concurrency

`AsyncRwLock` — concurrent readers, exclusive writers, isolate-safe.

---

## What ships in `dart_pytorch` (the embedder side)

Any of the following, loaded from HuggingFace `.safetensors`, running
pure-Dart on CPU or hand-written CUDA on GPU:

| Model | Params | Dim | Notes |
| --- | --- | --- | --- |
| `all-MiniLM-L6-v2` | 23 M | 384 | Workhorse, ~87 MB, the demo default |
| `BAAI/bge-small-en-v1.5` | 33 M | 384 | SOTA-quality drop-in replacement |
| CLIP-ViT-B/32 dual encoder | 151 M | 512 | Joint image + text — same SQL, swap the model |
| word2vec skip-gram | tiny | any | Trained from scratch on your corpus |
| Any HF BERT-family | any | any | `SentenceEncoder.wrap(BertModel(cfg))` |

Swap embedder — only `dim=` in the DDL changes. That is the whole
point of the separation.

---

## Who this is for

- **Flutter product teams** shipping RAG features into an app and
  refusing to pull in Python / Postgres / a separate search cluster.
- **Dart backend teams** who already picked Dart for the server and
  want RAG that doesn't require adopting a second language stack.
- **Desktop / CLI tool builders** who need semantic search
  bundled *inside* their binary — support-agent inbox indexers,
  personal-knowledge assistants, air-gapped enterprise search.
- **Embedded / on-device inference** where a Python microservice
  is a non-starter.
- **Prototyping** — one `dart pub get`, one `Database.open`, and you
  are at Qdrant/pgvector feature parity in fifty lines of code.

---

## The honest limits (this is a pitch, not a lie)

Truthful, so nobody's surprised in month two:

1. **Pure-Dart execution** — no SIMD, no BLAS, no PQ codebook
   inner-product intrinsics. On ≥1 M rows expect a few × slower per
   query than a C++/Rust engine at the same recall. **For RAG this is
   invisible** — LLM latency dwarfs it. For pure-ANN benchmarking
   competitions, it matters.
2. **JSON persistence** — great for portability + git-diffing your
   state, not the fastest write log format. Bulk ingest tops out
   below WAL-based stores.
3. **Single-node** — no built-in replication or sharding runtime. The
   correctness surface for sharding exists (payload filters = natural
   shard key) but you'd write the fanout yourself.
4. **CPU-only index build** — HNSW graph construction is not GPU-
   accelerated. Seconds for 1 M × 384, meaningful for 10 M+.
5. **Empirical recall** — no formal SLA. `PRAGMA vector_analyze`
   measures it on real data; you quote what you measured.

None of these block a production RAG deployment. All of them are
things a pgvector or Qdrant sales engineer would either have their
own answer to, or quietly avoid.

---

## Why "biggest asset" is the right framing

Not because it's the fastest. Not because it's the biggest. Because
**no other stack lets a Dart team ship RAG end-to-end without leaving
Dart**, and because inside that market the feature set is at parity
with the incumbents.

There is a specific class of developer for whom this is not "a
choice" — it is *the* choice. That's the definition of a moat.

- **Feature completeness vs vector DBs**: parity with pgvector,
  close to Qdrant.
- **Throughput vs C++/Rust engines**: slower per query, faster
  developer loop, zero deployment friction.
- **Ecosystem uniqueness**: owns the Dart vector-DB category outright.
- **End-to-end RAG story**: `dart_pytorch` + `dart_db_server` is a
  legitimately differentiated stack — offline-capable, no C deps,
  one process, runs everywhere Dart runs including Flutter and Web.

That combination is what makes it worth pitching.

---

## Call to action

Two commands, five minutes, real RAG results printed to your terminal:

```sh
git clone https://github.com/KellyKinyama/dart-pytorch
git clone https://github.com/KellyKinyama/dart-db-server ../dart-db-server
cd dart-pytorch
dart pub get
mkdir -p models/minilm && cd models/minilm
for f in config.json tokenizer_config.json vocab.txt model.safetensors; do
  curl -sSL -O \
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/$f"
done
cd -
dart run bin/db_rag_demo.dart --query "how do I turn on 2FA?"
```

Then read [doc/db_vector_rag.md](doc/db_vector_rag.md) for the
architecture, [commands.md § R13](commands.md) for the twelve-
subsection API reference, and the twelve
`dart-db-server/doc/*.md` recipes for the full engine surface.

Ship it.
