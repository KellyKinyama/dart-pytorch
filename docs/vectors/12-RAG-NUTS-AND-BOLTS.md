# **12. RAG NUTS AND BOLTS: TRACING A SINGLE QUERY END-TO-END**

Chapter 11 introduced the three integration patterns. This chapter
is the *code walkthrough*: we follow a single question —
`"What hormone regulates blood sugar?"` — from the raw string down
through every layer we have already built until an answer comes
back. Nothing new is introduced; every file mentioned is already in
the tree.

The runnable end-to-end version is
[../../bin/rag_qa_demo.dart](../../bin/rag_qa_demo.dart). Chapter 12
tells you *why* each of its ~300 lines is there.

## **12.1. What we already have**

The pieces of a working RAG system, laid out by directory:

```
┌────────────────────────────────────────────────────────────────────┐
│  models/distilgpt2/                                                │
│    model.safetensors        ── HF weights (500 MB fp32)            │
│    tokenizer.json           ── HF fast tokenizer                   │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│  Load path                                                         │
│    lib/core/nn/safetensors.dart       ── SafeTensors.loadFile      │
│    lib/core/nn/gpt2_hf_loader.dart    ── GPT2HFLoader.loadFile     │
│    lib/core/nn/gpt.dart               ── GPT / GPTConfig           │
│    lib/core/data/hf_bpe_tokenizer.dart── HFBpeTokenizer.loadFile   │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│  Encoder                                                           │
│    model.tokenEmb, posEmb, embedDrop, encoder                      │
│    lib/core/nn/masks.dart             ── causalMask(N)             │
│  Post-process                                                      │
│    last-token pooling  → Float32List                               │
│    corpus-mean centring                                            │
│    L2-normalise                                                    │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│  Index                                                             │
│    lib/core/vector_store/index_flat.dart  ── IndexFlatIP           │
│      .add(List<Float32List>)                                       │
│      .search(queries, k) → SearchResult(distances, ids)            │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│  Prompt template                                                   │
│    "Context:\n[1] ... \n[2] ... \nQuestion: ... \nAnswer:"         │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│  Generator (same model)                                            │
│    model.generate(prompt, maxNewTokens, temperature, topK)         │
│    lib/core/nn/kv_cache.dart          ── EncoderCache              │
│    tokenizer.decode(newIds)          → answer string               │
└────────────────────────────────────────────────────────────────────┘
```

Every box is code we already ship. RAG is the wiring, not new
machinery.

## **12.2. String → token ids**

The question `"What hormone regulates blood sugar?"` enters as
UTF-8 bytes. `HFBpeTokenizer.encode` from
[../../lib/core/data/hf_bpe_tokenizer.dart](../../lib/core/data/hf_bpe_tokenizer.dart)
does what `tokenizers` does at runtime:

1. **UTF-8 byte encode** the input string.
2. **Byte→unicode map** — GPT-2's `bytes_to_unicode`. Space (0x20)
   becomes `Ġ` (U+0120); other control/whitespace bytes get shifted
   into the printable private range starting at 0x100.
3. **Pre-tokenizer regex** — the canonical GPT-2 regex
   ```
   'll|'s|'re|... | ?\p{L}+ | ?\p{N}+ | ?[^\s\p{L}\p{N}]+ | \s+
   ```
   chops the byte-mapped string into "words".
4. **BPE merges** — each word (as a sequence of single-char
   symbols) is iteratively fused by the lowest-rank pair from
   `tokenizer.json`'s merges table.
5. **Vocab lookup** — each surviving symbol becomes an `int`.

The question comes out as roughly

```
[2061,  17969,  17632,  ates,   2910,  7543, ?]
 What   Ġhorm   Ġregul  ates   Ġblood Ġsugar ?
```

exact ids depend on the merges. What matters: we now have a
`List<int>` living in vocab-space 0..50256.

For a corpus of 10 documents plus 5 questions this happens 15
times, no learnable state anywhere. Everything downstream reads
ids, not strings.

## **12.3. LM as encoder: skipping the head**

`GPT` in [../../lib/core/nn/gpt.dart](../../lib/core/nn/gpt.dart)
exposes its sub-modules as `final` fields:

```dart
final Embedding                    tokenEmb;
final LearnedPositionalEmbedding   posEmb;
final Dropout                      embedDrop;
final TransformerEncoder           encoder;
final Linear?                      untiedHead;  // null when weights tied
```

The public `call(tokens)` runs the whole stack and multiplies by
either the tied embedding or `untiedHead` to produce logits. For
retrieval we want the **hidden state before the LM head**, so we
call the sub-modules directly:

```dart
var h = model.tokenEmb(tokens);                       // [N, D]
h = model.posEmb(h);                                   // [N, D]
h = model.embedDrop(h);
final mask = clipped.length > 1
    ? causalMask(clipped.length, device: h.device)
    : null;
h = model.encoder(h, mask: mask);                     // [N, D]
```

`causalMask(N)` from
[../../lib/core/nn/masks.dart](../../lib/core/nn/masks.dart) builds
the same triangular float mask the LM uses during generation — one
`0.0` on the diagonal, `-∞` above. Without it a token would attend
to the future and the encoder would leak information.

`h` is now `[N, D]` — `N` tokens in the input, `D = embedDim` =
768 for distilgpt2 / gpt2 / pythia-160m, 1024 for gpt2-medium and
pythia-410m, 4096 for gpt-j-6b. This is the state chapter 11
promised.

## **12.4. Last-token pooling, then the anisotropy trap**

We need `[D]`, not `[N, D]`. The obvious choice is mean-pool over
the `N` axis. **Do not do that on a causal LM.** Every mean-pooled
vector ends up cosine ~0.996 to every other one — a phenomenon the
literature calls the "anisotropy cone".

The runnable demo made both mistakes before landing on the fix.
The fix is a stack of two tricks:

**Trick 1 — last-token pooling.** Under the causal mask, only
position `N-1` has attended to every previous token. It is the
only row that saw the whole sequence, so it carries the summary:

```dart
final flat = h.toList();                              // length N*D
final base = (n - 1) * d;
final out = Float32List(d);
for (var j = 0; j < d; j++) {
  out[j] = flat[base + j].toDouble();
}
```

Better than mean, but by itself still not enough — on distilgpt2
the last-token vectors also cluster (cosine ~0.995) because the
final `.` / newline token pulls every doc towards a shared
representation.

**Trick 2 — corpus-mean centring.** Do a first pass to collect
every raw last-token vector, average them, subtract the mean from
every doc AND every query, *then* L2-normalise. This is the same
"BERT-flow" centring trick used in the sentence-encoder
literature.

```dart
final rawDocVecs = <Float32List>[];
for (final doc in docs) {
  rawDocVecs.add(lastTokenHidden(model, tokenizer.encode(doc)));
}
final mean = meanVector(rawDocVecs, D);
final index = IndexFlatIP(D);
for (final v in rawDocVecs) {
  index.add([centerAndNormalize(v, mean)]);
}
```

`centerAndNormalize` does `v ← (v − mean) / ‖v − mean‖₂`. The unit
vector lives on the sphere so `IndexFlatIP` (chapter 3) computing
dot products *is* computing cosine similarity.

Concrete impact on the demo corpus (10 docs, 5 questions):

| Pooling | Score range | Right passage in top-3 |
|---|---|---|
| Mean-pool only            | 0.995 – 0.998 | ~1/5 questions |
| Last-token only           | 0.994 – 0.996 | ~2/5 questions |
| Last-token + mean-centre  | 0.09 – 0.69   | 5/5 questions |

The API notes from user memory (`Tensor.mean()` has no `dim`
argument, `SearchResult.distances` holds both L2 and IP scores)
are baked into the demo — that's why the pooling is done in Dart
over `h.toList()` rather than in tensor space.

## **12.5. The index — why IP + unit vectors gives cosine**

`IndexFlatIP` from
[../../lib/core/vector_store/index_flat.dart](../../lib/core/vector_store/index_flat.dart)
is the same brute-force class from chapter 3, configured with
`Metric.innerProduct`. For unit vectors:

```
IP(q, v)  =  qᵀ v  =  ‖q‖ · ‖v‖ · cos(q, v)  =  cos(q, v)
                                    └─ 1 · 1 ─┘
```

so we never store L2 distances at all — the raw dot product ranks
correctly. `SearchResult.distances` still holds them (the field is
called "distances" for API symmetry with `IndexFlatL2`; the metric
is fixed at construction and the interpretation is fixed with it).

For the 10-doc corpus in the demo, brute force costs
`10 * 768 = 7680` multiply-adds per query — sub-millisecond.
Chapter 4-6 explain when to graduate to `IndexIVFFlat` /
`IndexIVFPQ` / `IndexHNSW`; the API is identical so the swap is a
one-liner.

## **12.6. Prompt assembly**

Retrieval done. Now we splice the top-`k` passages into a prompt
the LM can continue. The demo uses:

```
Context:
[1] Insulin is a peptide hormone produced by beta cells in the pancreas...
[2] Photosynthesis is the process by which green plants convert...
[3] Albert Einstein published the special theory of relativity...

Question: What hormone regulates blood sugar?
Answer:
```

Three design choices worth calling out:

- **Numbered passages** — `[1] ... [2] ... [3] ...`. GPT-2 has
  seen enough Wikipedia and Stack Overflow to associate bracketed
  numerals with citations; the completion is more likely to
  reference them coherently.
- **Explicit `Question:` / `Answer:` markers.** Plain base LMs
  aren't instruction-tuned; anchors like these are the closest
  thing to a system prompt available.
- **Answer marker with no trailing newline.** The next token the
  model emits is the first token of the answer.

The whole string goes back through `tokenizer.encode` to become
`List<int>`. We check `prompt.length + maxNewTokens <=
config.maxCtx` (1024 for GPT-2, 2048 for Pythia / GPT-J) before
calling `generate`; if it overflows the demo falls back to top-1
retrieval only.

## **12.7. Generation**

`GPT.generate` from
[../../lib/core/nn/gpt.dart](../../lib/core/nn/gpt.dart) is the
same call every runner in
[../../commands.md](../../commands.md) uses:

```dart
final full = model.generate(
  promptDouble,
  maxNewTokens: _maxNewTokens,
  temperature: _temperature,
  topK: _generatorTopK,
);
```

Under the hood it:

1. Runs `model.eval()` (freezes dropout to zero).
2. Enters a `Tensor.noGrad` scope so no autograd tape is built.
3. On the first step, forwards the whole prompt through the same
   sub-modules we just used for encoding, and populates an
   `EncoderCache` from
   [../../lib/core/nn/kv_cache.dart](../../lib/core/nn/kv_cache.dart)
   with K/V for every layer.
4. On every subsequent step, forwards **just the new token** and
   appends its K/V — O(N) per step instead of O(N²).
5. Divides logits by `temperature`, keeps the top-`topK` if
   non-zero, samples, appends.
6. Returns `prompt + generated` as a flat `List<double>` (the same
   float-index convention used by everything else in the repo).

We drop the first `promptDouble.length` entries and hand the tail
to `tokenizer.decode` — same class as encoding, run in reverse.
The demo prints the decoded string.

## **12.8. Timings from an actual run**

Numbers from the smoke-test that landed the current demo:

- Load safetensors — 5.4 s (distilgpt2, CPU).
- Build tokenizer from `tokenizer.json` — 0.5 s.
- Embed 10 documents (pass 1 + pass 2) — 38 s ≈ 3.8 s/doc.
- Retrieve top-3 for one question — < 1 ms.
- Generate 60 tokens — 73 s ≈ 1.2 s/token.

The generator dominates. Chapter 11's cheat sheet tells you which
model to switch to and what quality-vs-speed you buy. The
retrieval half is essentially free at 10 docs; you can add three
orders of magnitude to the corpus before search becomes the
bottleneck.

## **12.9. What to swap next**

The wiring above is deliberately the simplest thing that works.
Every layer has a documented upgrade path elsewhere in this
series:

| Bottleneck | Symptom | Upgrade | Chapter |
|---|---|---|---|
| Corpus too big for `IndexFlatIP` | Search > 100 ms | `IndexIVFFlat` (nlist ≈ √N) | 4 |
| Vectors don't fit in RAM     | OOM around 10 M docs × 4096 dim | `IndexIVFPQ` with 8-byte codes | 5 |
| Need sub-ms search           | 1 M docs, 5 ms budget | `IndexHNSW` | 6 |
| Embedding dim too large      | GPT-J's 4096-d vectors dominate memory | `PCA` pre-transform to 256 | 7 |
| Weak retrieval               | Anisotropy fix isn't enough | Fine-tune small `TextTransformer` with triplet loss | [bin/vector_store_demo.dart](../../bin/vector_store_demo.dart) |
| Need persistence             | Rebuilding the index every start-up | `saveIndex` / `readIndex` (FAISS binary format) | 8 |
| Multi-tenant ids             | Application ids ≠ 0..N-1 | Wrap in `IndexIDMap` | 7 |
| GPU search                   | CPU search saturated | `GpuIndexFlat` | 3 |

None of these change anything above the index API — the encoder,
prompt template, and generator stay identical.

## **12.10. What we haven't built (yet)**

Being honest about the gaps:

- **No re-ranker.** Retrieval returns top-`k`; the LM gets the
  top-`k` verbatim. A cross-encoder re-ranker between the two
  would lift quality — not implemented.
- **No streaming generation.** `generate` returns the full token
  list at the end. The HTTP runners in
  [../../bin/](../../bin) return one JSON blob per request; no
  server-sent-events variant yet.
- **No conversation memory.** Each question is independent.
  Multi-turn RAG needs its own scratch index of turn embeddings;
  none of that lives here.
- **No hybrid BM25 + dense retrieval.** Everything is pure
  vector.
- **No instruction-tuned checkpoint.** All four families are
  base LMs. Answers read like continuations, not chat replies.
  Loading an instruction-tuned checkpoint of the same
  architecture (e.g. a Pythia SFT release) requires only that its
  safetensors match the shapes the loader validates — no code
  changes.

Each of these is a chapter-sized addition, none of them require
rewiring the parts we already have.

## **12.11. Summary — one query, one paragraph**

The question is UTF-8 bytes ⇒ `HFBpeTokenizer.encode` gives a
`List<int>` ⇒ token embedding + positional embedding + causal
encoder from `GPT` gives `[N, 768]` hidden state ⇒ last row +
subtract corpus mean + L2-normalise gives a `Float32List(768)` ⇒
`IndexFlatIP.search` returns `top-k` ids into the corpus ⇒
prompt-template splices the passages ⇒ `HFBpeTokenizer.encode`
again ⇒ `model.generate` runs the same LM autoregressively with a
KV cache ⇒ `HFBpeTokenizer.decode` on the new ids gives an answer
string. Everything in that sentence is in the tree today.

<div align="right">

[&lt;&lt;&nbsp;&nbsp;Previous chapter: LANGUAGE MODEL INTEGRATION](./11-LANGUAGE-MODEL-INTEGRATION.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Home: README&nbsp;&nbsp;&gt;](./README.md)

</div>
