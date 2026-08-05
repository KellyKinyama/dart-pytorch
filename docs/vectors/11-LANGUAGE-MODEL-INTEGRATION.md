# **11. INTEGRATING VECTOR INDEXES WITH LANGUAGE MODELS**

By chapter 10 you have a working vector store. The Hugging Face
runners in [../../commands.md](../../commands.md) give you working
language models (distilgpt2 through gpt-j-6b). This chapter connects
them.

There are three practical wiring patterns, in order of increasing
scope:

1. **Semantic search** — encode docs into vectors, index them, cosine-
   nearest the query. No LM generation.
2. **LM-as-encoder** — reuse the pretrained transformer's final
   hidden state as the embedding. Better recall than a from-scratch
   encoder, no fine-tuning required.
3. **Retrieval-augmented generation (RAG)** — retrieve top-k
   documents, splice them into the LM's prompt, generate.

All three share the same index API (chapter 2). The only thing that
changes is where the embedding vectors come from and what happens
after search.

## 11.1 Pattern 1: from-scratch encoder + triplet fine-tune

Already implemented end-to-end in
[../../bin/vector_store_demo.dart](../../bin/vector_store_demo.dart).
Pipeline:

```
docs ─▶ word-hash tokenize ─▶ TextTransformer(embedDim=32)
                                       │
                                       ▼
                            mean-pool over seq dim
                                       │
                                       ▼
                              L2-normalize ─▶ [1, D]
                                       │
                                       ▼
                        add(payload, vector) to VectorStore
```

Search is a single `qT.matmul(indexT)` — the same math
`IndexFlatIP` runs, just implemented directly on `Tensor` so it
picks up whatever `Device` the encoder used.

Use this pattern when you want a small, self-contained embedder
that trains in seconds on labelled `(anchor, positive, negative)`
triplets. Swap the ad-hoc `VectorStore` for `IndexFlatIP` from
[../../lib/core/vector_store/index_flat.dart](../../lib/core/vector_store/index_flat.dart)
the moment you cross a few thousand documents — you get
persistence (chapter 8) and can promote to `IndexIVFFlat` /
`IndexIVFPQ` / `IndexHNSW` (chapters 4–6) with zero surface
changes.

## 11.2 Pattern 2: use a pretrained LM as the encoder

The GPT surface in
[../../lib/core/nn/gpt.dart](../../lib/core/nn/gpt.dart) exposes
its sub-modules directly — `tokenEmb`, `posEmb`, `embedDrop`,
`encoder`. That lets you run the transformer stack **without** the
LM head and use the final hidden state as your embedding:

```dart
import 'dart:typed_data';
import 'package:dart_pytorch/dart_pytorch.dart';

/// Mean-pool + L2-normalize the final hidden state of a GPT model.
/// Returns a `Float32List` of length `config.embedDim` ready for
/// `IndexFlatIP.add([...])`.
Float32List embedWithLM(GPT model, List<int> tokenIds) {
  return Tensor.noGrad(() {
    final tokens = Tensor.fromList(
      [tokenIds.length],
      tokenIds.map((i) => i.toDouble()).toList(),
      device: model.config.device,
    );

    var h = model.tokenEmb(tokens);                 // [N, D]
    h = model.posEmb(h);                             // [N, D]
    h = model.embedDrop(h);
    final mask = tokenIds.length > 1
        ? causalMask(tokenIds.length, device: h.device)
        : null;
    h = model.encoder(h, mask: mask);                // [N, D]

    // Mean-pool across the sequence axis → [D].
    final pooled = h.mean(dim: 0);                   // [D]

    // L2-normalize so IndexFlatIP == cosine similarity.
    final norm = pooled
        .mul(pooled)
        .sum()
        .add(Tensor.scalar(1e-12, device: pooled.device))
        .sqrt();
    final normalized = pooled.div(norm);

    final v = Float32List(model.config.embedDim);
    final flat = normalized.toList();
    for (var i = 0; i < v.length; i++) {
      v[i] = flat[i].toDouble();
    }
    return v;
  });
}
```

Wiring it into a `IndexFlatIP` looks like:

```dart
final model = /* load via loadGpt2Hf, loadPythiaHf, loadGptJHf */;
final tokenizer = /* loaded from tokenizer.json */;
final index = IndexFlatIP(model.config.embedDim);
final payloads = <String>[];

for (final doc in docs) {
  final ids = tokenizer.encode(doc);
  index.add([embedWithLM(model, ids)]);
  payloads.add(doc);
}

// Query
final qids = tokenizer.encode('What is a safe vehicle for kids?');
final result = index.search([embedWithLM(model, qids)], 5);
for (var j = 0; j < result.ids[0].length; j++) {
  print('${result.scores[0][j].toStringAsFixed(3)}  ${payloads[result.ids[0][j]]}');
}
```

**Model-size cheat sheet** (which runner to load — see
[../../commands.md](../../commands.md) for the CLI equivalents):

| Model | `embedDim` | Notes |
|---|---|---|
| distilgpt2      | 768  | Cheapest, ~200 ms per short doc on CPU. |
| gpt2            | 768  | Same dim, better quality. |
| pythia-160m     | 768  | NeoX/rotary, tokenizer is different — GPT-NeoX BPE. |
| gpt2-medium     | 1024 | Sweet spot for offline batch embedding. |
| pythia-410m     | 1024 | |
| gpt-j-6b        | 4096 | Slow (CPU) but strongest single-vector representation of anything in this repo. |

Watch the memory: at 4096-dim × 4 bytes × 1 M docs = 16 GB just for
the vectors. Cross that threshold and swap `IndexFlatIP` for
`IndexIVFPQ` (chapter 5) or add a `PCA` pre-transform (chapter 7) to
drop the dim to something like 256 first.

## 11.3 Pattern 3: retrieval-augmented generation

Given the search from §11.2, generation is a prompt-templating step
followed by the model's normal `generate(...)`:

```dart
List<int> ragPrompt(
  List<String> retrieved,
  String question, {
  required Object tokenizer, // your loaded HF tokenizer
}) {
  final ctx = retrieved
      .asMap()
      .entries
      .map((e) => '[${e.key + 1}] ${e.value}')
      .join('\n');
  final prompt = 'Context:\n$ctx\n\nQuestion: $question\nAnswer:';
  return (tokenizer as dynamic).encode(prompt) as List<int>;
}

// Retrieve
final qEmb = embedWithLM(model, tokenizer.encode(question));
final hits = index.search([qEmb], 4);
final retrieved = [for (final id in hits.ids[0]) payloads[id]];

// Generate
final promptIds = ragPrompt(retrieved, question, tokenizer: tokenizer);
final full = model.generate(
  promptIds,
  maxNewTokens: 128,
  temperature: 0.7,
  topK: 40,
  seed: 42,
);
final answerIds = full.sublist(promptIds.length).map((d) => d.toInt()).toList();
print(tokenizer.decode(answerIds));
```

Two knobs that dominate quality:

- **k** — how many docs to inject. Too low → miss the answer; too
  high → context window overflow. Start at 4.
- **prompt shape** — the `[i] passage` numbering above lets the LM
  cite sources back to you, which is worth the ~15 wasted tokens.

Context-window ceilings (`config.maxCtx`) for the runners in
[../../commands.md](../../commands.md):

| Model | `maxCtx` | Comfortable prompt budget |
|---|---|---|
| distilgpt2 / gpt2*   | 1024 | ~800 tokens of context + 200 for answer |
| pythia-*             | 2048 | ~1600 / 400 |
| gpt-j-6b             | 2048 | ~1600 / 400 |

Truncate the retrieved passages before you exceed the ceiling —
`generate` will throw once the prompt alone hits `maxCtx`.

## 11.4 Picking an index

The same table from chapter 10 applies, filtered by the LM
embedding dimensions of §11.2:

| Corpus size | Index | Why |
|---|---|---|
| < 10 k      | `IndexFlatIP`            | 100% recall, no tuning, milliseconds per query. |
| < 1 M       | `IndexIVFFlat` (nlist ~ √N) | Cell-probe cuts scan cost 100× at ~98% recall. |
| < 100 M     | `IndexIVFPQ`             | PQ codes compress vectors 32×–64×; only option that fits in RAM. |
| any + fast  | `IndexHNSW`              | Sub-ms queries, memory-hungry, hard to update in-place. |

Wrap any of them in `IndexPreTransform` (chapter 7) with a `PCA`
transform to knock GPT-J's 4096-dim vectors down to 256 before
they touch the index — recall drop is typically single-digit
percentage points, memory drops 16×.

## 11.5 Persistence

Everything above roundtrips through the FAISS-compatible binary
format (chapter 8):

```dart
saveIndex(index, 'my_corpus.faiss');
final loaded = readIndex('my_corpus.faiss');
```

Store `payloads` alongside as JSON — the index only sees ids
(0..N-1) unless you wrap it in `IndexIDMap` (chapter 7) with your
own application ids.

## 11.6 Summary

- **Small / labelled**: pattern 1 (train `TextTransformer`, use
  `IndexFlatIP`, promote index type as N grows).
- **Zero-shot, no labels**: pattern 2 (mean-pool GPT-2 / Pythia /
  GPT-J hidden state, `IndexFlatIP` → `IndexIVFPQ`).
- **Answering questions over docs**: pattern 3 (§11.2 for retrieval
  + `.generate()` on the same or a bigger LM).

Every pattern shares one property: nothing here is
model-family-specific. Swap the encoder, keep the index; swap the
index, keep the encoder.
