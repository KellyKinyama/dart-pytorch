# Runnable commands (small → large)

Every command below assumes you're in the repo root and have the
weights downloaded under `models/<name>/`. All GPU-targeting commands
prepend `LD_LIBRARY_PATH=/usr/lib/wsl/lib` — that's a WSL2-specific
fix so the CUDA driver stub is found. Drop it on native Linux.

| # | Model | Params | Device | Tokenizer  | Runner |
|---|---|---|---|---|---|
| 1 | distilgpt2       | 82M   | GPU    | ✅ local | [bin/distilgpt2/run_gpu_api.dart](bin/distilgpt2/run_gpu_api.dart) |
| 2 | gpt2 (small)     | 124M  | GPU    | ✅ local | [bin/gpt2/small/run_gpu.dart](bin/gpt2/small/run_gpu.dart) |
| 3 | pythia-160m      | 160M  | GPU    | ✅ local | [bin/pythia/run_gpu_api.dart](bin/pythia/run_gpu_api.dart) |
| 4 | gpt2-medium      | 355M  | GPU    | ✅ local | [bin/gpt2/medium/run_gpu_api.dart](bin/gpt2/medium/run_gpu_api.dart) |
| 5 | pythia-410m      | 410M  | GPU    | ✅ local | [bin/pythia/run_410m_gpu_api.dart](bin/pythia/run_410m_gpu_api.dart) |
| 6 | gpt2-large       | 774M  | GPU (tight) | ✅ local | [bin/gpt2/large/run_gpu.dart](bin/gpt2/large/run_gpu.dart) |
| 7 | pythia-1b        | 1.0B  | GPU    | ✅ local | [bin/pythia/run_1b_gpu_api.dart](bin/pythia/run_1b_gpu_api.dart) |
| 8 | gpt-j-6b (hybrid)| 6.05B | CPU + GPU | ✅ local | [bin/gptj/run_6b_hybrid_api.dart](bin/gptj/run_6b_hybrid_api.dart) |
| 9 | gpt-j-6b (CPU)   | 6.05B | CPU    | ✅ local | [bin/gptj/run_6b_cpu_api.dart](bin/gptj/run_6b_cpu_api.dart) |

All `tokenizer.json` files are already downloaded under
`models/<name>/`, so every command below runs fully offline — no
network access needed at runtime. `--text` works everywhere.

### Refresh tokenizers (only if you nuke `models/`)

```sh
curl -L -o models/gpt2/tokenizer.json         https://huggingface.co/gpt2/resolve/main/tokenizer.json
curl -L -o models/gpt2-medium/tokenizer.json  https://huggingface.co/gpt2-medium/resolve/main/tokenizer.json
curl -L -o models/gpt2-large/tokenizer.json   https://huggingface.co/gpt2-large/resolve/main/tokenizer.json
curl -L -o models/gpt2-large/vocab.json       https://huggingface.co/gpt2-large/resolve/main/vocab.json
```

## What the tokenizer does

Language models don't consume characters — they consume integer
**token ids** drawn from a fixed vocabulary (~50k entries for the
GPT-2 family, ~50k for Pythia/NeoX, ~50k for GPT-J). The tokenizer
is the reversible map between raw UTF-8 text and that id stream:

```
"The world is"  ──encode──▶  [464, 995, 318]  ──model──▶  [318, 257, ...]
                                                              │
                                       decode ◀──────────────┘
                                          │
                                          ▼
                                    " a great"
```

Two on-disk formats show up in `models/`:

- **`tokenizer.json`** — Hugging Face "fast" tokenizer. A single
  JSON file bundling the vocab **and** the merges/pre-tokenizer/
  post-processor rules. Required for `--text STR`: it's what
  encodes your prompt into ids and decodes generated ids back to
  a string that respects byte-level BPE (so `"Ġworld"` becomes
  `" world"`, emoji round-trip, etc.).
- **`vocab.json`** — legacy GPT-2 vocab (id → surface token, with
  `Ġ` marking a leading space). No merges, no encoder rules — good
  enough to *decode* an id stream approximately, but you can't
  encode arbitrary text with it. Kept alongside `tokenizer.json`
  for legacy runners that only understand this format.

Practical rules of thumb for this repo:

| You want to… | Requires |
|---|---|
| Pass a natural-language prompt via `--text "..."` | `tokenizer.json` |
| Pass raw ids via `--prompt 464,995,318` | nothing (encoding skipped) |
| Get readable output from the runner | `tokenizer.json` preferred; `vocab.json` works with `Ġ`→space fallback |
| Hit `POST /generate` with `{"text": "..."}` | `tokenizer.json` on the server |
| Hit `POST /generate` with `{"tokens": [...]}` | nothing (ids in, ids + text-via-fallback out) |

Runners resolve the tokenizer in this order:

1. `--vocab PATH` (explicit override, either format)
2. `<weights-dir>/tokenizer.json`
3. `<weights-dir>/vocab.json`
4. otherwise: `--text` is rejected; `--prompt` still works.

## 1. distilgpt2 (82M, GPU)

```sh
LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  dart run bin/distilgpt2/run_gpu_api.dart \
    --text "The quick brown fox" --max-tokens 12
```

## 2. gpt2 small (124M, GPU)

Legacy demo runner (fixed prompt baked in, no `--text` / `--serve`):

```sh
LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  dart run bin/gpt2/small/run_gpu.dart models/gpt2/model.safetensors
```

For the full API surface (`--text`, `--serve`, sampling knobs) point
the distilgpt2 shim at gpt2's weights + tokenizer (same family):

```sh
LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  dart run bin/distilgpt2/run_gpu_api.dart \
    --path models/gpt2/model.safetensors \
    --vocab models/gpt2/tokenizer.json \
    --text "The quick brown fox" --max-tokens 12
```

## 3. pythia-160m (160M, GPU)

```sh
LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  dart run bin/pythia/run_gpu_api.dart \
    --text "Once upon a time," --max-tokens 30
```

## 4. gpt2-medium (355M, GPU)

```sh
LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  dart run bin/gpt2/medium/run_gpu_api.dart \
    --text "The world is" --max-tokens 20 --temperature 0.8 --top-k 40 --seed 42
```

Or equivalently with raw ids (`[464, 995, 318]` = `"The world is"`):

```sh
LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  dart run bin/gpt2/medium/run_gpu_api.dart \
    --prompt 464,995,318 --max-tokens 20
```

## 5. pythia-410m (410M, GPU)

```sh
LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  dart run bin/pythia/run_410m_gpu_api.dart \
    --text "The world is" --max-tokens 20
```

## 6. gpt2-large (774M, GPU — tight, may OOM)

Legacy demo runner (fixed prompt, no API surface):

```sh
LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  dart run bin/gpt2/large/run_gpu.dart models/gpt2-large/model.safetensors
```

At fp32 the weights alone are ~3 GB and activations + KV cache push
close to the 6 GB card's limit — longer prompts or larger
`--max-tokens` will likely OOM. Tokenizer + vocab are also on disk
for future API-runner support:

```
models/gpt2-large/tokenizer.json
models/gpt2-large/vocab.json
```

## 7. pythia-1b (1.0B, GPU)

```sh
LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  dart run bin/pythia/run_1b_gpu_api.dart \
    --text "Once upon a time," --max-tokens 30
```

Load takes ~6 min (2 GB fp16 → decoded to fp32 in Dart), generation
~3 s/token on an RTX 3060 Laptop.

## 8. gpt-j-6b hybrid (6.05B, 4 GPU blocks + 24 CPU blocks)

Default `--gpu-layers 4`. Bump to 5 if you have any spare VRAM
(~805 MB per block).

```sh
LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  dart run bin/gptj/run_6b_hybrid_api.dart \
    --gpu-layers 4 --text "Once upon a time," --max-tokens 20
```

Requires ~28 GB free host RAM during load. Expect a modest speedup
(~1.2×) over pure CPU — see
[docs/huggingface/04-hybrid-placement.md](docs/huggingface/04-hybrid-placement.md).

## 9. gpt-j-6b (6.05B, pure CPU)

Slowest option. Use if the hybrid runner has any issue.

```sh
dart run bin/gptj/run_6b_cpu_api.dart \
    --text "Once upon a time," --max-tokens 20
```

No `LD_LIBRARY_PATH` needed — nothing touches CUDA.

## Common overrides

Any of the `*_api.dart` runners above accept:

```
--path PATH           override safetensors location
--vocab PATH          override tokenizer.json / vocab.json location
--prompt IDS          comma-separated token ids instead of --text
--max-tokens N        default 20
--temperature T       0.0 = greedy (default)
--top-k K             0 = disabled (default)
--seed S              deterministic sampling
--no-cache            disable KV-cache fast path (debug flag)
--serve --port 8080   run as HTTP server (GET /health, /info; POST /generate)
```

### Serve + curl

Original scratch snippet from this file, cleaned up:

```sh
LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  dart run bin/gpt2/medium/run_gpu_api.dart --serve --port 8080 &

curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/info
curl -s -X POST http://127.0.0.1:8080/generate \
  -H 'content-type: application/json' \
  -d '{"tokens":[464,995,318],"maxNewTokens":20,"temperature":0.8,"topK":40,"seed":42}'
```

`"tokens"` can be replaced with `"text": "The world is"` when a
`tokenizer.json` was loaded.

---

# Vision Transformer (ViT) demos

Everything under `bin/vit_*.dart` and `bin/train_face_folder.dart`.
No weights or tokenizer needed — they train from scratch on tiny
synthetic (or, for `train_face_folder.dart`, on-disk real) data.
Every demo accepts the same flag surface:

```
(no flag)          CPU (default)
--gpu              CUDA via the FFI backend
```

`train_face_folder.dart` adds `--synthetic`, `--tmp`,
`--all-classes` (see its section below).

> **Speed note** (from `memories/dart_pytorch_training.md`): on
> tiny models the GPU is *slower* than CPU because kernel-launch
> overhead dominates. Expect ~4× slowdown on a 250 k-param ViT.
> Run these demos on CPU unless you're specifically checking the
> CUDA path.

## V1. vit_demo — 3-class synthetic classification (smoke test)

16×16 grayscale patches, 3 classes (horizontal/vertical stripes,
checkerboard). Two encoder layers. Runs in a few seconds.

```sh
# CPU
dart run bin/vit_demo.dart

# GPU
LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  dart run bin/vit_demo.dart --gpu
```

## V2. vit_face_recognition_demo — synthetic triplet loss

Four synthetic "identities" (deterministic random patchified images),
triplet loss `relu(‖a-p‖² - ‖a-n‖² + margin)`. Prints the
cosine-similarity matrix over the four base identities before/after
training — diagonal should approach 1.0, off-diagonal should drop.

```sh
# CPU
dart run bin/vit_face_recognition_demo.dart

# GPU
LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  dart run bin/vit_face_recognition_demo.dart --gpu
```

## V3. vit_object_detection_demo — fixed-order detection

3 query slots over a 32×32 synthetic RGB input trained against a
fixed target (class 1 @ small box, class 2 @ mid box, background @ 0).
Loss = `crossEntropy + 0.25 · MSE(box)`. No bipartite matching.

```sh
# CPU
dart run bin/vit_object_detection_demo.dart

# GPU
LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  dart run bin/vit_object_detection_demo.dart --gpu
```

---

# DETR (DEtection TRansformer)

Three demos that build up to the full DETR training loop as
introduced by Carion et al., *End-to-End Object Detection with
Transformers* (Facebook AI, 2020). DETR reframes detection as
**set prediction**: a fixed pool of `N` learnable *object queries*
each produce a `(class, box)` prediction in parallel via a
transformer decoder over patch features, and a **bipartite
matching** loss pairs predictions to ground-truth boxes via the
Hungarian algorithm before scoring.

The three moving parts already live in the tree:

  * [lib/core/nn/vision/vit_object_detector.dart](lib/core/nn/vision/vit_object_detector.dart) — patchify → ViT encoder → `numQueries` learnable slots → `(logits, boxes)`.
  * [lib/core/utils/hungarian_algorithm.dart](lib/core/utils/hungarian_algorithm.dart) — `HungarianAlgorithm(costMatrix).getAssignment()` gives `assign[q] = gtIdx` in O(n³).
  * [test/vit_object_detector_test.dart](test/vit_object_detector_test.dart) — coverage for both.

D1 → D2 → D3 below walk the same architecture from "fixed
targets, no matching" to "variable-count GT per image inside a
mini-batch", which is the training loop DETR actually uses.

## D1 / V3. Fixed-order detection (warm-up)

Same as V3 above (kept for cross-references). 3 query slots, GT is
a **fixed triple** at every step, no Hungarian — just position-`q`
head predicts target-`q`. Loss = `CE(class) + 0.25 · MSE(box)`.
Establishes that the detector converges before we complicate the
loss with bipartite matching.

Command: see V3.

## D2 / V4. vit_hungarian_matching_demo — single-image bipartite matching

Same [ViTObjectDetector](lib/core/nn/vision/vit_object_detector.dart)
as V3, but every step draws a **variable-length** GT list
`0 .. numQueries`. At each step:

  1. Forward pass → `[logits, boxes]` for every query slot.
  2. Build the `numQueries × numQueries` cost matrix
     `-log p(class_gt) + λ · L1(box, gt_box)`, padding unmatched
     GT columns with a large constant so Hungarian never prefers
     them over a real match.
  3. `HungarianAlgorithm(cost).getAssignment()` → `assign[q] = gtIdx`.
  4. Backprop CE over all queries (unmatched → background class),
     mask the box regression to matched slots only.

```sh
# CPU
dart run bin/vit_hungarian_matching_demo.dart

# GPU
LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  dart run bin/vit_hungarian_matching_demo.dart --gpu
```

## D3 / V5. vit_hungarian_batch_demo — variable-count DETR batch

Extends D2 to a full **batch** where each image in the batch has
its own GT count `0..numQueries`. This is the realistic DETR
training setup. Runs one Hungarian assignment per image inside the
batch (they can't share since costs and GT counts differ), then
sums the per-image losses.

Uses `Tensor.abs` for the L1 box cost — its GPU backward kernel
(`abs_backward_op`) is already wired into the CUDA FFI backend.

```sh
# CPU
dart run bin/vit_hungarian_batch_demo.dart

# GPU
LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  dart run bin/vit_hungarian_batch_demo.dart --gpu
```

Together V3 → V4 → V5 (aka D1 → D2 → D3) cover the DETR loss
factors: **fixed targets** → **bipartite matching on a single
image** → **bipartite matching in a mini-batch**. Extending to
Deformable-DETR (multi-scale features, sparse attention) or DINO
would live in this same folder.

---

# ViT — real photo training

## V6. train_face_folder — real-photo face recognition

End-to-end pipeline: reads real celebrity photos, materializes an
`ImageFolder`-style layout, splits 75/25 train/val, trains a
`ViTClassifier` with cross-entropy on class ids (top-1 accuracy
reported before/after). Falls back to the cartoon 4-identity gallery
under `--synthetic` (triplet loss + cosine similarity gap).

Data source: hardcoded to `/mnt/c/Users/kkinyama/dart_cuda/Faces`
by default — override by prepping your own `faces_gallery/{name}/*.jpg`
tree and skipping the flat-dir stage.

```sh
# CPU, real celebrity photos, 8-class subset
dart run bin/train_face_folder.dart

# GPU
LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  dart run bin/train_face_folder.dart --gpu

# Cartoon synthetic gallery (triplet loss path)
dart run bin/train_face_folder.dart --synthetic

# Build the intermediate gallery in a /tmp dir that's deleted on exit
dart run bin/train_face_folder.dart --tmp

# Use every identity in the source folder instead of the 8-class subset
dart run bin/train_face_folder.dart --all-classes
```

Working defaults (from user memory): 8 classes × 16 samples ×
64×64×3, `ViTClassifier(embedDim=96, numLayers=2, numHeads=4,
patchSize=8)`, Adam `lr=1e-3`, 1500 steps → val top-1 ≈ 25 %
(2× uniform 12.5 %). `lr=3e-3` is too high — loss stays around
uniform.

---

# Document analysis — Retrieval-Augmented Q&A

End-to-end **RAG** pipeline that uses one of the HF language models
above as *both* the sentence encoder and the answer generator. See
the chapter 11 write-up in
[docs/vectors/11-LANGUAGE-MODEL-INTEGRATION.md](docs/vectors/11-LANGUAGE-MODEL-INTEGRATION.md).

## R1. rag_qa_demo (distilgpt2, CPU)

Ten hardcoded factual passages + five questions. For each question:
last-token encode → corpus-mean center → `IndexFlatIP.search(k=3)` →
stuff top-3 into prompt → `distilgpt2.generate(60 tokens)`.

```sh
dart run bin/rag_qa_demo.dart
```

## R2. rag_qa_demo (bigger model)

Point at any GPT-2-family checkpoint. `--preset` picks the matching
`GPTConfig`.

```sh
LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  dart run bin/rag_qa_demo.dart \
    --path models/gpt2-medium/model.safetensors \
    --vocab models/gpt2-medium/tokenizer.json \
    --preset medium --gpu
```

Presets: `distilgpt2` (default) | `small` | `medium` | `large`.
Encoder quality lifts visibly with model depth — distilgpt2 (6 blocks)
gets the right passage into top-3 for every demo question; medium
(24 blocks) usually to top-1.

To swap in your own corpus, edit the `_corpus` and `_questions`
constants at the top of [bin/rag_qa_demo.dart](bin/rag_qa_demo.dart).
For a production-grade encoder path, fine-tune a small
`TextTransformer` with triplet loss instead — see
[bin/vector_store_demo.dart](bin/vector_store_demo.dart).

---

# More vector-embedding applications

Same encoder recipe as the RAG demo (`distilgpt2` → last-token
hidden → corpus-mean centre → L2-normalise → `IndexFlatIP`), applied
to three more classic tasks. Shared encoder helpers live in
[bin/_lm_encoder.dart](bin/_lm_encoder.dart) so each demo file stays
focused on its own logic. All three accept the same
`--path / --vocab / --preset / --gpu` flags as `rag_qa_demo`.

## R3. Zero-shot text classification

Describe each candidate **label** as a short sentence, index the
label vectors, then classify each doc by argmax cosine similarity.
No training required — the "prompt engineering as classifier"
trick popularised by SBERT / instructor-xl.

```sh
dart run bin/vector_zero_shot_classify_demo.dart
```

Six labels (`astronomy`, `biology`, `cooking`, `finance`,
`programming`, `sports`) × twelve test docs. Prints top-3 label
scores per doc plus overall top-1 agreement with ground truth.
Editable label list at the top of
[bin/vector_zero_shot_classify_demo.dart](bin/vector_zero_shot_classify_demo.dart).

## R4. Semantic near-duplicate detection

Find paraphrase clusters in a mixed pile of documents.
`IndexFlatIP.rangeSearch(vecs, radius=cosine_threshold)` returns
every pair above the similarity cutoff; union-find groups them into
duplicate clusters.

```sh
dart run bin/vector_dedup_demo.dart
```

Seeded corpus contains three phrasings of "photosynthesis", two of
"Eiffel Tower", two of "insulin regulates blood sugar" plus three
unique passages — the demo should recover exactly those clusters.
Adjust `_threshold` in
[bin/vector_dedup_demo.dart](bin/vector_dedup_demo.dart) to trade
precision for recall.

## R5. Unsupervised topic clustering (k-means)

No labels, no queries — just run Lloyd's k-means over the corpus
embeddings. Each cluster centroid becomes a discovered topic.
Uses [lib/core/vector_store/kmeans.dart](lib/core/vector_store/kmeans.dart)
(k-means++ init + 30 Lloyd iters).

```sh
dart run bin/vector_cluster_demo.dart
```

24-passage mixed corpus (astronomy / cooking / programming /
sports) with `k=4`. Prints each cluster's members alongside their
ground-truth topic and an overall purity score (fraction assigned
to the majority topic of their cluster; chance = 25%). Editable
corpus at the top of
[bin/vector_cluster_demo.dart](bin/vector_cluster_demo.dart).

## R6. Chunked long-document RAG

R1/R2 kept each "document" to a single short paragraph. Real
articles run 5-50k tokens and won't fit a 1024-ctx window. This
demo does what production RAG actually does: chunks each long doc
into overlapping ~200-token windows, indexes chunks (not docs),
groups hits back by parent doc.

```sh
dart run bin/vector_chunked_rag_demo.dart
```

Three inline articles (Roman Empire / neural networks / Apollo
program) with topic drift within each doc; five questions target
material buried in the middle of an article — the case
doc-level embedding would miss. Editable at
[bin/vector_chunked_rag_demo.dart](bin/vector_chunked_rag_demo.dart).

## R7. Index-type benchmark

Chapters 3-8 of [docs/vectors](docs/vectors/README.md) explain
*why* ANN indexes exist. This is the one table that lets you see
the tradeoff numerically: same synthetic Gaussian vectors, same
queries, every index type. Reports build time, µs/query, recall@k
against exact `IndexFlatL2` ground truth, and rough bytes/vec.

```sh
dart run bin/vector_index_benchmark_demo.dart
```

Default `N=5000` × `d=128` × `nq=200` finishes in seconds. Tune
with `--n / --d / --nq / --k`. Uses synthetic vectors (worst case
for ANN) so the numbers are conservative; real embeddings cluster
naturally and score higher. Source:
[bin/vector_index_benchmark_demo.dart](bin/vector_index_benchmark_demo.dart).

## R8. Image-to-image similarity search

Same vector-store story off the text axis. Trains a small
`ViTClassifier` on `faces_gallery/` (~90 s CPU for 8 classes),
drops the head, uses the backbone's CLS vector as a face
embedding, `IndexFlatIP` over the gallery, queries with held-out
val-set images.

```sh
dart run bin/vector_image_search_demo.dart
```

Prints top-5 nearest gallery photos per query with cosine scores
and a "match / no match" marker, plus overall top-1 and same-
identity hit-rate vs class-uniform baseline. `--all-classes` uses
every subfolder; `--gallery PATH` points at a different
folder-per-class dataset. Source:
[bin/vector_image_search_demo.dart](bin/vector_image_search_demo.dart).

## R9. Copilot-style RAG chat server (browser UI + upload)

Local HTTP server that serves a drag-drop chat page and does
retrieval-augmented generation over the docs you upload. Reuses
the R6 chunked-RAG pipeline (200-token windows, 100-token stride,
last-token hidden state, corpus-mean centring, `IndexFlatIP` over
L2-normalised vectors, cosine = inner product). Prompts assemble
top-K retrieved chunks + recent history + user turn, then feeds
the pretrained HF GPT-2 checkpoint. State is in-memory (no auth,
single-user, no persistence).

```sh
# distilgpt2 on CPU (default)
dart run bin/rag_chat_server.dart --port 8090

# gpt2-medium on GPU
LD_LIBRARY_PATH=/usr/lib/wsl/lib dart run bin/rag_chat_server.dart \
  --path models/gpt2-medium/model.safetensors \
  --vocab models/gpt2-medium/tokenizer.json \
  --preset medium --gpu --port 8090

# Llama-3.2-1B-Instruct on GPU (same RAG pipeline, different arch)
LD_LIBRARY_PATH=/usr/lib/wsl/lib dart run bin/rag_chat_server.dart \
  --arch llama \
  --path models/llama-3.2-1b-instruct/model.safetensors \
  --vocab models/llama-3.2-1b-instruct/tokenizer.json \
  --preset llama-3.2-1b --gpu --port 8090
```

Then open `http://127.0.0.1:8090/` in a browser: drop text files
into the sidebar, chat in the main pane, expand the "sources"
panel under each reply to see which chunks were retrieved (doc
title, token span, cosine score, preview). Endpoints:

- `GET  /`          — HTML chat UI.
- `GET  /health`    — `{arch, model, device, embedDim, numLayers, maxCtx}`.
- `GET  /status`    — doc list + chunk counts.
- `POST /upload`    — `text/plain` body, `x-filename` header.
- `POST /chat`      — `{"message": "..."}` → `{reply, retrieved, ms, ...}`.
- `POST /reset`     — wipe docs, chunks and history.

`--arch gpt|llama` picks the backend (default `gpt`). Under
`--arch llama` the `--path` / `--vocab` / `--preset` defaults
switch to `models/llama-3.2-1b-instruct/…` and `llama-3.2-1b`;
presets are `llama-3.2-1b | llama-3.2-3b | llama-3.1-8b`.
Generation stops at the tokenizer's EOT (Llama-3 `<|eot_id|>` or
GPT-2 `<|endoftext|>`). See the R10 Llama section for VRAM math
before pointing this at the 3B/8B checkpoints on a 6 GB card.

Limits: 5 MB per upload, 500 chunks total, last 3 turn-pairs of
history retained, prompt overflow first drops history then
chunks. Text/markdown only (no PDF ingest). Source:
[bin/rag_chat_server.dart](bin/rag_chat_server.dart).

---

# Llama-3 chat (standalone CLI, no RAG)

## R10. llama_chat — interactive Llama-3 REPL

Companion to R9 without any of the retrieval plumbing: pure
chat. Loads a Llama-3 instruct checkpoint (default:
Llama-3.2-1B-Instruct), tokenizes each turn using the official
Llama-3 chat template built from special-token literals, calls
`Llama.generate`, and stops each reply at the first `<|eot_id|>`.
Multi-turn history is kept as a raw string and re-tokenized every
turn (so any turn can drop out cleanly on `:reset`).

```sh
# Llama-3.2-1B on CPU, defaults
dart run bin/llama_chat.dart

# Llama-3.2-1B on GPU (WSL2 needs the CUDA driver-stub prefix)
LD_LIBRARY_PATH=/usr/lib/wsl/lib dart run bin/llama_chat.dart --gpu

# Larger checkpoint (see VRAM note below before using --gpu on a 6 GB card)
dart run bin/llama_chat.dart --preset llama-3.2-3b \
  --path models/llama-3.2-3b-instruct/model.safetensors \
  --vocab models/llama-3.2-3b-instruct/tokenizer.json

# Custom system prompt, greedy decoding
dart run bin/llama_chat.dart \
  --system "You are a Dart expert." --temperature 0.0

# RAG over a folder of text/markdown notes
dart run bin/llama_chat.dart --docs notes/ --docs README.md

# RAG on GPU, chunkier windows, retrieve top-6
LD_LIBRARY_PATH=/usr/lib/wsl/lib dart run bin/llama_chat.dart --gpu \
  --docs docs/vectors/ --chunk-tokens 300 --top-k-docs 6
```

Flags:

```
--path PATH        safetensors weights (default: models/llama-3.2-1b-instruct/model.safetensors)
--vocab PATH       tokenizer.json      (default: models/llama-3.2-1b-instruct/tokenizer.json)
--preset NAME      llama-3.2-1b | llama-3.2-3b | llama-3.1-8b  (default: llama-3.2-1b)
--gpu              run on CUDA (default: CPU)
--system "..."     initial system prompt
--max-new N        max tokens per reply (default: 256)
--temperature F    0.0 = greedy (default 0.7)
--top-k K          0 = disabled (default 40)

# RAG (optional — no --docs = plain chat)
--docs PATH        (repeatable) file, or dir walked for *.txt / *.md
--chunk-tokens N   chunk size in tokens (default: 200)
--chunk-stride N   sliding stride       (default: 100 = 50% overlap)
--top-k-docs K     chunks retrieved per turn (default: 4)
```

REPL commands (inside the running process):

```
:quit              exit
:reset             wipe conversation history (system prompt kept)
:sys <text>        replace system prompt and reset
:sources           print the chunks retrieved for the last answer (RAG only)
```

**How RAG works here.** On startup, every file under `--docs` is
tokenized, split into ~200-token windows with 50 % overlap, and each
window is embedded via the same last-token-hidden trick the RAG
server uses (`lastTokenHiddenLlama`). Vectors are centred against
the corpus mean and L2-normalised, so an `IndexFlatIP` search
reports cosine similarity directly. Per user turn, the message is
embedded the same way, top-K chunks are retrieved, and they get
prepended to the user turn as an "Use the following excerpts…"
context block *inside* the Llama-3 chat template. `:sources` shows
which chunks fired for the last reply.

### VRAM note — will this fit on a 6 GB GPU?

`dart_pytorch` stores every `Tensor` as fp32 on both CPU and
GPU (`Float32List` under the hood, safetensors bf16 promoted on
load). That means weight memory is exactly **4 bytes × params**:

| preset | params | fp32 weights | 6 GB card? |
|---|---|---|---|
| `llama-3.2-1b` | ~1.24 B | ~4.96 GB | **borderline** — leaves ~500-800 MB for KV cache + activations + driver reserve |
| `llama-3.2-3b` | ~3.2 B  | ~12.8 GB | **no** — will OOM at load |
| `llama-3.1-8b` | ~8.0 B  | ~32 GB  | no |

Practical rules on a 6 GB / RTX 3060-class card:

- `llama-3.2-1b --gpu`: try it, keep prompts and `--max-new` modest.
- `llama-3.2-3b` and `llama-3.1-8b`: drop `--gpu` and run on CPU (slow but correct).

Source: [bin/llama_chat.dart](bin/llama_chat.dart).