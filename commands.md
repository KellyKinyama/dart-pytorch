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