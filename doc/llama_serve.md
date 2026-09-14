# `llama_serve` — HTTP server for Llama-3 (the instruction-tuned brain behind R15)

An HTTP wrapper around [`bin/llama_serve.dart`](../bin/llama_serve.dart) that
speaks the exact same `POST /generate` wire protocol as the GPT-2
`bin/*_api.dart` runners, but hosts Meta's post-trained
`llama-3.2-1b-instruct` (or any preset supported by
`LlamaHFLoader`). Drop it in behind [`db_rag_chat_server`](../bin/db_rag_chat_server.dart)
and the browser chat UI stops guessing and starts answering.

> Reference implementation: [bin/llama_serve.dart](../bin/llama_serve.dart)
> Command reference: [commands.md § R16](../commands.md)
> Consumed by: [bin/db_rag_chat_server.dart](../bin/db_rag_chat_server.dart)
> Consumed by: [bin/db_rag_http_demo.dart](../bin/db_rag_http_demo.dart)

## Why it exists

The R14 / R15 RAG pipeline is only as good as the LLM in shell A.
The pre-existing `bin/*_api.dart` runners all host **base LMs**:

- distilgpt2 (82 M) — text continuation, no chat behaviour
- gpt2-medium (355 M) — same
- pythia-1b (1 B) — same

Base LMs don't refuse. They continue whatever pattern is in the prompt,
which is why they keep pattern-matching the top retrieved passage
even when the user asked something orthogonal. That's fundamentally a
model-quality problem, not a prompting problem.

`llama_serve` hosts `llama-3.2-1b-instruct` — 1.2 B params, RLHF-tuned
by Meta — behind the same HTTP protocol so no other component has to
change. Result: the chat UI now honours "answer using only the
context; otherwise say you don't know" instead of hallucinating.

## Architecture — why the two "embeddings" never collide

A common (fair) question when swapping LLMs into a RAG stack is:
*doesn't the retriever's embedder disagree with the LLM's internal
embeddings?* The answer is that they live in **completely separate
vector spaces** and **never touch each other**. Only text crosses
between them.

```
┌─ shell B: db_rag_chat_server.dart ──────────────────────────────┐
│                                                                 │
│  user question ──► MiniLM.embed() ──► [0.11, -0.03, ...]       │
│                                       │                         │
│                                       │  (384-d, lives IN the   │
│                                       │   vector store; never   │
│                                       │   leaves this shell)    │
│                                       ▼                         │
│                       vec_hybrid_search  ──►  top-k chunk TEXT  │
│                                                     │           │
│                       ┌─────────────────────────────┘           │
│                       ▼                                         │
│              build prompt STRING out of that text               │
│                       │                                         │
└───────────────────────┼─────────────────────────────────────────┘
                        │
              HTTP { "text": "Answer using only the context below.\n..." }
                        │      ↑
                        │      this is a STRING of English characters —
                        │      no vectors go over the wire
                        ▼
┌─ shell A: llama_serve.dart ─────────────────────────────────────┐
│                                                                 │
│  incoming text ──► Llama.tokenize ──► token IDs [128000, 271...]│
│                                       │                         │
│                                       ▼                         │
│                    Llama's internal 2048-d embeddings           │
│                    (only Llama's forward pass sees these)       │
│                                       │                         │
│                                       ▼                         │
│                    forward pass → generated tokens              │
│                                       │                         │
│                                       ▼                         │
│                    Llama.decode ──► reply TEXT                  │
└───────────────────────┬─────────────────────────────────────────┘
                        │
              HTTP { "text": "clientPrompt + assistant reply" }
                        │
                        ▼
                back to shell B
```

That's the moat: **you can swap either half without touching the other.**

| Change you want | What you change | What stays the same |
| --- | --- | --- |
| Better retrieval | MiniLM → BGE-small (E1) + re-ingest | Llama, chat server, HTTP wire |
| Better generation | distilgpt2 → `llama_serve` (this doc) | MiniLM, corpus vectors, chat server |
| Different corpus | Re-run ingest | Both models |
| Multi-tenant | Add `tenant` filter col in DDL | Both models |

The one gotcha: the DDL declares `dim=384`, so if you swap embedders
you need to match the new output dimension (or point `--db-file` at a
fresh path).

## Startup

Prerequisites:

```sh
# Weights (2.4 GB — one-time)
mkdir -p models/llama-3.2-1b-instruct && cd models/llama-3.2-1b-instruct
for f in model.safetensors tokenizer.json config.json generation_config.json; do
  curl -L --progress-bar -o "$f" \
    "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/resolve/main/$f"
done
cd -
```

Serve:

```sh
# GPU — recommended, fits on 6 GB VRAM
LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  dart run bin/llama_serve.dart --gpu --port 8080

# CPU — slower, always works
dart run bin/llama_serve.dart --port 8080
```

On startup you'll see:

```
Building Llama (preset=llama-3.2-1b, device=gpu, embed=2048, layers=16, heads=32, kv=8)
Loading safetensors from models/llama-3.2-1b-instruct/model.safetensors ...
Loaded. LlamaLoadReport(consumed=147, unused=0)
Loading tokenizer from models/llama-3.2-1b-instruct/tokenizer.json
llama_serve: listening on http://127.0.0.1:8080
  GET  /health   |  GET  /info   |  POST /generate  (chat_template=on)
```

## Flags

```
Model
  --path PATH        weights (default: models/llama-3.2-1b-instruct/model.safetensors)
  --vocab PATH       tokenizer.json (default: same directory as --path)
  --preset NAME      llama-3.2-1b | llama-3.2-3b | llama-3.1-8b (default: llama-3.2-1b)
  --gpu              CUDA (default: CPU)

Server
  --host H           bind host (default 127.0.0.1)
  --port P           bind port (default 8080)
  --system "..."     default system prompt injected by the chat template
  --raw              disable chat-template wrapping (send `text` verbatim)

Sampling defaults
  --max-new N        (default 128)
  --temperature F    (default 0.7)
  --top-k K          (default 40; 0 = disabled)
  --seed S           deterministic sampling (optional)

One-shot smoke test
  --text "..."       generate one reply, print, exit (bypasses HTTP)
```

## Wire protocol

Same three endpoints as `bin/_gpt2_hf_api_common.dart` so it's a
drop-in replacement in every caller in the repo.

### `GET /health`

```json
{
  "status": "ok",
  "model": "llama-3.2-1b",
  "device": "gpu",
  "weights": "models/llama-3.2-1b-instruct/model.safetensors",
  "chat_template": true
}
```

Fail-fast pings from the chat server target this endpoint.

### `GET /info`

```json
{
  "model": "llama-3.2-1b",
  "device": "gpu",
  "embedDim": 2048,
  "numLayers": 16,
  "numHeads": 32,
  "numKvHeads": 8,
  "vocabSize": 128256,
  "maxCtx": 131072,
  "eot_id": 128009
}
```

`embedDim=2048` is Llama's internal hidden size — unrelated to
MiniLM's 384-d retrieval embeddings. See the architecture diagram
above.

### `POST /generate`

Two accepted request shapes. Pick either.

**Text form** — what [`db_rag_chat_server`](../bin/db_rag_chat_server.dart)
sends. The server wraps the text in a Llama chat-template user turn
automatically:

```sh
curl -sS -X POST http://127.0.0.1:8080/generate \
  -H 'content-type: application/json' \
  -d '{
    "text": "Explain BM25 in one sentence.",
    "maxNewTokens": 80,
    "temperature": 0.7,
    "topK": 40,
    "seed": 42
  }'
```

**Structured form** — hand over role-annotated turns; the server
applies the chat template as-is:

```sh
curl -sS -X POST http://127.0.0.1:8080/generate \
  -H 'content-type: application/json' \
  -d '{
    "messages": [
      {"role": "system",    "content": "You are a concise DB expert."},
      {"role": "user",      "content": "When would I use HNSW over IVFPQ?"},
      {"role": "assistant", "content": "HNSW is great below ~1M rows because ..."},
      {"role": "user",      "content": "And above that?"}
    ],
    "maxNewTokens": 128
  }'
```

Response:

```json
{
  "model": "llama-3.2-1b",
  "text": "<clientPrompt>+<assistant reply>",
  "newTokens": [40, 1418, 656, 358, ...],
  "elapsedMs": 812,
  "promptTokens": 189,
  "newTokensCount": 42
}
```

Notes:

- For text callers, `text` is `clientPrompt + assistantReply` so the
  `startsWith(prompt)` stripping trick in
  [`db_rag_chat_server`](../bin/db_rag_chat_server.dart) works unchanged.
- For `messages` callers, `text` is just the assistant reply (no
  wrapping to strip).
- `newTokens` is the raw generated ids from `Llama.generate`, already
  truncated at the first `<|eot_id|>`.

## Chat-template wrapping

Default ON — this is what turns Llama-instruct into an instruction
follower instead of a text continuer.

Every text prompt gets wrapped like:

```
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>

<system prompt>
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

<client text>
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

```

Generation stops at the first `<|eot_id|>` in the assistant reply so
the model can't drift into a synthetic follow-up user turn.

Pass `--raw` if your caller already emits Llama chat markers inside
`text` and you don't want a second layer of wrapping.

## Sampling defaults

Tuned for Llama-3.2's post-training:

| Flag | Default | Notes |
| --- | --- | --- |
| `--temperature` | `0.7` | Meta's own recommendation for the instruct models |
| `--top-k` | `40` | A good match for `temperature=0.7`; `0` disables |
| `--max-new` | `128` | Room for a paragraph; raise for long-form |

For deterministic output pin the RNG:

```sh
curl -sS -X POST http://127.0.0.1:8080/generate \
  -H 'content-type: application/json' \
  -d '{"text":"Say hi","seed":42,"temperature":0.0}'
```

## Sizing and memory

`llama-3.2-1b-instruct` at fp32 needs ~4 GB on the GPU. With the
CUDA driver + activation + KV cache overhead this fits comfortably
inside 6 GB VRAM.

Contexts up to ~4 K are cheap. The model's declared `maxCtx` is
128 K, but you'll hit memory or latency limits well before that on
consumer hardware. The server refuses jobs where
`prompt_tokens + maxNewTokens > maxCtx` with a 400 error.

## Wiring it into R15

The chat server is unchanged. Only shell A moves:

```sh
# shell A — was distilgpt2, now llama-3.2-1b-instruct
LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  dart run bin/llama_serve.dart --gpu --port 8080

# shell B — identical to before
dart run bin/db_rag_chat_server.dart \
    --llm http://127.0.0.1:8080 --port 8090 \
    --corpus data/support_faq.txt

# open http://127.0.0.1:8090/ in a browser
```

Sanity check while both are up:

```sh
curl -sS http://127.0.0.1:8080/health   # LLM
curl -sS http://127.0.0.1:8090/health   # chat server (must show llm points at 8080)
```

## Troubleshooting

**Exit code 66: weights not found.**
Check `--path` / `--vocab`. Defaults assume
`models/llama-3.2-1b-instruct/`. Download commands above.

**"CUDA out of memory" during load.**
You're on shared VRAM with something else (browser, Windows compositor).
Reboot the WSL guest (`wsl --shutdown`) and reload. Or fall back to
`--preset llama-3.2-1b` on CPU (no `--gpu`) — same wire protocol, ~15×
slower.

**Chat replies are still nonsense.**
Two most likely causes:

1. You're not actually hitting `llama_serve`. `curl http://127.0.0.1:8090/health`
   — the `llm` field should show the URL you configured. If the chat
   server is still pointing at a distilgpt2 instance, the model change
   didn't take effect.
2. Retrieval returned nothing relevant. Expand the `retrieved N passage(s), M used`
   section under a chat reply — if `M = 0` the chat server intentionally
   sent an empty context and Llama-instruct is answering blind. Either
   upload a doc that covers your question or lower the relevance floor
   in [`_relevanceDistanceCutoff`](../bin/db_rag_chat_server.dart).

**LLM stops mid-sentence.**
`maxNewTokens` too small. Bump via `--max-new` on the server (raises
the default) or per-request via `{"maxNewTokens": 256}`.

**Model repeats itself.**
Temperature too low with `topK=0`. Try `--temperature 0.7 --top-k 40`
(the defaults) or add explicit stop cues in the system prompt.

**Client keeps seeing "chat_template":false in `/health`.**
You started the server with `--raw`. Restart without it.

## Related

- [doc/db_vector_rag.md](db_vector_rag.md) — the retrieval half of the stack.
- [PITCH.md](../PITCH.md) — 60-second showcase of the whole pairing.
- [commands.md § R14–R16](../commands.md) — full command reference for the RAG stack.
- [bin/llama_chat.dart](../bin/llama_chat.dart) — the interactive REPL variant (no HTTP, no chat server) that shares the same Llama loading code path.
