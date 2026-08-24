# Layer-streaming inference (AirLLM-style)

Fits transformer LMs many times bigger than available RAM by keeping
only one block resident on the compute device and pulling every
other layer's weights off disk on demand. Same trick as
[lyogavin/airllm](https://github.com/lyogavin/airllm), implemented
on top of this repo's existing safetensors reader.

## Why

- A 6 GB GPU can inspect completions from Llama-3.1-8B (~16 GB fp16)
  or GPT-J-6B (~12 GB fp16) without OOM.
- A 16 GB laptop can run any of the small-to-medium HF LMs without
  worrying about "does the merged state dict + model init fit?".

## What it *isn't*

- Fast. Per-token latency ≈ `numLayers × (layer-bytes / disk-bandwidth)`.
  On a WSL laptop that's seconds per token for anything bigger than
  ~200 M params. Use it to verify the model runs on modest hardware,
  not for interactive chat.
- GPU-backed. `Tensor.adoptCpuStorageFrom` is CPU-only; a GPU
  streaming variant would need a per-layer H2D upload path.
- KV-cached. This first cut re-runs the full prefix through all
  layers for every generated token. Adding a per-layer KV cache
  (kept resident across layer swaps) is the obvious next step.

## Architecture

```
persistent (loaded once):     resident (swapped per layer):
  ┌──────────────┐              ┌────────────┐
  │ embed_tokens │              │ 1 × Block  │  ← weights adopted
  │ final norm   │              │ (LN,       │    from disk per
  │ lm_head?     │              │  MHA,      │    forward step
  │ RoPE cache   │              │  FFN)      │
  └──────────────┘              └────────────┘
        ▲                             ▲
        │        ShardedSafeTensorsReader.readTensor(name)
        │        ────────────────────────────────────────►
        │                                                  ▼
        └──── model.safetensors[.index.json] on disk ──────┘
```

For each `forward(tokens)`:

1. Embed tokens with the persistent `embed_tokens`.
2. For `i = 0..L-1`:
   - `_swapLayer(i)` — for every weight in layer `i`, seek into the
     safetensors file at that tensor's byte range, read only those
     bytes, decode (fp16 fast-path preserved), and
     `adoptCpuStorageFrom` it into the resident block's slot. Old
     buffer is GC'd.
   - Run the resident block once.
3. Final norm + `lm_head`.

Peak resident RAM ≈ `embed_bytes + lm_head_bytes + ~2×layer_bytes +
activations`. For Llama-3.1-8B fp16 that's ~1.05 GB + 1.05 GB + 872 MB
+ activations ≈ 3.3 GB. Embed and lm_head are loaded **directly** as
fp16 tensors from disk — never allocated as fp32 first — which is
what makes the peak match the fp16 checkpoint layout rather than
double it.

The [bin/llama_streaming_random_demo.dart](../bin/llama_streaming_random_demo.dart)
demo computes this estimate at startup, reads `MemAvailable` from
`/proc/meminfo`, and refuses to run if the prediction exceeds free
RAM. On WSL, grow the RAM limit via `%USERPROFILE%\.wslconfig`:

```ini
[wsl2]
memory=16GB
```

## Building blocks

- **`SafeTensorsReader`** in [lib/core/nn/safetensors_reader.dart](../lib/core/nn/safetensors_reader.dart)
  — opens one safetensors file, parses the header once, keeps the
  file handle open, and does per-tensor seek + read + decode via
  `SafeTensors.decodeBlob`.
- **`ShardedSafeTensorsReader`** — same API but transparently spans
  a multi-shard HF checkpoint (`model.safetensors.index.json`).
  Auto-detect: `ShardedSafeTensorsReader.open(path)` picks
  `fromIndex` for `.index.json`, `singleFile` otherwise.
- **`LlamaStreamingRunner`** in [lib/core/nn/llama_streaming.dart](../lib/core/nn/llama_streaming.dart)
  — Llama arch (RMSNorm + GQA + SwiGLU + RoPE + optional Q/K/V bias
  for Qwen2/2.5). Works for every preset in
  `LlamaHFLoader` (SmolLM2, Llama-3.x, Qwen2.5, DeepSeek-R1-Distill,
  DeepSeek-Coder, Qwen2.5-Coder, Qwen2.5-Math).
- **`GPTJStreamingRunner`** in [lib/core/nn/gptj_streaming.dart](../lib/core/nn/gptj_streaming.dart)
  — GPT-J arch (shared LN, no-bias QKV, GELU-tanh FFN with bias, biased
  lm_head, interleaved→half-split rotary permutation per head).

## Demos

Small (verified end-to-end on WSL):

```bash
dart run bin/llama_streaming_demo.dart \
  --weights models/smollm2-135m/model.safetensors \
  --tokenizer models/smollm2-135m/tokenizer.json \
  --preset smollm2-135m \
  --prompt "The capital of France is" \
  --max-new 10
```

Sample output (30 layers × ~200 ms swap on WSL SSD, 0.1 tok/s):

```
== completion ==
The capital of France is Paris. Paris is the largest city in France and
```

Big (compiled + wired up, not smoke-tested here — needs the fp16
checkpoint downloaded and enough disk space):

```bash
dart run bin/gptj_streaming_demo.dart \
  --weights models/gpt-j-6b/model.safetensors \
  --tokenizer models/gpt-j-6b/tokenizer.json \
  --prompt "Once upon a time," --max-new 5
```

**MoE end-to-end** — Qwen1.5-MoE-A2.7B-Chat, 14 B total / 2.7 B
active, ~1.5 GB resident while the checkpoint is 28.6 GB on disk.
Per token, only the top-K=4 of E=60 experts get streamed per layer
(≈93 % of per-layer expert bytes skipped):

```bash
# one-time download to WSL ext4 (NOT /mnt/c)
hf download Qwen/Qwen1.5-MoE-A2.7B-Chat \
  --local-dir ~/models/qwen1.5-moe-a2.7b-chat

# then run
dart run bin/qwen15_moe_streaming_demo.dart \
  --index     ~/models/qwen1.5-moe-a2.7b-chat/model.safetensors.index.json \
  --tokenizer ~/models/qwen1.5-moe-a2.7b-chat/tokenizer.json \
  --prompt    "The capital of France is" --max-new 10
```

Random-weight smoke test — no download needed, exercises the full
pipeline (layer swap + MHA + routing + per-expert stream + KV
cache) end-to-end in ~30 ms on a tiny preset:

```bash
dart run bin/qwen15_moe_streaming_random_demo.dart
# or the real A2.7B shape (28.6 GB temp write, several minutes)
dart run bin/qwen15_moe_streaming_random_demo.dart --preset full --keep
```

MoE per-expert streaming validator — measures byte savings at any
`(D, E, H, K)` without needing a real checkpoint:

```bash
# 90.6 % savings at DeepSeek-V2-Lite scale
dart run bin/moe_streaming_demo.dart --d 2048 --experts 64 --hidden 1408 --topk 6 --tokens 1
# 93.3 % savings at Qwen1.5-MoE scale
dart run bin/moe_streaming_demo.dart --d 2048 --experts 60 --hidden 1408 --topk 4 --tokens 1
```

## What's not implemented

- **Prefetching** — overlap layer *i+1* read with layer *i* compute
  (AirLLM's ~10% speedup). Would use a background `Future` or
  isolate.
- **KV cache** — done. `LlamaStreamingRunner.generate` and
  `GPTJStreamingRunner.generate` keep a persistent `EncoderCache`
  across layer swaps. Only visible speedup on big models where
  matmul (not disk I/O) dominates.
- **MoE per-expert streaming** — done. Library:
  [lib/core/nn/moe_streaming.dart](../lib/core/nn/moe_streaming.dart)
  (`MoeStreamingLayer.route`, `.loadRoutedExpert`, `MoeStreamingReport`).
  Runs the router for a batch, computes the union of top-K experts
  the tokens actually pick, and streams only those experts' weights
  from disk.
  * Synthetic validator:
    [bin/moe_streaming_demo.dart](../bin/moe_streaming_demo.dart) —
    75 % savings at E=16 K=4, **90.6 % at DeepSeek-V2-Lite scale
    (E=64 K=6)**, **93.3 % at Qwen1.5-MoE scale (E=60 K=4)**.
  * Real-weights runners (per-layer only, no MLA / full-model
    orchestration yet):
    * [bin/deepseek_v2_moe_layer_stream.dart](../bin/deepseek_v2_moe_layer_stream.dart)
      — loads one MoE layer of `deepseek-ai/DeepSeek-V2-Lite-Chat`
      and runs SwiGLU forward on real HF weight layout.
    * [bin/qwen15_moe_layer_stream.dart](../bin/qwen15_moe_layer_stream.dart)
      — same for `Qwen/Qwen1.5-MoE-A2.7B-Chat` (scalar-gated
      `shared_expert`). **Verified on real weights (layer 3):**
      T=1 → 4/60 experts streamed (**93.3 % saved**),
      T=4 → 16/60 (73.3 %), T=8 → 25/60 (58.3 %). Union grows with
      batch size, so the AirLLM headline savings apply hardest to
      single-token autoregressive generation.
  * Not yet: a full `DeepSeekV2StreamingRunner`. Qwen1.5-MoE E2E
    runner IS landed as [Qwen15MoEStreamingRunner](../lib/core/nn/qwen15_moe_streaming.dart)
    — plain MHA (Q/K/V bias, no O bias), RMSNorm × 2, router
    swap-in, scalar-gated `shared_expert`, top-K expert streaming
    per layer, KV cache across layer swaps. Random-weight validator
    [bin/qwen15_moe_streaming_random_demo.dart](../bin/qwen15_moe_streaming_random_demo.dart)
    runs a tiny preset E2E in ~50 ms (2 layers, D=128, E=8, K=2).
    Real-weight demo [bin/qwen15_moe_streaming_demo.dart](../bin/qwen15_moe_streaming_demo.dart)
    is ready — needs the 28.6 GB HF checkpoint on disk.
- **Block-wise int4/int8 compression** — AirLLM's on-disk quantized
  weight format. Would need a new safetensors dtype path plus a
  dequant-on-load step.
- **GPU streaming** — Llama-3.1-8B on a 6 GB GPU is the natural
  target; needs an H2D upload path for per-layer swap.
