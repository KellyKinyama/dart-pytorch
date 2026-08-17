# HuggingFace model interop

Everything in this folder documents how `dart-pytorch` loads and runs
pretrained HuggingFace checkpoints — the code lives under
[`lib/core/nn/*_hf_loader.dart`](../../lib/core/nn/), [`lib/core/data/hf_bpe_tokenizer.dart`](../../lib/core/data/hf_bpe_tokenizer.dart),
[`lib/core/nn/safetensors.dart`](../../lib/core/nn/safetensors.dart),
and the runner shims under [`bin/gpt2/`](../../bin/), [`bin/pythia/`](../../bin/pythia/),
and [`bin/gptj/`](../../bin/gptj/).

## What's supported today

All models load from a HuggingFace `model.safetensors` file (single-shard;
multi-shard is not implemented yet) and use HF's `tokenizer.json` when
present.

| Family | Model | Params | Loader | Runner | Fits 6 GB GPU? |
|---|---|---|---|---|---|
| GPT-2 | `distilgpt2` | 82M | `GPT2HFLoader.gpt2Config` | [`bin/distilgpt2/run_gpu_api.dart`](../../bin/distilgpt2/run_gpu_api.dart) | ✅ fp32 |
| GPT-2 | `gpt2` | 124M | `GPT2HFLoader.gpt2Config` | [`bin/gpt2/small/run_gpu_api.dart`](../../bin/) | ✅ fp32 |
| GPT-2 | `gpt2-medium` | 355M | `GPT2HFLoader.gpt2MediumConfig` | [`bin/gpt2/medium/run_gpu_api.dart`](../../bin/) | ✅ fp32 |
| Pythia (GPT-NeoX) | 14m / 70m | 14M / 70M | `PythiaHFLoader.pythia{14,70}mConfig` | — | ✅ |
| Pythia | 160m | 160M | `PythiaHFLoader.pythia160mConfig` | [`bin/pythia/run_gpu_api.dart`](../../bin/pythia/run_gpu_api.dart) | ✅ |
| Pythia | 410m | 410M | `PythiaHFLoader.pythia410mConfig` | [`bin/pythia/run_410m_gpu_api.dart`](../../bin/pythia/run_410m_gpu_api.dart) | ✅ |
| Pythia | 1b | 1.0B | `PythiaHFLoader.pythia1bConfig` | [`bin/pythia/run_1b_gpu_api.dart`](../../bin/pythia/run_1b_gpu_api.dart) | ✅ tight (~3.3 GB) |
| GPT-J | 6B | 6.05B | `GPTJHFLoader.gptJ6bConfig` | [`bin/gptj/run_6b_cpu_api.dart`](../../bin/gptj/run_6b_cpu_api.dart) | ❌ CPU only (24 GB fp32) |
| GPT-J | 6B (hybrid) | 6.05B | `GPTJHFLoader.gptJ6bHybridConfig` | [`bin/gptj/run_6b_hybrid_api.dart`](../../bin/gptj/run_6b_hybrid_api.dart) | 🟡 split (see [04-hybrid-placement.md](04-hybrid-placement.md)) |

## Contents of this folder

- [01-model-loaders.md](01-model-loaders.md) — architectural
  differences between GPT-2, Pythia (GPT-NeoX), and GPT-J; the HF
  key layout each loader expects.
- [02-bpe-tokenizer.md](02-bpe-tokenizer.md) — pure-Dart byte-level BPE
  encoder/decoder, compatible with both GPT-2 and GPT-NeoX
  `tokenizer.json` files.
- [03-rotary-conventions.md](03-rotary-conventions.md) — GPT-NeoX / LLaMA
  half-split RoPE vs GPT-J interleaved-pair RoPE, plus the weight
  permutation that lets one kernel serve both.
- [04-hybrid-placement.md](04-hybrid-placement.md) — placing some
  transformer blocks on GPU, others on CPU (`--gpu-layers N`).
- [05-runners.md](05-runners.md) — the shared CLI + HTTP surface used
  by every model runner.

## Quickstart

Download a checkpoint and run it. `distilgpt2` is the fastest way to
verify end-to-end plumbing:

```sh
mkdir -p models/distilgpt2
curl -L -o models/distilgpt2/model.safetensors \
  https://huggingface.co/distilgpt2/resolve/main/model.safetensors
curl -L -o models/distilgpt2/tokenizer.json \
  https://huggingface.co/distilgpt2/resolve/main/tokenizer.json

LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  dart run bin/distilgpt2/run_gpu_api.dart \
    --text "The quick brown fox" --max-tokens 12
```

`LD_LIBRARY_PATH=/usr/lib/wsl/lib` is a WSL2-specific fix so the
CUDA driver stub is found (see the top-level `README.md` for
details). Drop it on native Linux.

## What is *not* implemented

- **Multi-shard checkpoints** (`model-00001-of-000NN.safetensors` +
  `model.safetensors.index.json`). Every large model on HF currently
  is served as a single shard when you download the raw
  `.safetensors` file directly, but this cuts you off from some
  30B+ models that are shard-only.
- **fp16 / bf16 / int8 / int4 runtime.** The loader
  *reads* fp16 storage (that's how Pythia-1b works — HF ships fp16,
  we decode on the fly to fp32), but every matmul is fp32. This is
  why GPT-J-6B needs 24 GB.
- **Batching** — the tensor stack is single-sequence 2D `[N, D]`.
  All KV caches, all attention paths, all HTTP `POST /generate`
  requests operate on batch size 1.
- **Streaming HTTP responses** (SSE) — `/generate` currently blocks
  until the full continuation is ready.
- **Top-p / nucleus sampling, repetition penalty, stop tokens.**
  Only top-k + temperature are supported.
