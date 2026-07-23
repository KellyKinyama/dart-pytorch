# 05 — Runners: CLI + HTTP API

Every HuggingFace model has a thin runner "shim" in `bin/<family>/`
that composes:

- CLI flag parsing
- Model construction + weight load
- Either one-shot generation *or* a long-lived HTTP server
- BPE tokenizer discovery (auto-loads `tokenizer.json` next to the
  weights)

The shim itself is ~30 lines. All the machinery lives in three
shared helpers:

- [`bin/_gpt2_hf_api_common.dart`](../../bin/_gpt2_hf_api_common.dart)
  → `runGpt2Api(...)`  — for the GPT-2 family.
- [`bin/_pythia_hf_api_common.dart`](../../bin/_pythia_hf_api_common.dart)
  → `runPythiaApi(...)` — for Pythia.
- [`bin/_gptj_hf_api_common.dart`](../../bin/_gptj_hf_api_common.dart)
  → `runGPTJApi(...)`  — for GPT-J.

They diverge only where the underlying model API forces them to
(config factories have different fields; GPT-J's factory takes an
extra `gpuLayers` parameter for [hybrid placement](04-hybrid-placement.md)).

## Shared CLI flags

```
--path PATH           safetensors file to load
--vocab PATH          HF tokenizer.json (defaults to <weights-dir>/tokenizer.json)
--prompt IDS          comma-separated token ids
--text  STR           prompt as text (BPE-encoded; needs tokenizer.json)
--max-tokens N        new tokens to generate (default 20)
--temperature T       0.0 = greedy sampling (default 0.0)
--top-k K             0 = disabled (default 0)
--seed S              RNG seed for reproducible sampling
--no-cache            disable KV-cache fast path (slower, mostly a debug flag)
--cpu / --gpu         where to build the model (default varies by family)
--serve               run HTTP server instead of one-shot
--host H              HTTP bind host (default 127.0.0.1)
--port P              HTTP port    (default 8080)
-h, --help
```

GPT-J adds one:

```
--gpu-layers N        hybrid: first N transformer blocks on GPU, rest on CPU
```

## One-shot mode

```sh
LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  dart run bin/pythia/run_1b_gpu_api.dart \
    --text "Once upon a time," --max-tokens 30
```

Prints:

```
Building Pythia (pythia-1b, gpu, embed=2048, layers=16, heads=8)...
Loading safetensors from models/pythia-1b/model.safetensors ...
Loaded in 383272 ms. PythiaLoadReport(consumed=196, unused=48)
Loading BPE tokenizer from models/pythia-1b/tokenizer.json

Generating: prompt=[15752, 3402, 247, 673, 13] max=30 temp=0.0 topK=0 seed=none cache=true
  ids : [15752, 3402, ...]
  text: "Once upon a time, in a distant land, a young man named John Smith..."
  86148 ms for 30 tokens
```

## Server mode

```sh
LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  dart run bin/gpt2/medium/run_gpu_api.dart --serve --port 8080
```

Three endpoints:

### `GET /health`

Cheap liveness probe.

```json
{
  "status": "ok",
  "model": "gpt2-medium",
  "device": "gpu",
  "weights": "models/gpt2-medium/model.safetensors"
}
```

### `GET /info`

Configuration snapshot.

```json
{
  "model": "gpt2-medium",
  "device": "gpu",
  "embedDim": 1024,
  "numLayers": 24,
  "numHeads": 16,
  "vocabSize": 50257,
  "maxCtx": 1024
}
```

The Pythia and GPT-J endpoints also report `rotaryPct` / `rotaryDim`
and `ropeBase`. GPT-J's `/info` additionally reports per-layer
placement:

```json
{
  ...
  "placement": {
    "embedding": "Device.CPU",
    "lmHead":    "Device.CPU",
    "layers":    ["Device.GPU", "Device.GPU", "Device.GPU", "Device.GPU", "Device.GPU",
                  "Device.CPU", "Device.CPU", "Device.CPU", ... ]
  }
}
```

### `POST /generate`

Body accepts either raw ids or text (mutually exclusive; text
requires tokenizer):

```json
{
  "text": "The quick brown fox",           // OR "tokens": [464, 2068, 7586, 21831]
  "maxNewTokens": 20,
  "temperature": 0.8,
  "topK": 40,
  "seed": 42,
  "useCache": true
}
```

Response:

```json
{
  "tokens":     [464, 2068, 7586, 21831, ...],
  "newTokens":  [21831, ...],
  "text":       "The quick brown fox jumps out of the bushes...",
  "elapsedMs":  2318
}
```

Errors return HTTP 400 with `{"error": "..."}`. Handler exceptions
return HTTP 500 with the exception message.

### Example: curl

```sh
curl -s -X POST http://127.0.0.1:8080/generate \
  -H 'content-type: application/json' \
  -d '{"text":"The world is","maxNewTokens":20,"temperature":0.8,"topK":40,"seed":42}'
```

## Runner shims

Each concrete runner is ~30 lines of glue. Example
([`bin/pythia/run_1b_gpu_api.dart`](../../bin/pythia/run_1b_gpu_api.dart)):

```dart
Future<void> main(List<String> args) async {
  await runPythiaApi(
    modelName:   'pythia-1b',
    defaultPath: 'models/pythia-1b/model.safetensors',
    configFactory: ({required Device device}) =>
        PythiaHFLoader.pythia1bConfig(device: device),
    args: args,
  );
  exit(0);
}
```

The GPT-J hybrid shim
([`bin/gptj/run_6b_hybrid_api.dart`](../../bin/gptj/run_6b_hybrid_api.dart))
is nearly identical — it just consults the extra `gpuLayers`
parameter:

```dart
Future<void> main(List<String> args) async {
  await runGPTJApi(
    modelName:   'gpt-j-6b-hybrid',
    defaultPath: 'models/gpt-j-6b/model.safetensors',
    configFactory: ({required Device device, int? gpuLayers}) {
      final n = gpuLayers ?? 4;
      return GPTJHFLoader.gptJ6bHybridConfig(gpuLayers: n);
    },
    args: args,
  );
  exit(0);
}
```

## Adding a new runner

1. Write (or find) an HF loader that returns a `Config` + `Model`
   pair and consumes a safetensors state dict — see
   [01-model-loaders.md](01-model-loaders.md).
2. If your family already has an `_*_hf_api_common.dart`, add a new
   preset (e.g. `pythia1_4bConfig`) and a matching shim in
   `bin/<family>/`. Done.
3. If it's a new architecture family (e.g. LLaMA), copy one of the
   `_hf_api_common.dart` files and adapt: change the config type,
   swap the loader class, update the `/info` fields.

## Measured performance

Small table of real numbers from an RTX 3060 Laptop 6 GB under WSL2:

| Model | Device | Load | Gen (tokens/s greedy) |
|---|---|---|---|
| distilgpt2 (82M) | GPU | 22 s | ~5.0 |
| gpt2 (124M) | GPU | 25 s | ~4.0 |
| gpt2-medium (355M) | GPU | 114 s | ~1.2 |
| pythia-160m | GPU | 35 s | ~2.0 |
| pythia-410m | GPU | 412 s | ~1.3 |
| pythia-1b | GPU | 383 s | ~0.35 |
| gpt-j-6b | CPU | (not tested yet) | (expected ≪0.1) |

Load times are dominated by CPU-side `Tensor.fromList` copies —
optimising the load path is on the todo list.
