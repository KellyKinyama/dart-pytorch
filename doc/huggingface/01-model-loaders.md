# 01 — Model loaders

Three loader classes, one per architecture family:

- [`GPT2HFLoader`](../../lib/core/nn/gpt2_hf_loader.dart) — GPT-2
  and derivatives (`distilgpt2`, `gpt2`, `gpt2-medium`).
- [`PythiaHFLoader`](../../lib/core/nn/pythia_hf_loader.dart) —
  EleutherAI Pythia / GPT-NeoX (14m through 1b).
- [`GPTJHFLoader`](../../lib/core/nn/gptj_hf_loader.dart) —
  EleutherAI GPT-J-6B (also documents [the interleaved-rotary
  trick](03-rotary-conventions.md)).

All three consume a `Map<String, Tensor>` produced by
[`SafeTensors.loadFile()`](../../lib/core/nn/safetensors.dart) and
report which HF keys they consumed vs. left over. Unused keys are
almost always benign (e.g. rotary `inv_freq` tables we recompute, or
causal-mask buffers we build on the fly).

## Architectural cheat sheet

The three families share the pre-LN transformer skeleton but differ
in five places worth memorising when reading the loaders:

| Feature | GPT-2 | Pythia (GPT-NeoX) | GPT-J |
|---|---|---|---|
| Position encoding | learned `[maxCtx, D]` table | **RoPE** (25% of head dim) | **RoPE** (first 64 dims per head, **interleaved-pair**) |
| Block topology | sequential: `x + attn(ln1(x))`, then `+ mlp(ln2(...))` | **parallel**, two LNs: `x + attn(ln1(x)) + mlp(ln2(x))` | **parallel**, **one shared LN**: `x + attn(ln(x)) + mlp(ln(x))` |
| Q/K/V projection | fused `c_attn` `[3D, D]`, GPT-2 layout | fused `query_key_value` `[3D, D]`, per-head layout | **separate** `q_proj` / `k_proj` / `v_proj` `[D, D]` |
| Attention bias | yes (Q/K/V + out) | yes | **no** (Q/K/V/out have no bias) |
| Output head | **tied** to `wte` | **untied** `embed_out` (no bias) | **untied** `lm_head` **with bias** |
| Activation | `gelu_new` (tanh) | `gelu_new` (tanh) | `gelu_new` (tanh) |
| Vocab | 50257 | 50304 (padded) | 50400 (padded) |
| Tokenizer | GPT-2 BPE | GPT-NeoX BPE (same byte-level algorithm) | GPT-2 BPE |

Everything else — LayerNorm epsilon, Q/K scaling by `1/sqrt(head_dim)`,
softmax + dropout in attention, GELU-tanh MLP — is identical.

## HF key layouts

### GPT-2 family

```
wte.weight                                  [V, D]      (embedding, tied to output)
wpe.weight                                  [maxCtx, D] (learned positional table)
h.{i}.ln_1.{weight,bias}                    [D]
h.{i}.attn.c_attn.{weight,bias}             [D, 3D] / [3D]      # note transposed layout
h.{i}.attn.c_proj.{weight,bias}             [D, D]   / [D]      # also transposed
h.{i}.ln_2.{weight,bias}                    [D]
h.{i}.mlp.c_fc.{weight,bias}                [D, 4D]  / [4D]
h.{i}.mlp.c_proj.{weight,bias}              [4D, D]  / [D]
ln_f.{weight,bias}                          [D]
```

GPT-2 uses `Conv1D` internally, which stores weight as `[in, out]` —
we transpose on load. `distilgpt2` prefixes every key with
`transformer.` — the loader detects this pattern and strips.

### Pythia (GPT-NeoX)

```
gpt_neox.embed_in.weight                                        [V, D]
gpt_neox.layers.{i}.input_layernorm.{weight,bias}               [D]
gpt_neox.layers.{i}.post_attention_layernorm.{weight,bias}      [D]
gpt_neox.layers.{i}.attention.query_key_value.{weight,bias}     [3D, D] / [3D]
gpt_neox.layers.{i}.attention.dense.{weight,bias}               [D, D]
gpt_neox.layers.{i}.mlp.dense_h_to_4h.{weight,bias}             [4D, D] / [4D]
gpt_neox.layers.{i}.mlp.dense_4h_to_h.{weight,bias}             [D, 4D] / [D]
gpt_neox.final_layer_norm.{weight,bias}                         [D]
embed_out.weight                                                [V, D]
```

The fused `query_key_value` weight is grouped **per head first**:
`[num_heads, 3, head_dim, D]` in row-major order — *not*
`[3, num_heads, head_dim, D]` like GPT-2's layout. The loader
unpacks per-head Q/K/V slices manually.

Rotary applies to only the first `rotary_pct * head_dim` dims per head
(0.25 for every Pythia model — so `rotaryDim = head_dim / 4`). The
[`RopeCache`](../../lib/core/nn/rotary.dart) handles this via a
pass-through tail on cos/sin tables.

Pythia checkpoints ship two extra buffers we ignore: a U8
`attention.bias` causal-mask table (we build our own from
[`causalMask()`](../../lib/core/nn/masks.dart)) and a scalar
`attention.masked_bias`. The [SafeTensors reader](../../lib/core/nn/safetensors.dart)
was extended to handle U8/BOOL/I8 and empty-shape (scalar) tensors
specifically so these keys don't blow up the loader.

### GPT-J

```
transformer.wte.weight                                          [V, D]
transformer.h.{i}.ln_1.{weight,bias}                            [D]        # single shared LN
transformer.h.{i}.attn.q_proj.weight                            [D, D]     # no bias
transformer.h.{i}.attn.k_proj.weight                            [D, D]
transformer.h.{i}.attn.v_proj.weight                            [D, D]
transformer.h.{i}.attn.out_proj.weight                          [D, D]
transformer.h.{i}.mlp.fc_in.{weight,bias}                       [4D, D] / [4D]
transformer.h.{i}.mlp.fc_out.{weight,bias}                      [D, 4D] / [D]
transformer.ln_f.{weight,bias}                                  [D]
lm_head.{weight,bias}                                           [V, D] / [V]  # bias here is unusual
```

Note the *single* `ln_1` per block (Pythia has both `input_layernorm`
and `post_attention_layernorm`; GPT-J uses the same LN output for
both branches — see [`GPTJBlock`](../../lib/core/nn/gptj.dart)).

Rotary applies to the first 64 dims of each 256-dim head, but with
the **interleaved-pair** convention — the loader permutes Q/K weight
rows at load time so our half-split [`RopeCache`](../../lib/core/nn/rotary.dart)
produces bit-identical results. Full derivation in
[03-rotary-conventions.md](03-rotary-conventions.md).

## Presets

| Config | Layers | Embed | Heads | HeadDim | Rotary | fp32 size |
|---|---|---|---|---|---|---|
| `gpt2Config` | 12 | 768 | 12 | 64 | — | 500 MB |
| `gpt2MediumConfig` | 24 | 1024 | 16 | 64 | — | 1.4 GB |
| `pythia14mConfig` | 6 | 128 | 8 | 16 | 4 | 60 MB |
| `pythia70mConfig` | 6 | 512 | 8 | 64 | 16 | 300 MB |
| `pythia160mConfig` | 12 | 768 | 12 | 64 | 16 | 650 MB |
| `pythia410mConfig` | 24 | 1024 | 16 | 64 | 16 | 1.6 GB |
| `pythia1bConfig` | 16 | 2048 | 8 | 256 | 64 | 3.3 GB |
| `gptJ6bConfig` | 28 | 4096 | 16 | 256 | 64 | 24 GB |

All `*Config` factories accept a `device: Device.CPU|GPU` — the
GPT-J one additionally has [`gptJ6bHybridConfig(gpuLayers: N)`](../../lib/core/nn/gptj_hf_loader.dart)
for CPU/GPU split loading (see [04-hybrid-placement.md](04-hybrid-placement.md)).

## Load-time behaviour

Every loader returns a report of consumed / unused HF keys. Typical
output for a healthy Pythia load:

```
Loaded in 34580 ms. PythiaLoadReport(consumed=173, unused=24)
```

The 24 "unused" keys are the ignored rotary `inv_freq` buffers +
per-layer causal-mask + `masked_bias` tensors mentioned above.
Nonzero unused counts are informational only; missing *consumed*
keys throw immediately.

## Common pitfalls

- **fp16 storage**: Pythia-1b ships as fp16. Our [SafeTensors reader](../../lib/core/nn/safetensors.dart)
  auto-converts fp16 → fp32 during load. This roughly doubles load
  time vs a native fp32 checkpoint because every entry goes through a
  Dart-side decode loop.
- **PyTorch Linear layout is `[out, in]`.** All Pythia and GPT-J
  weights match this exactly (no transpose). GPT-2 is the odd one out
  because it uses `Conv1D` (which is `[in, out]`), which is why the
  GPT-2 loader has explicit transposes and the others don't.
- **Tied vs untied output head.** GPT-2's `lm_head` is just a
  transpose of `wte` — the loader shares one storage. Pythia has
  a separate `embed_out.weight`. GPT-J has both a separate
  `lm_head.weight` *and* an `lm_head.bias`. Do not tie them.
