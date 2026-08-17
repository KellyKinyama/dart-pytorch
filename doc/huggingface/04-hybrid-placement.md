# 04 — Hybrid CPU/GPU placement

Some checkpoints are close to fitting on the GPU but not quite there
— GPT-J-6B is 24 GB fp32 vs a 6 GB card; Pythia-2.8b is 11 GB vs a
6 GB card. Rather than fall back to pure CPU (which is *slow*), you
can put a subset of transformer blocks on GPU and leave the rest on
CPU. Activations shuttle across PCIe at the boundaries — cheap,
since they're `[N, D]` (a few hundred KB for a 30-token prompt).

Today this is implemented for GPT-J only. The `GPTJModel` code path
is the reference — porting to `PythiaModel` is straightforward when
needed.

## API

`GPTJConfig` gained three optional fields:

```dart
GPTJConfig({
  // ... existing required fields ...
  Device device = Device.CPU,     // primary/default device
  List<Device>? layerDevices,     // one per transformer block (length must equal numLayers)
  Device? embeddingDevice,        // override for wte (fallback: device)
  Device? lmHeadDevice,           // override for ln_f + lm_head (fallback: device)
})
```

When `layerDevices == null` and both overrides are null, the model
behaves identically to the pre-hybrid code (bit-exact — see the
[test](../../test/gptj_test.dart) that copies weights between a
uniform-CPU and a `layerDevices: all-CPU` model and asserts logits
agree to `< 1e-6`).

Two convenience methods:

```dart
cfg.deviceForLayer(i);   // resolved device for block i
cfg.hasMixedPlacement;   // true if any two components differ
```

## Preset for the common case

```dart
GPTJHFLoader.gptJ6bHybridConfig(
  gpuLayers: 5,             // blocks [0..5) on GPU, [5..28) on CPU
  embedOnGpu: false,        // wte on CPU (826 MB fp32)
  lmHeadOnGpu: false,       // ln_f + lm_head on CPU (826 MB fp32)
)
```

Yields `layerDevices = [GPU]*5 + [CPU]*23`.

## VRAM budget for GPT-J-6B

Per-block fp32 cost is roughly

$$
12 \cdot D^2 \cdot 4 \text{ bytes} \approx 805 \text{ MB}
$$

for `D = 4096` — that's 4 attention projections at `D²` and two MLP
projections at `4D²` each. On a 6 GB GPU:

| gpuLayers | Weights | Free (~5 GB budget after CUDA + activations) |
|---|---|---|
| 0 | 0 | full budget |
| 3 | 2.4 GB | comfortable |
| 5 | 4.0 GB | tight but works |
| 6 | 4.8 GB | risky |
| 7+ | 5.6 GB+ | OOM likely |

Add `embedOnGpu: true` or `lmHeadOnGpu: true` and each adds ~826 MB
(shift the table down by one row per flag).

## How it works internally

Model construction:

1. Every unique `Device` that appears across all blocks + `embed_device`
   + `lm_device` gets its own [`RopeCache`](../../lib/core/nn/rotary.dart)
   built on that device. Blocks look up their cache by device.
2. Each block's `LayerNorm`, `MultiHeadAttention`, and `Linear` MLP
   layers are constructed on `deviceForLayer(i)`. Existing module
   plumbing already accepts `device: …`, no changes needed there.
3. The [HF loader](../../lib/core/nn/gptj_hf_loader.dart) copies each
   weight into the destination tensor via `dst.assign(...)` — the
   destination already lives on the right device, so weight upload
   happens as a side effect of `Tensor.fromList(..., device: dst.device)`.

Forward pass ([`GPTJModel._forward`](../../lib/core/nn/gptj.dart)):

```dart
var h = wte(t0);                  // on embedDevice
Tensor? mask; Device? maskDev;
for (int i = 0; i < blocks.length; i++) {
  final blockDev = config.deviceForLayer(i);
  if (h.device != blockDev) h = h.to(blockDev);          // move activation
  if (n > 1 && maskDev != blockDev) {                     // rebuild mask per device
    mask = causalMask(n, device: blockDev);
    maskDev = blockDev;
  }
  h = blocks[i](h, mask: n > 1 ? mask : null, ...);
}
if (h.device != config.lmDevice) h = h.to(config.lmDevice);
return lmHead(finalLn(h));
```

Everything else (KV cache, greedy sampling, HTTP request handling) is
unchanged — the KV cache lives inside each block, on the same device
as the block, and needs no coordination across boundaries.

## Cost model

- **Activation moves.** Each cross-device transition copies
  `[N, D] · 4 bytes`. For GPT-J-6B (D=4096) at a 30-token prompt
  that's 480 KB per transition. WSL2 host↔device bandwidth is
  ~10–15 GB/s realistic → ~30–50 µs per move. Multiply by the number
  of transitions (2 per generation step: embed→first-GPU-block once,
  last-GPU-block→CPU once). Negligible.
- **Kernel launch overhead.** Each of the `gpuLayers` blocks runs 6
  matmuls on GPU. CUDA launch is ~5–20 µs, matmul at `D=4096` is
  hundreds of ms of compute → launch overhead invisible.
- **KV cache growth.** During autoregressive generation, each block's
  cache grows by one K row + one V row per token — on the same device
  as the block. GPU blocks accumulate GPU tensors, CPU blocks
  accumulate Float32Lists. No cross-device pressure.

## Expected speedup

Assume a hypothetical GPT-J-6B where GPU matmul is 30× faster than
CPU matmul (matmul-limited, WSL2, tiny prompt). If you put `k` of
`28` blocks on GPU:

$$
\text{speedup} = \frac{28}{k/30 + (28-k)}
$$

| gpuLayers | Speedup |
|---|---|
| 0 | 1.00× |
| 5 | 1.22× |
| 10 | 1.55× |
| 15 | 2.14× |
| 20 | 3.44× |
| 28 | 30× |

At `gpuLayers=5` (the max that comfortably fits), the win is modest
(~20%) — CPU is still doing most of the work. This is the fundamental
limitation of the approach: on a memory-constrained card, hybrid
buys you a fraction, not a factor.

For a bigger real win, look at fp16/int8 runtime (~2×/4× VRAM
reduction) or weight streaming — both are separate features.

## Runner

[`bin/gptj/run_6b_hybrid_api.dart`](../../bin/gptj/run_6b_hybrid_api.dart)
wraps the hybrid config behind a `--gpu-layers N` CLI flag (default 4):

```sh
LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  dart run bin/gptj/run_6b_hybrid_api.dart \
    --gpu-layers 5 --text "Once upon a time," --max-tokens 20
```

The runner prints a compact placement label like
`hybrid(gpu:0-4,cpu:5-27; embed=cpu, lm=cpu)` at load time; the
same information is available via `GET /info` when running in
`--serve` mode.

## Extending to other models

Porting to `PythiaModel` is a mechanical copy: add
`layerDevices`/`embeddingDevice`/`lmHeadDevice` to `PythiaConfig`,
replace the single `RopeCache` field with `Map<Device, RopeCache>`,
add `.to()` moves at block boundaries in `_forward`. The GPT-J commit
(`8889e09`) is the reference implementation.
