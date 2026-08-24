# SkyReels-V2 port — scope and design

**Status: aspirational.** This document describes what a full Dart port of
[SkyworkAI/SkyReels-V2](https://github.com/SkyworkAI/SkyReels-V2) would
entail. The repo currently contains a **skeleton** of the diffusion transformer
backbone only, in [lib/core/nn/skyreels_v2.dart](../lib/core/nn/skyreels_v2.dart) plus
[lib/core/nn/skyreels_v2_streaming.dart](../lib/core/nn/skyreels_v2_streaming.dart).
Everything else on the list below is deferred.

## Hardware reality check

The smallest SkyReels-V2 variant (`DF-1.3B-540P`) needs **14.7 GB VRAM** end-to-end.
The 14B variants need **43–51 GB VRAM**. On a 6 GB GPU + 15 GB WSL RAM box the
end-to-end pipeline cannot render video at all, even with layer streaming — the
VAE alone (`AutoencoderKLWan`, fp32) exceeds the GPU. The value of a port here
is therefore not "generate videos on my laptop"; it is:

- Validate that this repo's AirLLM-style layer-streaming pattern
  ([lib/core/nn/llama_streaming.dart](../lib/core/nn/llama_streaming.dart) etc.)
  generalises to video-DiT tensor layouts.
- Provide a reference for future work when better hardware is available.

## Reference architecture (Wan / SkyReels-V2 `WanModel`)

Source: [skyreels_v2_infer/modules/transformer.py](https://github.com/SkyworkAI/SkyReels-V2/blob/main/skyreels_v2_infer/modules/transformer.py).

```
video [B, in_dim=16, F, H, W] ──► patch_embedding: Conv3d(k=stride=(1,2,2))
                                  ──► x [B, N, dim]     (N = F * H/2 * W/2)

text  [B, text_len=512, text_dim=4096] ──► text_embedding: Linear→GELU→Linear
                                           ──► context [B, 512, dim]

t     [B] (or [B, F] for diffusion-forcing)
      ──► sinusoidal_embedding_1d(freq_dim=256)
      ──► time_embedding: Linear→SiLU→Linear                → e   [B, dim]
      ──► time_projection: SiLU→Linear(dim, 6*dim)          → e0  [B, 6, dim]

x ─► for i in 0..num_layers-1:
       block[i](x, e0, context, freqs, block_mask)
    ─► head(x, e)  ──► [B, N, prod(patch_size) * out_dim]
    ─► unpatchify ──► [B, out_dim=16, F, H/8, W/8]
```

**Per block (`WanAttentionBlock`):**

```
mod = block.modulation + e0                                 # [1, 6, dim] + [B, 6, dim]
[s0, s1, s2, s3, s4, s5] = mod.chunk(6, dim=1)

h = norm1(x) * (1 + s1) + s0                                # AdaLN pre-attn
h = self_attn(h, rope=3D)                                   # RoPE-3D, QK RMSNorm
x = x + h * s2                                              # gated residual

h = cross_attn(norm3(x), context)                           # T2V or I2V
x = x + h

h = norm2(x) * (1 + s4) + s3                                # AdaLN pre-FFN
h = Linear(dim, ffn_dim) → GELU(tanh) → Linear(ffn_dim, dim)
x = x + h * s5
```

**Config parameters** (defaults in `WanModel.__init__`):

| Key              | Default   | Notes                                          |
|------------------|-----------|------------------------------------------------|
| model_type       | 't2v'     | 'i2v' adds `img_emb` (MLPProj 1280→dim) and swaps `WanT2VCrossAttention` for `WanI2VCrossAttention` with a separate `k_img`/`v_img` for the first 257 image tokens |
| patch_size       | (1, 2, 2) | 3D Conv3D kernel=stride                        |
| text_len         | 512       | Fixed max text seq                             |
| in_dim / out_dim | 16 / 16   | Latent-video channels (from the VAE)           |
| dim              | 2048      | 1536 for `1.3B`, 5120 for `14B`                |
| ffn_dim          | 8192      | 8960 for `1.3B`, 13824 for `14B`               |
| freq_dim         | 256       | Sinusoidal timestep width                      |
| text_dim         | 4096      | T5 (`Qwen2.5-32B` prompt-enhance is optional)  |
| num_heads        | 16        | 12 for `1.3B`, 40 for `14B`                    |
| num_layers       | 32        | 30 for `1.3B`, 40 for `14B`                    |
| qk_norm          | True      | RMSNorm on Q, K post-projection                |
| cross_attn_norm  | True      | Elementwise-affine LN before cross-attn        |
| eps              | 1e-6      | All norms                                      |

## What is in the skeleton

| Piece                             | File                                                                 | Status |
|-----------------------------------|----------------------------------------------------------------------|--------|
| `SkyReelsV2Config`                | [lib/core/nn/skyreels_v2.dart](../lib/core/nn/skyreels_v2.dart)     | ✅ 1.3B and 14B presets. Verify against shipped `config.json`. |
| `SkyReelsV2Block` (self+cross+FFN, AdaLN modulation) | same                                              | ✅ AdaLN scale/shift applied. |
| `SkyReelsV2Model` (embed → blocks → head)            | same                                              | ✅ Forward runs on random weights, output shape matches unpatchified head. |
| `SkyReelsV2StreamingRunner`       | [lib/core/nn/skyreels_v2_streaming.dart](../lib/core/nn/skyreels_v2_streaming.dart) | ✅ Per-block weight swap via `ShardedSafeTensorsReader`, persistent embed/head. Key-name map matches HF layout (T2V). |
| Demo                              | [bin/skyreels_v2_random_demo.dart](../bin/skyreels_v2_random_demo.dart) | ✅ 1.3B forward on 16 patch tokens × 4 text tokens: ~2 min wall, all outputs finite. |

## What is deferred

**Blockers for real video output:**

- **Conv3D patch embedding.** Not implemented in this repo. The skeleton uses a
  `Linear(in_dim * prod(patch_size), dim)` placeholder that expects
  pre-patched `[N, in_dim * prod(patch_size)]` tokens.
- **3D RoPE (`rope_apply` with a ternary time/height/width frequency split).**
  Skipped entirely — self-attention runs without positional encoding.
- **Q/K RMSNorm inside attention** (`qk_norm=True`). This repo's `MultiHeadAttention`
  applies RoPE but does not expose a post-projection RMSNorm hook; adding it
  requires either a custom attention implementation or extending the shared one.
- **AutoencoderKLWan 3D VAE.** ~350 MB fp32 encoder + decoder, VAEResBlock,
  causal Conv3D, spatial upsample/downsample. Would be a separate port on the
  scale of `lib/core/nn/vae.dart` (which doesn't yet exist for video).
- **T5 text encoder.** The reference uses `google/t5-v1_1-xxl` or similar. This
  repo already has a T5 port in [lib/core/nn/t5.dart](../lib/core/nn/t5.dart)
  — encoder-only mode could be reused; would need loader glue.
- **UniPC scheduler + flow-matching sampler.** ~200 LOC of scheduler math with
  Runge-Kutta-style updates. Currently not in the repo.
- **I2V mode** — `WanI2VCrossAttention` with a separate `k_img`/`v_img` for the
  first 257 image tokens, plus `img_emb` (MLPProj 1280→dim) to consume CLIP
  image features.
- **Diffusion Forcing** — per-frame independent noise schedules and the
  block-wise causal attention mask that enables infinite-length video.
- **Teacache** — residual caching that skips computation on similar timesteps.
- **Video I/O** — FFmpeg bindings for reading input video (extension mode) and
  writing output MP4/GIF are not present in this repo at all.

**Non-blockers (nice to have):**

- HF safetensors loader (`SkyReelsV2HFLoader.loadFile(model, path)`) that maps
  the actual HF key names to model params, including the per-block Q/K/V
  weight tensors (which the loader has to slice `[dim, dim]` into
  `[num_heads, [head_dim, dim]]` for our per-head Linear layout).
- Prompt enhancer (Qwen2.5-32B) — +64 GB VRAM by itself, wildly out of scope.
- xDiT multi-GPU USP — pointless on single-GPU hobbyist hardware.

## Layer-streaming footprint

Same pattern as `LlamaStreamingRunner`. Peak resident RAM (fp16 checkpoint,
single-batch inference) ≈

```
patch_embed  + text_embed  + time_embed  + head        # persistent, small (~50 MB)
+ 1 × block  + swap-scratch                            # ~50 MB per block for 1.3B,
                                                         ~250 MB per block for 14B
+ activations                                          # depends on seq_len (video shape)
```

For **1.3B** the resident memory should be under **~1 GB fp16**, but activations
for the reference 540P shape (97 frames × 68 × 60 = 395 760 tokens) are
gigabytes on their own — a real run needs to shrink `--tokens` massively.

For **14B**, resident is ~4 GB fp16 — matches the "fits on 6 GB but only just"
regime that AirLLM targets. **Still bottlenecked by activations at production
video shapes.**

## Verifying without weights

Run the random-weight validator ([bin/skyreels_v2_random_demo.dart](../bin/skyreels_v2_random_demo.dart)):

```bash
dart run bin/skyreels_v2_random_demo.dart --preset 1.3b --tokens 16 --text-len 4
```

Expected: `output: [16, 64]` (that is, `[num_tokens, prod(patch_size) * out_dim]
= [16, 4 * 16]`), all values finite, ~2 min wall time on a WSL CPU.
