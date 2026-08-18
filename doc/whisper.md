# Whisper tiny.en in dart-pytorch

End-to-end port of [openai/whisper](https://github.com/openai/whisper)
`tiny.en`: WAV file → mel spectrogram → transformer encoder + decoder
→ tokenized transcript. Runs on CPU and GPU with the same code path;
GPU is ~5× faster end-to-end on the JFK sample.

- [`lib/core/audio/whisper_mel.dart`](../../lib/core/audio/whisper_mel.dart)
  — 80-bin log-mel front-end (bit-close to `openai-whisper/audio.py`).
- [`lib/core/nn/whisper.dart`](../../lib/core/nn/whisper.dart)
  — `WhisperEncoder`, `WhisperEncoderBlock` (Conv1d ×2, sinusoidal PE,
  N pre-LN transformer blocks).
- [`lib/core/nn/whisper_decoder.dart`](../../lib/core/nn/whisper_decoder.dart)
  — `WhisperDecoder`, `WhisperDecoderBlock` (learned PE, causal
  self-attn, cross-attn with per-block K/V cache).
- [`lib/core/nn/whisper_hf_loader.dart`](../../lib/core/nn/whisper_hf_loader.dart)
  — `WhisperHFLoader.loadFile` / `loadDecoderFile`: HuggingFace safetensors
  → our per-head Linear projections.
- [`bin/whisper_demo.dart`](../../bin/whisper_demo.dart) — CPU demo.
- [`bin/whisper_gpu_demo.dart`](../../bin/whisper_gpu_demo.dart) — GPU demo.
- [`test/whisper_test.dart`](../../test/whisper_test.dart),
  [`test/whisper_gpu_test.dart`](../../test/whisper_gpu_test.dart) —
  fast structural + full transcription tests.

## Quick start

Fetch the weights and tokenizer once (~150 MB):

```bash
mkdir -p models/whisper-tiny.en
curl -sSL -o models/whisper-tiny.en/model.safetensors \
  https://huggingface.co/openai/whisper-tiny.en/resolve/main/model.safetensors
curl -sSL -o models/whisper-tiny.en/tokenizer.json \
  https://huggingface.co/openai/whisper-tiny.en/resolve/main/tokenizer.json
```

The repo ships a ~11 s speech clip at
[`data/jfk.wav`](../../data/jfk.wav) (16 kHz mono PCM16) — the JFK
inaugural excerpt from OpenAI Whisper's test assets.

CPU:

```bash
dart run bin/whisper_demo.dart
```

GPU (WSL2 / Linux with the CUDA build of `libmat_mul.so` in place):

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  dart run bin/whisper_gpu_demo.dart
```

Both print:

```
== transcript ==
  " And so my fellow Americans ask not what your country can do for you, ask what you can do for your country."
```

Token stream is bit-identical to
`transformers.WhisperForConditionalGeneration.generate(num_beams=1,
do_sample=False)`:

```
[843, 523, 616, 5891, 3399, 1265, 407, 644, 534, 1499,
 460, 466, 329, 345, 11, 1265, 644, 345, 460, 466,
 329, 534, 1499, 13]
```

## Architecture

```
WAV (16 kHz mono)
  ▼   WhisperMel.logMelFromFile
[1, 80, 3000]   ← log-mel spectrogram, (x + 4) / 4 rescaled
  ▼   WhisperEncoder
    conv1  Cin=80  Cout=384  k=3 p=1        + GELU
    conv2  Cin=384 Cout=384  k=3 s=2 p=1    + GELU
    BCT → BTC
    + sinusoidal PE [1500, 384]
    × 4 blocks:
        + attn_ln → 6-head self-attn → residual
        + mlp_ln  → Linear(384→1536) → GELU → Linear(1536→384) → residual
    ln_post
[1, 1500, 384]  ← encoder memory
  ▼   WhisperDecoder.primeCrossAttn  (once)
    per block: crossK[h] = crossKHeads[h](memory)   ← 4 blocks × 6 heads
               crossV[h] = crossVHeads[h](memory)
  ▼   greedy loop: while len(tokens) < maxLen:
    forward([SOT, NOTIMESTAMPS, ...tokensSoFar])
      token_emb + positional_emb[0:T]
      × 4 blocks:
          + attn_ln → 6-head causal self-attn      → residual
          + cross_attn_ln → 6-head cross-attn      → residual
          + mlp_ln → Linear(384→1536) → GELU → Linear(1536→384) → residual
      ln
    logits[T-1] = hidden[T-1] @ token_emb.T
    tokens.append(argmax(logits[T-1]))
    (suppress token 220 " " and 50256 EOT at t=0)
    stop when argmax == 50256
```

## Key layout — HuggingFace ⇒ our fields

HuggingFace stores the fused `[C, C]` Q/K/V projections; our
`WhisperEncoderBlock` and `WhisperDecoderBlock` hold them as per-head
`Linear`s of shape `[headDim, C]` so both CPU and GPU share the same
code path (`Tensor.scaledDotProductAttention` operates on single-head
tensors).

| HuggingFace safetensors key                                  | Our field                    |
|---|---|
| `model.encoder.conv1.{weight,bias}`                          | `encoder.conv1`              |
| `model.encoder.conv2.{weight,bias}`                          | `encoder.conv2`              |
| `model.encoder.embed_positions.weight`                       | `encoder.positionalEmbedding` (overwrites sinusoids for bit-close reference match) |
| `model.encoder.layer_norm.{weight,bias}`                     | `encoder.lnPost`             |
| `model.encoder.layers.{i}.self_attn.q_proj.{weight,bias}`    | `block.qHeads[h]` for h ∈ 0..5 (row-split) |
| `model.encoder.layers.{i}.self_attn.k_proj.weight`           | `block.kHeads[h]` (no bias)  |
| `model.encoder.layers.{i}.self_attn.v_proj.{weight,bias}`    | `block.vHeads[h]`            |
| `model.encoder.layers.{i}.self_attn.out_proj.{weight,bias}`  | `block.outProj`              |
| `model.encoder.layers.{i}.self_attn_layer_norm.{weight,bias}` | `block.attnLn`              |
| `model.encoder.layers.{i}.fc1.{weight,bias}`                 | `block.mlp0`                 |
| `model.encoder.layers.{i}.fc2.{weight,bias}`                 | `block.mlp2`                 |
| `model.encoder.layers.{i}.final_layer_norm.{weight,bias}`    | `block.mlpLn`                |

The decoder mapping is analogous, with `model.decoder.embed_tokens`
→ `decoder.tokenEmbedding`, `model.decoder.embed_positions` →
`decoder.positionalEmbedding` (learned, unlike the encoder's
sinusoids), and each block also carrying `encoder_attn.*` →
`crossQHeads / crossKHeads / crossVHeads / crossOutProj /
crossAttnLn`.

Row-split rule for the fused `[C, C]` weight:
head `h` receives rows `[h * headDim, (h + 1) * headDim)`. The bias
`[C]` is sliced identically.

## Tiny.en constants (from `generation_config.json`)

| Token id | Symbol                    | Role                                       |
|---:|---|---|
| 50256 | `<|endoftext|>`              | EOT — stops greedy decode                  |
| 50257 | `<|startoftranscript|>`      | SOT — first token in the decoder prefix    |
| 50362 | `<|notimestamps|>`           | position 1 in the prefix (tiny.en is english-only, no `<|en|>` / `<|transcribe|>`) |
| 220   | ` ` (byte-BPE space)         | suppressed at first sampled token (`begin_suppress_tokens`) |

Prefix used by the greedy loop:

```dart
decoder.greedyDecode(
  startTokens: [50257, 50362],      // SOT, NOTIMESTAMPS
  eot: 50256,
  initialSuppress: [220, 50256],    // don't start with " " or EOT
)
```

## Numerical accuracy vs. HuggingFace `transformers`

Encoder log-mel front-end vs. numpy reference (`scripts/whisper_mel_reference.py`):

- mean abs diff `2.3e-5`, max `0.196` (max is all in the silence-padded
  tail where fftea rounds tiny values differently from numpy).

Encoder hidden state on `data/jfk.wav` vs. `WhisperModel.encoder`:

- our (device-agnostic per-head SDPA): mean abs diff `~8e-3`,
  max `~0.32` (0.03% of elements > 1e-1).

Greedy token stream vs. `WhisperForConditionalGeneration.generate`:

- **bit-exact** on both `models/silero_vad/warmup_audio.wav`
  (`" Audio for model warmup?"`) and `data/jfk.wav`
  (`" And so my fellow Americans..."`).

## Wall-clock — CPU vs. GPU (WSL2, RTX 3060)

Measured on `data/jfk.wav` (~11 s of audio):

| Phase                | CPU   | GPU   | Speedup |
|---|---|---|---|
| Log-mel front-end    | 115 ms | 115 ms | 1×      |
| Load encoder weights | 2.0 s  | 2.3 s  | ~1×     |
| Load decoder weights | 5.5 s  | 5.9 s  | ~1×     |
| **Encoder forward (1500 tokens)** | **43 s**  | **2.5 s** | **17×** |
| Prime cross-attn     | 2.3 s  | 0.12 s | 19×     |
| Greedy decode (24 tokens) | 8.7 s | 2.0 s | 4×      |
| **Total**            | **64 s** | **13.5 s** | **4.7×** |

The encoder self-attention dominates the CPU cost (T = 1500, N = 4
blocks, 6 heads). On GPU it becomes a single-digit-ms `matmul +
softmax + matmul` per head, hence the 17× speedup. Weight loading is
device-agnostic and stays on the host (safetensors decode + CPU→GPU
upload), so it doesn't benefit.

Load-time overhead can be roughly halved by loading the encoder and
decoder in parallel or (better) preserving decoded safetensors between
successive audio clips.

## Trying a different audio file

Any 16 kHz mono WAV works out of the box:

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  dart run bin/whisper_gpu_demo.dart \
    --wav path/to/your.wav \
    --max-len 200
```

For a non-16 kHz or stereo source, resample first — the mel front-end
assumes 16 kHz mono. Something like:

```bash
python3 -c "
import soundfile as sf, numpy as np
d, sr = sf.read('input.mp3')
if d.ndim > 1: d = d.mean(axis=1)
if sr != 16000:
    n = int(len(d) * 16000 / sr)
    xs = np.linspace(0, len(d)-1, n)
    x_lo = np.floor(xs).astype(int); x_hi = np.minimum(x_lo+1, len(d)-1)
    frac = xs - x_lo; d = d[x_lo]*(1-frac) + d[x_hi]*frac
d = np.clip(d, -1, 1)
sf.write('data/mine.wav', (d*32767).astype(np.int16), 16000, subtype='PCM_16')
"
```

## Known limitations

- **English-only** — this is `tiny.en`. The multilingual `tiny` /
  `base` / `small` variants use the same architecture but a longer
  prefix (`SOT, <|language|>, <|task|>, NOTIMESTAMPS`). The loader
  and modules will pick those up with just different config numbers,
  but the greedy-decode caller has to build the right prefix.
- **30 s clips only** — the log-mel truncates or pads to exactly 30 s
  (480 000 samples), producing a fixed `[80, 3000]` input. Chunk
  longer audio yourself before feeding the encoder.
- **Greedy decoding** — no beam search yet, no temperature fallback,
  no timestamp tokens. For the tiny.en single-30 s use case greedy
  is usually indistinguishable from `num_beams=5` in Whisper.
- **No KV cache** in the decoder — each greedy step re-runs the full
  `T`-token forward. For `tiny.en` with `T ≤ ~30` this is < 2 s on
  GPU (see wall-clock above), but a proper KV cache would drop it
  further.
