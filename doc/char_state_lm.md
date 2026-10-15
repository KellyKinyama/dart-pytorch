# Context-free, state-based character LM

A decoder-only transformer that predicts characters **without a context
window**. Instead of attending over the whole history, each block carries
a fixed-size recurrent **state** that summarises everything seen so far —
so there is no positional embedding and generation is **O(1) time and
memory per character** with unbounded effective context.

This is the recurrent form of **linear attention** ("Transformers are
RNNs", Katharopoulos et al. 2020), extended with **multiple heads** and a
per-head **RWKV/RetNet-style decay gate**.

- Model: [../lib/core/nn/char_state_lm.dart](../lib/core/nn/char_state_lm.dart)
- Demo: [../bin/char_state_lm_demo.dart](../bin/char_state_lm_demo.dart)

---

## Why "context-free"

A standard GPT re-reads the last `N` tokens every step (an `N`-wide
attention window, plus positional embeddings). This model instead keeps,
per block and per head, a state

$$
S \in \mathbb{R}^{d\times d}, \qquad z \in \mathbb{R}^{d}
$$

and updates it one character at a time. The state *is* the context, so:

- **no context window** — history length is irrelevant to cost;
- **no positional embedding** — order is encoded by the order of state
  updates;
- **O(1) per-character generation** — constant time and memory, any
  prompt length.

## The recurrence

Feature map `φ(x) = relu(x) + 1` (a positive, `elu+1`-style kernel). Per
head, with learnable decay `λ ∈ (0, 1)`:

$$
S \leftarrow \lambda\,S + \varphi(k_t)^\top v_t, \qquad
z \leftarrow \lambda\,z + \varphi(k_t)
$$
$$
\text{out}_t = \frac{\varphi(q_t)\,S}{\varphi(q_t)\cdot z + \varepsilon}
$$

Training unrolls this exact recurrence over a sequence (`forwardSeq`);
generation runs the identical step on a persistent state (`step`), so the
trained model and the streaming generator are numerically the same.

### Multi-head + decay

- **Heads** (`numHeads`): `D` is split into `numHeads` heads of size
  `D/numHeads`, each with its own `(S, z)`; outputs are concatenated and
  passed through an output projection.
- **Decay gate** (`useDecay`, default on): each head has a learnable
  scalar `λ_h = σ(decayRaw_h)`. Heads are initialised across a spread of
  decays (~0.90…0.999) so different heads remember at different
  timescales. `useDecay: false` gives plain linear attention (`λ = 1`,
  perfect memory).

## Architecture

```
ids [N]
  → Embedding [N, D]                     (no positional embedding)
  → RecurrentDecoderBlock × numLayers    pre-LN:
        h = x + LinearAttention(ln1(x))
        y = h + FFN(ln2(h))              (Linear → relu → Linear)
  → LayerNorm
  → tied head  (h @ Embedding.weightᵀ)   → logits [N, vocab]
```

## Run it

Trains on `data/tiny_shakespeare.txt` (via `loadCorpus`) then samples.
CPU by default; the per-character recurrence builds a small graph per
step, so the defaults keep a run to a few minutes.

```bash
# default quick run (≈300 steps), then generate from a prompt
dart run bin/char_state_lm_demo.dart

# your own prompt / length / temperature
dart run bin/char_state_lm_demo.dart --prompt "To be, or not to be" --gen 200 --temp 0.7

# a stronger (slower) run for more legible text
dart run bin/char_state_lm_demo.dart --chars 40000 --steps 2000 --dim 192 --layers 3 \
  --prompt "ROMEO:" --gen 400
```

### Flags

| flag | default | meaning |
|---|---|---|
| `--chars N` | 5000 | corpus characters used |
| `--steps N` | 300 | training steps |
| `--seq N` | 48 | training window length |
| `--dim N` | 96 | embedding / model width |
| `--layers N` | 2 | number of decoder blocks |
| `--heads N` | 4 | linear-attention heads (`dim % heads == 0`) |
| `--prompt "…"` | `ROMEO:` | seed text for generation |
| `--gen N` | 240 | characters to generate |
| `--temp T` | 0.8 | sampling temperature (`0` = greedy) |
| `--overfit` | off | memorize one window (sanity check) |
| `--save PATH` | — | write a JSON checkpoint after training |
| `--load PATH` | — | load a checkpoint and prompt it immediately (no training) |

## Save & reuse (train once, prompt many times)

`--save` writes config + tokenizer vocab + all weights to a JSON file;
`--load` rebuilds the model and copies the weights back, skipping
training entirely:

```bash
# train once, save
dart run bin/char_state_lm_demo.dart --chars 40000 --steps 2000 --dim 192 \
  --layers 3 --save shakes.json

# then prompt it instantly, as many times as you like
dart run bin/char_state_lm_demo.dart --load shakes.json --prompt "ROMEO:" --gen 400
dart run bin/char_state_lm_demo.dart --load shakes.json --prompt "To be" --temp 0.5
```

## Prompting

`CharStateLM.generate(promptIds, maxNewTokens, {temperature, rng})` feeds
the prompt through the recurrent state character-by-character, then
streams new characters. `temperature <= 0` is greedy/deterministic.
Prompt quality is bounded by training — the quick demo produces
Shakespeare rhythm and real words but not fluent prose; train longer and
larger for coherent output.

## Overfit sanity check

`--overfit` trains on a single fixed window every step and then **greedily
regenerates it from its first character**. A correct implementation drives
the loss to ~0 and reproduces the window exactly — greedy decoding feeds
the model's own predictions through the state, so an exact match proves
the training (`forwardSeq`) and streaming (`step`) paths are consistent.

```
$ dart run bin/char_state_lm_demo.dart --overfit --steps 400 --seq 48
step   0  loss 4.4175
step 399  loss 0.0003
target:  "First Citizen:\r\nBefore we proceed any further, h"
greedy:  "First Citizen:\r\nBefore we proceed any further, h"
chars matched: 48 / 48  — OK, memorized exactly
```

## Limitations

- **CPU-oriented** — the per-character autograd unroll is fine for a char
  model but not optimised for GPU; training is single-sequence (batch 1).
- Quality needs scale (more steps/data/width) for fluent text.
