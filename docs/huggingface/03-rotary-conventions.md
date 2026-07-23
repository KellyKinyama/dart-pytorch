# 03 — Rotary conventions

We ship *one* rotary implementation
([`RopeCache`](../../lib/core/nn/rotary.dart)) — the GPT-NeoX /
LLaMA / Mistral "half-split" flavour. GPT-J uses a different layout
("interleaved-pair"), but we still use the same kernel: the
[GPT-J loader](../../lib/core/nn/gptj_hf_loader.dart) permutes the
Q/K weight rows at load time so the downstream `Q @ K^T` produces
mathematically identical scores.

This page derives the permutation.

## Both conventions apply the same 2D rotation

Given a query row `q ∈ ℝ^{headDim}`, rotary embeds absolute position
`m` by rotating pairs of coordinates by angle `θ_i = m · f_i`, where
`f_i = base^{-2i/rotaryDim}` and `i` indexes pairs.

The rotation on pair `(a, b)` at angle `θ` is:

$$
\begin{pmatrix} a' \\ b' \end{pmatrix} =
\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}
\begin{pmatrix} a \\ b \end{pmatrix}
$$

Where the two conventions differ is **which two coordinates of `q`
constitute a pair**.

## Half-split (GPT-NeoX / LLaMA / Pythia / Mistral)

Pair `i` is `(q[i], q[i + H])`, where `H = rotaryDim / 2`. Both halves
of the first `rotaryDim` coordinates are stacked side by side, and
element `i` in the low half pairs with element `i` in the high half.

Written out for `i ∈ [0, H)`:

$$
\begin{aligned}
q'[i]     &= q[i]\cos\theta_i - q[i+H]\sin\theta_i \\
q'[i+H]   &= q[i+H]\cos\theta_i + q[i]\sin\theta_i
\end{aligned}
$$

Coordinates `[rotaryDim, headDim)` are pass-through (multiplied by 1).

The implementation encodes this as
`q' = q ⊙ cos + rotate_half(q) ⊙ sin` where
`rotate_half([a | b | pass]) = [-b | a | 0]` — see the
`_rotateHalfP` permutation matrix in [`RopeCache`](../../lib/core/nn/rotary.dart).

## Interleaved-pair (GPT-J)

Pair `i` is `(q[2i], q[2i + 1])` — pairs are adjacent, alternating.

For `i ∈ [0, H)`:

$$
\begin{aligned}
q'[2i]   &= q[2i]\cos\theta_i - q[2i+1]\sin\theta_i \\
q'[2i+1] &= q[2i+1]\cos\theta_i + q[2i]\sin\theta_i
\end{aligned}
$$

Same rotation, different coordinate assignment.

## The permutation that reconciles them

Define a permutation `π` on the head dimension that maps interleaved
positions to half-split positions:

$$
\pi(2i) = i, \qquad \pi(2i+1) = i + H \qquad \text{for } i \in [0, H)
$$

with `π(j) = j` for `j ∈ [rotaryDim, headDim)`.

Now suppose we take a query vector `q` (indexed in the GPT-J
convention), apply `π` to its coordinates to get `q_π`, and then run
our **half-split** rotary on `q_π`. What comes out?

Setting `a = q[2i]` and `b = q[2i+1]`, we have `q_π[i] = a` and
`q_π[i+H] = b`. Half-split rotary at position `i` gives:

$$
\begin{aligned}
(q_\pi')[i]   &= q_\pi[i]\cos\theta_i - q_\pi[i+H]\sin\theta_i
              = a\cos\theta_i - b\sin\theta_i
              = q'[2i] \\[2pt]
(q_\pi')[i+H] &= q_\pi[i+H]\cos\theta_i + q_\pi[i]\sin\theta_i
              = b\cos\theta_i + a\sin\theta_i
              = q'[2i+1]
\end{aligned}
$$

In other words, **`q_π'` is a coordinate-reordering of the reference
GPT-J `q'`**. Element `i` of the half-split output equals element
`2i` of the interleaved output; element `i+H` equals element `2i+1`.
Same values, different indexing.

## Why attention scores match exactly

Attention computes `Q K^\top`. If we apply the same permutation `π`
to both `Q` and `K`, the score at position `(m, n)` becomes:

$$
\sum_{d=0}^{H'\!-\!1} q_\pi'[m, d] \cdot k_\pi'[n, d]
= \sum_{d=0}^{H'\!-\!1} q'[m, \pi^{-1}(d)] \cdot k'[n, \pi^{-1}(d)]
= \sum_{d'=0}^{H'\!-\!1} q'[m, d'] \cdot k'[n, d']
$$

(reindex by `d' = π^{-1}(d)`). So the score matrix is **invariant**
under identical permutation of `Q` and `K`. Softmax → attention
weights → weighted sum of `V` all follow.

`V` is not touched — no rotary applies to V, and no permutation is
applied at load time. The output projection `W_o` reads V-shaped
activations and is unchanged too.

## What the loader does

Every GPT-J attention block ships `q_proj` and `k_proj` weights of
shape `[D, D]`. Splitting into heads gives per-head slices of shape
`[head_dim, D]`. Within each head's slice, [`GPTJHFLoader._permuteRotaryRows`](../../lib/core/nn/gptj_hf_loader.dart)
applies `π` to the first `rotaryDim=64` rows:

```dart
// out[i, :]     <- src[2i, :]
// out[i+H, :]   <- src[2i+1, :]
// out[j, :]     <- src[j, :]     for j in [rotaryDim, headDim)
```

`v_proj.weight` and `out_proj.weight` are copied straight through
without permutation.

## Verification

The permutation trick is validated end-to-end in
[`test/gptj_test.dart`](../../test/gptj_test.dart) — the test
"`half-split(π(q)) @ half-split(π(k))^T == interleaved(q) @ interleaved(k)^T`"
generates a random `[3, 8]` Q and K with `rotaryDim=4`, computes both
score matrices, and asserts every entry agrees to `< 1e-6`.

## Practical implications

- **No new rotary code was needed for GPT-J.** All the model-family
  differences live in the loader — the core `RopeCache` is untouched.
- **The same trick would work in reverse.** If you ever wanted to
  load a half-split checkpoint into an interleaved kernel, apply
  `π^{-1}` to the Q/K weight rows instead.
- **This *only* works because both conventions rotate the same pairs
  the same way — just under different indexing schemes.** If a future
  model changed the angles (e.g. YaRN, LongRoPE) rather than the
  layout, no permutation would save you — the rotation itself would
  differ.
