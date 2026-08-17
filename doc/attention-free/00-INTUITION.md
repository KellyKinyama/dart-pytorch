# **0. INTUITION**

## **0.1. Recap: what standard attention does**

Standard scaled dot-product attention computes, for a sequence of
`T` tokens each of width `D`:

```text
scores = Q @ K.T / sqrt(D)      # [T, T]  ("who attends to whom")
attn   = softmax(scores)        # [T, T]  (row-wise probabilities)
out    = attn @ V               # [T, D]
```

The **defining feature** is the `[T, T]` similarity matrix. It
carries the full pairwise interaction — an entry `attn[i, j]` for
every ordered pair of positions.

The cost:

- **Time:** `O(T^2 * D)` for `Q @ K.T` plus `attn @ V`.
- **Memory (activations):** `O(T^2)` — we have to store the softmax
  weights for the backward pass.

That `T^2` term is what makes 128k-token contexts a research
problem. It also dominates GPU memory in modern LLM inference.

## **0.2. What if we could skip the `[T, T]` matrix?**

Every position still needs to depend on every other position — you
can't just drop the pairwise interaction and hope for the best. But
**there is no law of physics** that says the pairwise interaction
has to go through a softmax over an explicit `[T, T]` scores matrix.

The AFT paper's central move: replace the pairwise weights
`attn[t, t']` with a formula that has **the same expressive shape**
(a softmax over `t'`, one per output position `t`) but is factored
in a way that avoids ever materialising the whole `[T, T]` tensor.

## **0.3. The AFT operator**

Here it is, verbatim from the top of [lib/core/tensor/aft.dart](../../lib/core/tensor/aft.dart):

    out[t, d] = sigmoid(Q[t, d]) *
                ( sum_{t'} exp(K[t', d] + W[t, t']) * V[t', d] )
                -----------------------------------------------
                (   sum_{t'} exp(K[t', d] + W[t, t'])          )

Read it slowly. Compare to standard attention:

| Ingredient        | Standard attention                     | AFT                                                    |
|-------------------|----------------------------------------|--------------------------------------------------------|
| Where do the weights come from? | `Q @ K.T` — dot product per pair | `K[t', d] + W[t, t']` — **add** a key and a position bias |
| What normalises them? | `softmax` over `t'`, computed on `[T, T]` scores | `softmax` over `t'`, computed **per feature `d`** |
| What role does `Q` play? | Enters the pair score               | Only appears as `sigmoid(Q[t, d])` — an **elementwise gate** |
| Pairwise `[T, T]` tensor? | Yes, `scores`                     | **No.** `W` is `[T, T]` but it's a **parameter**, not an activation to be checkpointed for backward |

The trick is that the "softmax over `t'`" is done **per feature
dimension `d`** rather than once over the whole vector. So instead
of one `[T, T]` softmax, you have `D` separate `[T]`-length
softmaxes per output position — but they never assemble into an
`[T, T]` scores matrix.

## **0.4. Why the gate?**

`sigmoid(Q[t, d])` is elementwise (no cross-position work). It's a
learned per-feature gate: for each output position, for each feature
dimension, it either lets the aggregated value through (~1) or
zeroes it out (~0).

This is the only role `Q` plays. In standard attention, `Q` is the
query that decides who to attend to. In AFT, "who to attend to" is
determined by `K + W` alone; `Q` just decides how much of the
aggregated result to keep.

This is a real reduction in expressive power vs. standard attention.
AFT compensates by making `W` learnable and per-position-pair, which
recovers a lot of the flexibility.

## **0.5. Complexity**

Reading the formula naively you might think:

- Compute `K[t', d] + W[t, t']` — that's `T * T * D` work.
- softmax over `t'` per `(t, d)` — same.
- weighted sum — same.

**Time:** `O(T^2 * D)` — same order as standard attention.

**But activation memory:** the intermediate `weights[t, t', d]`
tensor is `[T, T, D]`, which sounds worse. However this is the CPU
implementation's choice to keep the backward pass simple. In the
paper's original formulation you never form this tensor at all: you
can do the whole thing in a streaming fashion with `O(T * D)`
activation memory.

The current CPU code in this repo takes the memory-heavy option (see
[chapter 4](./04-NUMERICAL-STABILITY.md) — the `weights` array is
kept for the backward pass, chapter 5). The GPU kernel is
streamable. So the practical asymptotic is:

- **Time:** `O(T^2 * D)` — matches standard attention.
- **Memory (best case):** `O(T * D)` — better than standard
  attention's `O(T^2)`.

## **0.6. Is it actually "free"?**

Semantically, no — there are still `T^2` pairwise interactions
happening; they're just not routed through a materialised similarity
matrix. AFT is "free" in the sense that the `[T, T]` activation
tensor of standard attention is gone.

Practically, in this repo:

- Standard `MultiHeadAttention` runs on CPU **and** GPU, supports
  batched 3D inputs, and has a KV cache for O(N) generation.
- `AFTAttention` runs on CPU **and** GPU (via `aft_full_forward` /
  `aft_full_backward` kernels), is 2D-only, and has no cache.

For small models the two are drop-in comparable — see
[bin/aft_demo.dart](../../bin/aft_demo.dart) for a head-to-head. AFT
tends to have slightly fewer parameters at matched width and often
trains faster on very small sequences.

## **0.7. What comes next**

- [Chapter 1](./01-THE-AFT-OPERATOR.md) walks the `aftFull` op line
  by line.
- [Chapter 2](./02-POSITION-BIAS.md) explains the learned `W`
  matrix that carries all positional information.
- [Chapter 3](./03-CAUSAL-MASKING.md) shows how the `masked` flag
  turns AFT into a decoder-only variant.
- [Chapter 4](./04-NUMERICAL-STABILITY.md) covers the
  max-subtraction trick for exponentiating potentially large sums.
- [Chapter 5](./05-CUSTOM-BACKWARD.md) derives the analytical
  gradients — AFT is one of the few ops in this library that ships
  its own backward instead of composing from primitives.
- [Chapters 6-7](./06-AFT-ATTENTION-MODULE.md) show the module and
  language-model layers.

---

<div align="right">

[&lt;&lt;&nbsp;&nbsp;Previous: README](./README.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: THE AFT OPERATOR&nbsp;&nbsp;&gt;](./01-THE-AFT-OPERATOR.md)

</div>
