# **3. CAUSAL MASKING**

Standard attention causal-masks by **adding** a `[T, T]` mask
(`0`/`-1e9`) to the pre-softmax scores. AFT causal-masks by
**restricting the loop range** — the inner sum stops at `t' == t`
instead of running to `t' == T - 1`. No mask tensor is ever
allocated.

## **3.1. The `masked` flag**

`TensorAft.aftFull` takes a `bool masked` parameter:

<sup>from [lib/core/tensor/aft.dart](../../lib/core/tensor/aft.dart):</sup>

```dart
static Tensor aftFull(
  Tensor q,
  Tensor k,
  Tensor v,
  Tensor w, {
  bool masked = false,
}) {
  ...
}
```

- `masked: false` (default) — bidirectional; every output position
  attends to every context position. Suitable for an encoder or a
  BERT-style model.
- `masked: true` — causal; output position `t` only attends to
  context positions `t' <= t`. Suitable for a decoder / language
  model.

The `AFTLanguageModel` passes `masked: true` to every block; the
demo in `bin/aft_demo.dart` uses the causal variant.

## **3.2. How the mask shows up in the loop**

In the CPU forward, `masked` controls one line — the loop's upper
bound:

<sup>from [lib/core/tensor/aft.dart](../../lib/core/tensor/aft.dart):</sup>

```dart
for (int i = 0; i < t; i++) {
  final limit = masked ? i + 1 : t;
  for (int dd = 0; dd < d; dd++) {
    double maxVal = -1e30;
    for (int tp = 0; tp < limit; tp++) {
      final s = kd[tp * d + dd] + wd[i * t + tp];
      if (s > maxVal) maxVal = s;
    }
    double denom = 0.0;
    for (int tp = 0; tp < limit; tp++) {
      final e = math.exp(kd[tp * d + dd] + wd[i * t + tp] - maxVal);
      weights[(i * t + tp) * d + dd] = e;
      denom += e;
    }
    ...
```

`limit = masked ? i + 1 : t`. For output position `i`, the inner
loops sweep `t' = 0..limit-1`:

- `masked: false` — `limit = t`, so `t'` ranges `0..t-1` (every
  context position).
- `masked: true` — `limit = i + 1`, so `t'` ranges `0..i` (only
  past + current position).

For future positions `t' > i`, no work happens: no `exp` call, no
`weights` write, no contribution to `num`. Those entries of the
`weights` buffer stay at their initial zero.

## **3.3. Why this is better than an additive mask**

Standard attention with a `[T, T]` mask:

```text
scores = Q @ K.T / sqrt(D)          # [T, T], both triangles computed
scores = scores + mask              # [T, T], add -1e9 to upper triangle
attn   = softmax(scores)            # [T, T], softmax always runs on both triangles
out    = attn @ V
```

Every one of `T^2` score entries gets computed. The mask suppresses
half of them at softmax time, but the compute happened.

AFT with `masked: true`:

```text
for each output position i:
  for each feature d:
    sweep t' from 0 to i         # never touches t' > i
```

The forbidden entries are **never computed** in the first place.
Roughly halves the FLOPs for causal AFT vs. bidirectional AFT.

You could get the same effect with standard attention if the SDPA
op were re-written to accept a lower-triangular mask and skip the
suppressed positions in the kernel — but as long as the mask is a
generic additive tensor, the kernel doesn't know it can skip.

## **3.4. Interaction with position bias `W`**

The upper-triangular entries of `W` (i.e. `W[t, t']` for `t' > t`)
still get **initialized** and still live in the tensor, but under
`masked: true` they are **never read** on the forward pass and
therefore accumulate **no gradient** on the backward pass either.

Concretely, if you train `AFTLanguageModel` (always `masked: true`)
and inspect `posBias` after training, the upper triangle will still
be near its initial random values — the optimizer never had a reason
to touch it. The useful learned pattern sits in the lower triangle
plus the diagonal.

Some AFT variants (not implemented here) explicitly parameterize
only the lower triangle to save half the parameters. In this repo
we pay the full `maxSeqLen^2` in exchange for a simpler `Tensor`
type.

## **3.5. The masked backward**

Chapter 5 covers the backward in detail, but note the same
`limit = masked ? i + 1 : t` guard appears in the CPU backward:

<sup>from [lib/core/tensor/aft.dart](../../lib/core/tensor/aft.dart):</sup>

```dart
for (int i = 0; i < t; i++) {
  final limit = masked ? i + 1 : t;
  for (int dd = 0; dd < d; dd++) {
    ...
    double dot = 0.0;
    for (int tp = 0; tp < limit; tp++) {
      ...
    }
    for (int tp = 0; tp < limit; tp++) {
      ...
      gK[tp * d + dd] += ds;
      gW[i * t + tp] += ds;
      gV[tp * d + dd] += dWv * wt;
    }
  }
}
```

`gK[tp, dd]` only accumulates from output positions `i >= tp`
(because `tp < limit = i + 1` implies `i >= tp`). So each `K[t',
dd]`'s gradient is a sum over the future positions that saw it —
which is exactly the reverse-direction analog of the forward's
"attend to past" rule.

## **3.6. What happens if you flip the flag mid-run?**

Nothing catastrophic — the flag is passed per call. You could
imagine a curriculum where you train bidirectionally
(`masked: false`) then fine-tune causally (`masked: true`), or use
`masked: false` for encoding and `masked: true` for decoding in a
seq2seq setup.

Neither is exercised by the current codebase; the
`AFTLanguageModel` hard-codes `masked: true` in its block
construction, and the demo runs only the language-model path.

---

<div align="right">

[&lt;&lt;&nbsp;&nbsp;Previous: POSITION BIAS](./02-POSITION-BIAS.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: NUMERICAL STABILITY&nbsp;&nbsp;&gt;](./04-NUMERICAL-STABILITY.md)

</div>
