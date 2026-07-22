# **5. CUSTOM BACKWARD**

Most ops in `dart_pytorch` inherit their backward for free by
composing existing autograd-aware primitives (see
[chapter 2.6 of the self-attention docs](../self-attention/02-SCALED-DOT-PRODUCT-ATTENTION.md#26-autograd)).
AFT does not — the per-`d` softmax + weighted sum + sigmoid gate
would be tediously slow to express as a graph of primitives. So
`aftFull` ships a **hand-derived single-node backward** for all
four inputs (`Q`, `K`, `V`, `W`) in one pass.

## **5.1. The forward, mathematically**

Let me restate the forward compactly:

    s[i, t', d] = K[t', d] + W[i, t']
    weights[i, t', d] = softmax_over_t' ( s[i, t', d] )
    wv[i, d] = sum_{t'} weights[i, t', d] * V[t', d]
    sq[i, d] = sigmoid( Q[i, d] )
    out[i, d] = sq[i, d] * wv[i, d]

The backward has to compute `dQ`, `dK`, `dV`, `dW` from `dout`.

## **5.2. Reading `dout` and the cached state**

<sup>from [lib/core/tensor/aft.dart](../../lib/core/tensor/aft.dart):</sup>

```dart
out._setBackward([q, k, v, w], () {
  final gO = out._grad!._cpuData!;
  final gQ = Float32List(t * d);
  final gK = Float32List(t * d);
  final gV = Float32List(t * d);
  final gW = Float32List(t * t);

  for (int i = 0; i < t; i++) {
    final limit = masked ? i + 1 : t;
    for (int dd = 0; dd < d; dd++) {
      final gy = gO[i * d + dd];
      if (gy == 0.0) continue;
      final sq = sigQ[i * d + dd];
      final wvv = wv[i * d + dd];
      ...
```

Zero-init gradient buffers, then loop over the same `(i, dd)` grid
as the forward.

**The `if (gy == 0.0) continue`** is a small but important
optimization: if the downstream loss doesn't depend on output entry
`(i, dd)`, every gradient contribution from this cell is zero, so
we skip the entire inner loop. In practice `gy` is rarely exactly
zero, but this makes debugging easier — you can zero out a single
output row and confirm the backward stays put.

The forward stashed three things we need here:

- `sigQ[i, dd] = sq` — the sigmoid'd query. Needed for `dQ`.
- `wv[i, dd]` — the aggregated value before gating. Needed for
  `dQ`.
- `weights[i, t', dd]` — the normalized softmax weights. Needed for
  everything else.

Nothing else survives from the forward. These three arrays are all
that make the backward possible in a single pass.

## **5.3. `dQ`: sigmoid + product**

`out = sq * wv` and `sq = sigmoid(q)`. Standard chain rule:

    dout/dsq = wv
    dsq/dq = sq * (1 - sq)
    dout/dq = wv * sq * (1 - sq)

So

    gQ[i, dd] += gy * sq * (1 - sq) * wv[i, dd]

In code:

<sup>from [lib/core/tensor/aft.dart](../../lib/core/tensor/aft.dart):</sup>

```dart
// out = sq * wv
gQ[i * d + dd] += gy * sq * (1.0 - sq) * wvv;
final dWv = gy * sq;
```

The second line computes `dWv = d out / d wv * gy = sq * gy` —
the gradient with respect to the aggregated value `wv[i, dd]`,
which we're about to push through the softmax + weighted sum.

## **5.4. Setup for `dV`, `dK`, `dW`**

`wv = sum_{t'} weights[i, t', d] * V[t', d]` — a weighted sum. Two
gradients drop out immediately:

- **`dV[t', d] += dWv * weights[i, t', d]`** — each `V[t', d]`
  contributes to `wv[i, d]` weighted by `weights[i, t', d]`.
- **`dweights[i, t', d] = dWv * V[t', d]`** — each `weights[i, t',
  d]` contributes weighted by `V[t', d]`.

We now need to push `dweights` back through the softmax to get
`ds`, then further back through the pre-softmax `K + W` sum.

## **5.5. Backprop through per-`d` softmax**

The softmax backward for a single vector `y = softmax(s)`, given
downstream gradient `dy`, is

    ds[j] = ( dy[j] - dot(dy, y) ) * y[j]

We derived this in [self-attention chapter 2.4](../self-attention/02-SCALED-DOT-PRODUCT-ATTENTION.md#24-line-3---row-wise-softmax) but here's a fresh look. Because
`sum_j y[j] = 1`, differentiating the constraint gives `sum_j
dy[j]_gradient_of_j_through_ancestors = 0`; the `- dot(dy, y)` is
exactly the term that removes the component of `dy` orthogonal to
that constraint.

Applied to AFT, per-`(i, dd)`:

    dot = sum_{t'} dweights[i, t', d] * weights[i, t', d]
    ds[i, t', d] = ( dweights[i, t', d] - dot ) * weights[i, t', d]

Two loops in the code — one to accumulate `dot`, one to write out
`ds`:

<sup>from [lib/core/tensor/aft.dart](../../lib/core/tensor/aft.dart):</sup>

```dart
double dot = 0.0;
for (int tp = 0; tp < limit; tp++) {
  final wt = weights[(i * t + tp) * d + dd];
  final dwt = dWv * vd[tp * d + dd];
  dot += dwt * wt;
}
for (int tp = 0; tp < limit; tp++) {
  final wt = weights[(i * t + tp) * d + dd];
  final dwt = dWv * vd[tp * d + dd];
  final ds = (dwt - dot) * wt;
  gK[tp * d + dd] += ds;
  gW[i * t + tp] += ds;
  gV[tp * d + dd] += dWv * wt;
}
```

## **5.6. Backprop through `s = K + W`**

`s[i, t', d] = K[t', d] + W[i, t']`. Sum → gradient flows to both
inputs equally:

- **`dK[t', d] += ds[i, t', d]`** — but summed over `i` (every
  output position that used `K[t', d]`).
- **`dW[i, t'] += ds[i, t', d]`** — but summed over `d` (every
  feature that used `W[i, t']`).

Look at the code: `gK[tp * d + dd] += ds` accumulates into `[t',
d]`, and the outer `for (int i ...) for (int dd ...)` provides the
`i`-sum for `gK` and the `dd`-sum for `gW`.

`gV[tp * d + dd] += dWv * wt` is the direct gradient from the
weighted-sum step (section 5.4). Combined with the softmax
backprop, we've now accounted for all four downstream targets in
one nested loop.

## **5.7. Complexity of the backward**

Same as the forward: `O(T^2 * D)` in time (the `masked` version is
about half that). Activation memory during the backward is dominated
by the cached `weights` tensor, `[T, T, D]` — the biggest single
allocation in the whole op.

For long-context AFT, the `[T, T, D]` weights buffer is what makes
the CPU implementation memory-hungry. The GPU kernel avoids
materializing it (it recomputes on-the-fly using the same `K`, `V`,
`W`) — one of several places where the CPU path prioritizes clarity
over efficiency.

## **5.8. GPU backward: same math, different plumbing**

The GPU backward is a `aft_full_backward` CUDA kernel that takes the
same inputs (`Q`, `K`, `V`, `W`, `dout`, `masked`) and writes into
pre-allocated zero-init `dQ`, `dK`, `dV`, `dW` buffers using
`atomicAdd`. The kernel structure mirrors this file:

1. Recompute `weights[i, :, dd]` (or fetch from a stashed workspace).
2. Compute `dot` per `(i, dd)`.
3. Emit `ds` and use `atomicAdd` to scatter into `dK`, `dW`.
4. Emit `dV` directly and `atomicAdd` into `dV`.
5. Emit `dQ` from the sigmoid derivative directly.

`atomicAdd` is required because multiple `(i, t', dd)` triples can
target the same `dK[t', dd]` (and analogously for `dW`, `dV`).

## **5.9. Why not just use autograd primitives?**

You could, in principle:

- `s = K.unsqueeze(0).expand(t, t, d) + W.unsqueeze(-1).expand(t, t, d)`
- `weights = softmax(s, dim=1)`
- `wv = (weights * V.unsqueeze(0).expand(t, t, d)).sum(dim=1)`
- `sq = sigmoid(Q)`
- `out = sq * wv`

That expression graph would autograd for free — but it allocates the
full `[T, T, D]` intermediate several times over, doubles the memory
during backward, and each op is a separate kernel launch. The custom
single-node backward is roughly 4-8x faster and uses half the memory
on realistic sizes.

For teaching purposes the expanded expression is more transparent;
for production the fused backward wins. AFT was written for
production comparison, so it ships the fused version.

---

<div align="right">

[&lt;&lt;&nbsp;&nbsp;Previous: NUMERICAL STABILITY](./04-NUMERICAL-STABILITY.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: AFTAttention MODULE&nbsp;&nbsp;&gt;](./06-AFT-ATTENTION-MODULE.md)

</div>
