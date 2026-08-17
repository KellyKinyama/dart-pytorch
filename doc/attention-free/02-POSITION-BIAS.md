# **2. POSITION BIAS**

The `W` matrix in the AFT operator is where **all** positional
information lives. There is no sinusoidal encoding baked in, no
rotary transform, no attention on relative positions — just one
learned `[T, T]` matrix added directly into the pre-softmax scores.

## **2.1. Where `W` enters the formula**

Recall from [chapter 1](./01-THE-AFT-OPERATOR.md):

    out[t, d] = sigmoid(Q[t, d]) *
                sum_{t'} softmax_over_t' ( K[t', d] + W[t, t'] ) * V[t', d]

`W[t, t']` is a scalar bias added to the key score for context
position `t'` when the query is at position `t`. Two positions with
`W[t, t']` large and positive attend to each other strongly (all else
equal); positions with a large negative `W[t, t']` are effectively
disconnected.

Crucially, `W` does not depend on the input — it's the same matrix
for every sequence the model processes. It encodes **"which
positions matter for which"** as a learned bias, not as a function
of the token content.

## **2.2. Where `W` is created**

<sup>from [lib/core/nn/attention/aft_attention.dart](../../lib/core/nn/attention/aft_attention.dart):</sup>

```dart
class AFTAttention extends Module {
  ...
  /// Learnable position-bias matrix, shape `[maxSeqLen, maxSeqLen]`.
  final Tensor posBias;

  AFTAttention(
    this.embedDim, {
    required this.maxSeqLen,
    ...
  }) : ...
       posBias = _initPosBias(maxSeqLen, seed + 3000, device);

  static Tensor _initPosBias(int n, int seed, Device device) {
    final rng = math.Random(seed);
    final data = List<double>.generate(
      n * n,
      (_) => (rng.nextDouble() * 2 - 1) * 0.02,
    );
    return Tensor.fromList([n, n], data, requiresGrad: true, device: device);
  }
```

- **Shape**: `[maxSeqLen, maxSeqLen]`. `maxSeqLen` is fixed at
  construction time — the equivalent of a GPT `maxCtx`.
- **Init**: uniform in `[-0.02, +0.02]`. Small enough that at t=0
  the softmax is nearly uniform over positions (the bias barely
  perturbs the exponents), so the model starts by mixing all
  positions roughly equally and lets gradient descent carve out
  structure.
- **`requiresGrad: true`**: `W` is a **trainable parameter**. It
  appears in the module's `parameters()` list and gets updated by
  the optimizer.

## **2.3. Shorter sequences: `sliceTopLeft`**

`W` is sized for the worst case (`maxSeqLen`). Real inputs are often
shorter. Rather than allocate a new `W` per sequence length, the
module slices the top-left `[T, T]` sub-region:

<sup>from [lib/core/nn/attention/aft_attention.dart](../../lib/core/nn/attention/aft_attention.dart):</sup>

```dart
Tensor call(Tensor x) {
  ...
  final t = x.shape[0];
  ...
  final q = wq(x);
  final k = wk(x);
  final v = wv(x);
  final w = t == maxSeqLen
      ? posBias
      : TensorAft.sliceTopLeft(posBias, t, t);
  return TensorAft.aftFull(q, k, v, w, masked: masked);
}
```

`sliceTopLeft(posBias, t, t)` returns `posBias[:t, :t]` as a new
tensor, with autograd. Its backward routes gradients back into the
top-left `[t, t]` block of `posBias` — outside that block, no
gradient flows on that step.

This means only the top-left region of `W` gets exercised by short
sequences, but *any* row/column beyond `t` is untouched (until a
longer sequence comes through). Not a bug — just a natural
consequence of the position-indexed formulation.

## **2.4. `sliceTopLeft` implementation**

<sup>from [lib/core/tensor/aft.dart](../../lib/core/tensor/aft.dart):</sup>

```dart
static Tensor sliceTopLeft(Tensor t, int rows, int cols) {
  ...
  final src = t._cpuData!;
  final out = Float32List(rows * cols);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      out[i * cols + j] = src[i * c + j];
    }
  }
  final sliced = Tensor._cpu([rows, cols], out);
  if (t.requiresGrad) {
    sliced._setBackward([t], () {
      final gs = sliced._grad!._cpuData!;
      final gFull = Float32List(r * c);
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          gFull[i * c + j] = gs[i * cols + j];
        }
      }
      t._accumulateGrad(Tensor._cpu(t.shape, gFull));
    });
  }
  return sliced;
}
```

Forward is a plain copy. Backward writes the slice's gradient into
the top-left `[rows, cols]` of a full-sized zero tensor, then
accumulates that into the source's gradient. Everything outside the
top-left stays zero. Simple, but critical for correctness — a naive
in-place aliasing would corrupt autograd.

The GPU version has a matching `slice_top_left_forward` /
`slice_top_left_backward` kernel pair.

## **2.5. Parameter budget: `W` can dominate**

For `maxSeqLen = 32`, `embedDim = 32`:
- Q/K/V projections: `3 * embedDim^2 = 3072` scalars.
- `posBias`: `maxSeqLen^2 = 1024` scalars.
- Ratio: `posBias` is 25% of the attention parameters.

For `maxSeqLen = 1024`, `embedDim = 512`:
- Q/K/V projections: `3 * 512^2 = 786,432` scalars.
- `posBias`: `1024^2 = 1,048,576` scalars.
- Ratio: `posBias` is **larger** than all three projections combined.

For `maxSeqLen = 8192`, `embedDim = 512`:
- Q/K/V projections: 786,432.
- `posBias`: 67,108,864.
- Ratio: `posBias` is 85x larger than the projections.

This quadratic-in-`maxSeqLen` parameter blowup is the practical
reason AFT-full is unpopular for long contexts. The paper proposes
several variants (`AFT-simple`, `AFT-conv`, `AFT-local`) that
factorize or restrict `W` to sidestep this — none of them are
implemented in this repo. If you go past a few hundred tokens with
the current code, expect `posBias` to be the dominant memory cost.

## **2.6. How `W` learns different roles per position pair**

Because `W` is per-`(t, t')` and per-example-independent, gradient
descent can freely carve out structure like:

- **Recency bias**: `W[t, t']` decreases as `|t - t'|` grows.
- **Diagonals**: `W[t, t']` peaks along `t' = t - k` for some
  learned lag `k` — e.g. "always look at the token 5 positions back".
- **Boundary tokens**: `W[t, 0]` large for all `t` — every position
  attends to a distinguished start-of-sequence position.

None of these are hand-designed; they can all emerge from training
if they help the loss. The one thing `W` cannot do is depend on the
content — a token repeated at two positions gets the same bias
distribution as any other content at those positions. That's the
core capacity limitation vs. standard attention, and the reason `Q`
is retained (even in gated form) — to let the content contribute at
least one degree of freedom.

---

<div align="right">

[&lt;&lt;&nbsp;&nbsp;Previous: THE AFT OPERATOR](./01-THE-AFT-OPERATOR.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: CAUSAL MASKING&nbsp;&nbsp;&gt;](./03-CAUSAL-MASKING.md)

</div>
