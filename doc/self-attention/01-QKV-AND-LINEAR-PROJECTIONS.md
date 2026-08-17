# **1. Q, K, V AND LINEAR PROJECTIONS**

The Q/K/V projections in `dart_pytorch` are ordinary [`Linear`](../../lib/core/nn/linear.dart) layers. There is no
custom "attention weight" tensor — a query projection is just a
`Linear(embedDim, headDim, bias: false)` initialized with
Kaiming-uniform.

## **1.1. The `Linear` primitive**

<sup>from [lib/core/nn/linear.dart](../../lib/core/nn/linear.dart):</sup>

```dart
class Linear extends Module {
  final int inFeatures;
  final int outFeatures;
  final Tensor weight;
  final Tensor? bias;

  Linear(
    this.inFeatures,
    this.outFeatures, {
    bool bias = true,
    Device device = Device.CPU,
    int seed = 0,
  }) : weight = _initWeight(inFeatures, outFeatures, device, seed),
       bias = bias
           ? _initBias(inFeatures, outFeatures, device, seed + 1)
           : null;
```

Weight shape is `[outFeatures, inFeatures]` (PyTorch convention) so
the forward pass is `y = x @ W.T + b`. Attention layers pass
`bias: false` — the bias is redundant when a LayerNorm sits directly
in front of the projection.

## **1.2. One `Linear` per head, per role**

The `MultiHeadAttention` module holds **three lists of `Linear`
layers** — one list per role (Q, K, V), one entry per head:

<sup>from [lib/core/nn/attention/multi_head_attention.dart](../../lib/core/nn/attention/multi_head_attention.dart):</sup>

```dart
MultiHeadAttention(
  this.embedDim,
  this.numHeads, {
  bool bias = false,
  double dropoutP = 0.0,
  Device device = Device.CPU,
  int seed = 0,
}) : headDim = embedDim ~/ numHeads,
     wq = List<Linear>.generate(
       numHeads,
       (h) => Linear(embedDim, embedDim ~/ numHeads,
         bias: bias, device: device, seed: seed + h),
     ),
     wk = List<Linear>.generate(
       numHeads,
       (h) => Linear(embedDim, embedDim ~/ numHeads,
         bias: bias, device: device, seed: seed + 1000 + h),
     ),
     wv = List<Linear>.generate(
       numHeads,
       (h) => Linear(embedDim, embedDim ~/ numHeads,
         bias: bias, device: device, seed: seed + 2000 + h),
     ),
     wo = Linear(embedDim, embedDim,
       bias: bias, device: device, seed: seed + 3000),
     ...
```

Two things to notice.

**Each head projects `embedDim -> headDim`, not `embedDim ->
embedDim`.** With `H` heads and `headDim = embedDim / H`, the total
parameter count across the `H` per-head Q projections is
`H * embedDim * headDim = embedDim^2` — exactly the same as one
big `Linear(embedDim, embedDim)`. Splitting into heads is a
**re-parameterization**, not a size increase. The four `Linear`
groups together (`wq`, `wk`, `wv`, `wo`) hold `4 * embedDim^2`
parameters plus zero biases.

**Seeds are offset per role and per head** (`seed + h`,
`seed + 1000 + h`, ...) so no two of the `3H + 1` projections start
with byte-identical weights. This matters — if `wq[0]`, `wk[0]` and
`wv[0]` all started identical, head 0 would collapse to a symmetric
degenerate solution and never recover.

## **1.3. Why the divisibility check?**

<sup>from [lib/core/nn/attention/multi_head_attention.dart](../../lib/core/nn/attention/multi_head_attention.dart):</sup>

```dart
if (embedDim % numHeads != 0) {
  throw ArgumentError(
    'MultiHeadAttention: embedDim ($embedDim) must be divisible by '
    'numHeads ($numHeads)',
  );
}
```

`headDim` is computed with integer division `embedDim ~/ numHeads`.
If `embedDim` doesn't divide evenly, the concatenated head outputs
wouldn't add back up to `embedDim` and the output projection `wo`
would receive a wrong-sized tensor. Guard up front, fail loud.

## **1.4. Applying the projections**

Inside the forward pass, per head, per role, one `Linear` call:

```dart
for (int h = 0; h < numHeads; h++) {
  final q = wq[h](x);  // [N, embedDim] -> [N, headDim]
  var k = wk[h](x);    // [N, embedDim] -> [N, headDim]
  var v = wv[h](x);    // [N, embedDim] -> [N, headDim]
  ...
}
```

That's the full story of the Q/K/V projections. They are boring
`y = x @ W.T` matmuls; all the interesting geometry lives in the
next chapter, where those three tensors meet.

## **1.5. Output projection `wo`**

After each head runs its own scaled dot-product attention and
produces `[N, headDim]`, the `H` outputs are concatenated on the
last axis to form `[N, H * headDim] == [N, embedDim]`, then passed
through **one** more `Linear(embedDim, embedDim)` called `wo`:

```dart
final concatted = TensorConcat.concat(heads, axis: 1);
var out = wo(concatted);
```

`wo` is what lets the model **mix information across heads** — up to
this point each head has been staring at its own slice of the
embedding without ever seeing the others. Without `wo` the heads
would be completely independent columns; with it, head 3's output
can influence head 7's next-layer computation.

---

<div align="right">

[&lt;&lt;&nbsp;&nbsp;Previous: INTUITION](./00-INTUITION.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: SCALED DOT-PRODUCT ATTENTION&nbsp;&nbsp;&gt;](./02-SCALED-DOT-PRODUCT-ATTENTION.md)

</div>
