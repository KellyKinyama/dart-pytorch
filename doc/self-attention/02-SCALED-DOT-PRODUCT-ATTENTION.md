# **2. SCALED DOT-PRODUCT ATTENTION**

The single-head attention operator lives in one file, as one
extension method:

<sup>from [lib/core/tensor/attention.dart](../../lib/core/tensor/attention.dart):</sup>

```dart
extension TensorAttention on Tensor {
  Tensor scaledDotProductAttention(Tensor k, Tensor v, {Tensor? mask}) {
    ...
    final scale = 1.0 / math.sqrt(dk);
    var scores = matmul(k.transpose()) * scale;
    if (mask != null) {
      ...
      scores = scores + mask;
    }
    final attn = scores.softmax();
    return attn.matmul(v);
  }
}
```

The whole thing is four lines. Everything else in the file is
shape validation. Let's walk through it.

## **2.1. Shapes**

The op is deliberately 2D-only:

<sup>from [lib/core/tensor/attention.dart](../../lib/core/tensor/attention.dart):</sup>

```dart
/// Given `Q` shape `[N, Dk]`, `K` shape `[M, Dk]`, `V` shape `[M, Dv]`:
///
///     scores = Q @ K.T * (1 / sqrt(Dk))    // [N, M]
///     attn   = softmax(scores)             // [N, M], row-wise
///     out    = attn @ V                    // [N, Dv]
```

`Q` has `N` query rows. `K` and `V` share `M` context rows (one key
and one value per context position). The last dim of `Q` and `K`
must match (they need to dot-product together); the last dim of `V`
is independent and becomes the last dim of the output.

For self-attention `N == M` and `Dk == Dv`. For cross-attention
they can differ (chapter 7).

## **2.2. Line 1 - `Q @ K.T * scale`**

```dart
final scale = 1.0 / math.sqrt(dk);
var scores = matmul(k.transpose()) * scale;
```

`scores[i, j]` is the dot product of `Q[i]` with `K[j]` — a scalar
saying "how much does query `i` want to attend to context position
`j`?"

The `1 / sqrt(Dk)` scale is the "scaled" in "scaled dot-product
attention". Without it, dot products of `Dk`-dimensional random
vectors have variance `Dk`, so as `Dk` grows the softmax gets
sharper and sharper, saturating gradients. Dividing by `sqrt(Dk)`
normalises the variance back to 1 regardless of head width.

## **2.3. Line 2 - additive mask (optional)**

```dart
if (mask != null) {
  ...
  scores = scores + mask;
}
```

The mask is **added**, not multiplied. Convention across the library
(see [chapter 3](./03-CAUSAL-MASK.md)):

- `0` at allowed positions
- a large negative constant (default `-1e9`) at blocked positions

Adding `-1e9` before softmax sends `exp(-1e9)` to zero, so blocked
positions get probability ~0 and the allowed positions renormalise
among themselves. No division-by-zero, no branching, no `-inf`
fiddling — just an add.

The mask must be exactly the same shape as `scores` (`[N, M]`); no
broadcasting is implied. The op raises `ArgumentError` if you get
this wrong.

## **2.4. Line 3 - row-wise softmax**

```dart
final attn = scores.softmax();
```

`softmax` is defined in [lib/core/tensor/softmax.dart](../../lib/core/tensor/softmax.dart) as row-wise softmax
along the last axis, using the classic max-subtraction trick for
numerical stability:

<sup>from [lib/core/tensor/softmax.dart](../../lib/core/tensor/softmax.dart):</sup>

```dart
double m = xd[i * c];
for (int j = 1; j < c; j++) {
  if (xd[i * c + j] > m) m = xd[i * c + j];
}
double s = 0;
for (int j = 0; j < c; j++) {
  final e = math.exp(xd[i * c + j] - m);
  out[i * c + j] = e;
  s += e;
}
final inv = 1.0 / s;
for (int j = 0; j < c; j++) {
  out[i * c + j] *= inv;
}
```

Without the `- m`, `exp(scores[i, j])` on scores of ~30 already
overflows fp32. Subtracting the row max makes the largest exponent
exactly zero, so no exponential ever exceeds `1.0`.

The backward pass has a nice closed form:

    dScores[i] = (dAttn[i] - dot(dAttn[i], attn[i])) * attn[i]

On CPU it's the loop at the bottom of the same file; on GPU the
`softmax_backward` kernel runs it in one fused pass.

## **2.5. Line 4 - weighted sum of values**

```dart
return attn.matmul(v);
```

`attn` is `[N, M]` where each row sums to `1`. `V` is `[M, Dv]`.
The matmul is literally

    out[i] = sum_j attn[i, j] * V[j]

Every output row is a convex combination of value rows, with weights
that the model just computed from `Q` and `K`. The magic all
happened in the softmax.

## **2.6. Autograd**

Notice there is **no custom backward** for `scaledDotProductAttention`.
It's built entirely out of ops (`matmul`, `transpose`, `*`, `+`,
`softmax`) that already know how to differentiate themselves. That's
the whole point of an autograd engine — new layers compose for free
as long as their forward is written in existing ops.

## **2.7. Toy walk-through**

Two query tokens, three context tokens, `Dk = Dv = 4`:

```text
Q       [2, 4]
K       [3, 4]
V       [3, 4]

scores  [2, 3]   = Q @ K.T * (1/sqrt(4))     # {2 rows, 3 candidates}
attn    [2, 3]   = softmax(scores)           # each row sums to 1
out     [2, 4]   = attn @ V                  # 2 output rows, 4-dim
```

For self-attention plug in `Q = K = V = X` where `X` is `[N, D]` and
you get `[N, D]` back out — same shape as the input, every position
now aware of every other. That's the whole engine.

---

<div align="right">

[&lt;&lt;&nbsp;&nbsp;Previous: Q, K, V AND LINEAR PROJECTIONS](./01-QKV-AND-LINEAR-PROJECTIONS.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: CAUSAL MASK&nbsp;&nbsp;&gt;](./03-CAUSAL-MASK.md)

</div>
