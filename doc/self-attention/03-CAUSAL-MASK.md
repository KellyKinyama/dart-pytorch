# **3. CAUSAL MASK**

A language model must not let position `i` peek at positions `j > i`
during training — otherwise every next-token prediction becomes
trivial. The fix is a **causal mask**: zero out (in log-space) all
future scores before softmax.

## **3.1. The helper**

<sup>from [lib/core/nn/masks.dart](../../lib/core/nn/masks.dart):</sup>

```dart
Tensor causalMask(
  int n, {
  double blockValue = -1e9,
  Device device = Device.CPU,
}) {
  final data = Float32List(n * n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      data[i * n + j] = j > i ? blockValue : 0.0;
    }
  }
  return Tensor.fromList([n, n], data, device: device);
}
```

For `n = 4` the returned tensor is:

```text
[  0    -1e9  -1e9  -1e9 ]
[  0     0    -1e9  -1e9 ]
[  0     0     0    -1e9 ]
[  0     0     0     0   ]
```

Row `i` has zeros in columns `0..i` (positions the query is allowed
to see) and `-1e9` in columns `i+1..n-1` (positions in the future).

## **3.2. Why additive, why `-1e9`, why not `-inf`?**

The `scaledDotProductAttention` op adds the mask directly to the
pre-softmax scores:

<sup>from [lib/core/tensor/attention.dart](../../lib/core/tensor/attention.dart):</sup>

```dart
if (mask != null) {
  ...
  scores = scores + mask;
}
final attn = scores.softmax();
```

After the add, blocked positions become approximately `-1e9`. The
softmax's numerically-stable subtract-the-max step (see
[chapter 2.4](./02-SCALED-DOT-PRODUCT-ATTENTION.md)) turns those
into `exp(-1e9 - max) ~ 0`. Allowed positions keep their original
score minus the row max and renormalise among themselves.

Why not literally `-inf`? `-inf + <finite> = -inf`, and if the entire
row is blocked (never happens for causal, but happens with
padding masks) you'd get `exp(-inf) = 0` everywhere, then divide by
zero. `-1e9` gives the same practical effect (`exp(-1e9)` underflows
to zero anyway) without the edge case.

Why not multiplicative? A multiplicative mask would need to be
applied **after** softmax and then renormalised — two extra passes,
and if you naively multiply then divide you re-introduce the
division-by-zero problem. Additive-before-softmax is one op, no
special cases.

## **3.3. Where the mask enters**

Three places in the library call `causalMask`:

**1. [GPT.\_forward](../../lib/core/nn/gpt.dart)** — every forward
during training and every prompt-fill during generation:

<sup>from [lib/core/nn/gpt.dart](../../lib/core/nn/gpt.dart):</sup>

```dart
final mask = n > 1 ? causalMask(n, device: h.device) : null;
h = encoder(h, mask: mask, cache: cache);
```

Note the `n > 1` guard. When `n == 1` (a single-token generation
step) there is nothing to mask — the one query has exactly one key
to attend to. Passing a `[1, 1]` mask would be a no-op but wastes a
tensor allocation.

**2. `TransformerEncoder`** just forwards the mask to every block:

<sup>from [lib/core/nn/transformer_encoder.dart](../../lib/core/nn/transformer_encoder.dart):</sup>

```dart
for (int i = 0; i < blocks.length; i++) {
  h = blocks[i](h, mask: mask, cache: cache?.layers[i]);
}
```

**3. `TransformerBlock`** hands it to `MultiHeadAttention`:

<sup>from [lib/core/nn/transformer.dart](../../lib/core/nn/transformer.dart):</sup>

```dart
Tensor call(Tensor x, {Tensor? mask, MHACache? cache}) {
  final h = x + dropout(mha(ln1(x), mask: mask, cache: cache));
  ...
}
```

So a single `[N, N]` mask is shared across every head, every layer,
every batch element of one forward pass. This is cheap — an `[N, N]`
tensor of floats — but it's not free, so upstream callers cache it
per forward.

## **3.4. Interaction with the KV cache**

Once the KV cache holds `T` past positions and you feed in `1` new
token, `Q` is `[1, D]` and `K = V` are `[T + 1, D]`. Every cached
position is in the **past** by construction, so the mask is
unnecessary — the single query is allowed to see all `T + 1` keys.

The `MultiHeadAttention` module refuses to combine the two:

<sup>from [lib/core/nn/attention/multi_head_attention.dart](../../lib/core/nn/attention/multi_head_attention.dart):</sup>

```dart
if (cache != null && mask != null && cache.seqLen > 0) {
  throw ArgumentError(
    'MultiHeadAttention: cannot pass mask when appending to a '
    'non-empty cache (mask shape would be incompatible with '
    'grown K/V)',
  );
}
```

An `[N, N]` causal mask assumes `Q` and `K` are the same length. On
a cached step they aren't, and there's no future to hide from, so
the mask is both wrong-shaped and pointless. The check fires early
with a clear message rather than letting you produce garbage output.

## **3.5. Other mask shapes (not implemented here)**

Real-world attention has other flavours you might reach for:

- **Padding mask** — hide tokens beyond the true sequence length in a
  padded batch. Zero at real tokens, `-1e9` at pad slots. Same
  additive machinery, different pattern.
- **Prefix / bidirectional mask** — allow full attention within the
  first `k` prompt tokens, causal after that (used by prefix-LM and
  T5-style span decoders).
- **Sliding-window / local mask** — only allow attention within a
  fixed radius (Longformer, Mistral).

None of these are built in yet — `causalMask` is the only helper the
library ships. Any additive `[N, N]` tensor with the same convention
would work if you built it by hand.

---

<div align="right">

[&lt;&lt;&nbsp;&nbsp;Previous: SCALED DOT-PRODUCT ATTENTION](./02-SCALED-DOT-PRODUCT-ATTENTION.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: MULTI-HEAD ATTENTION&nbsp;&nbsp;&gt;](./04-MULTI-HEAD-ATTENTION.md)

</div>
