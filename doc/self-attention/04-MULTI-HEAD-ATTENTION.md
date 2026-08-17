# **4. MULTI-HEAD ATTENTION**

Single-head attention forces all `D` dimensions of the token vector
to agree on **one** notion of similarity. Multi-head attention runs
`H` attention operations in parallel, each on its own `Dh = D / H`
slice of the vector, and concatenates the results. Different heads
end up specialising on different relationships (syntax, coreference,
position, etc.) — for free, purely from the parallel structure and
gradient descent.

## **4.1. Class layout**

<sup>from [lib/core/nn/attention/multi_head_attention.dart](../../lib/core/nn/attention/multi_head_attention.dart):</sup>

```dart
class MultiHeadAttention extends Module {
  final int embedDim;
  final int numHeads;
  final int headDim;
  final List<Linear> wq;   // numHeads * Linear(embedDim, headDim)
  final List<Linear> wk;
  final List<Linear> wv;
  final Linear wo;         // one final mixing projection
  final Dropout? attnDropout;
  ...
}
```

- `wq[h]`, `wk[h]`, `wv[h]` — head `h`'s query / key / value
  projections. Each maps `[N, embedDim] -> [N, headDim]`.
- `wo` — the "output" projection, `[N, embedDim] -> [N, embedDim]`.
- `attnDropout` — an optional dropout applied to the concatenated
  head outputs (post `wo`).

Setup was covered in [chapter 1](./01-QKV-AND-LINEAR-PROJECTIONS.md).
Here we focus on the forward pass.

## **4.2. Forward pass (2D, no cache)**

<sup>from [lib/core/nn/attention/multi_head_attention.dart](../../lib/core/nn/attention/multi_head_attention.dart):</sup>

```dart
final heads = <Tensor>[];
for (int h = 0; h < numHeads; h++) {
  final q = wq[h](x);
  var k = wk[h](x);
  var v = wv[h](x);
  if (cache != null) {
    k = cache.appendK(h, k);
    v = cache.appendV(h, v);
  }
  heads.add(q.scaledDotProductAttention(k, v, mask: mask));
}
final concatted = TensorConcat.concat(heads, axis: 1);
var out = wo(concatted);
if (attnDropout != null) {
  out = attnDropout!(out);
}
return out;
```

That's the entire per-head loop. Per head:

1. Project `x` into `Q`, `K`, `V` — three `Linear` calls.
2. Run `scaledDotProductAttention` (chapter 2) with the shared mask.
3. Push the `[N, headDim]` output into a list.

Then:

4. Concatenate all `H` head outputs on the feature axis — `H` tensors
   of shape `[N, headDim]` become one `[N, H * headDim] == [N, embedDim]`.
5. Apply `wo` to mix across heads.
6. Optionally drop out for regularisation.

The concat step is intentionally not a `stack + reshape` — Dart's
`TensorConcat.concat(axis: 1)` gives us a clean autograd path (each
head keeps its own gradient stream) with no reshape overhead.

## **4.3. Complexity**

Per forward pass, per layer:

- Projections: `3 * H * (N * embedDim * headDim) + N * embedDim^2`
  FLOPs = `4 * N * embedDim^2` (the `wo` matches).
- Attention: `H * (N^2 * headDim + N^2 * headDim)` = `2 * N^2 * embedDim`
  FLOPs (dot products + weighted sums).
- Total: `4 * N * embedDim^2 + 2 * N^2 * embedDim`.

The projection term is linear in `N`; the attention term is
**quadratic** in `N`. For short sequences the projections dominate;
for long sequences the attention takes over. That crossover is what
the whole "efficient attention" literature (Flash, Linear, AFT, etc.)
is trying to move.

## **4.4. Batched 3D forward**

Most of the library is 2D-single-sequence, but for training you want
batched inputs. `MultiHeadAttention` accepts 3D `[B, N, embedDim]`
and dispatches to a separate path:

<sup>from [lib/core/nn/attention/multi_head_attention.dart](../../lib/core/nn/attention/multi_head_attention.dart):</sup>

```dart
Tensor _callBatched(Tensor x, {Tensor? mask}) {
  final b = x.shape[0];
  final s = x.shape[1];
  final heads = <Tensor>[];
  for (int h = 0; h < numHeads; h++) {
    final q = wq[h](x);           // [B, S, headDim]  (Linear folds leading dims)
    final k = wk[h](x);
    final v = wv[h](x);
    final qFlat = q.reshape([b * s, headDim]);
    final kFlat = k.reshape([b * s, headDim]);
    final vFlat = v.reshape([b * s, headDim]);
    final qSplits = TensorConcat.splitRows(qFlat, s);
    final kSplits = TensorConcat.splitRows(kFlat, s);
    final vSplits = TensorConcat.splitRows(vFlat, s);
    final perBatch = <Tensor>[];
    for (int i = 0; i < b; i++) {
      perBatch.add(
        qSplits[i].scaledDotProductAttention(
          kSplits[i], vSplits[i], mask: mask,
        ),
      );
    }
    heads.add(TensorConcat.concat(perBatch, axis: 0)); // [B*S, headDim]
  }
  final concatted = TensorConcat.concat(heads, axis: 1); // [B*S, embedDim]
  var out = wo(concatted.reshape([b, s, embedDim]));    // [B, S, embedDim]
  ...
}
```

Two important points about this path:

**Attention is computed per-sequence.** The dot products `Q_i @ K_i.T`
must not cross batch boundaries — token 3 of sequence A must not
attend to token 3 of sequence B. So we split the batched projection
into `B` per-sequence tensors, call `scaledDotProductAttention` `B`
times per head, and concat the results back.

**The mask is shared across batch elements.** All `B` sequences use
the same causal mask, which is safe because they all have the same
length `S`. If you ever add per-sequence padding masks you'd need a
`[B, S, S]` tensor and a broadcasting-aware SDPA.

## **4.5. Guardrails**

<sup>from [lib/core/nn/attention/multi_head_attention.dart](../../lib/core/nn/attention/multi_head_attention.dart):</sup>

```dart
if (isBatched && cache != null) {
  throw ArgumentError(
    'MultiHeadAttention: KV cache is not supported for batched '
    '(3D) input; run per-sequence for cached generation.',
  );
}
if (cache != null && mask != null && cache.seqLen > 0) {
  throw ArgumentError(
    'MultiHeadAttention: cannot pass mask when appending to a '
    'non-empty cache (mask shape would be incompatible with '
    'grown K/V)',
  );
}
```

Two invariants:

- **Batched + cache is not supported.** KV caching is per-sequence
  by definition (each sequence has its own past). If you want cached
  batched generation, run one call per sequence element or interleave
  them yourself.
- **Cache + mask is only allowed when the cache is empty.** An empty
  cache is the "prompt fill" case (chapter 6) where `Q` and `K` share
  a length and a causal mask makes sense. Non-empty cache is the
  "single-token append" case where the mask would be wrong-shaped
  and unnecessary.

## **4.6. What a head learns**

There's no direct supervision that says "head 3 should learn syntax".
Heads specialise emergently, purely because:

- The `wo` projection lets them combine, so redundancy is punished by
  gradient descent (two heads doing the same thing gives no benefit,
  so at least one gets pushed off to do something else).
- Random init gives every head a slightly different starting point,
  and once you diverge you tend to keep diverging.

Anthropic's / Elhage et al.'s interpretability work has catalogued
"induction heads", "copy heads", "positional heads" and so on — none
of them designed, all of them emergent. That's the elegance of the
architecture: the same block, replicated `H` times, discovers a
diverse mixture of specialists on its own.

---

<div align="right">

[&lt;&lt;&nbsp;&nbsp;Previous: CAUSAL MASK](./03-CAUSAL-MASK.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: PRE-LN TRANSFORMER BLOCK&nbsp;&nbsp;&gt;](./05-PRE-LN-TRANSFORMER-BLOCK.md)

</div>
