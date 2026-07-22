# **6. KV CACHE FOR GENERATION**

Training a language model works on fixed-length windows: feed `N`
tokens, compute `N` losses, done. **Generation** is different — the
model is asked for one token at a time, and each new token appends
to the context. Done naively, step `t` re-runs the whole prefix of
length `t`, giving `O(N^2)` work to generate `N` tokens.

The KV cache brings it down to `O(N)`.

## **6.1. What is cacheable?**

At step `t`, the model sees tokens `x[0..t-1]` and wants to compute
attention for the new token `x[t]`. Inside every attention layer:

- `Q = X @ Wq`  — depends on **every** token, but for step `t` we
  only need row `t`.
- `K = X @ Wk`  — every row depends only on its own token, so rows
  `0..t-1` are **exactly the same** as they were at the previous
  step.
- `V = X @ Wv`  — same story.

Rows `0..t-1` of `K` and `V` are pure functions of tokens `0..t-1`.
If we store them, we never have to recompute them. That's the KV
cache.

We do **not** cache `Q` because we only ever need row `t` of `Q` —
one row, computed once from `x[t]`, immediately used, then thrown
away.

## **6.2. `MHACache` — per-layer state**

<sup>from [lib/core/nn/kv_cache.dart](../../lib/core/nn/kv_cache.dart):</sup>

```dart
class MHACache {
  final int numHeads;

  /// Per-head running K, shape `[T_seen, headDim]`. `null` until the
  /// first append.
  final List<Tensor?> k;

  /// Per-head running V, shape `[T_seen, headDim]`. `null` until the
  /// first append.
  final List<Tensor?> v;

  MHACache.empty(this.numHeads)
    : k = List<Tensor?>.filled(numHeads, null, growable: false),
      v = List<Tensor?>.filled(numHeads, null, growable: false);

  int get seqLen => k[0]?.shape[0] ?? 0;
  ...
}
```

One `MHACache` per attention layer holds one `K` and one `V` tensor
per head. `seqLen` is how many tokens are currently cached (all
heads share the same T).

The tensors are created **without** `requiresGrad` — they're inference
state, not learnable parameters.

## **6.3. Appending: row-wise concat**

<sup>from [lib/core/nn/kv_cache.dart](../../lib/core/nn/kv_cache.dart):</sup>

```dart
Tensor appendK(int head, Tensor newK) {
  final prev = k[head];
  final next = prev == null
      ? newK
      : TensorConcat.concat([prev, newK], axis: 0);
  k[head] = next;
  return next;
}
```

`appendK` glues the new K rows onto the previously cached ones along
the sequence axis. If the cache is empty (`prev == null`) — that's
the prompt-fill case — it just stores the incoming tensor. Otherwise
it produces `[T + N_new, headDim]`.

`appendV` is identical in structure. The two are kept separate
(rather than a single `appendKV`) so the per-head loop in
`MultiHeadAttention` can call them at the natural point in the
projection sequence.

## **6.4. `EncoderCache` — one bag for the whole stack**

<sup>from [lib/core/nn/kv_cache.dart](../../lib/core/nn/kv_cache.dart):</sup>

```dart
class EncoderCache {
  final List<MHACache> layers;
  ...
  factory EncoderCache.empty(int numLayers, int numHeads) => EncoderCache(
    List<MHACache>.generate(
      numLayers,
      (_) => MHACache.empty(numHeads),
      growable: false,
    ),
  );

  int get seqLen => layers.isEmpty ? 0 : layers[0].seqLen;
}
```

Nothing clever — a list of per-layer caches, one entry per
transformer block. `TransformerEncoder.call` passes `cache.layers[i]`
to block `i` (see [chapter 5.5](./05-PRE-LN-TRANSFORMER-BLOCK.md)).

The overall `seqLen` is read from layer 0; every layer advances in
lockstep, so all layers have the same T after any successful
forward.

## **6.5. Using the cache in `MultiHeadAttention`**

The forward loop becomes:

<sup>from [lib/core/nn/attention/multi_head_attention.dart](../../lib/core/nn/attention/multi_head_attention.dart):</sup>

```dart
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
```

`q`, `k`, `v` are computed for the **new** tokens only. If a cache
is present, `k` and `v` get replaced by the full concatenated history
before hitting SDPA. The SDPA call then computes attention against
all `T + N_new` positions.

For single-token generation (`N_new == 1`):

- `q` is `[1, headDim]`.
- `k` and `v` become `[T + 1, headDim]` (post-append).
- Scores `q @ k.T` is `[1, T + 1]` — one row.
- Output is `[1, headDim]`.

Work per generation step: `O(T)` instead of `O(T^2)`. Over the
course of generating `N` tokens: `O(N^2)` total instead of `O(N^3)`.

## **6.6. Two modes at the GPT level**

<sup>from [lib/core/nn/gpt.dart](../../lib/core/nn/gpt.dart):</sup>

```dart
Tensor _forward(Tensor tokens, {required int startPos, EncoderCache? cache}) {
  ...
  final n = tokens.shape.last;
  var h = tokenEmb(tokens);
  h = posEmb(h, startPos: startPos);
  h = embedDrop(h);
  final mask = n > 1 ? causalMask(n, device: h.device) : null;
  h = encoder(h, mask: mask, cache: cache);
  ...
}
```

The `startPos` argument is what tells `posEmb` to start its
positional counter from `T` rather than `0` when we're appending to
a cache — otherwise the new token would get the position-0
embedding every step.

The three regimes:

| Regime           | tokens shape | startPos | cache        | mask                |
|------------------|--------------|----------|--------------|---------------------|
| Training / eval  | `[N]` or `[B, N]` | `0`      | `null`       | `causalMask(N)`     |
| Prompt fill      | `[N]`       | `0`      | fresh empty  | `causalMask(N)`     |
| Single-token step| `[1]`       | `T`      | non-empty T-long | `null` (safe: only one query, all keys are past) |

The generation loop alternates: one prompt fill, then many
single-token steps.

## **6.7. Generation loop**

<sup>from [lib/core/nn/gpt.dart](../../lib/core/nn/gpt.dart):</sup>

```dart
List<double> _generateCached(
  List<double> prompt,
  int maxNewTokens,
  double temperature,
  int? topK,
  math.Random rng,
) {
  final v = config.vocabSize;
  final cache = EncoderCache.empty(config.numLayers, config.numHeads);
  final out = List<double>.of(prompt);

  // Initial prompt fill.
  final promptCtx = Tensor.fromList([prompt.length], prompt, ...);
  var logits = _forward(promptCtx, startPos: 0, cache: cache).toList();
  var lastBase = (prompt.length - 1) * v;
  var row = List<double>.generate(v, (i) => logits[lastBase + i]);
  var next = _sampleFromLogits(row, temperature, topK, rng);
  out.add(next.toDouble());

  // Autoregressive single-token steps.
  for (int step = 1; step < maxNewTokens; step++) {
    if (cache.seqLen >= config.maxCtx) break;
    final tokTensor = Tensor.fromList([1], [next.toDouble()], ...);
    logits = _forward(tokTensor, startPos: cache.seqLen, cache: cache).toList();
    row = List<double>.generate(v, (i) => logits[i]);
    next = _sampleFromLogits(row, temperature, topK, rng);
    out.add(next.toDouble());
  }
  return out;
}
```

Notice the shape of `logits` differs between the two calls:

- Prompt fill: `[N, V]` — we only care about the last row (position
  `N - 1` predicts token `N`).
- Cached step: `[1, V]` — the whole thing is the one prediction we
  want.

Also notice `cache.seqLen >= config.maxCtx` as the termination
check — once the cache is full, we can't append any more without
overflowing the learned positional embedding table.

The whole loop runs inside `Tensor.noGrad(...)` so no gradient graph
is ever built during generation. This is critical: without it, the
concat-append pattern would keep growing an autograd tape that
never gets released until the loop ends.

## **6.8. Memory footprint**

For a model with `L` layers, `H` heads, `Dh = D/H`, generating up
to `N` tokens:

    cache memory = 2 * L * H * N * Dh * 4 bytes
                 = 2 * L * N * D * 4 bytes
                 = 8 * L * N * D  bytes

The `2` is K and V. For a `tiny` GPT (`L = 2`, `D = 32`, `N = 32`)
that's 16 KB — negligible. For a `small` model (`L = 4`, `D = 128`,
`N = 64`) that's 262 KB. For real GPT-3-scale models (`L = 96`,
`D = 12288`, `N = 2048`) it becomes gigabytes and starts to dominate
memory. That's why production inference systems care so much about
KV cache compression.

---

<div align="right">

[&lt;&lt;&nbsp;&nbsp;Previous: PRE-LN TRANSFORMER BLOCK](./05-PRE-LN-TRANSFORMER-BLOCK.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: CROSS-ATTENTION&nbsp;&nbsp;&gt;](./07-CROSS-ATTENTION.md)

</div>
