# **8. CONCLUSION**

Self-attention, from the ground up in this repo, is:

**One 4-line op** — [lib/core/tensor/attention.dart](../../lib/core/tensor/attention.dart)

```dart
final scale = 1.0 / math.sqrt(dk);
var scores = matmul(k.transpose()) * scale;
if (mask != null) scores = scores + mask;
final attn = scores.softmax();
return attn.matmul(v);
```

**One 20-line module** — [lib/core/nn/attention/multi_head_attention.dart](../../lib/core/nn/attention/multi_head_attention.dart)
that runs that op `numHeads` times in parallel and concatenates.

**One 3-line block** — [lib/core/nn/transformer.dart](../../lib/core/nn/transformer.dart)
that wraps it in pre-LayerNorm + a feed-forward, both residual.

**One 50-line generation loop** — inside [lib/core/nn/gpt.dart](../../lib/core/nn/gpt.dart)
that uses a KV cache to keep per-step work linear.

Everything else is shape validation, mask conventions and cache
bookkeeping.

## **8.1. The essential recipe**

If you had to reconstruct the whole architecture from memory,
this is enough:

```text
For each transformer block, for input x of shape [N, D]:

    n = LayerNorm(x)                          # per-position normalize
    q, k, v = Linear(n), Linear(n), Linear(n) # each [N, D]
    scores = q @ k.T / sqrt(D)                # [N, N]
    scores = scores + causalMask              # additive -1e9 for j > i
    a = softmax(scores)                       # row-wise
    h = x + dropout(Linear(a @ v))            # residual after wo

    n = LayerNorm(h)
    y = h + dropout(Linear2(relu(Linear1(n))))# FFN + residual

Repeat L times, apply one final LayerNorm, then a linear head.
```

Add "split D into H heads, run in parallel, concatenate before wo"
and you have the actual implementation. Add "cache K/V during
autoregressive generation" and you have the fast inference path.

## **8.2. What we skipped**

Deliberately outside this walkthrough (present in the codebase or
obvious extensions):

- **Positional encodings** — [lib/core/nn/positional.dart](../../lib/core/nn/positional.dart)
  covers both sinusoidal and learned embeddings; RoPE / ALiBi would
  slot in the same place.
- **Weight tying** — `GPT` optionally shares the token embedding
  with the output head. See [lib/core/nn/gpt.dart](../../lib/core/nn/gpt.dart).
- **Attention-free variants** — [lib/core/nn/attention/aft_attention.dart](../../lib/core/nn/attention/aft_attention.dart)
  replaces `softmax(QK.T)V` with an `O(N)` alternative. Same block
  layout, different mixer.
- **Grouped-query attention (GQA) / MQA** — share `K`/`V` across
  heads to shrink the cache. A one-file change to `MultiHeadAttention`
  would be enough.
- **Flash attention** — a memory-efficient tiling of SDPA. A
  drop-in kernel replacement for `scaledDotProductAttention` on GPU.
- **Rotary position embeddings (RoPE)** — instead of adding a
  positional vector, rotate `Q` and `K` in a way that makes the dot
  product position-relative. Would sit in the per-head loop just
  before `scaledDotProductAttention`.

None of these change the story. They're all optimizations or
re-parameterizations of the same core.

## **8.3. Reading order recap**

If you want to re-read this walkthrough in dependency order — how
each piece depends on the previous:

1. [Softmax](../../lib/core/tensor/softmax.dart) — the row-wise, max-subtracted primitive.
2. [`scaledDotProductAttention`](../../lib/core/tensor/attention.dart) — softmax + two matmuls.
3. [`causalMask`](../../lib/core/nn/masks.dart) — additive `[N, N]` triangular tensor.
4. [`Linear`](../../lib/core/nn/linear.dart) — the Q/K/V/O projections.
5. [`MultiHeadAttention`](../../lib/core/nn/attention/multi_head_attention.dart) — `numHeads` parallel SDPAs + concat + `wo`.
6. [`LayerNorm`](../../lib/core/nn/layer_norm.dart) + [`Dropout`](../../lib/core/nn/dropout.dart) — the pre-LN wrapping.
7. [`TransformerBlock`](../../lib/core/nn/transformer.dart) — attention + FFN with residual.
8. [`TransformerEncoder`](../../lib/core/nn/transformer_encoder.dart) — stack of `L` blocks + final norm.
9. [`Embedding`](../../lib/core/nn/embedding.dart) + [`LearnedPositionalEmbedding`](../../lib/core/nn/positional.dart) — token + position lookup.
10. [`GPT`](../../lib/core/nn/gpt.dart) — the full model.
11. [`MHACache`, `EncoderCache`](../../lib/core/nn/kv_cache.dart) — inference-time state.
12. [`GPT.generate`](../../lib/core/nn/gpt.dart) — sampling loop that ties everything together.

## **8.4. A parting observation**

The transformer works so well not because attention is a magically
correct inductive bias, but because it makes almost **no** inductive
bias — it's a token-mixing operation with `O(N^2)` connections and
learned weights. The model finds structure in the data because
softmax is smooth, the residual path preserves gradients, and
gradient descent has enough freedom to learn what specialist heads
want to look at.

The lesson generalises: give the optimiser a flexible mixer
(attention, or its many variants), stack it deep, feed it enough
data, and get out of the way.

Happy attending.

---

<div align="right">

[&lt;&lt;&nbsp;&nbsp;Previous: CROSS-ATTENTION](./07-CROSS-ATTENTION.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Home: index&nbsp;&nbsp;&gt;&gt;](./README.md)

</div>
