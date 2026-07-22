# **5. PRE-LN TRANSFORMER BLOCK**

Multi-head attention on its own is not a transformer. A **transformer
block** wraps attention in two extras that stabilise training and
give the model non-linear expressive power:

- **LayerNorm** before each sub-layer (the "pre-LN" recipe).
- A **feed-forward network** (position-wise MLP) after attention.

Both wrapped in a **residual connection** and an optional dropout.

## **5.1. The block**

<sup>from [lib/core/nn/transformer.dart](../../lib/core/nn/transformer.dart):</sup>

```dart
Tensor call(Tensor x, {Tensor? mask, MHACache? cache}) {
  final h = x + dropout(mha(ln1(x), mask: mask, cache: cache));
  final ff = ffn2(ffn1(ln2(h)).relu());
  return h + dropout(ff);
}
```

Two lines of forward. Read it as:

    h1 = x  + dropout(   MHA(  LayerNorm(x)  )   )
    h2 = h1 + dropout(   FFN(  LayerNorm(h1) )   )
    return h2

Both sub-layers share the same "pre-norm residual" pattern:

    output = input + dropout( SubLayer( LayerNorm( input ) ) )

Every arrow that flows out of the block has taken exactly one
gradient hop through a `+` (identity in the backward pass). That's
what makes pre-LN transformers trainable without warmup schedules or
learning-rate babysitting: gradients get to the earliest tokens by
walking straight down the residual stream.

## **5.2. Pre-LN vs post-LN**

The original 2017 transformer paper put LayerNorm **after** the
residual add:

    h = LayerNorm( x + MHA(x) )        # post-LN, original

This is unstable at depth: every LayerNorm re-standardises the
residual stream, so the direct-path signal keeps getting squashed.
Deep post-LN transformers need learning-rate warmup and even then
diverge easily.

Pre-LN — LayerNorm **on the branch, not the residual** — is what
GPT-2, GPT-3, LLaMA and essentially every modern transformer use:

    h = x + MHA( LayerNorm(x) )        # pre-LN, modern

Now the residual stream is untouched by any norm; only the
sub-layer's **input** is normalized. Gradient magnitudes stay bounded
and training just works.

`dart_pytorch` implements only the pre-LN form.

## **5.3. The feed-forward network**

<sup>from [lib/core/nn/transformer.dart](../../lib/core/nn/transformer.dart):</sup>

```dart
ffn1 = Linear(embedDim, ffnDim ?? embedDim * 4, ...),
ffn2 = Linear(ffnDim ?? embedDim * 4, embedDim, ...),
...
final ff = ffn2(ffn1(ln2(h)).relu());
```

The FFN is a two-layer MLP: expand to `ffnDim` (default `4 *
embedDim`), apply ReLU, contract back to `embedDim`. Applied
**position-wise** — the same weights act on every token independently.

Why the 4x expansion? It's the ratio that empirically balances
capacity against parameter count. About two-thirds of a transformer's
parameters live in these two matmuls (`2 * embedDim * 4 * embedDim
= 8 * embedDim^2` per block vs. `4 * embedDim^2` in the attention
projections).

The FFN is where the model does most of its "computation" — the
attention layer only moves information between positions; the FFN
transforms it. Recent interpretability work (e.g. Meng et al.'s ROME)
localizes factual knowledge to specific FFN neurons.

## **5.4. Why the extra dropout?**

Two dropouts appear in the forward:

    h = x + dropout(mha(...))
    y = h + dropout(ffn(...))

They regularize the **branch outputs** before they merge with the
residual. Not the residual itself — dropping activations on the
residual stream would keep zeroing out the token identity signal that
made it all the way down from the embedding.

The `MultiHeadAttention` module has its own `attnDropout` (also
optional, chapter 4). Both may be active simultaneously; the block's
dropout is what's guaranteed to run.

## **5.5. Stacking blocks: `TransformerEncoder`**

<sup>from [lib/core/nn/transformer_encoder.dart](../../lib/core/nn/transformer_encoder.dart):</sup>

```dart
Tensor call(Tensor x, {Tensor? mask, EncoderCache? cache}) {
  ...
  var h = x;
  for (int i = 0; i < blocks.length; i++) {
    h = blocks[i](h, mask: mask, cache: cache?.layers[i]);
  }
  if (finalNorm != null) {
    h = finalNorm!(h);
  }
  return h;
}
```

Every block is called with the **same** mask and its **own** cache
slot. The mask is shape-invariant across the stack (it depends on
`N`, not on the layer index), so we allocate it once at the top of
`GPT._forward` and reuse it.

`finalNorm` is a trailing LayerNorm applied **after** the last
block. It's a pre-LN quirk: the residual stream leaving the last
block hasn't been normalized (every prior norm was on the branch,
not on the trunk), so without a final norm the head sees an
un-normalized signal. Every serious pre-LN transformer ends with
this norm.

## **5.6. Putting it together: what one block actually does**

Given input `x` of shape `[N, D]`:

- `ln1(x)` — normalise every position's embedding to zero mean, unit
  variance across its `D` features.
- `mha(...)` — mix information across positions (attention).
- Residual add — keep the raw embedding around; attention only
  contributes an update.
- `ln2(h)` — normalise the updated positions.
- `ffn(...)` — mix information across features (an MLP on each
  position independently).
- Residual add — same story.

Attention moves stuff across the sequence axis. FFN moves stuff
across the feature axis. Alternate `L` times and you have the whole
architecture. It's stunningly little machinery for the amount of
work it does.

---

<div align="right">

[&lt;&lt;&nbsp;&nbsp;Previous: MULTI-HEAD ATTENTION](./04-MULTI-HEAD-ATTENTION.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: KV CACHE FOR GENERATION&nbsp;&nbsp;&gt;](./06-KV-CACHE-FOR-GENERATION.md)

</div>
