# **7. CROSS-ATTENTION**

Self-attention has `Q`, `K` and `V` all derived from the same
sequence. **Cross-attention** derives `Q` from one sequence and
`K`, `V` from another. Same math, two data streams.

Where does this come up?

- **Encoder-decoder translation** — decoder queries attend to encoder
  outputs.
- **Whisper / audio-conditioned models** — text decoder queries
  attend to audio-encoder outputs.
- **Perceiver / retrieval-augmented models** — learned queries
  attend to external memory.

`dart_pytorch` supplies `MultiHeadCrossAttention` for exactly this
pattern.

## **7.1. Class layout**

<sup>from [lib/core/nn/attention/multi_head_cross_attention.dart](../../lib/core/nn/attention/multi_head_cross_attention.dart):</sup>

```dart
class MultiHeadCrossAttention extends Module {
  final int embedDim;
  final int kvEmbedDim;
  final int numHeads;
  final int headDim;
  final List<Linear> wq;
  final List<Linear> wk;
  final List<Linear> wv;
  final Linear wo;
  ...

  MultiHeadCrossAttention(
    this.embedDim,       // width of the query side
    this.kvEmbedDim,     // width of the memory (K/V) side
    this.numHeads, {
    ...
  }) : headDim = embedDim ~/ numHeads,
       wq = List<Linear>.generate(
         numHeads,
         (h) => Linear(embedDim, embedDim ~/ numHeads, ...),
       ),
       wk = List<Linear>.generate(
         numHeads,
         (h) => Linear(kvEmbedDim, embedDim ~/ numHeads, ...),
       ),
       wv = List<Linear>.generate(
         numHeads,
         (h) => Linear(kvEmbedDim, embedDim ~/ numHeads, ...),
       ),
       wo = Linear(embedDim, embedDim, ...);
```

Two things differ from `MultiHeadAttention`:

- Two width parameters: `embedDim` (the decoder / query side) and
  `kvEmbedDim` (the encoder / memory side). They can be different.
- `wk` and `wv` project from `kvEmbedDim` (memory width) into
  `headDim`, while `wq` projects from `embedDim` (query width) into
  `headDim`. The **output** `headDim` matches on both sides so
  `Q @ K.T` is well-defined.

## **7.2. Shapes**

    xq   : [Sq, embedDim]        (or [B, Sq, embedDim])
    xkv  : [Skv, kvEmbedDim]     (or [B, Skv, kvEmbedDim])
    out  : [Sq, embedDim]        (or [B, Sq, embedDim])

The output has one row per **query** position, keeping the query
side's width. `Sq` and `Skv` can differ. `embedDim` and
`kvEmbedDim` can differ. The **only** dimension that must line up
is `headDim` on the Q side and K side (guaranteed by construction —
they both project into `embedDim ~/ numHeads`).

## **7.3. Forward pass**

Structurally identical to self-attention, except the inputs to the
K and V projections come from a different tensor:

<sup>from [lib/core/nn/attention/multi_head_cross_attention.dart](../../lib/core/nn/attention/multi_head_cross_attention.dart):</sup>

```dart
final heads = <Tensor>[];
for (int h = 0; h < numHeads; h++) {
  final q = wq[h](xq);            // from query side
  final k = wk[h](xkv);           // from memory side
  final v = wv[h](xkv);
  heads.add(q.scaledDotProductAttention(k, v));
}
```

Everything downstream (concat, `wo`, dropout) matches
`MultiHeadAttention` line-for-line.

## **7.4. No mask, no cache**

Two intentional differences from self-attention:

**Cross-attention never causal-masks.** The whole point is that the
decoder can see the entire encoder output at every step. There is no
"future" in the memory to hide — the memory was computed once from a
complete input sequence before decoding even started.

**Cross-attention doesn't use a KV cache.** The memory is fixed for
the whole decoding pass — the encoder ran once and its output
doesn't change per decoder step. So `K` and `V` are computed once,
then reused verbatim for every step. There's nothing to append.

If you look at how `TransformerDecoderBlock` uses cross-attention,
you'll see it never passes a mask or a cache to `crossAttn`:

<sup>from [lib/core/nn/transformer_decoder_block.dart](../../lib/core/nn/transformer_decoder_block.dart):</sup>

```dart
h1 = x     + dropout(selfAttn(ln1(x), causalMask))
h2 = h1    + dropout(crossAttn(ln2(h1), memory))
out = h2   + dropout(ffn(ln3(h2)))
```

Three pre-LN sub-layers per decoder block:

1. Masked self-attention over the decoder input (causal — no cheating).
2. **Unmasked** cross-attention where the decoder queries pull from
   the encoder's memory.
3. Feed-forward.

Compare to the encoder block ([chapter 5](./05-PRE-LN-TRANSFORMER-BLOCK.md))
which has only sub-layers 1 and 3, and its self-attention is
unmasked (the encoder sees its whole input all at once).

## **7.5. Where the parameter count goes**

Cross-attention has the same `4 * embedDim^2`-ish parameter budget
as self-attention when `kvEmbedDim == embedDim`. When
`kvEmbedDim != embedDim` the accounting shifts:

    wq : numHeads * embedDim   * headDim = embedDim^2
    wk : numHeads * kvEmbedDim * headDim = kvEmbedDim * embedDim
    wv : numHeads * kvEmbedDim * headDim = kvEmbedDim * embedDim
    wo : embedDim * embedDim              = embedDim^2

Total: `2 * embedDim^2 + 2 * kvEmbedDim * embedDim`. If your encoder
runs a much smaller model than your decoder (`kvEmbedDim <
embedDim`), the K/V matrices are cheap — a nice property for
retrieval-style architectures where the "memory" side is a compressed
database representation.

## **7.6. Why the same op works twice**

`scaledDotProductAttention` doesn't care where its arguments came
from. It sees three 2D tensors and does the same four operations. The
distinction between "self" and "cross" attention lives entirely in
**which tensor** you pass in — which is why the same op file
(`lib/core/tensor/attention.dart`) serves both `MultiHeadAttention`
and `MultiHeadCrossAttention` without a single conditional. All the
interesting differences are pushed up to the module layer, where they
belong.

---

<div align="right">

[&lt;&lt;&nbsp;&nbsp;Previous: KV CACHE FOR GENERATION](./06-KV-CACHE-FOR-GENERATION.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: CONCLUSION&nbsp;&nbsp;&gt;](./08-CONCLUSION.md)

</div>
