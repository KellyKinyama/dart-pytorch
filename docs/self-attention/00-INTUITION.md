# **0. INTUITION**

Before any code: what problem does self-attention actually solve, and
why is it the block that ate the deep-learning world?

## **0.1. The task: contextualize every token**

Say we have a sentence of `N` tokens, each already turned into a
`D`-dimensional embedding by a lookup table. Call the input
`X` of shape `[N, D]`.

The problem is that `X[i]` (the embedding of token `i`) doesn't know
anything about the other tokens yet. The word "bank" alone is
ambiguous; "river bank" and "bank account" want two very different
vectors.

We want an operation that produces `Y` of the same shape `[N, D]`
where every row `Y[i]` is a **context-aware** rewrite of `X[i]` — a
mix of every relevant token in the sequence, with the mixing weights
computed **from the data itself**.

## **0.2. Why not just average all tokens?**

A trivial contextualizer would be `Y[i] = mean(X[0..N-1])`. Every
row would then know about every other token. But the mix is
uniform — the word "bank" would be blended with "the", "of", "over"
just as strongly as with "river". Useless.

We want the weights `w[i, j]` — how much position `i` pulls in from
position `j` — to depend on **both** `X[i]` and `X[j]`. If those two
embeddings "match" on some semantic axis, `w[i, j]` should be large.
Otherwise it should be small.

## **0.3. The dot-product trick**

The cheapest way to score "do these two vectors match?" is their dot
product. If `X[i] . X[j]` is large and positive, the two tokens
agree on many dimensions. If it's zero, they're orthogonal.

So a first draft of self-attention is:

```text
scores[i, j] = X[i] . X[j]         # [N, N]
w[i]         = softmax(scores[i])  # each row sums to 1
Y[i]         = sum_j w[i, j] * X[j]
```

In matrix form:

```text
S = X @ X.T          # [N, N]
A = softmax(S)       # [N, N], row-wise
Y = A @ X            # [N, D]
```

That's it. Every row of `Y` is a soft-selected mixture of the rows
of `X`, with softmax turning raw scores into proper weights.

## **0.4. Why we split into Q, K, V**

The formulation above conflates three roles that the model actually
wants to keep distinct:

- **Query** — "what am I looking for?" (position `i` asks a question)
- **Key**   — "what do I advertise?" (position `j` publishes a summary)
- **Value** — "what do I hand over if I get chosen?" (position `j`'s payload)

Nothing forces these to live in the same subspace as the raw token
embedding. So we let the model learn three separate linear
projections:

```text
Q = X @ Wq     # [N, D] -- one query per token
K = X @ Wk     # [N, D] -- one key per token
V = X @ Wv     # [N, D] -- one value per token

Y = softmax(Q @ K.T / sqrt(D)) @ V
```

`Wq`, `Wk`, `Wv` are trainable `[D, D]` matrices. `Q @ K.T` is the
`[N, N]` compatibility matrix, and the `1/sqrt(D)` factor keeps the
scores from blowing up as `D` grows (more on that in
[chapter 2](./02-SCALED-DOT-PRODUCT-ATTENTION.md)).

The value projection `V` is what actually flows out. Splitting keys
from values lets the model use one representation to **decide** who
to attend to and a different representation to **carry** the
information. This is a surprisingly powerful separation.

## **0.5. Why "self" attention?**

Because `Q`, `K`, `V` all come from the same tensor `X`. The
sequence attends to itself.

If instead `Q` comes from a decoder and `K`, `V` come from an encoder,
you get **cross-attention** — the same math with different sources
(covered in [chapter 7](./07-CROSS-ATTENTION.md)).

## **0.6. What comes next**

The remaining chapters map this intuition onto the actual Dart
implementation:

- [Chapter 1](./01-QKV-AND-LINEAR-PROJECTIONS.md) — how the three
  projections show up as `Linear` layers.
- [Chapter 2](./02-SCALED-DOT-PRODUCT-ATTENTION.md) — the four-line
  `scaledDotProductAttention` op that runs it all.
- [Chapter 3](./03-CAUSAL-MASK.md) — the additive `-1e9` trick that
  turns a bidirectional sequence into a causal (language-model) one.
- [Chapters 4-5](./04-MULTI-HEAD-ATTENTION.md) — why we split `D`
  into `H` parallel subspaces and how a full transformer block wires
  attention, LayerNorm and a feed-forward together.
- [Chapters 6-7](./06-KV-CACHE-FOR-GENERATION.md) — the two big
  practical extensions: KV caching for O(N) generation, and
  cross-attention for encoder-decoder models.

---

<div align="right">

[&lt;&lt;&nbsp;&nbsp;Previous: README](./README.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: Q, K, V AND LINEAR PROJECTIONS&nbsp;&nbsp;&gt;](./01-QKV-AND-LINEAR-PROJECTIONS.md)

</div>
