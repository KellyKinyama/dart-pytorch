# **8. CONCLUSION**

## **8.1. The whole AFT recipe on one screen**

For an input `x` of shape `[T, D]`:

```text
Per block:
    n  = LayerNorm(x)
    q  = Linear(n)                  # [T, D]
    k  = Linear(n)                  # [T, D]
    v  = Linear(n)                  # [T, D]
    w  = posBias[:T, :T]            # [T, T] learned bias

    for each output position t:
        for each feature d:
            # (mask: sum only over t' <= t)
            s     = K[:, d] + W[t, :]           # [T]
            a     = softmax(s)                  # [T], stable
            wv    = sum(a * V[:, d])            # scalar
            out[t, d] = sigmoid(Q[t, d]) * wv

    h  = x + dropout(out)
    n2 = LayerNorm(h)
    y  = h + dropout(FFN2(ReLU(FFN1(n2))))

Repeat L times, apply final LayerNorm, then Linear head to vocab.
```

That's the whole thing. The per-`d` softmax loop is what replaces
`Q @ K.T`. Everything around it — pre-LN, residual, FFN, LM head —
is identical to a standard transformer.

## **8.2. AFT vs. standard attention: the essential trade-off**

| Property                        | Standard `MultiHeadAttention` | `AFTAttention`                    |
|---------------------------------|-------------------------------|-----------------------------------|
| Pairwise weights depend on      | `Q[t]` **and** `K[t']`        | `K[t']` **and** `W[t, t']` only  |
| Content-dependent mixing?       | Yes (via `Q @ K.T`)           | Weakly (via `sigmoid(Q)` gate)   |
| Position-dependent mixing?      | Only via input embeddings     | Directly via learned `W`          |
| Softmax normalizer              | One per `(head, output position)` | One per `(output position, feature)` |
| `[T, T]` activation tensor      | Yes (`scores` + `attn`)        | No (streaming possible on GPU)   |
| Per-forward FLOPs               | `O(T^2 * D)`                  | `O(T^2 * D)`                     |
| Parameter count (attn only)     | `~4 * D^2`                    | `~3 * D^2 + T_max^2`             |
| Batched input support           | Yes                           | No (2D only in this repo)         |
| KV cache for O(N) generation    | Yes                           | No                                |
| Custom backward                 | No (auto)                      | Yes (hand-derived, chapter 5)     |

The one row that captures it: **standard attention has content-content
pairwise interactions; AFT has content-position pairwise
interactions plus an elementwise content gate**. That is the whole
architectural difference, and everything else (efficiency,
implementation, expressive limits) follows from it.

## **8.3. When is AFT the right choice?**

- **Short sequences.** For `T < 256` or so, the parameter penalty of
  `posBias` is manageable and the missing KV cache doesn't hurt
  because you'd re-run the whole context anyway.
- **Research / ablations.** Being able to swap out attention for an
  attention-free operator without changing the surrounding stack is
  a valuable degree of freedom for studying what attention actually
  contributes to a task.
- **When you want to demystify the transformer.** The AFT block is a
  useful teaching foil: "here's the same model but with the
  softmax'd `Q @ K.T` replaced by a per-feature softmax over
  content-agnostic biases — how much worse does it do?"

## **8.4. When is AFT NOT the right choice?**

- **Long context.** `posBias`'s `O(T^2)` parameter cost blows up.
  AFT-simple, AFT-conv, and AFT-local (paper variants, not
  implemented here) address this.
- **Inference-heavy workloads.** No KV cache means every generated
  token pays for re-running the whole prefix. Standard attention
  wins by orders of magnitude in `tokens/second`.
- **Batched training at scale.** The 2D-only limitation forces
  per-sequence forward passes, throwing away the batched matmul
  parallelism that GPUs are built for. `MultiHeadAttention` has a
  batched path; `AFTAttention` doesn't.

For anything past "toy language model on a laptop", the standard
attention module in this repo will outperform. AFT is a fascinating
alternative, not a replacement.

## **8.5. What the paper does that we don't**

The 2021 Zhai et al. paper introduces four variants; this repo
implements only the first:

- **AFT-full** (this repo) — the `[T, T]` learned `W`. Most
  expressive, worst parameter scaling.
- **AFT-simple** — `W` factored as `w_i * w_j` (two length-`T`
  vectors). `O(T)` parameters instead of `O(T^2)`.
- **AFT-conv** — `W` restricted to a convolutional structure with a
  bounded receptive field. `O(kernel_size)` parameters.
- **AFT-local** — `W` restricted to a windowed pattern. Similar in
  spirit to sliding-window attention.

Each variant is a further restriction on `W` that trades expressive
power for efficiency. All three could be added to this repo as
lightweight subclasses of `AFTAttention` — the op layer is
architecturally ready (just build a differently-shaped `w` tensor
in the module and pass it to `TensorAft.aftFull`), but the module
layer would need small tweaks.

## **8.6. Reading order recap**

If you want to re-read this walkthrough in dependency order:

1. [Softmax](../../lib/core/tensor/softmax.dart) — the row-wise, max-subtracted primitive AFT reimplements inline.
2. [`TensorAft.aftFull` forward](../../lib/core/tensor/aft.dart) — three-sweep per-`d` softmax + weighted sum + sigmoid gate.
3. [`sliceTopLeft`](../../lib/core/tensor/aft.dart) — sub-region view with a proper backward.
4. [`AFTAttention`](../../lib/core/nn/attention/aft_attention.dart) — three `Linear` projections + owned `posBias` tensor.
5. [`AFTBlock`](../../lib/core/nn/aft_transformer.dart) — pre-LN residual wrapping (identical shape to `TransformerBlock`).
6. [`AFTLanguageModel`](../../lib/core/nn/aft_transformer.dart) — embedding + sinusoidal encoding + `numLayers * AFTBlock` + final norm + LM head.
7. [`bin/aft_demo.dart`](../../bin/aft_demo.dart) — training-loss + wall-clock comparison with `TransformerLM`.

## **8.7. Parting observation**

The interesting thing about AFT is not that it's "faster" (it isn't,
in the current implementation). It's that it demonstrates
**attention isn't magic**. The essential ingredients of a
transformer — a mixing operator across positions, a nonlinearity,
LayerNorm, and residuals — can be reassembled from very different
primitives. AFT keeps the residual scaffolding and swaps in a
mixer that has softmax's normalization properties but very
different structural priors.

The lesson is architectural rather than practical: think of "the
transformer" as a **template** (pre-LN + residual + some
token-mixer + some feature-mixer), of which softmax-attention is
one instance. AFT is another. There are many more waiting to be
tried.

Happy attending — even without attention.

---

<div align="right">

[&lt;&lt;&nbsp;&nbsp;Previous: AFT BLOCK AND LM](./07-AFT-BLOCK-AND-LM.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Home: index&nbsp;&nbsp;&gt;&gt;](./README.md)

</div>
