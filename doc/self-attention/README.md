# **Self-Attention Nuts and Bolts**

An end-to-end walkthrough of the self-attention machinery in
`dart_pytorch`, starting from the raw dot product and ending at a
KV-cached GPT-style generation loop. Every chapter cites the actual
Dart file it explains, so you can read the code and the prose side by
side.

The path from a token to a next-token distribution touches:

- The scaled dot-product attention op ([lib/core/tensor/attention.dart](../../lib/core/tensor/attention.dart))
- The row-wise softmax it depends on ([lib/core/tensor/softmax.dart](../../lib/core/tensor/softmax.dart))
- Additive causal masks ([lib/core/nn/masks.dart](../../lib/core/nn/masks.dart))
- Multi-head self-attention ([lib/core/nn/attention/multi_head_attention.dart](../../lib/core/nn/attention/multi_head_attention.dart))
- Pre-LN transformer block ([lib/core/nn/transformer.dart](../../lib/core/nn/transformer.dart))
- KV cache for autoregressive generation ([lib/core/nn/kv_cache.dart](../../lib/core/nn/kv_cache.dart))
- Cross-attention for encoder-decoder setups ([lib/core/nn/attention/multi_head_cross_attention.dart](../../lib/core/nn/attention/multi_head_cross_attention.dart))
- Everything glued together as `GPT.generate` ([lib/core/nn/gpt.dart](../../lib/core/nn/gpt.dart))

## Chapters

[0. INTUITION](./00-INTUITION.md)
<br>
[1. Q, K, V AND LINEAR PROJECTIONS](./01-QKV-AND-LINEAR-PROJECTIONS.md)
<br>
[2. SCALED DOT-PRODUCT ATTENTION](./02-SCALED-DOT-PRODUCT-ATTENTION.md)
<br>
[3. CAUSAL MASK](./03-CAUSAL-MASK.md)
<br>
[4. MULTI-HEAD ATTENTION](./04-MULTI-HEAD-ATTENTION.md)
<br>
[5. PRE-LN TRANSFORMER BLOCK](./05-PRE-LN-TRANSFORMER-BLOCK.md)
<br>
[6. KV CACHE FOR GENERATION](./06-KV-CACHE-FOR-GENERATION.md)
<br>
[7. CROSS-ATTENTION](./07-CROSS-ATTENTION.md)
<br>
[8. CONCLUSION](./08-CONCLUSION.md)

## Prerequisites

You should already be comfortable with:

- Matrix multiplication, transpose and elementwise ops
- Softmax and cross-entropy loss
- The idea of autograd / reverse-mode differentiation

You do **not** need to know anything about transformers going in —
that's what the walkthrough is for.

## Notation used throughout

| Symbol         | Meaning                                                         |
|----------------|-----------------------------------------------------------------|
| `N`            | Sequence length (number of tokens the model sees at once)       |
| `D` or `embedDim` | Model width (each token is a vector of this length)          |
| `H` or `numHeads` | Number of attention heads                                    |
| `Dh` or `headDim` | Per-head width; always `D / H`                               |
| `V`            | Vocabulary size                                                 |
| `B`            | Batch size (optional; most 2D examples treat `B = 1`)           |

Tensors are described `[dim0, dim1, ...]` in row-major order — same
convention as PyTorch. `[N, D]` means "N rows of D-dimensional
vectors".

---

<div align="right">

[&lt;&lt;&nbsp;&nbsp;Home: repo README](../../README.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: INTUITION&nbsp;&nbsp;&gt;](./00-INTUITION.md)

</div>
