# **Attention-Free Transformer Nuts and Bolts**

A walkthrough of the **Attention-Free Transformer** (AFT) as
implemented in `dart_pytorch`. AFT replaces the `softmax(QK.T) @ V`
core of standard attention (see the sister
[self-attention walkthrough](../self-attention/README.md)) with an
operator that never materialises an `[N, N]` similarity matrix, but
still lets every position influence every other one.

The reference is Zhai et al., "An Attention Free Transformer" (2021):
<https://arxiv.org/abs/2105.14103>.

## Where AFT lives in the codebase

| File                                                                                  | What it is                                             |
|---------------------------------------------------------------------------------------|--------------------------------------------------------|
| [lib/core/tensor/aft.dart](../../lib/core/tensor/aft.dart)                            | The `TensorAft.aftFull` op + custom backward + `sliceTopLeft` |
| [lib/core/nn/attention/aft_attention.dart](../../lib/core/nn/attention/aft_attention.dart) | `AFTAttention` module (`Linear` Q/K/V + learned `posBias`) |
| [lib/core/nn/aft_transformer.dart](../../lib/core/nn/aft_transformer.dart)            | `AFTBlock` + `AFTLanguageModel`                        |
| [bin/aft_demo.dart](../../bin/aft_demo.dart)                                          | Side-by-side speed/loss demo vs. `TransformerLM`       |
| [test/aft_test.dart](../../test/aft_test.dart)                                        | Numerical & gradient tests                             |

## Chapters

[0. INTUITION](./00-INTUITION.md)
<br>
[1. THE AFT OPERATOR](./01-THE-AFT-OPERATOR.md)
<br>
[2. POSITION BIAS](./02-POSITION-BIAS.md)
<br>
[3. CAUSAL MASKING](./03-CAUSAL-MASKING.md)
<br>
[4. NUMERICAL STABILITY](./04-NUMERICAL-STABILITY.md)
<br>
[5. CUSTOM BACKWARD](./05-CUSTOM-BACKWARD.md)
<br>
[6. AFTAttention MODULE](./06-AFT-ATTENTION-MODULE.md)
<br>
[7. AFT BLOCK AND LM](./07-AFT-BLOCK-AND-LM.md)
<br>
[8. CONCLUSION](./08-CONCLUSION.md)

## Prerequisites

Read the [self-attention walkthrough](../self-attention/README.md)
first if the words "Q, K, V", "softmax over context", or "causal
mask" don't already mean something. This walkthrough deliberately
draws the contrast rather than re-explaining the basics.

## Notation

Matches the self-attention docs:

| Symbol            | Meaning                                              |
|-------------------|------------------------------------------------------|
| `T` or `N`        | Sequence length (AFT uses `T` in its formula)        |
| `D` or `embedDim` | Model width; each token is a `D`-dim vector          |
| `t`               | Output position index (query position)               |
| `t'`              | Context position index                               |
| `d`               | Feature-axis index within `D`                        |
| `W`               | Learned `[T, T]` position-bias matrix                |

Tensors are described row-major as `[dim0, dim1, ...]`. All AFT
tensors in this codebase are 2D (`[T, D]` or `[T, T]`) — no leading
batch dim.

---

<div align="right">

[&lt;&lt;&nbsp;&nbsp;Home: repo README](../../README.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: INTUITION&nbsp;&nbsp;&gt;](./00-INTUITION.md)

</div>
