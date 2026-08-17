# **6. AFTAttention MODULE**

`TensorAft.aftFull` (chapters 1-5) is the raw op. `AFTAttention` is
the thin `Module` wrapper that adds trainable `Linear` projections
for Q/K/V and owns the learnable position bias `W`. It's the
attention-side drop-in replacement for `MultiHeadAttention`.

## **6.1. Class layout**

<sup>from [lib/core/nn/attention/aft_attention.dart](../../lib/core/nn/attention/aft_attention.dart):</sup>

```dart
class AFTAttention extends Module {
  final int embedDim;
  final int maxSeqLen;
  final bool masked;

  final Linear wq;
  final Linear wk;
  final Linear wv;

  /// Learnable position-bias matrix, shape `[maxSeqLen, maxSeqLen]`.
  final Tensor posBias;

  AFTAttention(
    this.embedDim, {
    required this.maxSeqLen,
    this.masked = false,
    bool bias = false,
    Device device = Device.CPU,
    int seed = 0,
  }) : wq = Linear(embedDim, embedDim, bias: bias, device: device, seed: seed),
       wk = Linear(embedDim, embedDim,
         bias: bias, device: device, seed: seed + 1000),
       wv = Linear(embedDim, embedDim,
         bias: bias, device: device, seed: seed + 2000),
       posBias = _initPosBias(maxSeqLen, seed + 3000, device);
```

Note the differences vs. `MultiHeadAttention`:

- **No `numHeads`.** AFT is single-head by construction — there is
  no `Q @ K.T` to factor across heads, and adding heads would
  require re-thinking the per-`d` softmax. The paper's `AFT-full`
  variant is single-head; `AFT-simple` and `AFT-local` are also
  single-head. AFT-multi-head is not a standard variant.
- **No output projection `wo`.** Standard attention needs `wo` to
  mix information across heads. AFT has one head, so there's
  nothing to mix — the value projection `wv` alone determines the
  output subspace.
- **Q/K/V project `embedDim -> embedDim`**, not `embedDim ->
  headDim`. Fewer, larger `Linear` layers instead of many small
  ones.

Total parameters (attention only):

- `wq`, `wk`, `wv`: `3 * embedDim^2`
- `posBias`: `maxSeqLen^2`

Compared to standard multi-head attention's `4 * embedDim^2` (Q,K,V
+ output projection), AFT is one `embedDim^2` matrix cheaper on
projections but pays `maxSeqLen^2` for the bias. For short contexts
(`maxSeqLen < 2 * embedDim`), AFT is smaller overall. For long
contexts the position bias dominates (see
[chapter 2.5](./02-POSITION-BIAS.md#25-parameter-budget-w-can-dominate)).

## **6.2. Forward pass**

<sup>from [lib/core/nn/attention/aft_attention.dart](../../lib/core/nn/attention/aft_attention.dart):</sup>

```dart
Tensor call(Tensor x) {
  if (x.shape.length != 2) {
    throw ArgumentError(
      'AFTAttention: expected 2D input [T, embedDim]; got ${x.shape}',
    );
  }
  if (x.shape[1] != embedDim) {
    throw ArgumentError(
      'AFTAttention: last dim ${x.shape[1]} != embedDim $embedDim',
    );
  }
  final t = x.shape[0];
  if (t > maxSeqLen) {
    throw ArgumentError(
      'AFTAttention: seqLen $t exceeds maxSeqLen $maxSeqLen',
    );
  }
  final q = wq(x);
  final k = wk(x);
  final v = wv(x);
  final w = t == maxSeqLen ? posBias : TensorAft.sliceTopLeft(posBias, t, t);
  return TensorAft.aftFull(q, k, v, w, masked: masked);
}
```

The whole forward is:

1. **Shape check.** 2D only, last dim must be `embedDim`, sequence
   length must not exceed `maxSeqLen`.
2. **Project** to `q`, `k`, `v` — three `Linear` calls, each
   `[T, embedDim] -> [T, embedDim]`.
3. **Slice** the position bias down to the runtime `[T, T]` if
   `t < maxSeqLen`. If `t == maxSeqLen` we pass `posBias` directly
   to avoid an unnecessary copy.
4. **Run `aftFull`** with the `masked` flag chosen at construction.

There is no dropout applied inside the module (unlike
`MultiHeadAttention`'s `attnDropout`). Dropout gets added at the
block level ([chapter 7](./07-AFT-BLOCK-AND-LM.md)).

## **6.3. What it doesn't do**

Deliberate omissions vs. the standard-attention module:

- **No batched (3D) input.** Everything is `[T, D]` — single
  sequence. If you want a batch, iterate outside. This is the
  biggest usability gap vs. `MultiHeadAttention` and is called out
  in the module's docstring.
- **No KV cache.** Autoregressive generation with AFT would need
  either a per-token recomputation (`O(T^2)` total for `T` steps)
  or a variant of the op that supports incremental appends. Neither
  is implemented; the `AFTLanguageModel` doesn't have a `generate()`
  method for exactly this reason.
- **No cross-attention.** The op's shape check forces `q`, `k`, `v`
  to share `[T, D]`, so cross-attention (where `Q` and `K/V` come
  from different sequences) isn't supported. AFT wasn't designed
  for encoder-decoder settings.

## **6.4. Parameter registration**

<sup>from [lib/core/nn/attention/aft_attention.dart](../../lib/core/nn/attention/aft_attention.dart):</sup>

```dart
@override
List<Tensor> parameters() => [
  ...wq.parameters(),
  ...wk.parameters(),
  ...wv.parameters(),
  posBias,
];

@override
List<Module> submodules() => [wq, wk, wv];
```

Two things to notice:

- **`posBias` appears in `parameters()`** — it's directly the
  learnable tensor, not wrapped in a helper module. The optimizer
  sees it and updates it every step.
- **`posBias` is NOT in `submodules()`** — because it's a `Tensor`,
  not a `Module`. Submodule traversal is used for `train()` /
  `eval()` mode switching (which affects dropout, etc.), and
  `posBias` has no such state.

This is the idiomatic way to add a learnable tensor to a `Module`
without inventing a `Parameter` wrapper class.

## **6.5. Comparing to `MultiHeadAttention`**

| Aspect                    | `MultiHeadAttention`               | `AFTAttention`                           |
|---------------------------|------------------------------------|------------------------------------------|
| Positional information    | External (positional embedding on the input) | Internal (`posBias` per position pair) |
| Number of heads           | Configurable, typically 4-32       | Always 1                                 |
| Output projection         | Yes (`wo`)                         | None                                     |
| Batched input             | Yes (3D `[B, T, D]`)               | No (2D only)                             |
| KV cache                  | Yes (see [self-attention ch. 6](../self-attention/06-KV-CACHE-FOR-GENERATION.md)) | No                                       |
| Cross-attention variant   | Yes (`MultiHeadCrossAttention`)    | No                                       |
| Mask mechanism            | Additive `[T, T]` tensor           | `masked: bool` flag on the op            |
| Custom backward           | No (composes from primitives)      | Yes (chapter 5)                          |
| Device support            | CPU + GPU                          | CPU + GPU                                |

The last row is the one that most surprises people who read the AFT
paper thinking "GPU-heavy": this Dart implementation has parity
across devices for both attention families, so you can honestly
benchmark them on either.

## **6.6. Where the module is used**

Two consumers in the codebase:

- **`AFTBlock`** in [lib/core/nn/aft_transformer.dart](../../lib/core/nn/aft_transformer.dart) —
  the pre-LN wrapper (next chapter). Every block owns its own
  `AFTAttention`, so a `numLayers` model has `numLayers`
  independent `posBias` tensors.
- **`bin/aft_demo.dart`** — the standalone demo that trains an
  `AFTLanguageModel` and a `TransformerLM` on the same task.

No other module or script imports `AFTAttention` directly. The
`CoopLM` factory ([lib/core/coop/lm_factory.dart](../../lib/core/coop/lm_factory.dart)) builds it via
`AFTLanguageModel`, not via `AFTAttention` directly.

---

<div align="right">

[&lt;&lt;&nbsp;&nbsp;Previous: CUSTOM BACKWARD](./05-CUSTOM-BACKWARD.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: AFT BLOCK AND LM&nbsp;&nbsp;&gt;](./07-AFT-BLOCK-AND-LM.md)

</div>
