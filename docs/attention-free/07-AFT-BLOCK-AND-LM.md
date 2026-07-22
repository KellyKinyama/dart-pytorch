# **7. AFT BLOCK AND LM**

`AFTAttention` is one sub-layer. To turn it into a working model we
wrap it in the same pre-LN residual pattern used by the standard
transformer ([self-attention chapter 5](../self-attention/05-PRE-LN-TRANSFORMER-BLOCK.md))
and stack it with token/positional embeddings and an LM head.

## **7.1. `AFTBlock` — pre-LN attention + FFN**

<sup>from [lib/core/nn/aft_transformer.dart](../../lib/core/nn/aft_transformer.dart):</sup>

```dart
class AFTBlock extends Module {
  final int embedDim;
  final int maxSeqLen;
  final int ffnDim;

  final LayerNorm ln1;
  final LayerNorm ln2;
  final AFTAttention attn;
  final Linear ffn1;
  final Linear ffn2;
  final Dropout dropout;

  AFTBlock(
    this.embedDim, {
    required this.maxSeqLen,
    bool masked = false,
    int? ffnDim,
    double dropoutP = 0.0,
    Device device = Device.CPU,
    int seed = 0,
  }) : ffnDim = ffnDim ?? embedDim * 4,
       ln1 = LayerNorm(embedDim, device: device),
       ln2 = LayerNorm(embedDim, device: device),
       attn = AFTAttention(
         embedDim,
         maxSeqLen: maxSeqLen,
         masked: masked,
         device: device,
         seed: seed,
       ),
       ffn1 = Linear(embedDim, ffnDim ?? embedDim * 4,
         device: device, seed: seed + 4000),
       ffn2 = Linear(ffnDim ?? embedDim * 4, embedDim,
         device: device, seed: seed + 5000),
       dropout = Dropout(dropoutP);

  Tensor call(Tensor x) {
    final h = x + dropout(attn(ln1(x)));
    final ff = ffn2(ffn1(ln2(h)).relu());
    return h + dropout(ff);
  }
```

The `call` method is identical to `TransformerBlock` from the
self-attention docs, character-for-character except for `attn` being
an `AFTAttention` instead of a `MultiHeadAttention`:

    h  = x + dropout( AFTAttention( LayerNorm(x) ) )
    y  = h + dropout( FFN(          LayerNorm(h) ) )

Same pre-LN residual pattern, same 4x-expanding ReLU FFN. That's
the point: the "attention" sub-layer is swappable. Everything
around it stays exactly the same.

## **7.2. What differs from `TransformerBlock`?**

Two small things:

- **No `mask` parameter on `call`.** Standard `TransformerBlock`
  takes an optional `mask: Tensor?` — for AFT the masking is a
  boolean flag baked into the module at construction time
  (chapter 3). Once you pick `masked: true` when creating the
  block, every forward is causal.
- **No `cache` parameter.** AFT has no KV cache, so there's no
  optional cache to pass through.

Both simplifications reflect AFT's inflexibility relative to
standard attention — you get one behavior per instance, no runtime
switching. Which is fine for training but why AFT is a poor fit for
production LLM serving.

## **7.3. `AFTLanguageModel` — the whole stack**

<sup>from [lib/core/nn/aft_transformer.dart](../../lib/core/nn/aft_transformer.dart):</sup>

```dart
class AFTLanguageModel extends Module {
  final int vocabSize;
  final int embedDim;
  final int numLayers;
  final int maxLen;

  final Embedding tokenEmb;
  final SinusoidalPositionalEncoding posEnc;
  final List<AFTBlock> blocks;
  final LayerNorm finalLn;
  final Linear head;

  AFTLanguageModel({
    required this.vocabSize,
    required this.embedDim,
    required this.numLayers,
    required this.maxLen,
    int? ffnDim,
    double dropoutP = 0.0,
    Device device = Device.CPU,
    int seed = 0,
  }) : tokenEmb = Embedding(vocabSize, embedDim, device: device, seed: seed),
       posEnc = SinusoidalPositionalEncoding(embedDim),
       blocks = List<AFTBlock>.generate(
         numLayers,
         (i) => AFTBlock(
           embedDim,
           maxSeqLen: maxLen,
           masked: true,
           ffnDim: ffnDim,
           dropoutP: dropoutP,
           device: device,
           seed: seed + 100000 + i * 10000,
         ),
       ),
       finalLn = LayerNorm(embedDim, device: device),
       head = Linear(embedDim, vocabSize, device: device, seed: seed + 900000);
```

Six sub-modules:

1. **`tokenEmb`** — vocabulary lookup, `[V, D]`.
2. **`posEnc`** — sinusoidal positional encoding. Fixed
   (non-trainable). Added directly to the embeddings.
3. **`blocks`** — `numLayers` `AFTBlock`s, all `masked: true`.
4. **`finalLn`** — trailing LayerNorm (standard for pre-LN
   architectures).
5. **`head`** — projection to logits, `[embedDim, V]`.

## **7.4. Wait — sinusoidal AND `posBias`?**

Yes, both. Two independent sources of positional information:

- **Sinusoidal encoding** on the input embeddings — the classic
  Vaswani et al. add-a-sinusoid trick. Content-agnostic, non-trainable,
  identical for every model.
- **`posBias`** inside every AFT block — the learned `[T, T]` matrix
  described in chapter 2. Different per block, trainable, entirely
  data-driven.

Why both? Historically, AFT's `posBias` is what lets attention weights
depend on position (there's no `Q @ K.T` to encode relative
positions like RoPE does). But the *values* being mixed still benefit
from knowing where they came from, so the sinusoidal encoding on the
embeddings is retained.

A model without the sinusoidal encoding would still work — `posBias`
alone can carry positional information into the attention weights,
and the values would just be permutation-equivariant. The demo runs
with both because that's the paper's setup and it makes head-to-head
comparison with `TransformerLM` cleanest.

## **7.5. The forward**

<sup>from [lib/core/nn/aft_transformer.dart](../../lib/core/nn/aft_transformer.dart):</sup>

```dart
Tensor call(Tensor tokens) {
  if (tokens.shape.length != 1) {
    throw ArgumentError(
      'AFTLanguageModel: tokens must be 1D [seqLen]; got ${tokens.shape}',
    );
  }
  final n = tokens.shape[0];
  if (n > maxLen) {
    throw ArgumentError('AFTLanguageModel: seqLen $n exceeds maxLen $maxLen');
  }
  var x = tokenEmb(tokens);
  x = posEnc(x);
  for (final b in blocks) {
    x = b(x);
  }
  x = finalLn(x);
  return head(x);
}
```

Six ops, one for each sub-module:

1. Look up token embeddings — `[N] -> [N, D]`.
2. Add positional encoding — `[N, D] -> [N, D]`.
3. Run through every block sequentially — `[N, D] -> [N, D]`.
4. Final LayerNorm — `[N, D] -> [N, D]`.
5. Project to logits — `[N, D] -> [N, V]`.

Input is **strictly 1D** — `AFTAttention`'s 2D-only restriction
propagates all the way up.

## **7.6. No `generate()` method**

The `AFTLanguageModel` has no autoregressive `generate()` method.
Reasons:

- No KV cache → every step would re-run the whole prefix.
- No incremental variant of `aftFull` → even a naive re-run
  approach would need shape adjustments per step.
- The demo compares training convergence, not generation.

If you want to sample from a trained `AFTLanguageModel`, you have to
build the loop yourself: call the model on `tokens[0..t-1]`, take
the last row of logits, sample a token, append, repeat. `O(T)` calls
each costing `O(T^2 * D)`, so `O(T^3 * D)` to generate `T` tokens.
Fine for demos with `T = 32`; painful for real generation.

## **7.7. Where `AFTLanguageModel` shows up**

Two callers:

- **`bin/aft_demo.dart`** — trains an `AFTLanguageModel` and a
  `TransformerLM` on the same 12-token counting task and reports
  loss/time side by side.
- **`lib/core/coop/lm_factory.dart`** — the `CoopLM` factory picks
  `AFTLanguageModel` when `--arch=aft` (see
  [docs/coop-training.md](../coop-training.md#two-model-families-archgptaft)).
  This is what lets the cooperative training scripts train an AFT
  model without any special-casing.

## **7.8. Model sizes preset by the CoopLM factory**

For reference, the two model sizes the coop scripts ship with map
onto `AFTLanguageModel` params like this:

| Preset | `maxLen` | `embedDim` | `numLayers` | `ffnDim` | Scalars   |
|--------|----------|-----------|-------------|----------|-----------|
| `tiny` | 32       | 32        | 2           | 128      | 26,776    |
| `small`| 64       | 128       | 4           | 512      | (larger)  |

The `tiny` GPT counterpart is 27,008 scalars — matched intentionally
so head-to-head comparisons control for parameter count. AFT is
slightly smaller because it avoids the per-head projection overhead
and the output projection `wo`, and the `maxLen = 32` case gives
`posBias` only 1024 scalars.

---

<div align="right">

[&lt;&lt;&nbsp;&nbsp;Previous: AFTAttention MODULE](./06-AFT-ATTENTION-MODULE.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: CONCLUSION&nbsp;&nbsp;&gt;](./08-CONCLUSION.md)

</div>
