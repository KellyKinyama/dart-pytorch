## Unreleased

- `GPT` — GPT-2 style causal language model, pure composition on top
  of `TransformerEncoder` (no new kernels).
  - `GPTConfig({vocabSize, maxCtx, embedDim, numLayers, numHeads,
    ffnDim, dropoutP, tieWeights = true, device, seed})` — small
    config object; `ffnDim` defaults to `4 * embedDim`.
  - `GPT(config)` — token embedding + `LearnedPositionalEmbedding` +
    embedding `Dropout` + `TransformerEncoder(finalNorm: true)` +
    output head. Head is **weight-tied** to the token embedding by
    default (`h @ W_e.T`, no separate `Linear`, no head bias). Set
    `tieWeights: false` to allocate a separate `Linear(embedDim,
    vocabSize, bias: false)` head. Because the tied weight tensor
    participates in both the embedding lookup and the head matmul,
    its gradient accumulates from both paths through the existing
    autograd graph — no bookkeeping needed.
  - `GPT.generate(prompt, {maxNewTokens, temperature, topK, rng})` —
    autoregressive sampler. `temperature <= 0` and `topK == 1` both
    collapse to greedy argmax; otherwise samples from a
    temperature-scaled, optionally top-k-filtered categorical.
    Truncates context to the trailing `maxCtx` tokens each step.
    Returns the full `prompt + generated` list of token indices
    (float32 convention). Puts the model in `eval()` for the duration
    and restores the previous `training` flag on exit.
  - `bin/gpt_demo.dart` — runnable char-level GPT demo. Overfits
    `"to be or not to be that is the question "` in a couple hundred
    Adam steps and prints greedy / T=0.8 / top-k=3 completions.
  - 12 new tests in `test/gpt_test.dart` covering forward shape,
    input validation, tied vs untied parameter counts, gradient
    accumulation through the tied weight, `train()/eval()`
    propagation into the embedding dropout and every block, an
    overfit convergence check, and `generate()`: length, greedy
    determinism, `topK=1` == argmax invariant, RNG reproducibility,
    context-window truncation, and training-flag restoration.
    Suite grows to 161 tests, all green.

- `TransformerEncoder` + `TransformerLM` + char-level LM demo — a full
  stackable transformer and a minimal language model built entirely by
  composition over the existing primitives (no new CUDA kernels).
  - `nn.TransformerEncoder(numLayers, embedDim, numHeads, {ffnDim,
    dropoutP, finalNorm = true, device, seed})` — stack of
    `TransformerBlock`s with an optional trailing `LayerNorm`
    (standard in pre-LN transformers). Forward accepts an optional
    `mask` passed through to every block's self-attention.
    `parameters()` / `submodules()` flatten in a stable order so
    optimizers see every block's params plus the final LN.
  - `nn.TransformerLM({vocabSize, embedDim, numLayers, numHeads,
    maxLen, ffnDim, dropoutP, device, seed})` — token embedding +
    sinusoidal positional encoding + causal encoder + linear head
    (`embedDim -> vocabSize`). Takes 1D `[seqLen]` token indices
    (stored as float32, same convention as `Embedding` /
    `crossEntropy`) and returns `[seqLen, vocabSize]` logits with a
    causal mask applied so position `i` only attends to `<= i`.
    Rejects non-1D input and `seqLen > maxLen`.
  - `nn.causalMask(n, {blockValue = -1e9, device})` — additive
    upper-triangular mask helper (`0` on-and-below diagonal,
    `blockValue` above). Non-trainable factory; supports CPU/GPU.
  - `bin/lm_demo.dart` — runnable char-level LM smoke test. Overfits
    the refrain `"hello world hello dart hello "` (vocab 10, len 28)
    in ~50 Adam steps with `clipGradNorm(1.0)`; reaches 100% next-char
    accuracy and prints a per-step loss trace. Invoke with
    `dart run bin/lm_demo.dart`.
  - 12 new tests in `test/transformer_lm_test.dart` covering the
    causal-mask shape / block-value / device options, encoder shape
    preservation, parameter counting, `finalNorm: false`, `train()`/
    `eval()` propagation into every block's dropout, gradient flow
    through every parameter under a mask, `TransformerLM` input
    validation, forward shape, parameter set, end-to-end overfit
    (loss < 0.1), and per-position greedy argmax matching the target.
    Suite grows to 149 tests, all green.

- Positional encodings — two `Module` variants for sequence models.
  - `nn.SinusoidalPositionalEncoding(embedDim)` — fixed, non-trainable
    encoding from "Attention Is All You Need". No parameters; the
    encoding table is recomputed per forward for the exact `seqLen`
    (`O(N * D)` — cheap enough that this avoids needing a slice op).
    Uploaded to the same device as the input before the residual add.
  - `nn.LearnedPositionalEmbedding(maxLen, embedDim)` — trainable
    `[maxLen, embedDim]` table wrapped as an `Embedding` submodule.
    Positions `0..seqLen-1` are gathered via the existing embedding
    op, so backward already scatter-adds into the position table.
    Rejects `seqLen > maxLen`.
  - Both take `[seqLen, embedDim]` and return `[seqLen, embedDim]`
    (single sequence — same 2D convention as the rest of the library).
  - 15 new tests in `test/positional_test.dart`: sinusoidal PE
    reference-formula check, additive semantics, gradient
    pass-through, seqLen-agnostic behavior, CPU/GPU parity;
    learned PE parameter shape, dim / maxLen validation, forward
    correctness, scatter-add gradient reaches only used rows,
    `train()/eval()` propagation; integration tests feeding both PE
    variants into a `TransformerBlock` (Adam-trained end-to-end).
    Suite grows to 137 tests, all green.

- `MultiHeadAttention` + `TransformerBlock` — first end-to-end
  transformer building block, pure composition over existing ops.
  - `TensorConcat.concat(tensors, {axis = 1})` — new op for last-axis
    2D concatenation. Backward slices the upstream gradient back into
    each input's column range. GPU inputs work by round-tripping
    through host memory (small MHA sizes make this acceptable; a
    fused GPU kernel can replace it later without changing the API).
    Enforces same device and matching row count.
  - `nn.MultiHeadAttention(embedDim, numHeads, {bias, dropoutP, device,
    seed})` — implemented as `numHeads` parallel per-head `Linear`
    projections for Q, K, V (each `embedDim -> headDim`), single-head
    SDPA per head, then `concat` + output `Linear`. Optional
    attention `Dropout` inherits `train()`/`eval()`. Rejects
    `embedDim % numHeads != 0`.
  - `nn.TransformerBlock(embedDim, numHeads, {ffnDim, dropoutP,
    device, seed})` — pre-LayerNorm encoder block:
    `h = x + dropout(mha(ln1(x)))`,
    `y = h + dropout(ffn2(relu(ffn1(ln2(h)))))`. Default
    `ffnDim = 4 * embedDim`. Every layer (LN, MHA, FFN, Dropout) is
    a submodule, so `block.eval()` / `block.train()` propagates
    through the whole subtree.
  - 17 new tests in `test/transformer_test.dart`: concat forward /
    device mismatch / row mismatch / axis / numerical-grad on
    weighted sum / GPU round-trip; MHA shape, `embedDim % numHeads`
    validation, end-to-end differentiability (every param gets a
    non-zero grad), synthetic Adam convergence, `train()/eval()`
    propagation to attention dropout; `TransformerBlock` shape and
    grad flow, deterministic output in `eval()` mode, Adam
    convergence on a synthetic target, and parameter count
    accounting. Suite grows to 122 tests, all green.

- Regularization — `Dropout` + gradient clipping + `Module` train/eval.
  - `Tensor.dropout(p, {training, rng})` — inverted dropout composed as
    `x * mask`, where `mask` is a plain constant tensor with entries
    `0` or `1/(1-p)`. Backward comes free from the existing multiply
    backward, so no custom gradient is needed. Works on CPU and GPU.
    Eval mode (or `p == 0`) is an identity that returns `this`.
    Accepts an optional seeded `math.Random` for reproducible masks.
  - `nn.Dropout(p, {rng})` — trivial module wrapper that reads its
    inherited `training` flag.
  - `Module` gained `bool training`, `train()`, `eval()`, and
    `submodules()`. `train()` / `eval()` recurse into `submodules()`
    so parent modules can toggle nested layers in one call.
    Existing modules (`LayerNorm`, `Linear`, `Embedding`) inherit
    the default (no submodules) — no behavior change.
  - `clipGradNorm(parameters, maxNorm)` in `core/optim/grad_utils.dart`
    — computes the global L2 norm across every non-null `.grad`, and
    if it exceeds `maxNorm`, rescales all grads in place using
    `Tensor.assign`. Returns the pre-clip norm for logging.
  - 14 new tests in `test/dropout_regularization_test.dart`: eval
    mode is identity, mask values are exactly `0` or `1/(1-p)`,
    expected mean is preserved over many draws, gradient flows only
    through survivors, GPU path works, `nn.Dropout.train()/eval()`
    switches behavior, parent `train()/eval()` propagates to
    submodules, and `clipGradNorm` no-op / scaling / multi-param /
    null-grad-skip / validation. Suite grows to 105 tests, all green.

- Scaled dot-product attention + `nn.Linear` — first attention primitive.
  - `Tensor.scaledDotProductAttention(k, v, {mask})` — 2D single-head
    SDPA composed from existing ops (`matmul`, `transpose`, scalar
    mul, `softmax`, `matmul`). No new kernel: autograd and CPU/GPU
    dispatch come free. `q` shape `[N, Dk]`, `k` shape `[M, Dk]`,
    `v` shape `[M, Dv]`, output `[N, Dv]`. Optional additive `mask`
    (same shape as `scores`) for causal / padding masks.
  - New `nn.Linear(inFeatures, outFeatures, {bias, device, seed})`
    module — `y = x @ W.T + b` with PyTorch-style Kaiming-uniform
    weight init (bound = `1 / sqrt(inFeatures)`) and matching bias
    init. `bias: false` trains weight only.
  - 9 new tests in `test/attention_test.dart`: SDPA forward
    correctness against a Dart reference (square + rectangular
    shapes), causal-mask semantics (row 0 sees only key 0),
    numerical-gradient check for `q` and `v`, Linear shape / forward
    correctness, Linear SGD regression convergence, and an end-to-end
    self-attention block (`Wq/Wk/Wv` + SDPA) trained with Adam.
    Suite grows to 91 tests, all green.

- Optimizers — `SGD` and `Adam` with per-parameter state, CPU + GPU.
  - New `Optimizer` base class in `lib/core/optim/optimizer.dart`.
  - `SGD(params, lr, {momentum = 0, weightDecay = 0})` — vanilla
    gradient descent with optional heavy-ball momentum and L2
    weight decay. Velocity buffers are allocated lazily on the
    first non-null grad and live on the same device as their param.
  - `Adam(params, {lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8,
    weightDecay = 0})` — Kingma & Ba with bias correction. Uses
    `pow(0.5)` for `sqrt(vHat)` so no new kernel is needed. Weight
    decay is decoupled (AdamW-style, applied post-update).
  - Updates never enter the autograd graph: they compose ops on
    `.detach()`ed tensors (so `_setBackward` never fires) and swap
    the resulting fresh tensor into each parameter via a new
    in-place `Tensor.assign(source)` helper (CPU: `Float32List.setAll`;
    GPU: dispose old handle, adopt source's handle). Parameter
    identity is preserved so the training-loop graph on the next
    forward pass sees the updated leaves.
  - 13 new tests in `test/optim_test.dart`: single-step formula
    checks against hand-computed reference, quadratic bowl
    convergence for both SGD and Adam, momentum accumulation over
    two steps, Adam bias-correction sanity (large grad yields update
    magnitude ~= lr), integration tests training `LayerNorm` gamma/beta
    with SGD and `Embedding` weights with Adam, plus `Tensor.assign`
    (CPU / GPU / shape mismatch / device mismatch). Suite grows to
    82 tests, all green.

- Softmax, fused cross-entropy, and embedding — CPU + GPU with autograd.
  - `Tensor.softmax()` — row-wise on 2D `[R, C]`, numerically stable
    (subtract row max). CPU inline; GPU `softmax_forward` /
    `softmax_backward` use block-per-row shared-memory reductions.
    Backward uses the standard `(gO - dot(gO, y)) * y` identity.
  - `Tensor.crossEntropy(targets)` — fused softmax + NLL. `this` is
    logits `[R, C]`, `targets` is `[R]` (integer class indices stored
    as float, rounded internally). Returns per-sample losses `[R, 1]`
    so callers control reduction with `.mean()` / `.sum()`. GPU
    `cross_entropy_forward` / `cross_entropy_backward` recompute
    softmax in backward (no cache); indices are treated as constants.
  - `Tensor.embedding(indices)` — table `[V, D]` gathered by `[N]` →
    `[N, D]`. Backward scatter-adds into a pre-allocated
    `table.grad` (atomicAdd on GPU); indices treated as constants.
    Out-of-range indices produce a zero row on both devices.
  - New `nn.Embedding(numEmbeddings, embeddingDim)` module — Gaussian
    init with std = `1 / sqrt(embeddingDim)` (Box–Muller, seedable),
    `call(indices) => weight.embedding(indices)`,
    `parameters() => [weight]`.
  - Six new FFI symbols: `softmax_forward`, `softmax_backward`,
    `cross_entropy_forward`, `cross_entropy_backward`,
    `embedding_forward`, `embedding_backward` (30 total).
  - Two new kernel headers: `lib/native/src/kernels/softmax.cuh`,
    `lib/native/src/kernels/embedding.cuh`.
  - 16 new tests in `test/softmax_ce_embedding_test.dart`: forward
    correctness against hand-computed reference, numerical gradient
    checks, CPU/GPU parity on forward and backward, scatter-add
    semantics for duplicated embedding indices, `nn.Embedding`
    module init and backward. Suite grows to 69 tests, all green.

- `LayerNorm` — first normalization op with full autograd on CPU + GPU.
  - `Tensor.layerNorm(gamma, beta, {eps})` extension on 2D `[R, C]`
    tensors (normalized over `C`).
  - CPU forward + backward inline (two-pass mean/variance; standard
    LayerNorm gradient formulas for `x`, `gamma`, `beta`).
  - GPU forward + backward via new `layernorm.cuh` kernels
    (`layernorm_fwd`, `layernorm_bwd` — block per row, shared-mem
    reduction; backward atomicAdds into pre-allocated `dGamma`/`dBeta`
    and writes a fresh `dX`).
  - Two new FFI symbols: `layernorm_forward`, `layernorm_backward`
    (24 total).
  - New minimal `nn` scaffold: `lib/core/nn/module.dart` exposes an
    abstract `Module { List<Tensor> parameters(); void zeroGrad(); }`.
    `lib/core/nn/layer_norm.dart` provides a trainable `LayerNorm(dim)`
    module (gamma=1, beta=0 init, `call(x)` forward).
  - 9 new tests in `test/layer_norm_test.dart`: manual forward
    correctness, numerical gradient check for `x` / `gamma` / `beta`,
    CPU/GPU parity on both forward and backward, module init and
    zeroGrad. Suite grows to 53 tests, all green.

- Reverse-mode autograd, Dart-side.
  - `Tensor.fromList` / `Tensor.fill` accept `requiresGrad: true`.
  - New `backward()`, `zeroGrad()`, `clone()`, `detach()`, plus a
    `grad` getter on every tensor.
  - Every op in `ops.dart` and `mat_mul.dart` sets a `_backward`
    closure when at least one input requires grad. Backward math is
    expressed as compositions of existing ops, so it runs on the input
    tensor's device automatically — no separate CPU/GPU gradient
    code paths.
  - Coverage: `+ - * /` (Tensor and `num` operands), matmul, sigmoid,
    tanh, log, pow, transpose, sum, mean — all CPU + GPU.
  - `relu` and `abs` backward are CPU-only (need `>0` / `sign` mask
    kernels); GPU calls throw with a `.to(Device.CPU)` hint.
  - Broadcast reduction handled for scalar (`length == 1`) case;
    row-broadcast backward `[1, N] <- [M, N]` is not wired yet
    (needs an axis-sum kernel).
  - 22 new tests in `test/autograd_test.dart`: analytic gradients for
    each op, numerical gradient check for sigmoid + tanh, CPU/GPU
    parity for matmul + sigmoid chains, and a linear-regression SGD
    loop that converges to the target weights.
- Wire GPU paths for every op that previously threw on GPU.
  - New `extern "C"` symbols in `lib/native/src/engine.cu` (18 added,
    22 total): `add_tensors`, `sub_tensors`, `mul_tensors`,
    `div_tensors`, `add_tensor_scalar`, `sub_tensor_scalar`,
    `mul_tensor_scalar`, `div_tensor_scalar`,
    `add_tensor_row_broadcast`, `abs_tensor`, `log_tensor`,
    `pow_tensor`, `relu_tensor`, `sigmoid_tensor`, `tanh_tensor`,
    `transpose_tensor`, `sum_tensor`, `mean_tensor`.
  - Pulled `kernels/elementwise.cuh` and `kernels/transpose.cuh`
    from `../dart_cuda` (forward slice only).
  - Full FFI bindings in `lib/core/tensor/cuda_engine.dart`.
  - `TensorOps` in `lib/core/tensor/ops.dart` now dispatches by
    input `Device`. Same-shape, scalar (`num` or `Tensor([1])`), and
    (for `+` only) `[1,N]` row-broadcast are supported on GPU;
    row-broadcast for `- * /` on GPU throws with a `.to(Device.CPU)`
    hint.
  - Test suite grows from 15 to 22 tests: added `GPU ops` group
    mirroring the CPU op tests plus a CPU/GPU parity chain
    (`((t * 2.0) + 1.5).relu().transpose()`).
- Device-aware `Tensor` with dual CPU/GPU backing.
  - `Tensor.fromList` and `Tensor.fill` now accept `device:`; default
    picks by size (threshold `Tensor.autoDeviceThreshold = 4096`).
  - Added `to(Device)` for explicit CPU<->GPU transfers.
  - `toList()` and `dispose()` are device-aware.
- CPU implementations for basic ops in `lib/core/tensor/ops.dart`:
  elementwise `+ - * /` (exact-shape, scalar, row broadcast), `relu`,
  `sigmoid`, `tanh`, `abs`, `log`, `pow`, 2D `transpose`, `sum`, `mean`.
- `matmul` now has both a CPU (naive loop) and GPU (tiled kernel)
  path; dispatch by input device. Mixed-device inputs raise
  `ArgumentError` (call `.to(...)` explicitly).
- Autograd scaffolding fields (`_backward`, `_children`,
  `requiresGrad`) reserved on `Tensor`; wiring is planned next.
- Add device-placement policy doc at `docs/device-placement.md`:
  per-op CPU vs GPU decisions, size thresholds, implementation
  status, and autograd-on-Dart recommendation.
- Wire up matrix multiplication end-to-end via `dart:ffi` + CUDA.
  - Add `Tensor` factories (`fromList`, `fill`), `toList()` readback,
    and `dispose()` for manual GPU cleanup.
  - Add `create_tensor`, `destroy_tensor`, `get_tensor_data`, and
    `matmul_tensors` C entry points in `lib/native/src/engine.cu`.
  - Pull in `kernels/common.cuh` + `kernels/matmul.cuh` from
    `../dart_cuda` (matmul-only subset).
  - Re-export `Tensor` from the package entry point.
  - Add correctness tests: 2x3 @ 3x2, 4x4 square, and non-tile-aligned
    33x17 @ 17x9 — all cross-checked against a naive CPU matmul.

## 1.0.0

- Initial version.
