# Device placement policy

**Goal.** Decide, per operation, whether to run on CPU or GPU — and make
gradient tracking work cleanly across both.

Status legend used below:
- **impl** — implemented in this repo
- **stub** — declared but not wired (throws `UnimplementedError`)
- **planned** — not started

## Guiding principles

1. **Don't ping-pong.** Every H↔D copy costs ~2 GB/s of PCIe bandwidth
   plus a synchronization stall. If a value is already on GPU, the next
   op should stay on GPU unless the CPU win is >10× the copy cost.
2. **Amortize the kernel launch.** A CUDA launch is ~5–20 µs. Any op
   with fewer than ~10k FLOPs is faster on CPU (loop + Float32List).
3. **Autograd tape is Dart-side and device-agnostic.** The graph is
   built in Dart; each backward closure is composed of the same ops as
   the forward pass, so gradients run on whichever device the tensors
   already live on. No separate CPU/GPU gradient code paths.
4. **Default device at construction** follows size at the factory,
   independent of op family. Threshold: `Tensor.autoDeviceThreshold`
   (4096 elements). Explicit `Tensor.fromList(..., device: X)` and
   `.to(Device.X)` are the escape hatches.
5. **Ops respect the input tensor's device.** No hidden promotion.
   Mixed-device inputs to a binary op raise an error — call `.to(...)`
   yourself. This keeps performance costs visible and simplifies the
   future autograd tape.

## Per-op decisions

Reference ops are the `DLLEXPORT` symbols in
`../dart_cuda/native/src/engine.cu`.

### matmul — device-respecting

| Path             | Status | Notes                                                   |
| ---------------- | ------ | ------------------------------------------------------- |
| CPU              | impl   | Naive triple loop; fine for `n <= ~64`                  |
| GPU (fwd)        | impl   | Tiled 32x32 kernel `matmul_fwd`                         |
| GPU (bwd)        | planned| Kernels are compiled (`matmul_bwd_dA_tiled`, `_dB_tiled`) but not exposed via FFI yet |
| Mixed CPU/GPU    | rejected | Throws `ArgumentError` — user must `.to()` explicitly |

### Elementwise binary (+ - * /) and scalar / row broadcast

Memory-bandwidth bound. Small tensors → CPU wins by far.

| Path        | Status | Location                      | Notes                                        |
| ----------- | ------ | ----------------------------- | -------------------------------------------- |
| CPU         | impl   | `ops.dart::_binaryCpu`        | Exact-shape, scalar (`length == 1`), and `[1, N]` row-broadcast into `[M, N]` |
| GPU (same shape) | impl | `add_tensors` / `sub_tensors` / `mul_tensors` / `div_tensors` | 256-thread block, elementwise |
| GPU (scalar broadcast) | impl | `add_tensor_scalar` etc. | Both `t + num` and `t + Tensor([1])`; `num` uploads a 1-elem GPU tensor and disposes it |
| GPU (row broadcast) | impl for `+` only | `add_tensor_row_broadcast` | Sub/mul/div row-broadcast on GPU throws — call `.to(Device.CPU)` |

### Unary elementwise: `abs`, `log`, `pow`

| Path | Status | Notes                                                    |
| ---- | ------ | -------------------------------------------------------- |
| CPU  | impl   | Straight `dart:math` calls; `pow` uses `math.pow`        |
| GPU  | impl   | `abs_tensor`, `log_tensor`, `pow_tensor` (takes a `float` exponent) |

### Activations: `relu`, `sigmoid`, `tanh` (+ planned `gelu`)

| Op       | CPU  | GPU  | Notes                                    |
| -------- | ---- | ---- | ---------------------------------------- |
| `relu`   | impl | impl | `relu_tensor`                            |
| `sigmoid`| impl | impl | `sigmoid_tensor`                         |
| `tanh`   | impl | impl | `tanh_tensor`; CPU uses `exp(2x)` form   |
| `gelu`   | planned | planned | Reference kernel exists in `elementwise.cuh` |

### Rearrangement

| Op            | CPU  | GPU  | Notes                                     |
| ------------- | ---- | ---- | ----------------------------------------- |
| `transpose()` (2D) | impl | impl | Strided copy on CPU; 32x32 tile kernel `transpose_tensor` on GPU |
| `slice()`     | planned | planned | Pure view / pointer arithmetic       |
| `concat`      | planned | planned | Fold on CPU; GPU only for large tiles |

### Reductions: `sum`, `mean`

| Op       | CPU  | GPU  | Notes                                    |
| -------- | ---- | ---- | ---------------------------------------- |
| `sum()`  | impl | impl | Result shape `[1, 1]`; GPU uses `atomicAdd` into a scalar buffer |
| `mean()` | impl | impl | GPU: `atomicAdd` sum, then divide by `n` in a follow-up kernel |

### Init: zero / xavier

| Op                    | Status  | Notes                                       |
| --------------------- | ------- | ------------------------------------------- |
| `Tensor.fill(v)`      | impl    | CPU or GPU depending on size / explicit dev |
| `Tensor.fromList(vs)` | impl    | Same                                        |
| Xavier / He init      | planned | CPU by default (reproducible)               |

### Bridge / bookkeeping

| Op                | Status | Notes                                                        |
| ----------------- | ------ | ------------------------------------------------------------ |
| `to(Device)`      | impl   | CPU↔GPU copy; `identical(x, x.to(x.device))`                 |
| `toList()`        | impl   | Device-aware; D→H copy from GPU                              |
| `dispose()`       | impl   | Free GPU buffer; no-op on CPU; idempotent                    |
| `get_tensor_grad` | planned| Needed once autograd is wired                                |
| `set_tensor_data` | planned| For weight loading                                           |

### Attention / normalization / embedding / conv

All planned as **GPU-only** wrappers around the reference kernels — no
CPU implementations planned since they're only useful at model scale.

| Op                     | CPU  | GPU  | Notes                                     |
| ---------------------- | ---- | ---- | ----------------------------------------- |
| `layernorm`            | impl | impl | 2D `[R, C]`, normalized over `C`. GPU uses `layernorm_forward` / `layernorm_backward` (block-per-row reduction; backward atomicAdds into pre-allocated `dGamma`/`dBeta`). Autograd wired end-to-end for `x`, `gamma`, `beta`. Also exposed as `nn.LayerNorm` module. |
| `softmax`              | impl | impl | Row-wise on 2D `[R, C]`. Numerically stable (subtract row max). GPU `softmax_forward` / `softmax_backward` are block-per-row. |
| `cross_entropy`        | impl | impl | Fused softmax + NLL with integer targets (`Tensor([R])` of floats, rounded to int). Returns per-sample losses `[R, 1]` — pair with `.mean()` / `.sum()`. GPU `cross_entropy_forward` / `cross_entropy_backward` recompute softmax in backward instead of caching. |
| `embedding`            | impl | impl | Table `[V, D]` gathered by indices `[N]` → `[N, D]`. Backward scatter-adds into pre-allocated `table.grad` (atomicAdd on GPU). Exposed as `nn.Embedding(numEmbeddings, embeddingDim)` with Gaussian init. |
| `aft` / attention      | none planned | planned | `attention.cuh`                   |
| `scaledDotProductAttention` | impl (composed) | impl (composed) | 2D `[N,Dk] x [M,Dk] x [M,Dv]`, no fused kernel — composes matmul + transpose + scalar mul + softmax + matmul, so autograd and device dispatch come free. Optional additive `mask` of shape `[N,M]`. |
| `concat(axis=0 or 1)`  | impl | impl (round-trip) | 2D concat along either axis, done in host memory. GPU inputs are downloaded, concatenated, and the result uploaded — acceptable for MHA + KV-cache use; a fused kernel can replace this later. Backward slices upstream grad back into each input along the concat axis. |
| `MultiHeadAttention`   | impl (composed) | impl (composed) | Per-head `Linear` projections + per-head SDPA + `concat` + output `Linear`. Optional attention `Dropout`. Optional `MHACache` (per-head K/V) enables the autoregressive fast path: new K/V are appended (axis-0 concat) and SDPA sees the full history. |
| `TransformerBlock` (pre-LN) | impl (composed) | impl (composed) | `x + dropout(mha(ln1(x))); h + dropout(ffn2(relu(ffn1(ln2(h)))))`. Composition of every op above. |
| `SinusoidalPositionalEncoding` | impl | impl | Non-trainable; PE table recomputed per forward for the exact `seqLen` (`O(N*D)`, no slice op needed). Uploaded to matching device before the residual add. |
| `LearnedPositionalEmbedding` | impl | impl | Trainable `[maxLen, embedDim]` table wrapped as an `Embedding` submodule; positions `startPos..startPos+N-1` gathered via existing embedding op, then added. `startPos` lets a cached single-token step get the encoding for its true absolute position. |
| `causalMask(n)`        | impl | impl (host build → upload) | Additive upper-triangular mask (`0` on-and-below diagonal, `blockValue` above, default `-1e9`). Built as a `Float32List` on the host; `device: Device.GPU` uploads the result. Non-trainable. |
| `TransformerEncoder`   | impl (composed) | impl (composed) | Stack of `TransformerBlock`s + optional final `LayerNorm`. Forward threads an optional `mask` through every block. Composition only — no new kernel. |
| `TransformerLM`        | impl (composed) | impl (composed) | `Embedding` (tokens) + `SinusoidalPositionalEncoding` + `TransformerEncoder` (with causal mask) + `Linear` head. Takes 1D `[N]` tokens → `[N, vocabSize]` logits. Composition only. |
| `GPT` (tied)           | impl (composed) | impl (composed) | GPT-2 style: `Embedding` + `LearnedPositionalEmbedding` + embedding `Dropout` + `TransformerEncoder` + weight-tied head (`h @ W_e.T`, no separate Linear, no head bias). Untied variant adds a `Linear(embedDim, vocabSize, bias=false)`. Composition only. |
| `GPT.generate`         | impl (host loop) | impl (host loop) | Autoregressive sampler; runs Dart-side around per-step `call()`. Greedy / temperature / top-k. Sampling itself is CPU (softmax + CDF draw on a `List<double>`) — the forward pass respects the model's device. `useCache: true` (default) uses `EncoderCache` (`List<MHACache>`) so each new token is a single-row forward (O(N) per step vs O(N²)); `useCache: false` re-runs the full context each step and supports sliding-window truncation. |
| `MHACache` / `EncoderCache` | impl | impl | Inference-time per-head K/V buffers, grown by `Tensor.concat(axis=0)`. `MHACache.appendK/V` returns the concatenated tensor and stores it. `EncoderCache.empty(numLayers, numHeads)` allocates a fresh stack. Non-trainable — cached tensors carry no gradient. |
| `dropout(p)` / `nn.Dropout` | impl (composed) | impl (composed) | Builds a mask tensor (`0` or `1/(1-p)`) and multiplies — backward comes free from `*`. Training-only; eval mode / `p=0` is identity. Optional seeded `math.Random`. Also exposed as `nn.Dropout(p)`. |
| `clipGradNorm`         | impl | impl | Rescales in-place across a list of parameters so their global L2 grad norm ≤ `maxNorm`. Uses `Tensor.assign` to swap the scaled grad into each param's `.grad`. |
| `im2col` / `col2im`    | none planned | planned | `conv_misc.cuh`                   |
| `SGD.step` / `Adam.step` | impl | impl | Composed from existing tensor ops on parameters' native device; state buffers (velocity, m/v) live alongside their parameter. No dedicated kernel. |

## Size threshold (default)

Elementwise + reductions default to CPU below **4096 elements**, GPU at
or above. Exposed as `Tensor.autoDeviceThreshold`. This is a starting
heuristic; benchmark and adjust.

## Current implementation snapshot

- `Tensor` has dual backing: `_cpuData: Float32List?` xor `_handle: Pointer<Void>?`.
- Factories: `Tensor.fromList`, `Tensor.fill`, both accept `device:`.
- Bridge: `to(Device)`, `toList()`, `dispose()`.
- Ops file `lib/core/tensor/ops.dart` implements CPU + GPU paths for
  every op listed above. Dispatch is by input `Device`; mixed-device
  binary ops throw `ArgumentError`. GPU row-broadcast is `+` only; the
  other three throw with a `.to(Device.CPU)` hint.
- Autograd: wired end-to-end for `+ - * /`, matmul, sigmoid, tanh, log,
  pow, transpose, sum, mean (all CPU + GPU). `relu` and `abs` backward
  are CPU-only — GPU calls throw with a hint. See
  [Autograd](#autograd) below for details.

## Autograd

Reverse-mode automatic differentiation is implemented in Dart. The
graph is built implicitly as ops execute; every op with at least one
input that has `requiresGrad = true` sets a `_backward` closure on its
output and records its input tensors as `_children`.

**Calling `backward()`:**
1. Runs a post-order DFS from the loss tensor to build a topological
   order (parents before self).
2. Initializes the root's gradient to `ones_like(loss)` if not already
   set. Typical use: loss is scalar, so grad becomes `[[1]]`.
3. Walks the ordered list in reverse, calling each node's
   `_backward()` closure.

**Gradient math uses the same ops as forward.** For example,
`sigmoid`'s backward computes `dOut * (y - y*y)` using the existing
`*` and `-` operators — which means it runs on GPU when `y` is on GPU,
CPU when `y` is on CPU, with no per-device code. Composition is the
device-agnostic contract.

**Per-op backward status:**

| Op | Backward math | GPU-ready |
| --- | --- | --- |
| `+` | `dA = dOut`, `dB = dOut` (with broadcast reduction) | yes |
| `-` | `dA = dOut`, `dB = -dOut` | yes |
| `*` | `dA = dOut * B`, `dB = dOut * A` | yes |
| `/` | `dA = dOut / B`, `dB = -dOut * A / (B*B)` | yes |
| `+ - * / num` | scalar variants — only left operand grad | yes |
| `matmul` | `dA = dOut @ B.T`, `dB = A.T @ dOut` | yes |
| `sigmoid` | `dX = dOut * (y - y*y)` | yes |
| `tanh` | `dX = dOut - dOut * y * y` | yes |
| `log` | `dX = dOut / x` | yes |
| `pow(e)` | `dX = dOut * e * x^(e-1)` | yes |
| `transpose` | `dX = dOut.T` | yes |
| `sum` | `dX = ones_like(x) * dOut` | yes |
| `mean` | `dX = ones_like(x) * (dOut / n)` | yes |
| `relu` | `dX = dOut * (x > 0)` — needs mask | **CPU only** |
| `abs` | `dX = dOut * sign(x)` — needs sign | **CPU only** |

**Broadcast reduction in backward.** When `y = a + b` with `b` scalar
(`length == 1`), the backward for `b` is `dOut.sum()`. Handled by
`_reduceForBroadcast`. Row-broadcast backward (`[1, N]` from `[M, N]`)
is **not wired** — would need an axis-sum kernel. Users combining row
broadcast with `requiresGrad = true` should keep the graph on CPU or
implement the bias term differently.

**Known limitations (step 1):**
- **`relu` / `abs` backward on GPU throws.** Add a `relu_bwd_tensor` /
  `sign_tensor` kernel to fix.
- **Gradient accumulation allocates a new tensor per step.** The old
  `grad` tensor becomes garbage; on GPU the handle leaks until the
  parent is disposed. In-place add or a per-`backward()` arena is the
  next step.
- **Row-broadcast backward not implemented.** See above.
- **No `no_grad()` context** — you can `detach()` a tensor to break the
  graph.

## Open questions (still open)

- Autograd tape in Dart (recommended, planned) vs C++ (as `dart_cuda`
  does). Sticking with Dart for portability and CPU-op cheapness.
- Whether large-tensor elementwise should auto-promote to GPU when
  called on a GPU tensor once we bind the kernels (yes — it will, by
  virtue of "stay on current device").
