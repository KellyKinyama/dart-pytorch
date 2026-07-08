# dart_pytorch

A minimal PyTorch-style tensor library for Dart, backed by hand-written CUDA
kernels via `dart:ffi`.

Matmul, elementwise binary ops (+ - * /, scalar and row-broadcast), unary
activations (`relu`, `sigmoid`, `tanh`, `abs`, `log`, `pow`), 2D
`transpose`, `sum`/`mean` reductions, `LayerNorm`, row-wise `softmax`,
fused `crossEntropy`, `embedding`, scaled dot-product `attention`,
`concat`, and `dropout` all run end-to-end on either CPU or GPU.
Reverse-mode autograd is wired for the whole op set (with `relu`/`abs`
backward CPU-only for now). A small `nn.Module` scaffolding exposes
`Linear`, `LayerNorm`, `Embedding`, `Dropout`, `MultiHeadAttention`,
a pre-LN `TransformerBlock`, `TransformerEncoder` (stacked blocks with
optional final LN), `TransformerLM` (token embed + positional encoding
+ causal encoder + linear head), and both `SinusoidalPositionalEncoding`
/ `LearnedPositionalEmbedding` as trainable / regularization layers
with `train()` / `eval()` mode toggling; `SGD` / `Adam` optimizers
update parameters in place on their native device, and `clipGradNorm`
bounds the global gradient L2. A runnable char-level LM demo at
`bin/lm_demo.dart` overfits a short refrain end-to-end (`dart run
bin/lm_demo.dart`).

## Layout

```
lib/
  dart_pytorch.dart              # package entry point (re-exports Tensor)
  core/tensor/
    tensor.dart                  # Tensor class + factories (fromList, fill) + to()
    ops.dart                     # elementwise / activation / reduction ops, CPU + GPU (part)
    mat_mul.dart                 # matmul (CPU loop + GPU tiled kernel) (part)
    layer_norm.dart              # LayerNorm forward + backward, CPU + GPU (part)
    softmax.dart                 # softmax + fused crossEntropy, CPU + GPU (part)
    embedding.dart               # table lookup with scatter-add backward (part)
    attention.dart               # scaled dot-product attention (composition) (part)
    dropout.dart                 # inverted dropout via mask multiply (part)
    concat.dart                  # last-axis 2D concat with slice-back backward (part)
    cuda_engine.dart             # dart:ffi bindings to libmat_mul.so
  core/nn/
    module.dart                  # abstract Module base (parameters, zeroGrad, train/eval)
    linear.dart                  # trainable Linear (y = x @ W.T + b)
    layer_norm.dart              # trainable LayerNorm module wrapper
    embedding.dart               # trainable Embedding module wrapper
    dropout.dart                 # nn.Dropout wrapper with train/eval toggle
    multi_head_attention.dart    # per-head Linear + SDPA + concat + out proj
    transformer.dart             # pre-LN TransformerBlock (MHA + MLP + residuals)
    positional.dart              # sinusoidal + learned positional encodings
    masks.dart                   # causalMask(n) additive attention mask helper
    transformer_encoder.dart     # stacked TransformerBlocks + optional final LN
    transformer_lm.dart          # token embed + posEnc + causal encoder + head
  core/optim/
    optimizer.dart               # abstract Optimizer base (step, zeroGrad)
    sgd.dart                     # SGD with optional momentum + weight decay
    adam.dart                    # Adam with bias correction + decoupled WD
    grad_utils.dart              # clipGradNorm (global L2 clip, in place)
  native/
    src/
      engine.cu                  # extern "C" DLLEXPORT wrappers (30 symbols)
      kernels/
        common.cuh               # CUDA includes, DLLEXPORT macro, reductions
        matmul.cuh               # tiled matmul_fwd/bwd kernels
        elementwise.cuh          # add/sub/mul/div, scalar/row-bcast, activations, abs/log/pow
        transpose.cuh            # 32x32 tile transpose
        layernorm.cuh            # layernorm_fwd/bwd (block-per-row)
        softmax.cuh              # softmax_fwd/bwd + cross_entropy_fwd/bwd (fused)
        embedding.cuh            # embedding_fwd + scatter-add embedding_bwd
    lib/                         # populated by the nvcc build (gitignored)
native/lib/libmat_mul.so         # actual load path used by cuda_engine.dart
docs/device-placement.md         # per-op CPU vs GPU decisions + implementation status
test/dart_pytorch_test.dart      # matmul + CPU-op correctness tests
```

> Note: `cuda_engine.dart` loads `${cwd}/native/lib/libmat_mul.so`, so the
> `.so` lives at the repo root, not under `lib/native/lib/`.

## Requirements

- Dart SDK `^3.9.3`
- NVIDIA GPU + CUDA toolkit (tested with CUDA 12.0, driver 576.x)
- Linux / WSL2

## Build the native library

```bash
nvcc --shared -Xcompiler -fPIC \
     -o native/lib/libmat_mul.so \
     lib/native/src/engine.cu
```

## Run the tests

```bash
dart pub get
dart test
```

Expected output: `+137: All tests passed!` (matmul CPU + GPU paths,
mixed-device rejection, device round-trip, every CPU op, every GPU op,
a CPU/GPU parity chain, 22 autograd tests, 9 LayerNorm tests, 16
softmax / cross-entropy / embedding tests, 13 optimizer tests, 9
attention / Linear tests, 14 regularization tests, 17
concat / MultiHeadAttention / TransformerBlock tests, plus 15
positional-encoding tests including PE + TransformerBlock trained
end-to-end with Adam).

## Usage

```dart
import 'package:dart_pytorch/dart_pytorch.dart';

void main() {
  // Small tensors default to CPU (below Tensor.autoDeviceThreshold = 4096).
  final a = Tensor.fromList([2, 3], [1, 2, 3, 4, 5, 6]);
  final b = Tensor.fromList([3, 2], [7, 8, 9, 10, 11, 12]);
  final c = a.matmul(b);            // CPU path, shape [2, 2]
  print(c.toList());                // [58, 64, 139, 154]

  // CPU elementwise + activations.
  final x = Tensor.fromList([4], [-1.0, 0.0, 1.0, 2.0]);
  print((x + 1).toList());          // [0, 1, 2, 3]
  print(x.relu().toList());         // [0, 0, 1, 2]

  // Autograd.
  final w = Tensor.fromList([1, 2], [0.5, -0.3], requiresGrad: true);
  final xVec = Tensor.fromList([2, 1], [1.0, 2.0]);
  final loss = (w.matmul(xVec) - 4.0).pow(2).sum();
  loss.backward();
  print(w.grad!.toList()); // gradient wrt w

  // Opt in to GPU for large workloads.
  final big = Tensor.fromList([64, 64],
      List<double>.generate(4096, (i) => i.toDouble()),
      device: Device.GPU);
  final bigResult = big.matmul(big); // tiled 32x32 kernel
  big.dispose();
  bigResult.dispose();

  // Or transfer explicitly.
  final gpuVersion = a.to(Device.GPU);
  gpuVersion.dispose();
}
```

## Currently supported

See `docs/device-placement.md` for the full policy and per-op status.
Short version:

| Area                                    | CPU  | GPU  | Notes                                              |
| --------------------------------------- | ---- | ---- | -------------------------------------------------- |
| `Tensor.fromList` / `Tensor.fill`       | yes  | yes  | Device chosen by size (threshold 4096) or explicit |
| `to(Device)`, `toList()`, `dispose()`   | yes  | yes  |                                                    |
| `matmul` (forward)                      | yes  | yes  | CPU naive loop; GPU tiled 32x32                    |
| `matmul` (backward)                     | —    | —    | GPU kernels compiled, not wired to Dart            |
| Elementwise `+ - * /` (same shape)      | yes  | yes  |                                                    |
| Scalar broadcast (`t + num` or `t + [1]`) | yes | yes  | `num` on GPU uploads a 1-elem tensor + disposes    |
| Row broadcast `[1,N]` into `[M,N]`      | yes (all) | `+` only | Sub/mul/div row-bcast on GPU throws — use `.to(Device.CPU)` |
| `relu`, `sigmoid`, `tanh`, `abs`, `log`, `pow` | yes | yes |                                                  |
| `transpose` (2D)                        | yes  | yes  | CPU strided copy; GPU 32x32 tile kernel            |
| `sum`, `mean`                           | yes  | yes  | GPU: `atomicAdd`-based reduction into `[1,1]`      |
| `layerNorm`                             | yes  | yes  | 2D `[R,C]` over `C`; also `nn.LayerNorm(dim)`      |
| `softmax`                               | yes  | yes  | Row-wise on 2D `[R,C]`, numerically stable         |
| `crossEntropy(targets)`                 | yes  | yes  | Fused softmax + NLL; returns `[R,1]` per-sample    |
| `embedding(indices)`                    | yes  | yes  | `[V,D]` table gathered by `[N]`; also `nn.Embedding` |
| Autograd graph                          | yes  | yes  | Dart-side tape; `relu`/`abs` backward CPU-only |
| `SGD` / `Adam` optimizers               | yes  | yes  | State (velocity, m/v) lives on parameter's device; `Tensor.assign` swaps updates in-place |
| `scaledDotProductAttention`             | yes  | yes  | 2D single-head SDPA; composed from matmul/transpose/softmax/matmul with optional additive mask |
| `nn.Linear(in, out)`                    | yes  | yes  | `y = x @ W.T + b`, Kaiming-uniform init; `bias: false` supported |
| `dropout(p)` / `nn.Dropout`             | yes  | yes  | Inverted dropout via `mask` multiply; `train()`/`eval()` toggles it |
| `TensorConcat.concat(list, axis=1)`     | yes  | yes  | Last-axis 2D concat; GPU round-trips through host |
| `nn.MultiHeadAttention(D, H)`           | yes  | yes  | Per-head Linear + SDPA + concat + out proj; optional attn Dropout |
| `nn.TransformerBlock(D, H, {ffnDim})`   | yes  | yes  | Pre-LN encoder block: MHA + residual + MLP + residual, all submodules toggled by `train()`/`eval()` |
| `nn.SinusoidalPositionalEncoding(D)`    | yes  | yes  | Fixed sin/cos PE, no params, recomputed per forward for the exact seqLen |
| `nn.LearnedPositionalEmbedding(maxLen, D)` | yes  | yes  | Trainable position table gathered via `Embedding`; scatter-add backward comes free |
| `clipGradNorm(params, maxNorm)`         | yes  | yes  | Global L2 grad clip, in-place via `Tensor.assign` |

**Placement rule:** ops respect the input tensor's device. Mixed-device
inputs to a binary op raise `ArgumentError` — call `.to(...)`
yourself so the copy cost stays visible.

## FFI surface (C entry points)

Defined in `lib/native/src/engine.cu` (30 symbols):

| Category       | Symbols                                                                    |
| -------------- | -------------------------------------------------------------------------- |
| Lifecycle      | `create_tensor`, `destroy_tensor`, `get_tensor_data`                       |
| Matmul         | `matmul_tensors`                                                           |
| Elementwise    | `add_tensors`, `sub_tensors`, `mul_tensors`, `div_tensors`                 |
| Scalar bcast   | `add_tensor_scalar`, `sub_tensor_scalar`, `mul_tensor_scalar`, `div_tensor_scalar` |
| Row bcast      | `add_tensor_row_broadcast`                                                 |
| Unary math     | `abs_tensor`, `log_tensor`, `pow_tensor` (`float exp`)                     |
| Activations    | `relu_tensor`, `sigmoid_tensor`, `tanh_tensor`                             |
| Rearrangement  | `transpose_tensor`                                                         |
| Reductions     | `sum_tensor`, `mean_tensor` (both return a `[1,1]` handle)                 |
| LayerNorm      | `layernorm_forward`, `layernorm_backward`                                  |
| Softmax / CE   | `softmax_forward`, `softmax_backward`, `cross_entropy_forward`, `cross_entropy_backward` |
| Embedding      | `embedding_forward`, `embedding_backward` (scatter-add into `gTable`)      |

All wrappers return `void*` (Tensor handle) except lifecycle helpers.

## Provenance

The CUDA kernels and FFI patterns were lifted from `../dart_cuda`,
keeping only forward-mode wrappers: `kernels/common.cuh`,
`kernels/matmul.cuh`, `kernels/elementwise.cuh`, `kernels/transpose.cuh`,
and the forward-only slice of `engine.cu`.
