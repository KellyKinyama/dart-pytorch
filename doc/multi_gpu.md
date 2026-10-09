# Multi-GPU: one model across several GPUs (in one process)

**Goal.** Run a single model whose weights are too big for one card, or
too slow on one card, by spreading it across every GPU in the machine —
all inside one Dart process, no sockets, no separate ranks.

Two independent strategies are provided and can be combined:

| Strategy | What splits | Who holds it | Boundary traffic |
|---|---|---|---|
| **Pipeline parallelism** | whole layers | one GPU per stage | the activation between stages |
| **Tensor parallelism** | a single big matmul | all GPUs share each layer | gather / all-reduce per layer |

Both are **CUDA-only** (NVIDIA) and single-host: they use the GPUs in
*one* machine. For multiple machines, see the socket-based DDP data
parallel path in [../bin/ddp_dist.dart](../bin/ddp_dist.dart).

---

## What was added

### Native CUDA — [../lib/native/src/engine.cu](../lib/native/src/engine.cu)
- Each GPU `Tensor` now records the physical device ordinal its memory
  lives on (captured with `cudaGetDevice` at allocation).
- New exported symbols:
  - `dp_device_count()` — visible GPU count.
  - `dp_get_device()` / `dp_set_device(int)` — read/pin the current CUDA
    device for the calling thread.
  - `copy_tensor_to_device(void*, int dst)` — allocate a copy on `dst`
    and `cudaMemcpyPeer` into it (direct GPU→GPU when peer access is
    available, host-staged otherwise).

### FFI layer — [../lib/core/tensor/cuda_engine.dart](../lib/core/tensor/cuda_engine.dart)
- Binds the new symbols **defensively**: an older prebuilt
  `libmat_mul` without them still loads and simply reports a single
  device (`supportsMultiGpu == false`).
- Caches the current device Dart-side so tagging fresh outputs costs no
  extra FFI call: `deviceCount`, `currentDevice`, `setDevice`,
  `copyTensorToDevice`, `supportsMultiGpu`.

### Tensor — [../lib/core/tensor/tensor.dart](../lib/core/tensor/tensor.dart)
- `Tensor.gpuIndex` — which physical GPU a GPU tensor is on.
- `Tensor.onGpu(k, () => ...)` — scope that pins device `k` for the body
  (like `with torch.cuda.device(k)`), restoring the previous device
  after. Every GPU tensor created inside lands on card `k`.
- `Tensor.toGpu(k)` — move a tensor to GPU `k` (peer copy GPU→GPU, or
  upload CPU→GPU). No-op if already there.
- `Tensor.gpuCount` / `Tensor.currentGpu` accessors.
- Binary ops and `matmul` reject operands that live on different GPUs
  with a clear "move one with `.toGpu(...)`" error.

### Tensor-parallel layers — [../lib/core/nn/parallel_linear.dart](../lib/core/nn/parallel_linear.dart)
The standard Megatron-LM scheme:
- `ColumnParallelLinear` — weight `[out, in]` split along output rows;
  each GPU computes a slice of the output columns, results are
  **gathered** (concat) into the full activation.
- `RowParallelLinear` — weight `[out, in]` split along input columns;
  each GPU consumes a slice of the input features and produces a partial
  sum, **all-reduced** (added) across GPUs.
- `TensorParallelMLP` — `down(relu(up(x)))` with a column-parallel up
  projection feeding a row-parallel down projection: the wide hidden
  layer never fully materialises on any single card.

> These layers are **inference-oriented** — weights are held detached
> (no autograd). A correct distributed backward (gradient all-reduce
> across shards) is not implemented yet.

---

## Build the native library (required)

The multi-GPU symbols are new, so the prebuilt library downloaded by
`ensureNativeLib()` does **not** contain them. Rebuild once on a machine
with the CUDA toolkit:

```bash
nvcc --shared -Xcompiler -fPIC \
  -o native/lib/libmat_mul.so lib/native/src/engine.cu
```

On WSL2, prepend `LD_LIBRARY_PATH=/usr/lib/wsl/lib` to the run commands
below so the CUDA driver stub is found (drop it on native Linux). On
Windows the loader looks for `native/lib/mat_mul.dll`.

Check how many GPUs the process sees:

```dart
import 'package:dart_pytorch/dart_pytorch.dart';
Future<void> main() async {
  await ensureNativeLib();
  print('visible GPUs: ${Tensor.gpuCount}');
}
```

If this prints `1` on a multi-GPU box, the library was not rebuilt with
the new symbols (or `CUDA_VISIBLE_DEVICES` is restricting the process).

---

## Run the demos

Both demos degrade gracefully to a single GPU, so they run anywhere the
native lib loads.

### Pipeline parallelism — [../bin/multi_gpu_pipeline_demo.dart](../bin/multi_gpu_pipeline_demo.dart)
Splits a stack of linear layers into contiguous stages, one per GPU, and
passes the activation from one card to the next.

```bash
dart run bin/multi_gpu_pipeline_demo.dart
```

Prints the layer→GPU assignment, the output, and how many devices the
model ran across.

### Tensor parallelism — [../bin/tensor_parallel_mlp_demo.dart](../bin/tensor_parallel_mlp_demo.dart)
Shards a 256→1024→256 MLP across every visible GPU and checks the result
against a single-device reference.

```bash
dart run bin/tensor_parallel_mlp_demo.dart
```

Expected tail:

```
max abs diff vs single-GPU reference: <~1e-6>
OK — tensor-parallel output matches the reference.
```

### Tensor-parallel attention — [../bin/tensor_parallel_attention_demo.dart](../bin/tensor_parallel_attention_demo.dart)
Shards a multi-head attention layer's heads across every visible GPU
(each card runs SDPA for its own heads), then row-parallelises the
output projection. Checks the result against the single-device
`MultiHeadAttention`. Supports GQA and an additive causal mask.

```bash
dart run bin/tensor_parallel_attention_demo.dart
```

### Training a tensor-parallel MLP — [../bin/tensor_parallel_train_demo.dart](../bin/tensor_parallel_train_demo.dart)
Shards an MLP with `trainable: true` and overfits it to a fixed target
with SGD. Each shard's weight gradient is computed **on its own card**;
only the activation and its gradient cross GPU boundaries (via the
differentiable `Tensor.toGpu`). Prints the loss curve and confirms every
shard received a gradient.

```bash
dart run bin/tensor_parallel_train_demo.dart
```

### Tensor-parallel transformer block — [../bin/tensor_parallel_transformer_demo.dart](../bin/tensor_parallel_transformer_demo.dart)
Shards a full pre-LN encoder block (attention + MLP) across every visible
GPU — `h = x + mha(ln1(x)); y = h + mlp(ln2(h))` — with the LayerNorms and
residuals replicated on the output device. Checks against the
single-device `TransformerBlock`.

```bash
dart run bin/tensor_parallel_transformer_demo.dart
```

### Multi-block stack (tensor + pipeline parallel) — [../bin/tensor_parallel_stack_demo.dart](../bin/tensor_parallel_stack_demo.dart)
Stacks several transformer blocks. By default every block is
tensor-parallel across all GPUs; with `--pipeline N` the blocks are also
split into `N` pipeline stages across disjoint device groups (TP within a
stage, PP across stages), with the activation handed between groups
automatically.

```bash
dart run bin/tensor_parallel_stack_demo.dart              # TP only
dart run bin/tensor_parallel_stack_demo.dart --pipeline 2 # TP + pipeline
```

### Scaling benchmark — [../bin/_tp_bench.dart](../bin/_tp_bench.dart)
Times a tensor-parallel transformer stack's forward across every GPU
count (1..N) and both gather modes (GPU-native peer copy vs host-staged),
reporting tokens/s, speedup, and the per-GPU weight footprint (which
should be ~`total / GPUs`, confirming the weights are sharded).

```bash
dart run bin/_tp_bench.dart
dart run bin/_tp_bench.dart --tokens 256 --depth 8 --embed 1024
```

---

## Writing your own sharded model

**Pipeline parallelism** — put each stage's weights on its own card and
move the activation across the boundary:

```dart
final w0 = Tensor.onGpu(0, () => Tensor.fromList(shape, vals, device: Device.GPU));
final w1 = Tensor.onGpu(1, () => Tensor.fromList(shape, vals, device: Device.GPU));

var x = input.toGpu(0);
x = Tensor.onGpu(0, () => x.matmul(w0).relu());
x = x.toGpu(1);                                   // hand off to the next card
x = Tensor.onGpu(1, () => x.matmul(w1).relu());
```

**Tensor parallelism** — shard the big linears from full host weights:

```dart
final mlp = TensorParallelMLP.fromWeights(
  upWeight: upW,     // [hidden, model] on CPU
  downWeight: downW, // [model, hidden] on CPU
  upBias: upB, downBias: downB,
  // devices: [0, 1, 2],   // defaults to all visible GPUs
);
final y = mlp(x);    // x: [tokens, model]
```

Rules of thumb:
- Wrap a layer's compute in `Tensor.onGpu(k, ...)` so **all** its ops —
  including specialised ones (layernorm, attention, …) — run on card `k`.
- Cross a GPU boundary only with `toGpu(...)`; keep everything else
  on-card to avoid PCIe ping-pong.
- Binary ops / matmul require both operands on the same GPU.

---

## Limitations

- **NVIDIA/CUDA only**, single host. Not AMD/Intel/Apple; not across
  machines (use DDP for that).
- Cross-GPU transfers use peer access (`cudaDeviceEnablePeerAccess`) for
  GPU-direct copies when the topology allows, falling back to UVA
  host-staging otherwise. The column-parallel gather copies shard
  columns straight between cards with `cudaMemcpy2D` (no host round-trip);
  set `Tensor.useGpuCollectives = false` to force the portable host path.
- Tensor-parallel **MLP** layers are trainable (`trainable: true`): weight
  gradients are local to each card, only the activation gradient crosses
  GPUs via the differentiable `Tensor.toGpu`. Row-parallel input-gradient
  flow (needed only when an earlier trainable layer feeds a row-parallel
  one) is not wired through the host feature-split yet.
- Tensor-parallel **attention** is inference-only so far (training wiring
  mirrors the MLP and is a follow-up).
- The tensor-parallel **transformer block** (`TensorParallelTransformerBlock`)
  is inference-only, 2D `[N, embedDim]`, ReLU FFN; LayerNorms/residuals are
  replicated on the output device.
- Tensor-parallel attention covers the 2D `[N, embedDim]` path with GQA,
  an additive mask, **RoPE**, and a **per-shard KV cache** for
  autoregressive decoding (`newCache()` on the attention, block, or
  stack; `call(..., cache:/caches:, startPos:)`); dropout and the batched
  3D path are not sharded yet.
- Pinning is per-thread; drive GPU ops from a single Dart isolate (the
  default), or set the device inside each isolate.

---

## What's next

See [multi_gpu_roadmap.md](multi_gpu_roadmap.md) for the phased plan:
tensor-parallel **attention**, **distributed backward** for training,
GPU-native collectives, a full TP transformer layer, and 3D
(data × tensor × pipeline) parallelism.
