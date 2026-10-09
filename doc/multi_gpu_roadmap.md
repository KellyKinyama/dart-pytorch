# Multi-GPU roadmap

A phased plan to extend the in-process multi-GPU foundation (see
[multi_gpu.md](multi_gpu.md)) into a complete, trainable, tensor-parallel
transformer stack — plus the supporting work that makes it fast and
correct.

## Where we are (done)

- Device primitives: `Tensor.gpuIndex`, `onGpu`, `toGpu`, `gpuCount`;
  native `dp_set_device` / `dp_device_count` / `copy_tensor_to_device`
  ([../lib/core/tensor/tensor.dart](../lib/core/tensor/tensor.dart),
  [../lib/native/src/engine.cu](../lib/native/src/engine.cu)).
- Pipeline parallelism demo ([../bin/multi_gpu_pipeline_demo.dart](../bin/multi_gpu_pipeline_demo.dart)).
- Tensor-parallel linears + MLP, **inference-only**
  ([../lib/core/nn/parallel_linear.dart](../lib/core/nn/parallel_linear.dart),
  [../bin/tensor_parallel_mlp_demo.dart](../bin/tensor_parallel_mlp_demo.dart)).

## Guiding constraints

- **NVIDIA/CUDA, single host** for tensor parallelism (peer copy). Keep
  the existing socket DDP ([../bin/ddp_dist.dart](../bin/ddp_dist.dart))
  as the cross-machine path; the two compose later (Phase 6).
- **Autograd is device-agnostic.** Backward reuses forward ops, so any
  layer composed of existing ops differentiates on whichever GPU its
  tensors live on — no new backward kernels needed for Phases 1–2.
- **Correctness before speed.** Each phase ships with a single-GPU
  reference-equivalence test (max-abs-diff) before optimisation.

---

## Phase 1 — Tensor-parallel attention block

**Status: done (first cut).** `TensorParallelMultiHeadAttention`
([../lib/core/nn/attention/tensor_parallel_attention.dart](../lib/core/nn/attention/tensor_parallel_attention.dart))
shards KV heads (and their Q heads) across GPUs, runs SDPA per head
on-card, and row-parallelises the output projection. Validated against
single-GPU `MultiHeadAttention` for plain MHA, GQA, and a causal mask
([../test/tensor_parallel_attention_test.dart](../test/tensor_parallel_attention_test.dart),
[../bin/tensor_parallel_attention_demo.dart](../bin/tensor_parallel_attention_demo.dart)).
Remaining for a later pass: RoPE, dropout, KV-cache, batched 3D path.

**Objective.** A `ColumnParallel`/`RowParallel` multi-head attention so
one attention layer's QKV and output projection span all GPUs, matching
Megatron: split attention **heads** across GPUs (column-parallel QKV),
compute SDPA per head on-card, then row-parallel the output projection.

**Touchpoints.**
- New: `lib/core/nn/attention/tensor_parallel_attention.dart`.
- Reference the non-parallel layer in
  [../lib/core/nn/attention/multi_head_attention.dart](../lib/core/nn/attention/multi_head_attention.dart)
  (heads are already separate `List<Linear>`, `headDim = embedDim/numHeads`).
- Reuse `scaledDotProductAttention` ([../lib/core/tensor/attention.dart](../lib/core/tensor/attention.dart)) — runs on GPU, composed ops.
- Reuse `ColumnParallelLinear` / `RowParallelLinear`.

**Design decisions.**
- Assign `numHeads` to GPUs in contiguous groups (`heads/G` per card);
  require `numHeads % G == 0` for v1, else fall back to uneven split.
- Each GPU holds its heads' Q/K/V weights and runs SDPA for those heads
  entirely on-card (no cross-GPU attention traffic).
- Output projection is **row-parallel**: each GPU projects its heads'
  concat slice and the partials are all-reduced (added).
- GQA (`numKvHeads < numHeads`): keep each KV head's weights on the GPU(s)
  owning the Q heads that map to it; replicate a KV head if it's shared
  across GPUs.
- Causal mask precomputed per device inside the `onGpu` scope.

**Tasks.**
1. `ColumnParallelQKV`: shard per-head Q/K/V onto devices; forward yields
   per-GPU head outputs `[N, headsOnCard*headDim]`.
2. Per-GPU SDPA over that card's heads (loop or batched).
3. `RowParallelLinear` output projection + all-reduce.
4. `TensorParallelMultiHeadAttention` wrapper (Module).
5. KV-cache variant: cache lives per-GPU alongside its heads.

**Risks.** Small `headDim` makes per-head kernels launch-bound;
gather/all-reduce through host (current `concat`) may dominate — tracked,
fixed in Phase 3.

**Acceptance.** `bin/tensor_parallel_attention_demo.dart` matches a
single-GPU `MultiHeadAttention` within `1e-3` max-abs-diff; works on 1
and N GPUs; GQA config validated.

---

## Phase 2 — Tensor-parallel training (distributed backward)

**Status: done (MLP, attention, block).** `Tensor.toGpu` is now a
differentiable cross-device transfer (grad moves back to the source
card), the parallel linears, attention, and transformer block/stack all
take `trainable: true` to expose their shards as autograd leaves, and
Adam allocates moment buffers on each shard's GPU (`Tensor.zerosLike`).
Validated by overfitting a sharded MLP, attention layer, and full block
([../test/tensor_parallel_train_test.dart](../test/tensor_parallel_train_test.dart),
[../test/tensor_parallel_attention_test.dart](../test/tensor_parallel_attention_test.dart),
[../test/tensor_parallel_transformer_test.dart](../test/tensor_parallel_transformer_test.dart),
[../bin/tensor_parallel_train_demo.dart](../bin/tensor_parallel_train_demo.dart)).
Follow-up: flow input
gradients through the row-parallel host feature-split.

**Objective.** Make the tensor-parallel layers (Phases 1 + the MLP)
trainable, with correct gradients across shards and a synchronized
optimizer step.

**Touchpoints.**
- [../lib/core/nn/parallel_linear.dart](../lib/core/nn/parallel_linear.dart) — un-detach weight shards; expose them via `parameters()`.
- [../lib/core/optim/](../lib/core/optim) — SGD/Adam already step
  per-parameter via `Tensor.assign`; buffers live on the parameter's
  device, so sharded params train in place.
- New: a TP gradient-sync helper (the on-GPU analogue of
  `allReduceMean` in [../bin/ddp_dist.dart](../bin/ddp_dist.dart)).

**Design decisions (the TP backward math).**
- **Column-parallel** (gather forward): forward concat → backward is a
  slice of the output grad to each shard. Each shard's weight grad is
  purely local; **no all-reduce** on weights. The *input* gradient is a
  sum across shards (all-reduce of `dX`).
- **Row-parallel** (all-reduce forward): forward sums partials → each
  shard already sees the full output grad; weight grads are local; the
  input was split, so input grad is just each shard's slice — **no
  all-reduce** on weights, concat the input-grad slices.
- Net: weight gradients are local by construction (the whole point of
  TP); only the **activation** gradient crosses GPUs, mirroring the
  forward boundary traffic. This means the current detached-weight
  concat/all-reduce just needs its backward closures wired — autograd
  does the rest since concat and `+` already have backward.
- Make `TensorConcat.concat` and the row-parallel sum carry grad across
  devices: today `concat` requires same-device inputs; add a
  device-aware gather whose backward scatters slices back to each shard's
  GPU via `toGpu`.

**Tasks.**
1. Flip `parallel_linear.dart` weights to `requiresGrad: true`; return
   them from `parameters()`; keep the `onGpu` scoping in forward so
   backward closures execute on the right card.
2. Device-aware `concat`/gather with a backward that `toGpu`s each slice
   back to its shard.
3. Row-parallel reduce backward: broadcast the output grad to each shard
   (already correct — `+` backward passes grad through).
4. Optimizer verification: confirm `Adam`/`SGD` buffers allocate on each
   shard's GPU (they follow `p.device`); add a test.
5. `bin/tensor_parallel_train_demo.dart`: overfit a tiny TP-MLP to a
   fixed target; assert loss decreases and matches a single-GPU run.

**Risks.** Grad accumulation ordering across `onGpu` scopes; ensure the
tape's backward restores each node's device (wrap backward closures in
`onGpu` too). Finalizer/VRAM pressure during backward (reuse `freeGraph`).

**Acceptance.** TP-MLP and TP-attention train to the same loss curve
(within tolerance) as the equivalent single-GPU model over N steps.

---

## Phase 3 — GPU-native collectives (performance)

**Status: done (peer gather).** Native `dp_enable_peer_access` +
`copy_block_2d` (cross-device `cudaMemcpy2D`, correct via UVA even
without P2P) are wired through `engine.enablePeerAccess` /
`engine.copyBlock2d`. `TensorConcat.gatherColumns` copies shard columns
straight between cards (no host round-trip) with a symmetric scatter
backward; the column-parallel forward uses it, and `Tensor.toGpu` now
enables peer access before its peer copy. Gated by `Tensor.useGpuCollectives`
(default on) with a host-staged fallback. NCCL bindings remain optional
future work.

**Objective.** Remove the host round-trip from gather/all-reduce.

**Touchpoints.** [../lib/native/src/engine.cu](../lib/native/src/engine.cu),
[../lib/core/tensor/cuda_engine.dart](../lib/core/tensor/cuda_engine.dart),
`concat`/parallel layers.

**Tasks.**
1. `cudaDeviceEnablePeerAccess` setup (best-effort, cached per pair).
2. Native `concat_peer` (device-to-device gather) and an on-GPU
   `add_peer` all-reduce; fall back to host staging when P2P unavailable.
3. Optional: NCCL FFI bindings behind the same Dart API for
   ring/tree all-reduce at scale (opt-in, like the ring toggle in DDP).

**Acceptance.** Same numerical results as Phase 1–2; measurable drop in
PCIe traffic / step time in the benchmark (Phase 5).

---

## Phase 4 — Full tensor-parallel transformer layer + model

**Status: done (block, first cut).** `TensorParallelTransformerBlock`
([../lib/core/nn/tensor_parallel_transformer.dart](../lib/core/nn/tensor_parallel_transformer.dart))
composes TP attention + TP MLP + output-device LayerNorms/residuals
(pre-LN), built from a reference `TransformerBlock` and validated against
it ([../test/tensor_parallel_transformer_test.dart](../test/tensor_parallel_transformer_test.dart),
[../bin/tensor_parallel_transformer_demo.dart](../bin/tensor_parallel_transformer_demo.dart)).
Inference-only, 2D, ReLU FFN. `TensorParallelTransformerStack` stacks
blocks with optional pipeline parallelism (`pipelineStages > 1` splits
devices + blocks into stages; TP within a stage, PP across stages) and is
validated end to end ([../bin/tensor_parallel_stack_demo.dart](../bin/tensor_parallel_stack_demo.dart)).
The block and stack are also **trainable** (`trainable: true`) end to end.
Remaining: a real-weights (HF) loader path.

**Objective.** Compose Phases 1–2 into a drop-in parallel block and a
runnable model.

**Touchpoints.** Mirror [../lib/core/nn/transformer.dart](../lib/core/nn/transformer.dart)
(pre-LN: `x + mha(ln1(x))` then `x + mlp(ln2(x))`).

**Tasks.**
1. `TensorParallelTransformerBlock` = TP-attention + TP-MLP + (replicated)
   LayerNorms + residuals. LayerNorm/residual are cheap → replicate on
   the output device, not sharded.
2. Stack N blocks; combine with **pipeline** split across GPUs for very
   deep models (reuse Phase 0 pipeline pattern): TP within a block, PP
   across block groups.
3. A real-weights path: shard a loaded checkpoint (HF loader) across
   GPUs; validate logits vs single-GPU/CPU reference on a small model.

**Acceptance.** End-to-end forward (and train for a small config) of a
multi-block TP transformer matching a reference within tolerance.

---

## Phase 5 — Benchmarks, tests, tooling (cross-cutting)

**Status: in progress.** `bin/_tp_bench.dart` times the TP transformer
stack's forward across GPU counts (1..N) and both gather modes (peer vs
host-staged), reporting tokens/s, speedup, and per-GPU weight footprint.
GPU-guarded equivalence + training tests exist for the linears, MLP,
attention, block, and stack. Remaining: a PCIe-bytes counter and a
memory-residency assertion.

**Tasks.**
- Equivalence tests under `test/`: extend
  `test/attention_test.dart`, add `test/parallel_linear_test.dart`,
  `test/tensor_parallel_attention_test.dart`. Use `devices: [0]` so they
  run on CI without multiple GPUs; a guarded multi-GPU variant runs when
  `Tensor.gpuCount > 1`.
- `bin/_tp_bench.dart`: tokens/s and PCIe bytes vs GPU count and shard
  strategy; compare host-staged (Phase 1) vs peer (Phase 3).
- Memory report: assert per-GPU resident bytes ≈ `1/G` of weights.
- Docs: keep [multi_gpu.md](multi_gpu.md) current; add a "training"
  section after Phase 2.

---

## Phase 6 — 3D parallelism (data × tensor × pipeline), multi-host

**Objective.** Combine the socket DDP data-parallel path with in-process
TP+PP for cluster-scale runs.

**Design.** Each DDP rank owns one host; within a host, TP+PP spans that
host's GPUs. Gradient all-reduce across hosts stays in
[../bin/ddp_dist.dart](../bin/ddp_dist.dart); TP grads are local, so only
the data-parallel replica grads cross machines — the existing
`allReduceMean` applies unchanged per replica.

**Tasks.**
1. Rank→(host, local GPU set) topology config.
2. Launcher update ([../bin/ddp_cluster.dart](../bin/ddp_cluster.dart)):
   pass per-host device lists.
3. Validate 2 hosts × 2 GPUs on a tiny model.

---

## Sequencing & dependencies

```
Phase 1 (TP attention)  ─┐
                         ├─> Phase 4 (TP block/model) ─> Phase 6 (3D)
Phase 2 (TP backward) ───┘            ▲
Phase 3 (GPU collectives) ────────────┘  (perf, not a correctness gate)
Phase 5 (tests/bench) runs alongside every phase.
```

**Recommended order:** Phase 1 → 2 → 5 (lock correctness) → 3 (speed) →
4 → 6. Phases 1 and 2 are the highest-value next steps and are mostly
Dart (no new CUDA kernels) because autograd is device-agnostic and TP
weight grads are local.

## Open questions

- NCCL dependency: optional FFI (Phase 3) or stay pure peer-copy for
  portability?
- Uneven head/feature splits: support from v1 or require divisibility?
- Checkpoint sharding format: shard on load, or a persisted sharded
  layout to avoid re-slicing each run?
