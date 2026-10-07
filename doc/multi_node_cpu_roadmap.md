# Multi-Node CPU Training — Roadmap

Scope: scale `dart_pytorch` data-parallel training across **multiple machines
using CPU only**, building on the socket-based DDP that already works. GPU and
NCCL are explicitly **out of scope** here (tracked separately as a future
option); nothing in this roadmap depends on a GPU.

The design goal throughout: keep it **pure Dart, zero external dependencies**,
so a run needs nothing but the Dart SDK on each host.

---

## Where we are today (landed, verified)

Data-parallel training with a pure-Dart TCP all-reduce:

- [`bin/ddp_dist.dart`](../bin/ddp_dist.dart) — `Dist`: rank 0 is the master
  (binds a port); every other rank connects. Primitives:
  - `allReduceMean(Float32List)` — averages a flat buffer across all ranks.
  - `broadcastFromMaster(Float32List)` — syncs rank 0's weights to everyone.
- [`bin/ddp_train.dart`](../bin/ddp_train.dart) — nanoGPT-style loop on
  `GPT` + `Adam` + `LinearWarmupCosineDecay`: each rank trains its own data
  shard, gradients are averaged every step, then the optimizer steps.
- [`bin/ddp_launch.dart`](../bin/ddp_launch.dart) — single-host launcher that
  spawns N ranks (stand-in for `torchrun`).

**Verified:** 2 and 3 ranks finish with *identical* parameter checksums, i.e.
the replicas stay bit-for-bit in sync. Loss decreases with the larger effective
batch.

### Current protocol (the contract to preserve)

- **Transport:** one TCP connection per worker to the master. `TCP_NODELAY` on.
- **Framing:** every message is `[uint32 little-endian length][payload bytes]`.
  Payloads are raw `Float32List` bytes. `sendFrame` copies the payload into a
  stable buffer first (a view over the reused gradient buffer would otherwise be
  mutated before the socket flushes — this was a real bug, now fixed).
- **Handshake:** a worker's first 4 bytes are its integer rank.
- **Collective:** `allReduceMean` = reduce-to-master + broadcast
  (`O(worldSize)` messages through rank 0 per call).
- **Config via env (torchrun-style):** `RANK`, `WORLD_SIZE`, `LOCAL_RANK`,
  `MASTER_ADDR`, `MASTER_PORT`.

### Known limitations (what this roadmap addresses)

1. Master is a **bandwidth bottleneck and single point of failure** (gather +
   broadcast all funnel through rank 0).
2. Launch is **single-host only** (`ddp_launch.dart`); no per-node launcher.
3. Data sharding is **by RNG seed**, not disjoint shards — ranks can overlap/miss
   samples (fine for the demo, wrong for real training).
4. **No checkpoint/resume**, no barriers, no timeouts, weak failure diagnostics.
5. Only **one core per process** is used for compute.

---

## Phase 1 — Multi-node bring-up

Goal: run the *existing* master-gather DDP across real machines, reliably.

- [ ] **Per-node launcher** `bin/ddp_run.dart` (torchrun equivalent): takes
      `--nnodes`, `--node-rank`, `--nproc-per-node`, `--master-addr`,
      `--master-port`; computes global `RANK = node_rank * nproc_per_node +
      local_rank` and `WORLD_SIZE = nnodes * nproc_per_node`, then spawns the
      local processes with the right env.
- [ ] **Hostfile + SSH helper** (`scripts/`): read a `hostfile` (one host per
      line, optional slots), SSH into each, and invoke `ddp_run.dart` with the
      correct `--node-rank`. One command brings up the whole cluster.
- [ ] **Connectivity hardening:** bounded connect retry with backoff (exists,
      make the limit/timeout configurable); master `accept` timeout; clear,
      actionable errors naming the unreachable peer.
- [ ] **Operational docs:** firewall/port notes, binding to a routable
      interface, and a 2-machine smoke-test recipe.

Exit criteria: a 2-host × 2-proc run (world_size 4) converges with identical
checksums across all four ranks.

---

## Phase 2 — Correctness & robustness

Goal: make distributed results *correct and resumable*, not just in-sync.

- [ ] **Distributed data sharding** (a `DistributedSampler` equivalent): shard
      the dataset by global rank so shards are disjoint and cover the data once
      per epoch; reshuffle deterministically per epoch with a shared seed.
- [ ] **Barriers:** a `Dist.barrier()` collective for epoch boundaries and
      before/after checkpointing.
- [ ] **Checkpoint save/resume:** rank 0 writes `Checkpoint` + optimizer state;
      on resume, rank 0 loads and `broadcastFromMaster` syncs weights (and Adam
      moments) to all ranks so everyone restarts identically.
- [ ] **Uneven/last-batch handling:** pad or drop-last consistently so every
      rank performs the same number of all-reduces (a mismatch deadlocks).
- [ ] **Failure handling:** per-collective timeout; if a rank drops, fail fast
      on all ranks with a diagnostic instead of hanging.
- [ ] **Parity test:** a loopback integration test (world_size ≥ 2 on one host)
      asserting checksum parity after N steps — formalizes the manual check.

Exit criteria: kill a run mid-epoch, resume from checkpoint, and reach the same
loss trajectory; the parity test runs in CI.

---

## Phase 3 — Scalable all-reduce (algorithmic, still CPU/sockets)

Goal: remove the rank-0 bottleneck so throughput holds as nodes grow.

- [ ] **Ring all-reduce** behind the same `allReduceMean` API: arrange ranks in
      a ring, each talks only to its two neighbors; `2·(worldSize−1)`
      reduce-scatter + all-gather steps move bandwidth-optimal data. Pure
      sockets, no master funnel. This is the single highest-value change for
      scaling and is the natural successor to master-gather.
- [ ] **Gradient bucketing:** we already all-reduce one flat buffer; split into
      fixed-size buckets so comms can **overlap** with backward as each bucket
      fills (reduces wall-clock by hiding comm under compute).
- [ ] **fp16 payload compression:** send gradients as fp16 over the wire
      (`dart_pytorch` already has fp16 encode/decode), ~halving bandwidth; sum
      in fp32 on receipt. Guard with a tolerance test vs the fp32 path.

Exit criteria: measured scaling efficiency with ring all-reduce beats
master-gather at world_size ≥ 8, and improves with bucket overlap.

---

## Phase 4 — Per-node CPU performance

Goal: use all cores on each box, since GPU is off the table.

- [ ] **Intra-node parallelism:** parallelize the hot ops (matmul,
      layernorm, softmax) across cores via Dart isolates / a worker pool, or a
      BLAS backing. This multiplies per-node throughput independently of the
      distributed layer.
- [ ] **Thread/affinity controls:** env knobs for compute threads per process;
      guidance on `nproc_per_node` vs cores and NUMA pinning.
- [ ] **Benchmark harness:** report tokens/sec per node, all-reduce time per
      step, and scaling efficiency (ideal vs actual) to locate comms-vs-compute
      bottlenecks.

Exit criteria: near-linear speedup from 1→C cores per node on a fixed model.

---

## Phase 5 — Operability

Goal: make real runs reproducible and observable.

- [ ] **Run config:** a single config (file + env overrides) capturing model,
      data, optimizer, schedule, and cluster topology; logged at startup.
- [ ] **Metrics & logging:** rank-0 structured logs; per-rank metrics to files;
      optional periodic throughput summary.
- [ ] **Reproducibility:** explicit global seed derived per rank; record it in
      the checkpoint so resume is deterministic.

---

## Protocol/API stability

`allReduceMean` and `broadcastFromMaster` are the stable seams. Ring all-reduce
(Phase 3), fp16 compression, and bucketing all land **behind these signatures**,
so `ddp_train.dart` (and any real trainer) does not change as the transport
evolves. The env-var contract (`RANK`/`WORLD_SIZE`/`LOCAL_RANK`/`MASTER_ADDR`/
`MASTER_PORT`) is also frozen.

## Explicitly out of scope (future, separate track)

- GPU execution and `CUDA_VISIBLE_DEVICES` per-rank pinning.
- NCCL FFI bindings for GPU-direct all-reduce.
- Model/tensor/pipeline parallelism (this roadmap is **data parallel** only).

## Suggested order

Phase 1 → Phase 2 → Phase 3 (ring all-reduce) → Phase 4 → Phase 5. Phases 1–2
make multi-node runs real and trustworthy; Phase 3 is where it starts to *scale*.
