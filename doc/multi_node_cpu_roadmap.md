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
- [`bin/ddp_run.dart`](../bin/ddp_run.dart) — **per-node** launcher (Phase 1):
  each node runs it with its own `--node-rank`; computes global
  `RANK = node_rank * nproc_per_node + local_rank` and rendezvouses at
  `--master-addr:--master-port`.
- [`bin/ddp_cluster.dart`](../bin/ddp_cluster.dart) — **cluster** launcher
  (Phase 1): reads a hostfile and SSHes into every node to start `ddp_run.dart`
  with the correct `--node-rank`. One command brings up the whole run.

**Verified:** 2 and 3 ranks (single host), and a **2-node × 2-proc simulation**
(world_size 4 over loopback) all finish with *identical* parameter checksums —
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

- [x] **Per-node launcher** [`bin/ddp_run.dart`](../bin/ddp_run.dart) (torchrun
      equivalent): takes `--nnodes`, `--node-rank`, `--nproc-per-node`,
      `--master-addr`, `--master-port`; computes global `RANK = node_rank *
      nproc_per_node + local_rank` and `WORLD_SIZE = nnodes * nproc_per_node`,
      then spawns the local processes with the right env. Verified via a
      2-node × 2-proc loopback run (world_size 4, all checksums identical).
- [ ] **Hostfile + SSH helper** (`scripts/`): read a `hostfile` (one host per
      line, optional slots), SSH into each, and invoke `ddp_run.dart` with the
      correct `--node-rank`. One command brings up the whole cluster.
      — **done:** [`bin/ddp_cluster.dart`](../bin/ddp_cluster.dart) +
      [`scripts/hostfile.example`](../scripts/hostfile.example) (verified via
      `--dry-run`).
- [x] **Connectivity hardening:** bounded connect retry with backoff and a
      configurable budget (`DDP_CONNECT_TIMEOUT_MS`); master rendezvous timeout
      (`DDP_INIT_TIMEOUT_MS`) that fails with how many of N connected; errors
      name the unreachable peer `addr:port`.
- [x] **Operational docs:** see “Running on a cluster” below.

Exit criteria: a 2-host × 2-proc run (world_size 4) converges with identical
checksums across all four ranks.

### Running on a cluster

Prereqs on **every** node: the Dart SDK and a copy of this repo at the same
path, plus passwordless SSH from the launch host to each node.

1. Write a hostfile (see [`scripts/hostfile.example`](../scripts/hostfile.example)):
   ```
   10.0.0.1   2      # node 0 — also the rendezvous master by default
   10.0.0.2   2      # node 1
   ```
2. Preview the exact commands, then launch:
   ```
   dart run bin/ddp_cluster.dart --hostfile scripts/hostfile \
     --workdir /opt/dart-pytorch --master-port 29500 --dry-run
   dart run bin/ddp_cluster.dart --hostfile scripts/hostfile \
     --workdir /opt/dart-pytorch --master-port 29500
   ```
   Per node it runs `ddp_run.dart --node-rank <i> --nproc-per-node <slots>`.

**Networking notes**
- The master binds `0.0.0.0:<MASTER_PORT>`; open that TCP port between nodes.
- `--master-addr` must be an interface on node 0 that the other nodes can
  route to (not `127.0.0.1` for real multi-host).
- Tunables: `DDP_CONNECT_TIMEOUT_MS` (worker connect budget) and
  `DDP_INIT_TIMEOUT_MS` (master rendezvous wait).

**Single-host smoke test** (no SSH; simulates 2 nodes over loopback): run
`ddp_run.dart` twice, `--node-rank 0` and `--node-rank 1`, with
`--master-addr 127.0.0.1` — all ranks should print the same `param_checksum`.

---

## Phase 2 — Correctness & robustness

Goal: make distributed results *correct and resumable*, not just in-sync.

- [x] **Distributed data sharding** (a `DistributedSampler` equivalent): each
      epoch does an identical shared shuffle, then rank `r` takes the disjoint
      stride `r, r+W, r+2W, …` (trimmed to an equal per-rank size) so shards
      don't overlap and cover the data once per epoch.
- [x] **Barriers:** `Dist.barrier()` (gather-at-master + release), used at each
      epoch boundary and around checkpointing.
- [x] **Checkpoint save/resume:** `DDP_SAVE` — rank 0 writes a `Checkpoint`
      after a barrier; `DDP_RESUME` — rank 0 loads, then `broadcastFromMaster`
      syncs weights to all ranks so everyone restarts identically. (Optimizer
      moment state resume is a future add; weights resume today.)
- [x] **Uneven/last-batch handling:** every rank runs exactly
      `perRank // microBatch` batches per epoch, so all ranks perform the same
      number of all-reduces (a mismatch would deadlock).
- [ ] **Failure handling:** per-collective timeout; if a rank drops, fail fast
      on all ranks with a diagnostic instead of hanging. (Connect/rendezvous
      timeouts exist from Phase 1; mid-run collective timeouts are still TODO.)
- [x] **Parity test:** [`test/ddp_parity_test.dart`](../test/ddp_parity_test.dart)
      launches a loopback multi-rank run and asserts identical checksums.

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
