# Cooperative training

This library ships three ways to train a model across more than one
process — from a single-machine research sanity check up to a fully
decentralized gossip mesh. Pick the one that matches your setup.

The overall approach follows **DiLoCo** (Douillard et al. 2024,
[arXiv:2311.08105](https://arxiv.org/abs/2311.08105)): every replica
runs `K` local Adam steps on its own shard of the data, then all
replicas periodically **average their parameters element-wise** and
adopt the average as the new starting point. The bandwidth cost is
one full model shipment per `K` local steps, not per gradient step —
which is what makes it usable across slow (or free!) internet links.

| Phase | Mode                                    | Topology                    | Best for                                                                             |
|-------|-----------------------------------------|-----------------------------|--------------------------------------------------------------------------------------|
| 0     | Single-process, in-memory averaging     | N replicas in one Dart VM   | Verifying the math; measuring how N and `K` affect convergence vs. a baseline        |
| 1     | HTTP coordinator + workers              | Star (1 coord + N workers)  | LAN training; friends who can port-forward to one machine; Colab room                |
| 2     | HTTP peer-to-peer gossip                | Full or partial mesh        | Fully decentralized volunteer compute; no single point of failure                    |

All three modes share the same underlying primitives:

- **DPTC checkpoint format** (`Checkpoint.saveBytes` / `loadIntoBytes`
  in `lib/core/nn/serialize.dart`) — a compact binary snapshot of
  every parameter tensor.
- **`averageCheckpoints(List<Uint8List>, {weights})`** in
  `lib/core/coop/param_avg.dart` — the one function every phase
  reduces down to. It byte-validates that all inputs describe the
  same architecture and then computes a weighted fp32 average of the
  data blob.
- **`CharVocab`** + **`shardSlice`** in `lib/core/coop/shard.dart` —
  a toy character-level tokenizer plus a helper that hands each
  replica a disjoint slice of the token stream.

### Two model families (`--arch=gpt|aft`)

Every phase can train either transformer family:

- `--arch=gpt` (default) — the attention-based `GPT` in
  `lib/core/nn/gpt.dart`. Runs on CPU or GPU.
- `--arch=aft` — the attention-free `AFTLanguageModel` in
  `lib/core/nn/aft_transformer.dart`. **CPU only** — passing
  `--device=gpu --arch=aft` logs a warning and falls back to CPU.

Both families are wrapped behind a common `CoopLM` handle
(`lib/core/coop/lm_factory.dart`) so the coop scripts are model
agnostic. The `tiny` and `small` presets are sized to have near
identical scalar counts across families (e.g. `tiny` is 27008 for
GPT and 26776 for AFT) so you can do apples-to-apples runs.

Because `averageCheckpoints` byte-validates the DPTC header before
averaging, **every process in a run MUST agree on `--arch`**:

- **Phase 0** — all replicas are spawned by the same process from
  the same `--arch`, so this is automatic.
- **Phase 1** — the coordinator advertises its `--arch` via
  `GET /config`, and workers inherit it. Workers do **not** take an
  `--arch` flag. If you point a stale `--arch=gpt` binary at an
  `--arch=aft` coordinator, its `POST /submit` will be rejected
  with HTTP 400 `invalid checkpoint`.
- **Phase 2** — every peer takes its own `--arch`. Mixing families
  in one mesh will throw at the first gossip round when
  `averageCheckpoints` sees a header mismatch. Keep them consistent.

---

## Phase 0 — single-process (`bin/coop_train_phase0.dart`)

Runs N replicas inside one Dart VM. Each replica has its own model,
optimizer, and shard. Every "communication round" happens in-memory:
we grab each replica's bytes, average them with `averageCheckpoints`,
and load the average back into every replica.

Use it to sanity-check the math and to compare against a **baseline**
that trains on the full corpus for `K * N` steps per round (the
apples-to-apples FLOP budget).

```bash
dart run bin/coop_train_phase0.dart --replicas=3 --rounds=8 --local-steps=30
```

Flags:

| Flag             | Default | Meaning                                    |
|------------------|---------|--------------------------------------------|
| `--replicas=N`   | 3       | How many virtual replicas                  |
| `--rounds=N`     | 8       | Communication rounds                       |
| `--local-steps=K`| 30      | Adam steps between averages                |
| `--seed=N`       | 42      |                                            |
| `--device=cpu\|gpu`| cpu   |                                            |
| `--arch=gpt\|aft`| gpt     | Transformer family (see "Two model families" below) |
| `--model=tiny\|small` | tiny |                                          |
| `--corpus=toy\|shakespeare` | toy | `shakespeare` needs `data/tiny_shakespeare.txt` |

Expect the averaged fleet to trail the baseline (baseline sees the
whole corpus every step), but stay stable and monotonically improve.

---

## Phase 1 — HTTP coordinator + workers (`bin/coop_coordinator.dart`, `bin/coop_worker.dart`)

One coordinator process holds the current **global** checkpoint and
the round counter. N workers pull the global, do `K` local Adam steps
on their shard, and POST the result back. When the coordinator has
`--target` fresh contributions **plus** the current global, it
averages them (weighted by `X-Coop-Local-Steps`, with the global
counted at the median weight so no single worker can yank the mean),
bumps the round counter, and starts the next round.

Stale submissions are rejected with HTTP 409 — workers that were
still training when the round rolled over just discard their update
and pull the new global on the next iteration. This is by design and
matches Hivemind's semantics.

### Run

Terminal 1 — coordinator:

```bash
dart run bin/coop_coordinator.dart --port=8765 --target=2
# or, for the attention-free family:
dart run bin/coop_coordinator.dart --port=8765 --target=2 --arch=aft
```

Terminals 2, 3, ... — one per worker (no `--arch` flag; the worker
reads it from the coordinator's `/config`):

```bash
dart run bin/coop_worker.dart \
    --coordinator=http://127.0.0.1:8765 \
    --id=alice --shard-id=0 --num-shards=2 \
    --rounds=8 --local-steps=15
```

```bash
dart run bin/coop_worker.dart \
    --coordinator=http://127.0.0.1:8765 \
    --id=bob   --shard-id=1 --num-shards=2 \
    --rounds=8 --local-steps=15
```

### Coordinator HTTP surface

| Route              | Meaning                                                                                  |
|--------------------|------------------------------------------------------------------------------------------|
| `GET /config`      | JSON: `{version, arch, model, corpus, vocab, maxLen, seed, target}` — workers use `arch` + `model` to build a matching local `CoopLM` |
| `GET /corpus`      | text/plain: full corpus (workers shard it locally via `shardSlice`)                      |
| `GET /checkpoint`  | binary DPTC bytes; response header `X-Coop-Round: N` says which round they belong to     |
| `GET /status`      | JSON: `{round, queueSize, target, workersSeen[]}`                                        |
| `POST /submit`     | Body: DPTC bytes. Headers: `X-Coop-Worker-Id`, `X-Coop-Round`, `X-Coop-Local-Steps`      |

Responses to `/submit`:
- **200** `{accepted:true, aggregated:bool, currentRound:N}` — enqueued (aggregated=true means this submission triggered the average)
- **400** `{accepted:false, reason:"invalid checkpoint: <e>"}` — DPTC failed to load into a fresh model
- **409** `{accepted:false, reason:"stale round", workerRound, currentRound}` — worker is behind; pull fresh global and retry

---

## Phase 2 — peer-to-peer gossip (`bin/coop_peer.dart`)

No coordinator. Each peer runs its own tiny HTTP server exposing
`GET /health` and `GET /checkpoint`, knows a bootstrap peer set via
`--peers=host:port,host:port,...`, and every `--gossip-every` rounds
picks one random peer, downloads their checkpoint, and averages it
into its own with equal weights. Both sides converge to the fleet
mean via diffusion, exactly like Hivemind's `DecentralizedSGD`.

Bootstrap peers only need to know **one** other live peer to join;
they'll gossip with the wider mesh transitively.

### Run 3 peers on one box

```bash
dart run bin/coop_peer.dart --listen=9001 --id=p1 --shard-id=0 --num-shards=3 \
    --peers=127.0.0.1:9002,127.0.0.1:9003 --rounds=20 --warmup-secs=20 &

dart run bin/coop_peer.dart --listen=9002 --id=p2 --shard-id=1 --num-shards=3 \
    --peers=127.0.0.1:9001,127.0.0.1:9003 --rounds=20 --warmup-secs=18 &

dart run bin/coop_peer.dart --listen=9003 --id=p3 --shard-id=2 --num-shards=3 \
    --peers=127.0.0.1:9001,127.0.0.1:9002 --rounds=20 --warmup-secs=16
```

`--warmup-secs=N` makes each peer wait N seconds **after binding its
port** before starting to train. On a single box Dart's cold start
takes 5-10 seconds; without warmup the first several rounds fail
with `Connection refused` because their target peer hasn't finished
`HttpServer.bind()` yet. On a real deployment with peers on separate
machines you can set it to 0.

### Peer HTTP surface

| Route            | Meaning                                                          |
|------------------|------------------------------------------------------------------|
| `GET /health`    | JSON: `{id, round, peers, vocabSize, arch, model}`               |
| `GET /checkpoint`| binary DPTC bytes + `X-Coop-Round: N` + `X-Coop-Peer-Id` + `X-Coop-Arch` |

### Flags

| Flag                | Default              | Meaning                                                  |
|---------------------|----------------------|----------------------------------------------------------|
| `--listen=PORT`     | 9001                 | our TCP port                                             |
| `--host=HOST`       | 127.0.0.1            | bind host                                                |
| `--peers=h:p,h:p`   | empty                | bootstrap peer list (empty = solo mode, no gossip)       |
| `--id=NAME`         | `peer-<pid>`         |                                                          |
| `--shard-id=N`      | 0                    | which corpus shard to train on                           |
| `--num-shards=N`    | `peers.length + 1`   | total shards                                             |
| `--local-steps=K`   | 30                   | Adam steps between gossip rounds                         |
| `--gossip-every=R`  | 1                    | gossip once every R rounds                               |
| `--rounds=N`        | 20                   | max rounds (-1 = forever)                                |
| `--warmup-secs=N`   | 0                    | delay training start by N seconds                        |
| `--arch=gpt\|aft`   | gpt                  | Transformer family. All peers in the mesh must agree.    |
| `--model=tiny\|small`| tiny                |                                                          |
| `--corpus=toy\|shakespeare` | toy         |                                                          |
| `--lr=F`            | 3e-3                 | Adam learning rate                                       |
| `--seed=N`          | 42                   | identical across peers so they start from same weights   |
| `--device=cpu\|gpu` | cpu                  |                                                          |

---

## Which phase should I use?

- **Just trying it?** — Phase 0. One command, no networking, no ports.
- **You and friends on the same LAN with one designated "server"
  machine?** — Phase 1. Easier to reason about (one authoritative
  round counter), lower overhead.
- **A donated-GPU network where anyone can join and any node can
  vanish?** — Phase 2. No single point of failure, no elected leader.

---

## Bandwidth math

Every gossip / aggregation ships one DPTC checkpoint per direction
per round. For the current `tiny` model:

- `27_008` parameters × 4 bytes (fp32) = **108 KiB** per checkpoint
- Small header (a few hundred bytes) on top

So one round for a Phase 1 worker is ~216 KiB (pull + push), and one
gossip exchange for a Phase 2 peer is ~108 KiB (pull only; the peer
you pulled from doesn't get anything from you until it independently
gossips back). A `--local-steps=30` round takes ~1 second on CPU for
this tiny model, so a saturated worker uses about **0.2 MB/s** — well
inside every free tier and every home connection.

For the `small` model (~300k parameters), scale by ~11x — still under
3 MB/s. Real LLMs (100M+ params) would push this to hundreds of MB
per round, which is where Hivemind's gradient-compression tricks
(1-bit quantization, quantile-based sparsification) start to matter.

---

## Security caveats (read this before you open a port to the internet)

**This is a proof of concept**, not a production system. All three
network phases assume a trusted network. Specifically, none of them
have:

- **No auth.** Anyone who can reach the coordinator's `/submit` or a
  peer's `/checkpoint` can push arbitrary weights or pull the current
  model. Currently mitigated only by binding to `127.0.0.1` and
  running on a LAN.
- **No TLS.** All traffic is plaintext HTTP. A network attacker can
  snoop or MITM the checkpoints.
- **No worker validation.** A malicious worker can submit a
  checkpoint that "loads" (correct shapes) but is deliberately
  poisoned to steer the model in a bad direction, or one that
  overshoots to yank the average. The `Phase 1` coordinator does clip
  the effective weight via the median-weight trick, but doesn't
  compare against expected gradient norm or run any consistency
  checks.
- **No sandboxing.** Loading arbitrary DPTC bytes is currently safe
  (it's a fixed binary format, not code), but any future extension
  that ships optimizer state or JIT'd kernels could open a code-exec
  path. Keep an eye on that.
- **No sybil resistance.** Nothing stops one attacker from spinning
  up 100 fake workers to overwhelm the average.

Concrete steps you'd take to open this to the wider internet:

1. Put every HTTP endpoint behind a bearer token. Rotate per-round.
2. Wrap the transport in TLS (or run everything through a Wireguard
   / Tailscale tunnel so peers only see each other).
3. On the receiving side (coordinator, or any peer), reject any
   incoming checkpoint whose L2 distance from the current global is
   more than K standard deviations of the recent update norms.
4. Require a small proof-of-work or a signed staking deposit before
   accepting a new peer/worker.
5. Cross-validate: have each round's average pass a held-out eval
   loss check; roll back if it degrades.

None of that is built yet.

---

## References

- Douillard et al. 2024. *DiLoCo: Distributed Low-Communication
  Training of Language Models.* arXiv:2311.08105.
- Ryabinin et al. 2020-2023. *Hivemind: decentralized deep learning
  in PyTorch.* [github.com/learning-at-home/hivemind](https://github.com/learning-at-home/hivemind)
- Lian et al. 2017. *Can Decentralized Algorithms Outperform
  Centralized Algorithms? A Case Study for Decentralized Parallel
  Stochastic Gradient Descent.* NeurIPS 2017.
