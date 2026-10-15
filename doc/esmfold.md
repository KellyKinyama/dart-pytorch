# Native ESMFold (protein structure prediction) in pure Dart

An in-progress, pure-Dart port of **ESMFold** — Meta's single-sequence,
AlphaFold2-class protein structure predictor — built on `dart_pytorch`. It
turns an amino-acid sequence into 3D backbone coordinates without a Python
runtime, MSAs, or external services.

ESMFold = an **ESM-2** protein language model (the 3B checkpoint for
`esmfold_v1`) feeding a **folding trunk** (Evoformer-style triangular
self-attention) and an AlphaFold2 **structure module** (Invariant Point
Attention) that emits per-residue rigid frames, which place the backbone atoms.

Everything here mirrors the layouts in `esm.esmfold.v1` and HuggingFace
`transformers/models/esm/modeling_esmfold.py` so a pretrained
`facebook/esmfold_v1` checkpoint maps onto these modules.

## Files

```
lib/core/nn/fold/
  geometry.dart            # Vec3 / Rot3 / Rigid frames, quaternion->rotation,
                           #   Gram-Schmidt frame-from-3-atoms, ideal backbone
  structure_module.dart    # Invariant Point Attention (IPA), backbone update,
                           #   shared-weight structure-module block stack
  trunk.dart               # TriangularSelfAttentionBlock: gated sequence
                           #   attention, SequenceToPair / PairToSequence,
                           #   triangle multiplication (out/in), triangle
                           #   self-attention (start/end), residue MLPs
  esmfold_loader.dart      # HF `facebook/esmfold_v1` key mapping (load + dump)
lib/core/nn/
  esm2.dart                # ESM-2 LM + representations() per-layer stack
  esm2_hf_loader.dart      # configs incl. esm2_3bConfig (ESMFold's stem)
test/fold/                 # geometry, structure module, trunk, representations,
                           #   and loader round-trip tests
```

## Design notes

- **Geometry is pure Dart** (double precision). `Rigid` composition implements
  AlphaFold's `compose_q_update_vec` exactly: the rotation is right-multiplied
  by the quaternion update and the translation is updated in the current frame.
- **Trunk and structure module are pure Dart** (`Float32List` + explicit loops)
  on top of the geometry module — deterministic and unit-testable without any
  weights. The IPA output concatenation order is
  `(o, o_pt_x, o_pt_y, o_pt_z, o_pt_norm, o_pair)`, matching OpenFold so the
  pretrained `linear_out` transfers.
- **The ESM-2 stem uses the `Tensor` core** (CPU or GPU). `representations()`
  returns the full per-layer hidden-state stack (`numLayers + 1`, index 0 the
  raw embedding), which ESMFold combines with a learned per-layer softmax.

## Correctness: invariance / equivariance tests

The structure is validated by symmetry properties rather than reference weights:

- IPA's single-representation update is **invariant** to a global rigid motion
  of the input frames.
- Backbone coordinates are **equivariant** under a global rotation.
- The trunk block is **equivariant to a joint residue permutation** — a single
  test that exercises every `i/j` index across the triangle operations.
- `finalLn(representations(tokens).last) == call(tokens)` for the ESM-2 stem.
- The loader round-trips a module through the HF key map bit-for-bit.

Run them with:

```bash
dart test test/fold/
```

## Pretrained weights

`esmfold_loader.dart` maps the HF checkpoint's state dict onto these modules
using key strings taken verbatim from `modeling_esmfold.py` (e.g.
`trunk.blocks.{i}.tri_att_start.mha.linear_q.weight`,
`trunk.structure_module.ipa.head_weights`). HF `Linear` weights are stored
`[out, in]` row-major, matching the internal `Dense` layout, so tensors copy
directly. Sharded safetensors load via `SafeTensors.loadSharded(index.json)`;
the ESM-2 3B stem loads through `ESM2HFLoader` with the `esm.` prefix.

## Status

| Stage | Component | State |
|-------|-----------|-------|
| Geometry | rigid frames, quaternions | done |
| Structure module | Invariant Point Attention | done (backbone N/Cα/C) |
| Folding trunk | triangular self-attention | done |
| ESM-2 stem | representation stack + 3B config | done |
| Weight loader | trunk + structure module keys | done |
| Assembly | end-to-end forward + PDB output | pending |
| Sidechains | angle resnet / full atom14 | deferred |

The end-to-end wiring (ESM-2 → `esm_s_combine`/`esm_s_mlp` → trunk → `trunk2sm`
→ structure module → PDB) and the top-level glue weight loader are the remaining
work. Running the full model requires the multi-GB `facebook/esmfold_v1`
checkpoint and is heavy on CPU.
