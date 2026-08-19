"""Convert a SAM (Segment Anything) `.pth` checkpoint to safetensors.

Prereqs (one-time):
    python3 -m pip install --break-system-packages --user torch safetensors

Usage:
    # Convert original Meta checkpoint (from
    # https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth):
    python3 scripts/convert_sam_pt_to_safetensors.py \\
        sam_vit_b_01ec64.pth models/sam-vit-b/model.safetensors

    # Optional: also dump for ViT-L (~2.4 GB fp32) or ViT-H (~2.5 GB fp32).

The Meta checkpoints are plain `torch.save(model.state_dict())` files —
no wrapper dict. Every key already matches what our `SamHFLoader`
expects (image_encoder.*, prompt_encoder.*, mask_decoder.*), so the
script is essentially a fp32 → safetensors format converter.

Ignored keys: `image_encoder.blocks.{i}.attn.rel_pos_h/w` for GLOBAL
attention layers use a shape `[2*grid-1, head_dim] = [127, 64]` for
ViT-B; the loader expects the exact per-block shape and picks the
right window_size at load time. No transformation needed here.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    from safetensors.numpy import save_file
except ImportError as e:
    print(f"missing dependency: {e}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("in_path", help="input .pth file (Meta SAM checkpoint)")
    p.add_argument("out_path", help="output model.safetensors path")
    p.add_argument(
        "--key",
        default=None,
        help="if the .pth is a dict wrapper, key to unwrap (default: None; "
        "for Meta's SAM the .pth already IS the state_dict).",
    )
    args = p.parse_args()

    obj = torch.load(args.in_path, map_location="cpu")
    if isinstance(obj, dict) and args.key and args.key in obj:
        sd = obj[args.key]
    else:
        sd = obj

    tensors = {}
    for name, t in sd.items():
        # Only serialize tensor entries. Meta SAM checkpoints have no
        # non-tensor bookkeeping, but be defensive.
        if not hasattr(t, "detach"):
            continue
        arr = t.detach().cpu().numpy().astype(np.float32)
        tensors[name] = np.ascontiguousarray(arr)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path))
    print(f"wrote {len(tensors)} tensors -> {out_path}")

    # Print a small summary so users can spot obvious shape drift.
    key_groups = {"image_encoder.": 0, "prompt_encoder.": 0, "mask_decoder.": 0}
    for k in tensors:
        for g in key_groups:
            if k.startswith(g):
                key_groups[g] += 1
                break
    for g, n in key_groups.items():
        print(f"  {g}* : {n} tensors")


if __name__ == "__main__":
    main()
