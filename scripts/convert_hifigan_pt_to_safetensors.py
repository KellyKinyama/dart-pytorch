"""Convert a HiFi-GAN V1 generator checkpoint (jik876/hifi-gan or
speechbrain/tts-hifigan-* layout) to a safetensors file that our
`HiFiGanLoader` can consume.

Prereqs (one-time):
    python3 -m pip install --break-system-packages --user \\
        torch safetensors

The upstream jik876/hifi-gan checkpoint is a `.pth` PyTorch state_dict
wrapped by `torch.save({'generator': generator.state_dict()}, path)`.
This script:

  1. Reads the .pth file and extracts the 'generator' submodel.
  2. **Folds weight_norm** — jik876/hifi-gan applies
     `torch.nn.utils.weight_norm` to every conv, so each conv weight
     is split into `weight_g` (magnitude, `[Cout, 1, 1]`) and
     `weight_v` (direction, `[Cout, Cin, K]`). We reconstruct the
     dense weight as `w = weight_v * (weight_g / ||weight_v||)` and
     write it back under the plain `weight` key.
  3. Writes a flat safetensors file with the folded weights.

Usage:
    python3 scripts/convert_hifigan_pt_to_safetensors.py INPUT.pth \\
        models/hifigan-ljspeech/model.safetensors

The dumped keys match what `HiFiGanLoader.loadMap` expects:
  conv_pre.{weight,bias}
  ups.{i}.{weight,bias}
  resblocks.{n}.convs{1,2}.{k}.{weight,bias}
  conv_post.{weight,bias}
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


def _fold_weight_norm(sd: dict) -> dict:
    """Merge `*.weight_g` / `*.weight_v` pairs into a single `.weight`.

    weight_g: [Cout, 1, 1]  (per-out-channel norm target)
    weight_v: [Cout, Cin, K]
    """
    out: dict = {}
    fold_keys = set()
    for k in list(sd.keys()):
        if k.endswith(".weight_g"):
            base = k[: -len(".weight_g")]
            wg = sd[base + ".weight_g"].detach().cpu().numpy().astype(np.float32)
            wv = sd[base + ".weight_v"].detach().cpu().numpy().astype(np.float32)
            # Normalise along all axes except the output-channel axis.
            axes = tuple(range(1, wv.ndim))
            norm = np.sqrt(np.sum(wv * wv, axis=axes, keepdims=True))
            w = wv * (wg / (norm + 1e-12))
            out[base + ".weight"] = w
            fold_keys.add(base + ".weight_g")
            fold_keys.add(base + ".weight_v")
    for k, t in sd.items():
        if k in fold_keys:
            continue
        if k.endswith(".weight") and (k + "_g") in sd:
            # Already folded above; skip the raw copy.
            continue
        arr = t.detach().cpu().numpy().astype(np.float32)
        out[k] = np.ascontiguousarray(arr)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("in_path", help="input .pth or .pt file")
    p.add_argument("out_path", help="output model.safetensors path")
    p.add_argument(
        "--key",
        default="generator",
        help="key inside the .pth dict holding the generator state_dict "
        "(default: 'generator'). Pass '' if the .pth already IS the "
        "state_dict.",
    )
    args = p.parse_args()

    obj = torch.load(args.in_path, map_location="cpu")
    if isinstance(obj, dict) and args.key and args.key in obj:
        sd = obj[args.key]
    else:
        sd = obj

    folded = _fold_weight_norm(sd)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: np.ascontiguousarray(v) for k, v in folded.items()}
    save_file(tensors, str(out_path))
    print(f"wrote {len(tensors)} tensors -> {out_path}")

    for name in sorted(tensors)[:8]:
        print(f"  {name}: {tensors[name].shape}")
    print("  ...")
    for name in sorted(tensors)[-3:]:
        print(f"  {name}: {tensors[name].shape}")


if __name__ == "__main__":
    main()
