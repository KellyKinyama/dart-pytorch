"""Convert facenet-pytorch's `20180402-114759-vggface2.pt` checkpoint
to a safetensors file that our WhisperHFLoader-style Dart loader can
consume.

Prereqs (one-time):
    python3 -m pip install --break-system-packages --user facenet-pytorch safetensors torch

Usage:
    python3 scripts/convert_facenet_pt_to_safetensors.py \\
        models/facenet-vggface2/model.safetensors

The PyTorch checkpoint is downloaded on first use via `facenet-pytorch`
(cached at ~/.cache/torch/checkpoints/). This script:

  1. Instantiates `InceptionResnetV1(pretrained='vggface2', classify=False)`
  2. Iterates `state_dict()` — every conv is followed by a BN and every
     BN carries `weight` (gamma), `bias` (beta), `running_mean`,
     `running_var`, and a scalar `num_batches_tracked` (which we drop).
  3. Writes a flat safetensors file keyed by the original param names.

The Dart loader (`facenet_loader.dart`) then walks the module tree and
pairs each conv with its BN sibling to run the fold.
"""

import sys
import numpy as np

try:
    import torch
    from facenet_pytorch import InceptionResnetV1
    from safetensors.numpy import save_file
except ImportError as e:
    print(f"missing dependency: {e}", file=sys.stderr)
    print(
        "install with:\n  python3 -m pip install --break-system-packages "
        "--user facenet-pytorch safetensors torch",
        file=sys.stderr,
    )
    sys.exit(1)


def main(out_path: str) -> None:
    model = InceptionResnetV1(pretrained="vggface2", classify=False)
    model.eval()
    state = model.state_dict()

    tensors = {}
    for name, t in state.items():
        if name.endswith(".num_batches_tracked"):
            continue
        arr = t.detach().cpu().numpy().astype(np.float32)
        tensors[name] = np.ascontiguousarray(arr)

    save_file(tensors, out_path)
    print(f"wrote {len(tensors)} tensors -> {out_path}")

    # Handy report of the key shapes.
    for name in sorted(tensors)[:10]:
        print(f"  {name}: {tensors[name].shape}")
    print("  ...")
    for name in sorted(tensors)[-5:]:
        print(f"  {name}: {tensors[name].shape}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: convert_facenet_pt_to_safetensors.py OUTPUT_PATH")
        sys.exit(1)
    main(sys.argv[1])
