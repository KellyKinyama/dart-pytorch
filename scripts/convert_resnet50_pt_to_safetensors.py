"""Convert torchvision's `resnet50` (ImageNet weights) to a safetensors
file that our `ResNetLoader` can consume.

Prereqs (one-time):
    python3 -m pip install --break-system-packages --user torch torchvision safetensors

Usage:
    python3 scripts/convert_resnet50_pt_to_safetensors.py \\
        models/resnet50/model.safetensors

By default uses the modern torchvision `ResNet50_Weights.IMAGENET1K_V2`
weights (80.86% top-1). Pass `--v1` to use the classic V1 weights
(76.15% top-1) which some older papers reference. Both share the same
architecture — only the training recipe differs.

Also drops the ImageNet class labels JSON alongside the safetensors
so `bin/resnet50_demo.dart` can print human-readable names.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torchvision
    from torchvision.models import ResNet50_Weights, resnet50
    from safetensors.numpy import save_file
except ImportError as e:
    print(f"missing dependency: {e}", file=sys.stderr)
    print(
        "install with:\n  python3 -m pip install --break-system-packages "
        "--user torch torchvision safetensors",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out_path", help="destination model.safetensors path")
    p.add_argument(
        "--v1",
        action="store_true",
        help="use classic ResNet50_Weights.IMAGENET1K_V1 (76.15% top-1) "
        "instead of the default V2 (80.86% top-1)",
    )
    args = p.parse_args()

    weights = (
        ResNet50_Weights.IMAGENET1K_V1
        if args.v1
        else ResNet50_Weights.IMAGENET1K_V2
    )
    print(f"torchvision {torchvision.__version__}: using {weights.name}")
    model = resnet50(weights=weights)
    model.eval()
    state = model.state_dict()

    tensors = {}
    for name, t in state.items():
        if name.endswith(".num_batches_tracked"):
            continue
        arr = t.detach().cpu().numpy().astype(np.float32)
        tensors[name] = np.ascontiguousarray(arr)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path))
    print(f"wrote {len(tensors)} tensors -> {out_path}")

    # Also dump ImageNet class labels next to the safetensors so the
    # Dart demo can map argmax -> "golden retriever" etc.
    categories = weights.meta.get("categories")
    if categories:
        labels_path = out_path.parent / "imagenet_classes.json"
        labels_path.write_text(json.dumps(categories, indent=2))
        print(f"wrote {len(categories)} labels -> {labels_path}")

    for name in sorted(tensors)[:5]:
        print(f"  {name}: {tensors[name].shape}")
    print("  ...")
    for name in sorted(tensors)[-3:]:
        print(f"  {name}: {tensors[name].shape}")


if __name__ == "__main__":
    main()
