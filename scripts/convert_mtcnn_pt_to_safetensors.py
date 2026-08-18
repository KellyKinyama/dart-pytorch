"""Convert facenet-pytorch's bundled MTCNN weights to safetensors.

    python3 scripts/convert_mtcnn_pt_to_safetensors.py \\
        models/mtcnn/pnet.safetensors \\
        models/mtcnn/rnet.safetensors \\
        models/mtcnn/onet.safetensors

facenet-pytorch ships the P/R/O-Net `.pt` files inside its package
(no external download); we just instantiate the three modules,
grab `state_dict()`, and dump as flat safetensors. The Dart loader
walks the tensor map and pairs each Conv2d with its PReLU slope.
"""

import sys
import numpy as np

try:
    import torch
    from facenet_pytorch.models.mtcnn import PNet, RNet, ONet
    from safetensors.numpy import save_file
except ImportError as e:
    print(f"missing dep: {e}", file=sys.stderr)
    sys.exit(1)


def dump(model: torch.nn.Module, out_path: str, label: str) -> None:
    model.eval()
    state = model.state_dict()
    tensors = {}
    for name, t in state.items():
        arr = t.detach().cpu().numpy().astype(np.float32)
        tensors[name] = np.ascontiguousarray(arr)
    save_file(tensors, out_path)
    print(f"[{label}] wrote {len(tensors)} tensors -> {out_path}")
    for k in sorted(tensors)[:5]:
        print(f"    {k}: {tensors[k].shape}")
    if len(tensors) > 5:
        print(f"    ... ({len(tensors) - 5} more)")


def main(p_out: str, r_out: str, o_out: str) -> None:
    dump(PNet(pretrained=True), p_out, "PNet")
    dump(RNet(pretrained=True), r_out, "RNet")
    dump(ONet(pretrained=True), o_out, "ONet")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: convert_mtcnn_pt_to_safetensors.py "
              "PNET_OUT RNET_OUT ONET_OUT")
        sys.exit(1)
    main(*sys.argv[1:])
