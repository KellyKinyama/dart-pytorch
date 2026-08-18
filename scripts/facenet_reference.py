"""Run facenet-pytorch on a chosen face crop and dump the 512-d embedding.

    python3 scripts/facenet_reference.py "faces_gallery/Brad Pitt/sample_0.jpg"

Writes:
    /tmp/facenet_input.raw   (1 * 3 * 160 * 160, fp32, MTCNN-style crop)
    /tmp/facenet_ref.raw     (512 fp32, L2-normalized)
    /tmp/facenet_ref.txt     (embedding stats + first 5 values)

The input crop is what facenet-pytorch's own preprocessing produces:
resize to 160×160 (Image.LANCZOS), convert to `(img - 127.5) / 128.0`,
[H, W, 3] → [3, H, W]. Feeding that raw file into the Dart pipeline
gives us a like-for-like comparison.
"""

import sys
import numpy as np
from PIL import Image

try:
    import torch
    from facenet_pytorch import InceptionResnetV1
except ImportError as e:
    print(f"missing dep: {e}", file=sys.stderr)
    sys.exit(1)


def prep(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((160, 160), Image.LANCZOS)
    a = np.asarray(img, dtype=np.float32)
    a = (a - 127.5) / 128.0
    return a.transpose(2, 0, 1)


def main(path: str) -> None:
    x = prep(path)
    x.tofile("/tmp/facenet_input.raw")
    print(f"input: {x.shape}  min={x.min():+.4f}  max={x.max():+.4f}  "
          f"mean={x.mean():+.4f}")

    model = InceptionResnetV1(pretrained="vggface2", classify=False)
    model.eval()

    with torch.no_grad():
        emb = model(torch.from_numpy(x).unsqueeze(0))
    e = emb.detach().cpu().numpy().astype(np.float32).ravel()
    e.tofile("/tmp/facenet_ref.raw")
    print(f"emb   : {e.shape}  norm={np.linalg.norm(e):.6f}  "
          f"min={e.min():+.4f}  max={e.max():+.4f}")
    print(f"first 5: {e[:5]}")

    with open("/tmp/facenet_ref.txt", "w") as f:
        f.write(f"path: {path}\n")
        f.write(f"shape: {list(e.shape)}\n")
        f.write(f"norm: {float(np.linalg.norm(e)):.6f}\n")
        f.write(f"mean: {float(e.mean()):.6f}\n")
        f.write(f"first5: {e[:5].tolist()}\n")
    print("wrote /tmp/facenet_input.raw, /tmp/facenet_ref.raw, "
          "/tmp/facenet_ref.txt")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: facenet_reference.py PATH_TO_FACE_IMAGE")
        sys.exit(1)
    main(sys.argv[1])
