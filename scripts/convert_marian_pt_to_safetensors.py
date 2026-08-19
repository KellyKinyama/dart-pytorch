#!/usr/bin/env python3
"""Convert HF MarianMT pytorch_model.bin -> safetensors.

Usage:
    python3 scripts/convert_marian_pt_to_safetensors.py \
        models/opus-mt-en-de/pytorch_model.bin \
        models/opus-mt-en-de/model.safetensors

Reads the pickled state_dict from `pytorch_model.bin` and re-serialises
each tensor as fp32 into safetensors. No key renaming — the Dart-side
`MarianHFLoader` expects HF's original layout
(`model.encoder.layers.{i}.self_attn.*`, `model.shared.weight`, etc.).
"""
import sys
import torch
from safetensors.torch import save_file


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(64)
    in_path, out_path = sys.argv[1], sys.argv[2]

    state = torch.load(in_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Marian ships the same shared embedding under several aliases
    # (model.shared, model.encoder.embed_tokens, model.decoder.embed_tokens,
    # lm_head). safetensors refuses to serialise aliased storages, so
    # keep only model.shared and drop the aliases — the loader will
    # resolve them.
    aliases_to_drop = (
        "model.encoder.embed_tokens.weight",
        "model.decoder.embed_tokens.weight",
        "lm_head.weight",
    )
    cleaned = {}
    for k, v in state.items():
        if k in aliases_to_drop:
            continue
        if not isinstance(v, torch.Tensor):
            continue
        cleaned[k] = v.detach().clone().contiguous().float()

    save_file(cleaned, out_path)
    print(f"wrote {len(cleaned)} tensors -> {out_path}")

    # Terse category summary.
    categories = {}
    for k in cleaned:
        prefix = k.split(".")[0] if "." in k else k
        if k.startswith("model.encoder"):
            prefix = "model.encoder.*"
        elif k.startswith("model.decoder"):
            prefix = "model.decoder.*"
        elif k.startswith("model.shared"):
            prefix = "model.shared"
        categories[prefix] = categories.get(prefix, 0) + 1
    for k, v in sorted(categories.items()):
        print(f"  {k}: {v} tensors")


if __name__ == "__main__":
    main()
