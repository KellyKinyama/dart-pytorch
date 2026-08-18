"""Extract Silero VAD v5 weights from the official ONNX file into a
compact binary that dart_pytorch's SileroVAD loader can read.

Wire format (all little-endian):
    magic:   4 bytes "SVAD"
    version: uint32 = 1
    n_tensors: uint32
    for each tensor:
        name_len: uint32
        name:     name_len bytes (UTF-8)
        n_dims:   uint32
        dims:     n_dims × int32
        dtype:    uint32 (1 = float32, only float32 supported here)
        data:     product(dims) × 4 bytes (fp32, little-endian)

Only the named weight tensors (matching PyTorch parameter names)
are exported. Bias vectors are 1-D. Conv kernels are stored in
PyTorch's [out, in, k] layout.

Run from the dart-pytorch repo root:
    python3 scripts/extract_silero_vad.py \\
        models/silero_vad/silero_vad.onnx \\
        models/silero_vad/silero_vad.dpt
"""
import struct
import sys

import onnx
import numpy as np


NAMED_PREFIXES = (
    "stft.forward_basis_buffer",
    "encoder.0.reparam_conv.weight",
    "encoder.0.reparam_conv.bias",
    "encoder.1.reparam_conv.weight",
    "encoder.1.reparam_conv.bias",
    "encoder.2.reparam_conv.weight",
    "encoder.2.reparam_conv.bias",
    "encoder.3.reparam_conv.weight",
    "encoder.3.reparam_conv.bias",
    "decoder.decoder.2.weight",
    "decoder.decoder.2.bias",
    "decoder.rnn.weight_ih",
    "decoder.rnn.weight_hh",
    "decoder.rnn.bias_ih",
    "decoder.rnn.bias_hh",
)


def all_constants(graph):
    """Yield (name, ndarray) for every Constant node in `graph`, recursively descending into If subgraphs."""
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    t = attr.t
                    yield node.output[0], onnx.numpy_helper.to_array(t)
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                yield from all_constants(attr.g)


def strip_prefix(name):
    """The ONNX weights come out with a giant scope prefix like
    'If_0_then_branch__Inline_0__stft.forward_basis_buffer'. We only
    care about the PyTorch parameter suffix."""
    marker = "Inline_0__"
    idx = name.rfind(marker)
    return name[idx + len(marker):] if idx >= 0 else name


def write_dpt(path, tensors):
    with open(path, "wb") as f:
        f.write(b"SVAD")
        f.write(struct.pack("<II", 1, len(tensors)))
        for name, arr in tensors:
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            f.write(struct.pack("<I", len(name)))
            f.write(name.encode("utf-8"))
            f.write(struct.pack("<I", arr.ndim))
            for d in arr.shape:
                f.write(struct.pack("<i", int(d)))
            f.write(struct.pack("<I", 1))
            f.write(arr.tobytes())


def main():
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <in.onnx> <out.dpt>")
    onnx_path, dpt_path = sys.argv[1], sys.argv[2]
    model = onnx.load(onnx_path)

    # Top-level graph has one If node comparing `sr == 16000`. The
    # `then_branch` is the 16 kHz path — that's the only one we want.
    then_branch = None
    for node in model.graph.node:
        if node.op_type == "If":
            for attr in node.attribute:
                if attr.name == "then_branch":
                    then_branch = attr.g
                    break
    if then_branch is None:
        raise SystemExit("no If node in top-level graph — unexpected model layout")

    named = {}
    for raw_name, arr in all_constants(then_branch):
        clean = strip_prefix(raw_name)
        for pref in NAMED_PREFIXES:
            if clean.endswith(pref):
                if pref in named:
                    continue
                named[pref] = arr
                break

    print(f"Named tensors extracted: {len(named)} / {len(NAMED_PREFIXES)}")
    for k in NAMED_PREFIXES:
        v = named.get(k)
        print(f"  {k:40s} {list(v.shape) if v is not None else '(MISSING)'}")

    missing = [k for k in NAMED_PREFIXES if k not in named]
    if missing:
        raise SystemExit(f"missing tensors: {missing}")

    tensors = [(k, named[k]) for k in NAMED_PREFIXES]

    write_dpt(dpt_path, tensors)
    total_params = sum(v.size for _, v in tensors)
    total_bytes = total_params * 4 + 4 + 8 + sum(4 + len(n) + 4 + 4 * a.ndim + 4 for n, a in tensors)
    print(f"Wrote {dpt_path}: {len(tensors)} tensors, "
          f"{total_params:,} params ({total_bytes / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
