"""Modify silero_vad.onnx to expose intermediate outputs, then run it
on the same chunk-1 input we use in the Dart verifier."""
import copy
import numpy as np
import onnx
import onnxruntime as ort


ONNX_PATH = "/mnt/c/Users/kkinyama/dart-pytorch/models/silero_vad/silero_vad.onnx"


def make_chunk(i, n=512):
    if i == 0:
        return np.zeros((1, n), dtype=np.float32)
    t = (np.arange(n) + i * n) / 16000.0
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32).reshape(1, n)


# Find intermediate tensors we want to expose (any output name we like).
# The names have `Inline_0__` prefixes because of the If-inlining.
INTEREST = [
    ("stft_mag_output", "/stft/Sqrt_output_0"),
    ("enc0_relu", "/encoder/0/activation/Relu_output_0"),
    ("enc3_relu", "/encoder/3/activation/Relu_output_0"),
    ("lstm_Y", "/decoder/rnn/LSTM_output_0"),  # all-timestep hidden output
    ("head_conv", "/decoder/decoder/2/Conv_output_0"),
    ("head_sigmoid", "/decoder/decoder/3/Sigmoid_output_0"),
]


def find_producer(graphs, tail):
    for g in graphs:
        for n in g.node:
            for o in n.output:
                if o.endswith(tail):
                    return o
        for n in g.node:
            for a in n.attribute:
                if a.type == onnx.AttributeProto.GRAPH:
                    r = find_producer([a.g], tail)
                    if r is not None:
                        return r
    return None


def main():
    model = onnx.load(ONNX_PATH)
    graph = model.graph

    # Enter 16k branch, since we're doing 16 kHz.
    then_branch = None
    for n in graph.node:
        if n.op_type == "If":
            for a in n.attribute:
                if a.name == "then_branch":
                    then_branch = a.g
                    break

    # Add each intermediate as a graph output on the then_branch. To
    # surface at the top level, we also need to add matching outputs
    # to the else_branch (with a stub value) or ORT will refuse to run.
    # Cheaper approach: add outputs to the outer graph via new Identity
    # nodes reading directly from the intermediate (with `Inline_0__`
    # prefix).
    print("Available intermediate tensor names:")
    for key, tail in INTEREST:
        full = find_producer([graph], tail)
        print(f"  {key:15s} → {full}")


if __name__ == "__main__":
    main()
