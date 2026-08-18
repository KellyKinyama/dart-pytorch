"""Surgery on silero_vad.onnx: keep the STFT+magnitude subgraph
(nodes ~20 through 69 of the 16 kHz then_branch), rewrite it into
a standalone ONNX with 'input' as the sole input and mag_output as
the sole output. Run it via ORT for a known chunk and print the
result — then we compare vs Dart's mag."""
import onnx
import numpy as np
import onnxruntime as ort


SRC = "/mnt/c/Users/kkinyama/dart-pytorch/models/silero_vad/silero_vad.onnx"
DST = "/tmp/silero_vad_stft_only.onnx"

TARGET_MAG_TAIL = "/encoder/3/activation/Relu_output_0"


def then_branch(g):
    for n in g.node:
        if n.op_type == "If":
            for a in n.attribute:
                if a.name == "then_branch":
                    return a.g
    return None


def main():
    model = onnx.load(SRC)
    sub = then_branch(model.graph)

    # Find the target node whose output we want (Relu for encoder.3).
    sqrt_out_name = None
    for n in sub.node:
        if n.op_type in ("Sqrt", "Relu"):
            if n.output[0].endswith(TARGET_MAG_TAIL):
                sqrt_out_name = n.output[0]
                break
    if sqrt_out_name is None:
        raise SystemExit("could not find target node")
    print("target output name:", sqrt_out_name)

    # Build a new graph by KEEPING every node whose output ancestry contains
    # `sqrt_out_name` OR is a Constant, and rewrite the parent-graph input.
    # In practice: keep every node up to and including the Sqrt.
    needed = set()

    producers = {}
    for n in sub.node:
        for o in n.output:
            producers[o] = n

    stack = [sqrt_out_name]
    while stack:
        name = stack.pop()
        n = producers.get(name)
        if n is None:
            continue  # graph input like "input"
        if id(n) in needed:
            continue
        needed.add(id(n))
        for inp in n.input:
            if inp and inp not in producers:  # graph input
                continue
            stack.append(inp)

    kept_nodes = [n for n in sub.node if id(n) in needed]
    print(f"Kept {len(kept_nodes)} nodes")

    # Build new graph.
    new_graph = onnx.helper.make_graph(
        nodes=kept_nodes,
        name="stft_only",
        inputs=[
            onnx.helper.make_tensor_value_info(
                "input", onnx.TensorProto.FLOAT, [1, "L"]
            )
        ],
        outputs=[
            onnx.helper.make_tensor_value_info(
                sqrt_out_name, onnx.TensorProto.FLOAT, [1, 129, "T"]
            )
        ],
    )
    new_model = onnx.helper.make_model(new_graph, opset_imports=[onnx.helper.make_opsetid("", 16)])
    new_model.ir_version = 8
    onnx.save(new_model, DST)
    print(f"Wrote {DST}")

    # Run the mini model on chunk 1.
    def make_chunk(i, n=512):
        if i == 0:
            return np.zeros((1, n), dtype=np.float32)
        t = (np.arange(n) + i * n) / 16000.0
        return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32).reshape(1, n)

    ctx = np.zeros((1, 64), dtype=np.float32)
    chunk = make_chunk(1)
    inp = np.concatenate([ctx, chunk], axis=1)
    print("input shape", inp.shape)
    sess = ort.InferenceSession(DST, providers=["CPUExecutionProvider"])
    (mag,) = sess.run(None, {"input": inp})
    print("ONNX output shape:", mag.shape)
    print("ONNX max:", mag.max(), "min:", mag.min())
    print("ONNX mean:", mag.mean(), "std:", mag.std())


if __name__ == "__main__":
    main()
