"""Build a mini ONNX that computes just STFT + magnitude, using the
same weights, so we can compare its output tensor with the Dart port."""
import numpy as np
import onnx
import onnxruntime as ort


def load_stft_weight():
    m = onnx.load("/mnt/c/Users/kkinyama/dart-pytorch/models/silero_vad/silero_vad.onnx")
    for g in _walk(m.graph):
        for n in g.node:
            if n.op_type == "Constant":
                for a in n.attribute:
                    if a.name == "value":
                        if n.output[0].endswith("stft.forward_basis_buffer"):
                            arr = onnx.numpy_helper.to_array(a.t)
                            if arr.shape == (258, 1, 256):
                                return arr
    raise SystemExit("stft weight not found")


def _walk(g):
    yield g
    for n in g.node:
        for a in n.attribute:
            if a.type == onnx.AttributeProto.GRAPH:
                yield from _walk(a.g)


def make_chunk(i, n=512):
    if i == 0:
        return np.zeros((1, n), dtype=np.float32)
    t = (np.arange(n) + i * n) / 16000.0
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32).reshape(1, n)


def numpy_conv(x, w, stride=1):
    B, Cin, L = x.shape
    Cout, _, K = w.shape
    Lout = (L - K) // stride + 1
    out = np.zeros((B, Cout, Lout), dtype=np.float32)
    for t in range(Lout):
        window = x[:, :, t * stride:t * stride + K]
        out[:, :, t] = np.einsum("bik,oik->bo", window, w)
    return out


def main():
    w = load_stft_weight()
    print("STFT weight shape", w.shape)

    ctx = np.zeros((1, 64), dtype=np.float32)
    chunk = make_chunk(1)
    inp = np.concatenate([ctx, chunk], axis=1)  # [1, 576]
    inp_c = inp[:, None, :]  # [1, 1, 576]

    # NumPy conv (assuming no additional padding — the ONNX Pad node
    # may add reflect-padding; we test both).
    stft_plain = numpy_conv(inp_c, w, stride=128)
    print("plain conv stft shape", stft_plain.shape)
    real = stft_plain[:, :129, :]
    imag = stft_plain[:, 129:, :]
    mag_plain = np.sqrt(real ** 2 + imag ** 2 + 1e-9)
    print("plain mag [:, 0, :] =", mag_plain[0, 0, :])   # DC bin over 3 frames
    print("plain mag [:, 20, :] =", mag_plain[0, 20, :]) # ~1250 Hz (bin 20 @ 62.5 Hz spacing)

    # If we reflect-pad by 128 on each side of the 576-input, we get 832 samples,
    # so Lout = (832 - 256) / 128 + 1 = 5. That doesn't match the LSTM expecting 3 frames.
    # So the ONNX likely does NOT add reflect padding to the raw stream — the 576
    # input is what's fed. Our plain computation should match.

    # Also try REFLECT padding of size 64 on both sides (matching the paper's spec):
    inp_ref = np.pad(inp_c, ((0, 0), (0, 0), (64, 64)), mode='reflect')
    stft_ref = numpy_conv(inp_ref, w, stride=128)
    real2, imag2 = stft_ref[:, :129, :], stft_ref[:, 129:, :]
    mag_ref = np.sqrt(real2 ** 2 + imag2 ** 2 + 1e-9)
    print("reflect-pad mag shape", mag_ref.shape)
    print("reflect-pad mag [:, 0, :] =", mag_ref[0, 0, :])
    print("reflect-pad mag [:, 20, :] =", mag_ref[0, 20, :])


if __name__ == "__main__":
    main()
