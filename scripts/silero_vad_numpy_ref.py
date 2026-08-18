"""Faithful NumPy reimplementation of Silero VAD v5 using the raw
weights we exported. Prints intermediate tensor summaries so we can
compare against the Dart port layer-by-layer."""
import struct
import numpy as np


DPT = "/mnt/c/Users/kkinyama/dart-pytorch/models/silero_vad/silero_vad.dpt"


def load_dpt(path):
    with open(path, "rb") as f:
        raw = f.read()
    assert raw[:4] == b"SVAD"
    off = 4
    (version, n) = struct.unpack_from("<II", raw, off)
    off += 8
    tensors = {}
    for _ in range(n):
        (name_len,) = struct.unpack_from("<I", raw, off)
        off += 4
        name = raw[off:off + name_len].decode()
        off += name_len
        (ndim,) = struct.unpack_from("<I", raw, off)
        off += 4
        dims = struct.unpack_from(f"<{ndim}i", raw, off)
        off += 4 * ndim
        (dtype,) = struct.unpack_from("<I", raw, off)
        off += 4
        count = int(np.prod(dims))
        arr = np.frombuffer(raw[off:off + count * 4], dtype=np.float32).reshape(dims).copy()
        off += count * 4
        tensors[name] = arr
    return tensors


def conv1d(x, w, b=None, stride=1, padding=0):
    """PyTorch-style Conv1d: x [B, Cin, L], w [Cout, Cin, K], b [Cout].
    Output shape [B, Cout, Lout]."""
    B, Cin, L = x.shape
    Cout, _, K = w.shape
    Lout = (L + 2 * padding - K) // stride + 1
    if padding:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding)))
    out = np.zeros((B, Cout, Lout), dtype=np.float32)
    for t in range(Lout):
        window = x[:, :, t * stride:t * stride + K]  # [B, Cin, K]
        out[:, :, t] = np.einsum("bik,oik->bo", window, w)
    if b is not None:
        out += b[None, :, None]
    return out


def lstm_step(x, h, c, w_ih, w_hh, b_ih, b_hh):
    """One PyTorch LSTMCell step (gate order i, f, g, o)."""
    gates = x @ w_ih.T + b_ih + h @ w_hh.T + b_hh  # [B, 4H]
    H = h.shape[1]
    i = 1 / (1 + np.exp(-gates[:, 0:H]))
    f = 1 / (1 + np.exp(-gates[:, H:2*H]))
    g = np.tanh(gates[:, 2*H:3*H])
    o = 1 / (1 + np.exp(-gates[:, 3*H:4*H]))
    c_new = f * c + i * g
    h_new = o * np.tanh(c_new)
    return h_new, c_new


def make_chunk(i, n=512):
    if i == 0:
        return np.zeros((1, n), dtype=np.float32)
    t = (np.arange(n) + i * n) / 16000.0
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32).reshape(1, n)


def print_summary(name, arr):
    a = arr.ravel()
    print(f"  {name:30s} shape={list(arr.shape)}  min={a.min():.5f}  max={a.max():.5f}  mean={a.mean():.5f}  norm={np.linalg.norm(a):.4f}")


def main():
    t = load_dpt(DPT)
    print("Loaded", len(t), "tensors")

    # Simulate streaming, print at each chunk
    ctx = np.zeros((1, 64), dtype=np.float32)
    h = np.zeros((1, 128), dtype=np.float32)
    c = np.zeros((1, 128), dtype=np.float32)

    for chunk_idx in [0, 1]:
        print(f"\n=== chunk {chunk_idx} ===")
        chunk = make_chunk(chunk_idx)  # [1, 512]

        # Concat ctx and chunk, then reflect-pad the RIGHT side by 64
        # samples so the STFT produces 4 frames (matches silero_vad.onnx).
        full = np.concatenate([ctx, chunk], axis=1)  # [1, 576]
        full = np.pad(full, ((0, 0), (0, 64)), mode="reflect")  # [1, 640]
        print_summary("input(ctx+chunk+padR64)", full)

        # STFT: conv1d with stride=128, kernel=256, no padding
        full_c = full[:, None, :]  # [1, 1, 576]
        w_stft = t["stft.forward_basis_buffer"]  # [258, 1, 256]
        stft = conv1d(full_c, w_stft, stride=128, padding=0)  # [1, 258, 3]
        print_summary("stft raw", stft)

        # Magnitude: sqrt(re^2 + im^2 + eps)
        real = stft[:, :129, :]
        imag = stft[:, 129:, :]
        mag = np.sqrt(real ** 2 + imag ** 2 + 1e-9)  # [1, 129, 3]
        print_summary("stft mag", mag)

        # Encoder — strides 1, 2, 2, 1 from silero_vad.onnx.
        strides = [1, 2, 2, 1]
        h_enc = mag
        for i in range(4):
            w = t[f"encoder.{i}.reparam_conv.weight"]
            b = t[f"encoder.{i}.reparam_conv.bias"]
            h_enc = conv1d(h_enc, w, b, stride=strides[i], padding=1)
            h_enc = np.maximum(h_enc, 0)
            print_summary(f"encoder.{i}", h_enc)
        # h_enc: [1, 128, 3]

        # LSTM: run 3 steps (one per encoder frame)
        w_ih = t["decoder.rnn.weight_ih"]  # [512, 128]
        w_hh = t["decoder.rnn.weight_hh"]  # [512, 128]
        b_ih = t["decoder.rnn.bias_ih"]
        b_hh = t["decoder.rnn.bias_hh"]
        head_seq = []
        for step in range(h_enc.shape[2]):
            x = h_enc[:, :, step]  # [1, 128]
            h, c = lstm_step(x, h, c, w_ih, w_hh, b_ih, b_hh)
            head_seq.append(h)
        head_input = np.stack(head_seq, axis=2)  # [1, 128, 3]
        print_summary("lstm out (h stack)", head_input)

        # Head: ReLU + Conv1d(128->1, k=1) + Sigmoid + ReduceMean over T
        h_head = np.maximum(head_input, 0)
        w_head = t["decoder.decoder.2.weight"]  # [1, 128, 1]
        b_head = t["decoder.decoder.2.bias"]
        h_head = conv1d(h_head, w_head, b_head)  # [1, 1, 3]
        h_head = 1 / (1 + np.exp(-h_head))
        prob = h_head.mean(axis=2)  # [1, 1]
        print(f"  → prob = {float(prob[0, 0]):.6f}")

        ctx = chunk[:, -64:]


if __name__ == "__main__":
    main()
