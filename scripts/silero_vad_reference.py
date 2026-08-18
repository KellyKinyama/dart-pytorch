"""Run Silero VAD on identical deterministic input via ONNX Runtime.

Prints per-chunk probabilities that a matching Dart run can compare
against directly."""
import numpy as np
import onnxruntime as ort


def make_chunk(i, n=512):
    """Sine burst at 440 Hz for chunk index i.

    Chunk 0 = silence, chunks 1..5 = short 440 Hz sine wave scaled to
    0.3. Gives the network a mix of silent and voice-like inputs.
    """
    if i == 0:
        return np.zeros((1, n), dtype=np.float32)
    t = (np.arange(n) + i * n) / 16000.0
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32).reshape(1, n)


def main():
    sess = ort.InferenceSession(
        "models/silero_vad/silero_vad.onnx",
        providers=["CPUExecutionProvider"],
    )
    ctx = np.zeros((1, 64), dtype=np.float32)
    state = np.zeros((2, 1, 128), dtype=np.float32)
    sr = np.array(16000, dtype=np.int64)

    print("chunks (ONNX Runtime, deterministic sine at 440 Hz):")
    for i in range(6):
        chunk = make_chunk(i)
        inp = np.concatenate([ctx, chunk], axis=1)
        out, state = sess.run(None, {"input": inp, "state": state, "sr": sr})
        ctx = chunk[:, -64:]
        print(f"  {i}: p={float(out[0, 0]):.6f}")


if __name__ == "__main__":
    main()
