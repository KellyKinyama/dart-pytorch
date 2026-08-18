"""Faithful numpy port of openai-whisper's log_mel_spectrogram.

Replicates whisper/audio.py::log_mel_spectrogram at 16 kHz, using the
same mel_filters.npz we've extracted. Prints a summary + dumps the
tensor to /tmp/mel_ref.raw for a Dart comparison.
"""
import struct
import sys
import numpy as np

# All constants come from openai-whisper/whisper/audio.py.
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = SAMPLE_RATE * CHUNK_LENGTH  # 480000
N_FRAMES = N_SAMPLES // HOP_LENGTH       # 3000


def pad_or_trim(x, n=N_SAMPLES):
    if len(x) > n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))


def stft_torchlike(waveform, n_fft, hop_length):
    """`torch.stft(waveform, n_fft, hop_length, window=hann, return_complex=True)`.

    Reflection-pads by n_fft // 2 on each side. Uses centered frames.
    Output shape: (n_fft/2 + 1, n_frames)
    """
    hann = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n_fft) / n_fft))
    # Reflect-pad by n_fft // 2 on each side.
    pad = n_fft // 2
    padded = np.pad(waveform, (pad, pad), mode="reflect")
    n_frames = (len(padded) - n_fft) // hop_length + 1
    bins = n_fft // 2 + 1
    result = np.empty((bins, n_frames), dtype=np.complex64)
    for i in range(n_frames):
        s = i * hop_length
        w = padded[s:s + n_fft] * hann
        F = np.fft.rfft(w, n=n_fft)
        result[:, i] = F.astype(np.complex64)
    return result


def load_wav_mono_16k(path):
    """Minimal 16 kHz mono 16-bit PCM WAV loader (no resample)."""
    with open(path, "rb") as f:
        raw = f.read()
    assert raw[0:4] == b"RIFF" and raw[8:12] == b"WAVE"
    off = 12
    num_ch = sr = bps = 0
    data_off = data_len = 0
    while off + 8 <= len(raw):
        tag = raw[off:off + 4].decode("ascii")
        size = struct.unpack("<I", raw[off + 4:off + 8])[0]
        if tag == "fmt ":
            num_ch = struct.unpack("<H", raw[off + 10:off + 12])[0]
            sr = struct.unpack("<I", raw[off + 12:off + 16])[0]
            bps = struct.unpack("<H", raw[off + 22:off + 24])[0]
        elif tag == "data":
            data_off = off + 8
            data_len = size
            break
        off += 8 + size
    assert sr == SAMPLE_RATE and bps == 16
    samples = np.frombuffer(
        raw[data_off:data_off + data_len], dtype=np.int16
    ).astype(np.float32) / 32768.0
    if num_ch > 1:
        samples = samples.reshape(-1, num_ch).mean(axis=1)
    return samples


def main():
    wav_path = sys.argv[1] if len(sys.argv) > 1 else "models/silero_vad/warmup_audio.wav"
    audio = load_wav_mono_16k(wav_path)
    padded = pad_or_trim(audio)                       # (480000,)
    stft = stft_torchlike(padded, N_FFT, HOP_LENGTH)  # (201, N_FRAMES + 1)
    # Whisper drops last frame: stft[..., :-1]
    stft = stft[:, :-1]
    magnitudes = (np.abs(stft) ** 2).astype(np.float32)  # (201, N_FRAMES)

    filters = np.load("/tmp/mel_filters.npz")["mel_80"].astype(np.float32)  # (80, 201)
    mel_spec = filters @ magnitudes                                        # (80, N_FRAMES)

    log_spec = np.log10(np.clip(mel_spec, a_min=1e-10, a_max=None))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    print(f"log-mel shape: {log_spec.shape}, dtype: {log_spec.dtype}")
    print(f"stats: min={log_spec.min():.4f}, max={log_spec.max():.4f}, "
          f"mean={log_spec.mean():.4f}")
    print(f"log-mel[0, :5]:  {log_spec[0, :5]}")
    print(f"log-mel[40, 40:45]: {log_spec[40, 40:45]}")

    with open("/tmp/mel_ref.raw", "wb") as f:
        f.write(log_spec.astype(np.float32).tobytes())
    print(f"wrote /tmp/mel_ref.raw ({log_spec.nbytes} bytes)")


if __name__ == "__main__":
    main()
