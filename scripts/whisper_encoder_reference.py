"""Reference encoder output for the warmup wav, using HuggingFace transformers.

Writes:
  /tmp/whisper_enc_ref.raw   (1 * 1500 * 384 * fp32)
  /tmp/whisper_enc_stats.txt
"""

import numpy as np
import soundfile as sf
import sys

try:
    import torch
    from transformers import WhisperFeatureExtractor, WhisperModel
except ImportError as e:
    print(f"missing dep: {e}", file=sys.stderr)
    sys.exit(1)

WAV = "models/silero_vad/warmup_audio.wav"
MODEL = "openai/whisper-tiny.en"

audio, sr = sf.read(WAV)
if sr != 16000:
    print(f"expected 16 kHz, got {sr}", file=sys.stderr)
    sys.exit(1)
if audio.ndim > 1:
    audio = audio.mean(-1)
audio = audio.astype(np.float32)

fe = WhisperFeatureExtractor.from_pretrained(MODEL)
model = WhisperModel.from_pretrained(MODEL)
model.eval()

feats = fe(audio, sampling_rate=16000, return_tensors="pt")
mel = feats.input_features            # [1, 80, 3000]
print(f"mel input shape:  {tuple(mel.shape)}  dtype={mel.dtype}")

with torch.no_grad():
    enc = model.encoder(mel)          # BaseModelOutput
    hidden = enc.last_hidden_state    # [1, 1500, 384]

print(f"enc out shape:    {tuple(hidden.shape)}  dtype={hidden.dtype}")
arr = hidden.detach().cpu().numpy().astype(np.float32)
print(f"stats: mean={arr.mean():.6f} std={arr.std():.6f} "
      f"min={arr.min():.4f} max={arr.max():.4f}")

arr.tofile("/tmp/whisper_enc_ref.raw")
with open("/tmp/whisper_enc_stats.txt", "w") as f:
    f.write(f"shape={list(arr.shape)}\n")
    f.write(f"mean={arr.mean():.9f}\n")
    f.write(f"std={arr.std():.9f}\n")
    f.write(f"min={arr.min():.9f}\n")
    f.write(f"max={arr.max():.9f}\n")
print("wrote /tmp/whisper_enc_ref.raw")
