/// Whisper's exact log-mel-spectrogram frontend.
///
/// Ports the reference implementation from
/// `openai-whisper/whisper/audio.py::log_mel_spectrogram` at 16 kHz:
///
///   1. Pad or trim the waveform to `nSamples = sampleRate *
///      chunkLength` (30 s window = 480 000 samples).
///   2. Windowed STFT: Hann window, n_fft = 400 (25 ms), hop = 160
///      (10 ms). Reflection-padded on both sides by n_fft/2 = 200
///      samples so the number of frames is `nSamples / hop = 3000`.
///   3. Take the magnitude-squared of each bin, DROP the last frame
///      (Whisper's `[:, :-1]`) → 3000 frames × 201 bins.
///   4. Multiply by Whisper's fixed 80×201 mel filterbank (Slaney
///      style with fmin=0, fmax=8000, htk=False).
///   5. `log10(clamp(mel_power, min=1e-10))`, clamped again to the
///      per-utterance max minus 8, and rescaled `(x + 4) / 4`.
///
/// Output shape: `[80 mels, 3000 frames]`, dtype fp32, values in
/// ~[-1, 1]. This is exactly what the encoder's first Conv1d expects.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:fftea/fftea.dart';

import 'audio_io.dart';
import 'dsp.dart' as dsp;

class WhisperMelConfig {
  final int sampleRate;
  final int nFft;
  final int hopLength;
  final int nMels;
  final int chunkLength; // seconds
  const WhisperMelConfig({
    this.sampleRate = 16000,
    this.nFft = 400,
    this.hopLength = 160,
    this.nMels = 80,
    this.chunkLength = 30,
  });

  int get nSamples => sampleRate * chunkLength; // 480 000 for 30 s @ 16 kHz
  int get nFrames => nSamples ~/ hopLength; // 3000
}

class WhisperMel {
  final WhisperMelConfig cfg;
  final Float64List _window; // Hann, length nFft
  final List<Float64List> _melFilters; // [nMels][nFft/2 + 1]

  WhisperMel([this.cfg = const WhisperMelConfig()])
    : _window = _hannWindow(cfg.nFft),
      _melFilters = dsp.createMelFilterbank(
        sampleRate: cfg.sampleRate,
        nFft: cfg.nFft,
        nMels: cfg.nMels,
        fMin: 0.0,
        fMax: cfg.sampleRate / 2.0,
      );

  /// End-to-end from a raw waveform. Returns a `[nMels × nFrames]`
  /// Float32List in row-major order.
  Float32List logMelFromSamples(List<double> samples) {
    final trimmed = _padOrTrim(samples, cfg.nSamples);
    final padded = _reflectPad(trimmed, cfg.nFft ~/ 2);
    final power = _stftPower(padded); // [nFramesFull][nFft/2 + 1]
    // Whisper drops the last frame ([:, :-1] in Python).
    final nFrames = power.length - 1;
    final nBins = power[0].length;
    if (nFrames != cfg.nFrames) {
      // 30-s trim guarantees this; sanity-check.
      throw StateError(
        'unexpected frame count: got $nFrames, want ${cfg.nFrames}',
      );
    }

    // Mel: for each mel row, sum over bins of (filter × power).
    final mel = List.generate(cfg.nMels, (_) => Float64List(nFrames));
    for (int m = 0; m < cfg.nMels; m++) {
      final row = _melFilters[m];
      for (int t = 0; t < nFrames; t++) {
        double s = 0.0;
        final p = power[t];
        for (int k = 0; k < nBins; k++) {
          s += row[k] * p[k];
        }
        mel[m][t] = s;
      }
    }

    // Log10 clamp.
    var maxLog = -double.infinity;
    for (int m = 0; m < cfg.nMels; m++) {
      for (int t = 0; t < nFrames; t++) {
        final v = mel[m][t] < 1e-10 ? 1e-10 : mel[m][t];
        final l = math.log(v) / math.ln10;
        mel[m][t] = l;
        if (l > maxLog) maxLog = l;
      }
    }
    // Clamp to max - 8, then rescale.
    final floor = maxLog - 8.0;
    final out = Float32List(cfg.nMels * nFrames);
    for (int m = 0; m < cfg.nMels; m++) {
      for (int t = 0; t < nFrames; t++) {
        var v = mel[m][t] < floor ? floor : mel[m][t];
        v = (v + 4.0) / 4.0;
        out[m * nFrames + t] = v;
      }
    }
    return out;
  }

  /// Convenience wrapper that also loads / resamples a WAV file.
  Future<Float32List> logMelFromFile(String wavPath) async {
    final audio = await loadAudio(wavPath, cfg.sampleRate);
    return logMelFromSamples(audio.samples);
  }

  List<Float64List> _stftPower(Float64List paddedSamples) {
    final stft = STFT(cfg.nFft, _window);
    final out = <Float64List>[];
    stft.run(paddedSamples, (Float64x2List freq) {
      final bins = freq.discardConjugates();
      final mag = bins.magnitudes();
      // magnitude-squared
      for (int i = 0; i < mag.length; i++) {
        mag[i] *= mag[i];
      }
      out.add(mag);
    }, cfg.hopLength);
    return out;
  }

  static Float64List _padOrTrim(List<double> x, int n) {
    if (x.length == n) {
      return Float64List.fromList(x);
    }
    if (x.length > n) {
      return Float64List.fromList(x.sublist(0, n));
    }
    final out = Float64List(n);
    for (int i = 0; i < x.length; i++) {
      out[i] = x[i];
    }
    return out;
  }

  static Float64List _reflectPad(Float64List x, int pad) {
    if (pad == 0) return x;
    final out = Float64List(x.length + 2 * pad);
    // Left reflect (excluding boundary, matching torch/numpy reflect).
    for (int i = 0; i < pad; i++) {
      out[i] = x[pad - i];
    }
    // Body.
    for (int i = 0; i < x.length; i++) {
      out[pad + i] = x[i];
    }
    // Right reflect.
    for (int i = 0; i < pad; i++) {
      out[pad + x.length + i] = x[x.length - 2 - i];
    }
    return out;
  }

  static Float64List _hannWindow(int n) {
    final w = Float64List(n);
    for (int i = 0; i < n; i++) {
      w[i] = 0.5 * (1.0 - math.cos(2.0 * math.pi * i / n));
    }
    return w;
  }
}
