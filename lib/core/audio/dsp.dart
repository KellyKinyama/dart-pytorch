import 'dart:math';
import 'dart:typed_data';
import 'package:fftea/fftea.dart';

/// Computes the power spectrogram via Short-Time Fourier Transform (STFT).
///
/// Returns a list of Float64List, where each element represents the power spectrum
/// of a frame.
///
/// [samples]: The audio signal.
/// [hopSize]: Number of samples between frames. Defaults to [fftSize] / 4.
/// [fftSize]: Size of the FFT window.
List<Float64List> calculateSpectrogram(
  List<double> samples, {
  int? hopSize,
  int fftSize = 2048,
}) {
  final hop = hopSize ?? fftSize ~/ 4;
  final padAmount = fftSize ~/ 2;

  if (samples.length < padAmount) {
    throw ArgumentError(
      'Signal length (${samples.length}) must be >= fftSize // 2 ($padAmount).',
    );
  }
  final paddedSamples = Float64List.fromList(_padReflect(samples, padAmount));

  final window = Float64List(fftSize);
  for (int i = 0; i < fftSize; i++) {
    window[i] = 0.5 * (1.0 - cos(2.0 * pi * i / fftSize));
  }

  final stft = STFT(fftSize, window);
  final spectrogram = <Float64List>[];

  stft.run(paddedSamples, (Float64x2List freq) {
    final bins = freq.discardConjugates();
    final expectedBins = fftSize ~/ 2 + 1;
    if (bins.length != expectedBins) {
      throw StateError('STFT bin mismatch.');
    }
    final mags = bins.magnitudes();
    for (int i = 0; i < mags.length; i++) {
      mags[i] *= mags[i];
    }
    spectrogram.add(mags);
  }, hop);

  return spectrogram;
}

List<double> _padReflect(List<double> signal, int padSize) {
  final len = signal.length;
  if (len == 0) return [];
  if (padSize > len - 1) {
    throw ArgumentError('Pad size too large for reflect padding.');
  }

  final padded = Float64List(len + 2 * padSize);
  for (int i = 0; i < padSize; i++) {
    padded[i] = signal[padSize - i];
  }
  for (int i = 0; i < len; i++) {
    padded[padSize + i] = signal[i];
  }
  for (int i = 0; i < padSize; i++) {
    padded[padSize + len + i] = signal[len - 2 - i];
  }
  return padded;
}

const double _minLogHz  = 1000.0;
const double _minLogMel = 15.0;
const double _linearSp  = 200.0 / 3.0;
final double _logStep   = log(6.4) / 27.0;

double hzToMel(double hz) {
  if (hz >= _minLogHz) {
    return _minLogMel + log(hz / _minLogHz) / _logStep;
  }
  return hz / _linearSp;
}

double melToHz(double mel) {
  if (mel >= _minLogMel) {
    return _minLogHz * exp((mel - _minLogMel) * _logStep);
  }
  return mel * _linearSp;
}

/// Creates a Mel filterbank matrix.
///
/// Returns a list of [nMels] filters, where each filter is a [Float64List]
/// of length `nFft ~/ 2 + 1`.
///
/// [sampleRate]: Sampling rate of the audio.
/// [nFft]: FFT size.
/// [nMels]: Number of Mel bands.
/// [fMin]: Minimum frequency in Hz.
/// [fMax]: Maximum frequency in Hz (defaults to Nyquist).
List<Float64List> createMelFilterbank({
  required int sampleRate,
  required int nFft,
  int nMels = 128,
  double fMin = 0.0,
  double? fMax,
}) {
  fMax ??= sampleRate / 2.0;
  final nFftBins = nFft ~/ 2 + 1;

  final lowMel  = hzToMel(fMin);
  final highMel = hzToMel(fMax);

  final melPoints = List.generate(
    nMels + 2,
    (i) => lowMel + i * (highMel - lowMel) / (nMels + 1),
  );
  final hzPoints = melPoints.map(melToHz).toList();

  final fftFreqs = Float64List(nFftBins);
  for (int i = 0; i < nFftBins; i++) {
    fftFreqs[i] = i * sampleRate / nFft;
  }

  final fbank = List.generate(nMels, (_) => Float64List(nFftBins));

  for (int m = 0; m < nMels; m++) {
    final lower  = hzPoints[m];
    final center = hzPoints[m + 1];
    final upper  = hzPoints[m + 2];

    final norm = 2.0 / (upper - lower);
    final iMin = (lower * nFft / sampleRate).floor();
    final iMax = (upper * nFft / sampleRate).ceil();

    for (int i = iMin; i <= iMax; i++) {
      if (i < 0 || i >= nFftBins) continue;
      final freq = fftFreqs[i];
      double weight = 0.0;

      if (freq >= lower && freq <= center) {
        weight = (freq - lower) / (center - lower);
      } else if (freq > center && freq <= upper) {
        weight = (upper - freq) / (upper - center);
      }

      if (weight > 0) {
        fbank[m][i] = weight * norm;
      }
    }
  }
  return fbank;
}

/// Applies a Mel filterbank to a power spectrogram.
///
/// [spectrogram]: The power spectrogram (frames x freq_bins).
/// [filterbank]: The Mel filterbank matrix (mels x freq_bins).
/// Returns the Mel spectrogram (mels x frames).
List<Float64List> applyMelFilterbank(
  List<Float64List> spectrogram,
  List<Float64List> filterbank,
) {
  final nMels   = filterbank.length;
  final nFrames = spectrogram.length;
  final melSpec = List.generate(nMels, (_) => Float64List(nFrames));

  for (int m = 0; m < nMels; m++) {
    final filterRow = filterbank[m];
    for (int t = 0; t < nFrames; t++) {
      final frame = spectrogram[t];
      double sum  = 0.0;
      final len   = min(frame.length, filterRow.length);
      for (int k = 0; k < len; k++) {
        sum += frame[k] * filterRow[k];
      }
      melSpec[m][t] = sum;
    }
  }
  return melSpec;
}

/// Converts a power spectrogram (amplitude squared) to decibel (dB) units.
///
/// [spectrogram]: The input spectrogram (linear power).
/// [topDb]: The dynamic range of the output. Values below `max - topDb` are clipped.
///
/// Returns the spectrogram in dB.
List<Float64List> powerToDb(
  List<Float64List> spectrogram, {
  double topDb = 80.0,
}) {
  double maxVal = 0.0;
  for (final row in spectrogram) {
    for (final val in row) {
      if (val > maxVal) maxVal = val;
    }
  }

  const minThreshold = 1e-10;
  final ref    = maxVal > minThreshold ? maxVal : 1.0;
  final log10  = log(10);
  final floor  = -topDb;

  return spectrogram.map((row) {
    final newRow = Float64List(row.length);
    for (int i = 0; i < row.length; i++) {
      final val = row[i] < minThreshold ? minThreshold : row[i];
      final db  = 10.0 * log(val / ref) / log10;
      newRow[i] = db < floor ? floor : db;
    }
    return newRow;
  }).toList();
}