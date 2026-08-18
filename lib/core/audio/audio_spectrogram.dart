import 'dart:typed_data';
import 'audio_io.dart';
import 'dsp.dart';
import 'visualization.dart';

export 'audio_io.dart' show ProcessedAudio, loadAudio;
export 'dsp.dart';
export 'visualization.dart';

/// Loads a WAV file and computes its Power spectrogram via STFT in decibels (dB).
///
/// Returns a 2D list (list of frames, where each frame is a list of frequency bins).
/// The number of frequency bins is `nFft / 2 + 1`.
Future<List<Float64List>> powerSpectrogram(
  String filePath, {
  int sampleRate = 22050,
  int nFft       = 2048,
  int? hopLength,
  double topDb   = 80.0,
}) async {
  final audio      = await loadAudio(filePath, sampleRate);
  final powerSpec  = calculateSpectrogram(
    audio.samples,
    fftSize: nFft,
    hopSize: hopLength,
  );
  return powerToDb(powerSpec, topDb: topDb);
}

/// Loads a WAV file and computes its Mel-frequency spectrogram in decibels (dB).
///
/// This is a high-level convenience function that combines audio loading,
/// Short-Time Fourier Transform (STFT), Mel filterbank application, and
/// power-to-dB conversion.
///
/// Returns a 2D list (list of frames, where each frame is a list of Mel bins).
///
/// Matches the behavior of `librosa.feature.melspectrogram` followed by `librosa.power_to_db`.
Future<List<Float64List>> melSpectrogram(
  String filePath, {
  int sampleRate = 22050,
  int nFft       = 2048,
  int? hopLength,
  int nMels      = 128,
  double fMin    = 0.0,
  double? fMax,
  double topDb   = 80.0,
}) async {
  final audio      = await loadAudio(filePath, sampleRate);
  final powerSpec  = calculateSpectrogram(
    audio.samples,
    fftSize: nFft,
    hopSize: hopLength,
  );
  final filterbank = createMelFilterbank(
    sampleRate: sampleRate,
    nFft: nFft,
    nMels: nMels,
    fMin: fMin,
    fMax: fMax,
  );
  final melSpec    = applyMelFilterbank(powerSpec, filterbank);
  return powerToDb(melSpec, topDb: topDb);
}

/// Generates a Mel-frequency spectrogram from a WAV file and saves it as a PNG image.
///
/// This function performs the full pipeline:
/// 1. Load and resample audio.
/// 2. Compute STFT power spectrogram.
/// 3. Apply Mel filterbank.
/// 4. Convert to dB.
/// 5. Render to an image using the Magma colormap and save to disk.
Future<void> saveSpectrogramPng(
  String wavPath,
  String pngPath, {
  int sampleRate = 22050,
  int nFft       = 2048,
  int? hopLength,
  int nMels      = 128,
  double fMin    = 0.0,
  double? fMax,
  double topDb   = 80.0,
  String title   = 'Mel-frequency spectrogram',
  int plotWidth  = 1000,
  int plotHeight = 400,
  bool useNearest = true,
}) async {
  final hop     = hopLength ?? nFft ~/ 4;

  final audio   = await loadAudio(wavPath, sampleRate);

  final power   = calculateSpectrogram(audio.samples, fftSize: nFft, hopSize: hop);

  final fbank   = createMelFilterbank(
    sampleRate: sampleRate, nFft: nFft, nMels: nMels, fMin: fMin, fMax: fMax,
  );

  final melSpec = applyMelFilterbank(power, fbank);

  final melDb   = powerToDb(melSpec, topDb: topDb);

  await saveSpectrogramImage(
    melDb, pngPath,
    sampleRate: sampleRate,
    hopLength: hop,
    fMin:       fMin,
    fMax:       fMax ?? sampleRate / 2.0,
    topDb:      topDb,
    title:      title,
    plotWidth:  plotWidth,
    plotHeight: plotHeight,
    useNearest: useNearest,
  );
}

/// Generates a Power spectrogram (linear frequency) from a WAV file and saves it as a PNG.
///
/// [wavPath]: Input audio file.
/// [pngPath]: Output image file.
Future<void> savePowerSpectrogramPng(
  String wavPath,
  String pngPath, {
  int sampleRate = 22050,
  int nFft       = 2048,
  int? hopLength,
  double topDb   = 80.0,
  String title   = 'Power Spectrogram',
  int plotWidth  = 1000,
  int plotHeight = 400,
  bool useNearest = true,
}) async {
  final hop = hopLength ?? nFft ~/ 4;

  final audio = await loadAudio(wavPath, sampleRate);

  // STFT returns [Frames][FrequencyBins]
  final power = calculateSpectrogram(audio.samples, fftSize: nFft, hopSize: hop);

  final powerDb = powerToDb(power, topDb: topDb);

  // Transpose to [FrequencyBins][Frames] for the visualization logic
  final transposed = _transpose(powerDb);

  await saveSpectrogramImage(
    transposed, pngPath,
    sampleRate: sampleRate,
    hopLength: hop,
    fMin: 0.0,
    fMax: sampleRate / 2.0,
    topDb: topDb,
    title: title,
    plotWidth: plotWidth,
    plotHeight: plotHeight,
    useNearest: useNearest,
    isMel: false,
  );
}

List<Float64List> _transpose(List<Float64List> matrix) {
  if (matrix.isEmpty) return [];
  final rows = matrix.length;
  final cols = matrix[0].length;
  final transposed = List.generate(cols, (_) => Float64List(rows));
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      transposed[j][i] = matrix[i][j];
    }
  }
  return transposed;
}