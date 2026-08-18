import 'dart:io';
import 'dart:typed_data';
import 'package:wav/wav.dart';

/// Container for PCM audio data loaded from a file.
///
/// Holds the [samples] as a normalized mono signal (-1.0 to 1.0) and the
/// [sampleRate] in Hz.
class ProcessedAudio {
  /// The sample rate of the audio in Hz.
  final int sampleRate;

  /// The mono audio samples.
  final List<double> samples;

  ProcessedAudio(this.samples, this.sampleRate);
}

/// Decodes a WAV file into raw PCM samples (mono), resampled to [sampleRate].
Future<ProcessedAudio> loadAudio(String filePath, int sampleRate) async {
  final file = File(filePath);
  if (!await file.exists()) {
    throw Exception('Audio file not found at $filePath');
  }

  if (!filePath.toLowerCase().endsWith('.wav')) {
    throw Exception('Unsupported format. Only .wav files are supported.');
  }
  // if (!filePath.toLowerCase().endsWith('.flac')) {
  //   throw Exception('Unsupported format. Only .wav files are supported.');
  // }

  final bytes = await file.readAsBytes();

  final wav = Wav.read(bytes);

  var samples = wav.toMono();

  if (wav.samplesPerSecond <= 0) {
    throw Exception('Invalid sample rate in WAV file: ${wav.samplesPerSecond}');
  }

  if (wav.samplesPerSecond != sampleRate) {
    samples = _resampleLinear(samples, wav.samplesPerSecond, sampleRate);
  }

  return ProcessedAudio(samples, sampleRate);
}

/// Resamples a signal using linear interpolation.
Float64List _resampleLinear(List<double> input, int srcRate, int targetRate) {
  if (srcRate == targetRate) {
    return input is Float64List ? input : Float64List.fromList(input);
  }

  if (targetRate <= 0)
    throw ArgumentError('Target sample rate must be positive.');
  if (srcRate <= 0) throw ArgumentError('Source sample rate must be positive.');

  final ratio = srcRate / targetRate;
  final newLength = (input.length / ratio).ceil();
  final output = Float64List(newLength);

  for (int i = 0; i < newLength; i++) {
    final position = i * ratio;
    final index = position.floor();
    final fraction = position - index;

    if (index >= input.length - 1) {
      output[i] = input.last;
    } else {
      output[i] = input[index] * (1.0 - fraction) + input[index + 1] * fraction;
    }
  }
  return output;
}
