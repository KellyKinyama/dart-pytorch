/// F5-TTS voice pipeline demo — text → mel → waveform.
///
/// Wires together the fresh F5-TTS DiT stack ([F5TextEncoder] +
/// [F5DiT] + [F5DurationPredictor]), the flow-matching sampler
/// ([FlowMatchingSampler]), and our existing HiFi-GAN vocoder
/// ([HiFiGanGenerator]) into a single text-to-speech pipeline:
///
///   1. Tokenise input text as ASCII byte codes (placeholder tokenizer;
///      real F5-TTS uses a phoneme dictionary + language-specific rules).
///   2. Encode chars with [F5TextEncoder] → `[T_chars, textDim]`.
///   3. Predict per-char durations, expand to per-frame text
///      conditioning `[T_mel, textDim]`.
///   4. Sample the mel spectrogram via flow-matching: start from
///      Gaussian noise `[T_mel, melDim]`, integrate `dx/dt = F5DiT(x,
///      textCond, t)` over `numSteps` timesteps.
///   5. Feed the mel through [HiFiGanGenerator] → waveform.
///   6. Save the resulting `[T_samples]` as a 22050 Hz 16-bit mono WAV.
///
///   dart run bin/f5_tts_demo.dart --text "Hello world" [--out out.wav]
///       [--num-steps 32] [--f5-weights PATH] [--vocoder-weights PATH]
///
/// **Status**: this demo is a **plumbing skeleton**. Without real
/// F5-TTS + HiFi-GAN weights the output waveform will be uncorrelated
/// noise; the demo still exercises every arch stage end-to-end and
/// prints per-stage timing, which is what you want when validating
/// the loader path against a real checkpoint.
///
/// One-time weight download (placeholders — F5-TTS's official
/// SafeTensors packaging is model-specific; adapt paths as needed):
///
///   mkdir -p models/f5-tts models/hifigan-universal
///   # F5-TTS: from https://huggingface.co/SWivid/F5-TTS
///   # HiFi-GAN: from https://github.com/jik876/hifi-gan releases
///   python3 scripts/convert_hifigan_pt_to_safetensors.py \\
///       hifigan_v1.pth models/hifigan-universal/model.safetensors
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

const _melDim = 100;
const _textDim = 512;
const _embedDim = 1024;
const _numLayers = 22;
const _numHeads = 16;
const _mlpDim = 2048;
const _freqDim = 256;
const _sampleRate = 22050;
const _vocabSize = 2545;

Future<void> main(List<String> args) async {
  var text = 'The quick brown fox jumps over the lazy dog.';
  var outPath = 'out.wav';
  var numSteps = 32;
  String? f5WeightsPath;
  String? vocoderWeightsPath;
  var noiseSeed = 42;
  for (int i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--text':
        text = args[++i];
        break;
      case '--out':
        outPath = args[++i];
        break;
      case '--num-steps':
        numSteps = int.parse(args[++i]);
        break;
      case '--f5-weights':
        f5WeightsPath = args[++i];
        break;
      case '--vocoder-weights':
        vocoderWeightsPath = args[++i];
        break;
      case '--seed':
        noiseSeed = int.parse(args[++i]);
        break;
    }
  }

  print('== text ==');
  print('  "$text"');
  print('');

  // ---------- 1. Tokenise ----------
  final swTok = Stopwatch()..start();
  final tokenIds = <int>[];
  for (final code in text.toLowerCase().codeUnits) {
    tokenIds.add(code % _vocabSize);
  }
  final tokens = Tensor.fromList([
    tokenIds.length,
  ], tokenIds.map((i) => i.toDouble()).toList());
  swTok.stop();
  print(
    'tokenise: ${swTok.elapsedMilliseconds} ms  '
    '(${tokenIds.length} chars → ${tokens.shape})',
  );

  // ---------- 2. Build models ----------
  print('');
  print('Building F5TextEncoder + F5DurationPredictor + F5DiT + HiFiGAN');
  final swBuild = Stopwatch()..start();
  final textEnc = F5TextEncoder(
    vocabSize: _vocabSize,
    dim: _textDim,
    intermediateDim: 2048,
    numLayers: 4,
  );
  final durationPredictor = F5DurationPredictor(
    textDim: _textDim,
    intermediateDim: 512,
    numLayers: 2,
  );
  final dit = F5DiT(
    melDim: _melDim,
    textDim: _textDim,
    embedDim: _embedDim,
    numLayers: _numLayers,
    numHeads: _numHeads,
    mlpDim: _mlpDim,
    freqDim: _freqDim,
  );
  final vocoder = HiFiGanGenerator(const HiFiGanV1Config(melChannels: _melDim));
  swBuild.stop();
  print('  build: ${swBuild.elapsedMilliseconds} ms');

  // ---------- 3. Load weights (optional) ----------
  if (f5WeightsPath != null && File(f5WeightsPath).existsSync()) {
    print('');
    print('Loading F5-TTS weights from $f5WeightsPath ...');
    final sw = Stopwatch()..start();
    final report = F5TtsHFLoader.loadFile(
      textEncoder: textEnc,
      dit: dit,
      durationPredictor: durationPredictor,
      path: f5WeightsPath,
    );
    sw.stop();
    print('  $report  (${sw.elapsedMilliseconds} ms)');
  } else if (f5WeightsPath != null) {
    stderr.writeln(
      'warning: --f5-weights $f5WeightsPath not found; '
      'using random init',
    );
  } else {
    print('');
    print(
      '(no F5-TTS weights provided; using random init — output '
      'will be uncorrelated noise)',
    );
  }
  if (vocoderWeightsPath != null && File(vocoderWeightsPath).existsSync()) {
    print('Loading HiFi-GAN weights from $vocoderWeightsPath ...');
    final sw = Stopwatch()..start();
    final report = HiFiGanLoader.loadFile(vocoder, vocoderWeightsPath);
    sw.stop();
    print('  $report  (${sw.elapsedMilliseconds} ms)');
  }

  // ---------- 4. Text → text features → durations → expanded text ----
  print('');
  print('Encoding text ...');
  final swT = Stopwatch()..start();
  final textFeatures = textEnc(tokens); // [T_chars, textDim]
  swT.stop();
  print('  ${swT.elapsedMilliseconds} ms  → ${textFeatures.shape}');

  print('');
  print('Predicting durations ...');
  final swD = Stopwatch()..start();
  final durations = durationPredictor(textFeatures); // [T_chars]
  final durVals = durations.toList();
  double totalFrames = 0;
  for (final v in durVals) {
    totalFrames += v;
  }
  swD.stop();
  final tMel = totalFrames.round().clamp(8, 4096);
  print('  ${swD.elapsedMilliseconds} ms  → total frames ≈ $tMel');

  // With random init, durationPredictor output is close to
  // softplus(0) = ln(2). For a proper demo we'd apply this alignment,
  // but at random init we'd get ~0.69 frames per char. Instead use a
  // fixed 5 frames per char so downstream shapes are reasonable.
  final fixedDur = Tensor.fromList([
    tokenIds.length,
  ], List<double>.filled(tokenIds.length, 5.0));
  final textCond = F5DurationPredictor.expandTextToFrames(
    textFeatures,
    f5WeightsPath == null ? fixedDur : durations,
  );
  print('  text conditioning: ${textCond.shape}');

  // ---------- 5. Flow-matching sample mel ----------
  final tMelActual = textCond.shape[0];
  print('');
  print('Sampling mel via flow-matching ($numSteps steps) ...');
  final swS = Stopwatch()..start();
  final noise = gaussianNoise([tMelActual, _melDim], seed: noiseSeed);
  final sampler = FlowMatchingSampler(numSteps: numSteps);
  final mel = sampler.sample(
    initialNoise: noise,
    velocityField: (x, t) => dit(x, textCond, t),
  );
  swS.stop();
  print('  ${swS.elapsedMilliseconds} ms  → mel ${mel.shape}');

  // ---------- 6. Mel → waveform ----------
  print('');
  print('Vocoding mel with HiFi-GAN ...');
  final swV = Stopwatch()..start();
  // HiFi-GAN wants [N, melDim, T]. Our mel is [T, melDim]; transpose
  // and add batch dim.
  final melT = _transposeMelToNCT(mel);
  final wav = vocoder(melT); // [1, 1, T·256]
  swV.stop();
  final wavData = wav.toList();
  print(
    '  ${swV.elapsedMilliseconds} ms  → ${wav.shape} '
    '(${wavData.length ~/ _sampleRate} s at $_sampleRate Hz)',
  );

  // ---------- 7. Write WAV ----------
  _writeWav(outPath, wavData, _sampleRate);
  print('');
  print('wrote $outPath');
}

/// `[T, melDim]` -> `[1, melDim, T]`.
Tensor _transposeMelToNCT(Tensor mel) {
  final t = mel.shape[0];
  final c = mel.shape[1];
  final data = mel.toFloat32List();
  final out = Float32List(c * t);
  for (int ti = 0; ti < t; ti++) {
    for (int ci = 0; ci < c; ci++) {
      out[ci * t + ti] = data[ti * c + ci];
    }
  }
  return Tensor.fromFloat32List([1, c, t], out, device: mel.device);
}

/// Write a mono 16-bit PCM WAV file.
void _writeWav(String path, List<double> samples, int sampleRate) {
  final numSamples = samples.length;
  final bytes = BytesBuilder();
  void writeU32(int v) {
    bytes.addByte(v & 0xff);
    bytes.addByte((v >> 8) & 0xff);
    bytes.addByte((v >> 16) & 0xff);
    bytes.addByte((v >> 24) & 0xff);
  }

  void writeU16(int v) {
    bytes.addByte(v & 0xff);
    bytes.addByte((v >> 8) & 0xff);
  }

  bytes.add('RIFF'.codeUnits);
  writeU32(36 + numSamples * 2);
  bytes.add('WAVE'.codeUnits);
  bytes.add('fmt '.codeUnits);
  writeU32(16); // fmt chunk size
  writeU16(1); // PCM
  writeU16(1); // mono
  writeU32(sampleRate);
  writeU32(sampleRate * 2); // byte rate = sr * blockAlign
  writeU16(2); // block align
  writeU16(16); // bits per sample
  bytes.add('data'.codeUnits);
  writeU32(numSamples * 2);
  for (final v in samples) {
    final clamped = v.clamp(-1.0, 1.0);
    final s = (clamped * 32767.0).round();
    writeU16(s & 0xffff);
  }
  File(path).writeAsBytesSync(bytes.toBytes());
  // Silence unused imports
  (math.pi);
}
