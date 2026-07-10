/// Multi-modal transformer demo: image + audio + video + text prompt
/// → text output (like a tiny Gemini). Uses [MultiModalGenerator].
///
/// Setup:
///
///   * Vocab is character-level over the ASCII letters, plus 3 special
///     tokens (PAD, BOS, EOS). Each class name is 3 letters, so the
///     target sequence is [BOS, c1, c2, c3, EOS] (5 tokens).
///
///   * A "class" is drawn from {0..7}. Each modality carries a
///     redundant, class-specific signal:
///
///       - image  : an 8x8 RGB image filled with per-class RGB palette
///                  colour + noise. Patchified into 4 patches of 4x4.
///       - audio  : 6-frame "spectrogram" of shape [6, 16] with a
///                  Gaussian peak at bin ~2*c + noise.
///       - video  : 4 frames of length-12 pattern vectors, each frame
///                  a sinusoid at class-specific phase + noise.
///       - text   : a fixed 2-token prompt ([BOS, ASK]). Text is
///                  present mainly to exercise the text encoder path.
///
///   * Target: the class's 3-letter name (see `_classNames`).
///
/// The model is trained with per-step next-token cross-entropy
/// (teacher forcing on decoder input) via Adam. After training we
/// greedy-generate the output for each class and count exact-string
/// matches on a held-out val batch (fresh noise draws with the same
/// per-class palettes / peaks / phases).
///
/// Run:
///
///     dart run bin/train_multimodal_demo.dart          # CPU
///     dart run bin/train_multimodal_demo.dart --gpu    # GPU
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

import 'shakespeare_util.dart' show paramScalarCount;

// -------------------------------------------------------------------
// Vocab + class definitions
// -------------------------------------------------------------------
const int _pad = 0;
const int _bos = 1;
const int _eos = 2;
const int _charBase = 3; // 'a' -> 3, 'b' -> 4, ...
const int _vocabSize = _charBase + 26; // 29
const int _numClasses = 8;
const int _nameLen = 3;

const List<String> _classNames = [
  'red', 'sun', 'sky', 'sea', // 0..3
  'cat', 'dog', 'owl', 'orb', // 4..7
];

// Per-class RGB palette (0..1). Chosen distinct.
const List<List<double>> _palette = [
  [0.90, 0.10, 0.10], // red
  [0.95, 0.85, 0.10], // sun
  [0.20, 0.55, 0.95], // sky
  [0.10, 0.35, 0.55], // sea
  [0.55, 0.35, 0.20], // cat
  [0.60, 0.45, 0.30], // dog
  [0.75, 0.65, 0.45], // owl
  [0.85, 0.55, 0.20], // orb
];

int _charToken(int letterIndex) => _charBase + letterIndex;

List<int> _nameToTokens(String name) {
  final out = <int>[_bos];
  for (final ch in name.codeUnits) {
    out.add(_charToken(ch - 'a'.codeUnitAt(0)));
  }
  out.add(_eos);
  return out;
}

String _tokensToString(List<int> tokens) {
  final buf = StringBuffer();
  for (final t in tokens) {
    if (t == _bos) {
      buf.write('<bos>');
    } else if (t == _eos) {
      buf.write('<eos>');
    } else if (t == _pad) {
      buf.write('<pad>');
    } else if (t >= _charBase && t < _vocabSize) {
      buf.writeCharCode('a'.codeUnitAt(0) + (t - _charBase));
    } else {
      buf.write('?');
    }
  }
  return buf.toString();
}

// -------------------------------------------------------------------
// Model hyperparameters
// -------------------------------------------------------------------
const int _imageSize = 8;
const int _patchSize = 4;
const int _numChannels = 3;
const int _numPatches = 4;
const int _patchPixels = _patchSize * _patchSize * _numChannels; // 48

const int _audioSeq = 6;
const int _audioFeat = 16;

const int _videoFrames = 4;
const int _videoFeat = 12;

const int _promptLen = 2; // [BOS, ASK]
const int _promptAsk = _charBase; // reuse 'a' (letter index 0) as the ASK token

const int _jointDim = 32;
const int _decoderLayers = 2;
const int _decoderHeads = 4;
const int _fusionLayers = 1;
const int _fusionHeads = 4;
const int _encoderLayers = 1;
const int _encoderHeads = 4;

const int _maxTargetLen = 8;
const int _steps = 500;
const int _logEvery = 50;
const double _lr = 3e-3;

// -------------------------------------------------------------------
// Synthetic multi-modal sample generator
// -------------------------------------------------------------------

typedef _Sample = ({
  Tensor imagePatches,
  Tensor audioFeatures,
  Tensor videoFrames,
  Tensor textPrompt,
  Tensor decIn,
  Tensor decTgt,
  int label,
});

/// Materialize a single 4-modality training example for class `c`
/// with independent Gaussian noise on all continuous modalities.
_Sample _mkSample(
  int c,
  math.Random rng, {
  required Device device,
  double noise = 0.08,
}) {
  // -------- Image: fill with palette colour + per-pixel noise, then
  //          patchify into [_numPatches, _patchPixels]. --------
  final img = Float32List(_numPatches * _patchPixels);
  final rgb = _palette[c];
  for (int p = 0; p < _numPatches; p++) {
    for (int px = 0; px < _patchSize * _patchSize; px++) {
      for (int ch = 0; ch < _numChannels; ch++) {
        img[p * _patchPixels + px * _numChannels + ch] =
            rgb[ch] + (rng.nextDouble() * 2 - 1) * noise;
      }
    }
  }
  final imagePatches = Tensor.fromList(
    [_numPatches, _patchPixels],
    img,
    device: device,
  );

  // -------- Audio: [seqA, featureDim]. Each frame is a Gaussian
  //          bump around bin (2*c) with per-cell noise. --------
  final audio = Float32List(_audioSeq * _audioFeat);
  final peakBin = (2 * c) % _audioFeat;
  for (int t = 0; t < _audioSeq; t++) {
    for (int f = 0; f < _audioFeat; f++) {
      final d = ((f - peakBin).abs()).toDouble();
      final gauss = math.exp(-d * d / 4.0);
      audio[t * _audioFeat + f] = gauss + (rng.nextDouble() * 2 - 1) * noise;
    }
  }
  final audioFeatures = Tensor.fromList(
    [_audioSeq, _audioFeat],
    audio,
    device: device,
  );

  // -------- Video: [numFrames, frameFeatureDim]. Each frame is
  //          a sinusoid at a class-specific phase, plus noise. --------
  final vid = Float32List(_videoFrames * _videoFeat);
  final phase = (c / _numClasses) * 2 * math.pi;
  for (int t = 0; t < _videoFrames; t++) {
    for (int f = 0; f < _videoFeat; f++) {
      final angle = phase + (f / _videoFeat) * 2 * math.pi + t * 0.1;
      vid[t * _videoFeat + f] =
          math.sin(angle) + (rng.nextDouble() * 2 - 1) * noise;
    }
  }
  final videoFrames = Tensor.fromList(
    [_videoFrames, _videoFeat],
    vid,
    device: device,
  );

  // -------- Text prompt: a fixed 2-token ask ([BOS, 'a']). --------
  final textPrompt = Tensor.fromList(
    [_promptLen],
    [_bos.toDouble(), _promptAsk.toDouble()],
    device: device,
  );

  // -------- Target text: [BOS, c1, c2, c3, EOS]. --------
  final tgtTokens = _nameToTokens(_classNames[c]); // length 5
  final decIn = Tensor.fromList(
    [tgtTokens.length - 1],
    tgtTokens.sublist(0, tgtTokens.length - 1).map((i) => i.toDouble()).toList(),
    device: device,
  );
  final decTgt = Tensor.fromList(
    [tgtTokens.length - 1],
    tgtTokens.sublist(1).map((i) => i.toDouble()).toList(),
    device: device,
  );

  return (
    imagePatches: imagePatches,
    audioFeatures: audioFeatures,
    videoFrames: videoFrames,
    textPrompt: textPrompt,
    decIn: decIn,
    decTgt: decTgt,
    label: c,
  );
}

// -------------------------------------------------------------------
// Eval — greedy generate for each class and check exact match.
// -------------------------------------------------------------------
({int correct, int total, List<String> preds}) _evalGenerate(
  MultiModalGenerator gen,
  Device device,
  math.Random rng, {
  int samplesPerClass = 4,
}) {
  final preds = <String>[];
  int correct = 0;
  int total = 0;
  for (int c = 0; c < _numClasses; c++) {
    String? firstPred;
    for (int rep = 0; rep < samplesPerClass; rep++) {
      final s = _mkSample(c, rng, device: device);
      final out = gen.generate(
        imagePatches: s.imagePatches,
        audioFeatures: s.audioFeatures,
        videoFrames: s.videoFrames,
        textIn: s.textPrompt,
        prompt: [_bos],
        maxNewTokens: _nameLen + 1, // 3 chars + EOS
        eosId: _eos,
        device: device,
      );
      // Extract chars between BOS and (optional) EOS.
      final chars = <int>[];
      for (int i = 1; i < out.length; i++) {
        if (out[i] == _eos) break;
        chars.add(out[i]);
      }
      final predStr = _tokensToString(chars);
      firstPred ??= predStr;
      if (predStr == _classNames[c]) correct++;
      total++;
    }
    preds.add('${_classNames[c]} -> "$firstPred"');
  }
  return (correct: correct, total: total, preds: preds);
}

// -------------------------------------------------------------------
// Main
// -------------------------------------------------------------------
void main(List<String> args) {
  final device = args.contains('--gpu') ? Device.GPU : Device.CPU;
  print('=== train_multimodal_demo (${device.name}) ===');
  print('task:      given (image + audio + video + text-prompt), output');
  print('           the 3-letter class name.');
  print('classes:   $_classNames');
  print('vocab:     $_vocabSize (PAD, BOS, EOS + 26 letters)');
  print(
    'jointDim=$_jointDim  decoderLayers=$_decoderLayers  '
    'fusionLayers=$_fusionLayers',
  );

  // 1. Build model.
  final gen = MultiModalGenerator(
    image: ViTBackbone(
      imageSize: _imageSize,
      patchSize: _patchSize,
      numChannels: _numChannels,
      embedDim: _jointDim,
      numLayers: _encoderLayers,
      numHeads: _encoderHeads,
      device: device,
      seed: 1,
    ),
    audio: AudioTransformer(
      featureDim: _audioFeat,
      maxSeqLen: _audioSeq,
      embedDim: _jointDim,
      numLayers: _encoderLayers,
      numHeads: _encoderHeads,
      device: device,
      seed: 2,
    ),
    video: VideoTransformer(
      frameFeatureDim: _videoFeat,
      maxFrames: _videoFrames,
      embedDim: _jointDim,
      numLayers: _encoderLayers,
      numHeads: _encoderHeads,
      device: device,
      seed: 3,
    ),
    text: TextTransformer(
      vocabSize: _vocabSize,
      maxSeqLen: _promptLen,
      embedDim: _jointDim,
      numLayers: _encoderLayers,
      numHeads: _encoderHeads,
      device: device,
      seed: 4,
    ),
    targetVocabSize: _vocabSize,
    maxTargetLen: _maxTargetLen,
    jointEmbedDim: _jointDim,
    decoderLayers: _decoderLayers,
    decoderHeads: _decoderHeads,
    fusionLayers: _fusionLayers,
    fusionHeads: _fusionHeads,
    device: device,
    seed: 5,
  );
  final params = gen.parameters();
  print('params:    ${paramScalarCount(params)} scalars');

  final opt = Adam(params, lr: _lr);
  final rng = math.Random(0);

  // 2. Baseline eval — untrained model.
  final before = _evalGenerate(gen, device, math.Random(100));
  print(
    '\nBEFORE  exact-name accuracy = '
    '${(before.correct / before.total * 100).toStringAsFixed(1)}% '
    '(${before.correct}/${before.total})',
  );
  for (final p in before.preds) print('        $p');

  // 3. Train.
  print(
    '\ntraining $_steps steps '
    '(lr=$_lr, cross-entropy over target sequence, teacher forcing)...',
  );
  final sw = Stopwatch()..start();
  double lossSum = 0;
  for (int step = 1; step <= _steps; step++) {
    opt.zeroGrad();
    final c = rng.nextInt(_numClasses);
    final s = _mkSample(c, rng, device: device);
    final logits = gen(
      imagePatches: s.imagePatches,
      audioFeatures: s.audioFeatures,
      videoFrames: s.videoFrames,
      textIn: s.textPrompt,
      targetTokens: s.decIn,
    );
    final loss = logits.crossEntropy(s.decTgt).mean();
    loss.backward();
    clipGradNorm(params, 1.0);
    opt.step();

    final lossVal = loss.toList()[0];
    lossSum += lossVal;

    if (step == 1 || step % _logEvery == 0 || step == _steps) {
      final ms = sw.elapsedMilliseconds / step;
      final avg = lossSum / step;
      print(
        '  step ${step.toString().padLeft(4)}  '
        'ce=${lossVal.toStringAsFixed(4)}  '
        'avg=${avg.toStringAsFixed(4)}  '
        '(${ms.toStringAsFixed(1)} ms/step)',
      );
    }
  }
  sw.stop();

  // 4. Final eval — greedy generation on fresh noise draws.
  gen.eval();
  final after = _evalGenerate(gen, device, math.Random(200));
  print(
    '\nAFTER   exact-name accuracy = '
    '${(after.correct / after.total * 100).toStringAsFixed(1)}% '
    '(${after.correct}/${after.total})',
  );
  for (final p in after.preds) print('        $p');

  print(
    '\naccuracy change: '
    '${(after.correct - before.correct) >= 0 ? '+' : ''}'
    '${((after.correct - before.correct) / after.total * 100).toStringAsFixed(1)}%',
  );
  if (after.correct >= (0.75 * after.total).round()) {
    print('✅ multimodal transformer generates the correct class name.');
  } else if (after.correct > before.correct) {
    print('⚠️  training improved but did not reach 75% exact match.');
  } else {
    print('❌ training did not improve exact-match accuracy.');
  }
}
