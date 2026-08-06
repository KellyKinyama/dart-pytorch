/// Tiny end-to-end demo of the [LlamaVision] wiring: train a
/// [VisionProjector] that translates our from-scratch [ViTBackbone]
/// features into a from-scratch [Llama] decoder's embedding space,
/// so the decoder can name the class of a synthetic coloured image.
///
/// This is the spiritual sibling of `bin/train_multimodal_demo.dart`
/// (which uses four modalities feeding a bespoke MultiModalLM). Here
/// we exercise the *Llama-shaped* multi-modal path: vision tokens
/// prepended to text tokens, single causal stack, weight-tied output.
///
/// Setup
///
/// * 8 classes, each with a distinct RGB palette + 3-letter name.
/// * Image: 8×8×3, patched to 4 patches of 4×4×3 → 5 image tokens
///   (4 patches + CLS).
/// * Vocab: PAD, BOS, EOS + 26 letters = 29.
/// * Target sequence: `[BOS, c1, c2, c3, EOS]` — teacher-forced.
///
/// The full Llama+ViT+Projector stack is trained end-to-end. On a
/// dataset this simple the vision encoder is easy to memorise; on a
/// larger corpus you would freeze `vit` and only train `projector`
/// (LLaVA-style stage-1 alignment). This demo just proves the
/// plumbing works: gradients flow ViT → projector → Llama end to end
/// and the model learns to caption the synthetic images.
///
/// Run:
///
///     dart run bin/train_llama_vision_demo.dart          # CPU
///     dart run bin/train_llama_vision_demo.dart --gpu    # GPU
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

import 'shakespeare_util.dart' show paramScalarCount;

// -------------------------------------------------------------------
// Vocab + class definitions (same shape as train_multimodal_demo)
// -------------------------------------------------------------------
const int _pad = 0;
const int _bos = 1;
const int _eos = 2;
const int _charBase = 3;
const int _vocabSize = _charBase + 26; // 29

const int _numClasses = 8;
const int _nameLen = 3;

const List<String> _classNames = [
  'red', 'sun', 'sky', 'sea', //
  'cat', 'dog', 'owl', 'orb', //
];

const List<List<double>> _palette = [
  [0.90, 0.10, 0.10],
  [0.95, 0.85, 0.10],
  [0.20, 0.55, 0.95],
  [0.10, 0.35, 0.55],
  [0.55, 0.35, 0.20],
  [0.60, 0.45, 0.30],
  [0.75, 0.65, 0.45],
  [0.85, 0.55, 0.20],
];

int _charToken(int letterIdx) => _charBase + letterIdx;

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
// Image + text config
// -------------------------------------------------------------------
const int _imageSize = 8;
const int _patchSize = 4;
const int _numChannels = 3;
const int _patchPixels = _patchSize * _patchSize * _numChannels; // 48
const int _numPatches = 4;
const int _numImageTokens = _numPatches + 1; // + CLS

// ViT hyperparams
const int _vitEmbedDim = 24;
const int _vitLayers = 2;
const int _vitHeads = 4;

// Llama hyperparams
const int _llamaEmbedDim = 32;
const int _llamaLayers = 2;
const int _llamaHeads = 4;
const int _llamaFfnDim = 64;
const int _llamaMaxCtx = 32;

// Training
const int _steps = 1200;
const int _logEvery = 100;
const double _lr = 2e-3;

// -------------------------------------------------------------------
// Synthetic sample generation
// -------------------------------------------------------------------
typedef _Sample = ({
  Tensor imagePatches,
  Tensor decIn,
  Tensor decTgt,
  int label,
});

_Sample _mkSample(
  int c,
  math.Random rng, {
  required Device device,
  double noise = 0.08,
}) {
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

  final tgtTokens = _nameToTokens(_classNames[c]); // 5 tokens
  final decIn = Tensor.fromList(
    [tgtTokens.length - 1],
    tgtTokens
        .sublist(0, tgtTokens.length - 1)
        .map((i) => i.toDouble())
        .toList(),
    device: device,
  );
  final decTgt = Tensor.fromList(
    [tgtTokens.length - 1],
    tgtTokens.sublist(1).map((i) => i.toDouble()).toList(),
    device: device,
  );
  return (imagePatches: imagePatches, decIn: decIn, decTgt: decTgt, label: c);
}

// Slice out only the text-prediction rows of the full logits using
// the differentiable `embedding` op (rows past the image-token
// prefix). Gradient flows only into those rows.
Tensor _textLogits(Tensor fullLogits, int targetStart, int targetLen) {
  final idxData = Float32List(targetLen);
  for (int i = 0; i < targetLen; i++) {
    idxData[i] = (targetStart + i).toDouble();
  }
  final indices = Tensor.fromList(
    [targetLen],
    idxData,
    device: fullLogits.device,
  );
  return fullLogits.embedding(indices);
}

// -------------------------------------------------------------------
// Eval — greedy generation, count exact-name matches.
// -------------------------------------------------------------------
({int correct, int total, List<String> preds}) _evalGenerate(
  LlamaVision lv,
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
      final out = lv.generate(
        s.imagePatches,
        [_bos.toDouble()],
        maxNewTokens: _nameLen + 1,
        temperature: 0.0,
        stopId: _eos,
      );
      final chars = <int>[];
      for (int i = 1; i < out.length; i++) {
        final tok = out[i].toInt();
        if (tok == _eos) break;
        chars.add(tok);
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
  print('=== train_llama_vision_demo (${device.name}) ===');
  print('arch:    ViTBackbone -> VisionProjector -> Llama decoder');
  print('         (image tokens prepended to text; causal mask)');
  print('task:    given an 8x8 coloured image, output the 3-letter class name');
  print('classes: $_classNames');
  print(
    'vocab:   $_vocabSize (PAD, BOS, EOS + 26 letters)  '
    'imageTokens=$_numImageTokens',
  );

  // 1. Build ViT + Llama + wrapper.
  final vit = ViTBackbone(
    imageSize: _imageSize,
    patchSize: _patchSize,
    numChannels: _numChannels,
    embedDim: _vitEmbedDim,
    numLayers: _vitLayers,
    numHeads: _vitHeads,
    device: device,
    seed: 1,
  );
  final llama = Llama(
    LlamaConfig(
      vocabSize: _vocabSize,
      maxCtx: _llamaMaxCtx,
      embedDim: _llamaEmbedDim,
      numLayers: _llamaLayers,
      numHeads: _llamaHeads,
      ffnDim: _llamaFfnDim,
      device: device,
      seed: 2,
    ),
  );
  final lv = LlamaVision.build(vit: vit, llama: llama, seed: 3);

  final params = lv.parameters();
  print('params:  ${paramScalarCount(params)} scalars total');
  print(
    '         vit=${paramScalarCount(vit.parameters())}  '
    'projector=${paramScalarCount(lv.projector.parameters())}  '
    'llama=${paramScalarCount(llama.parameters())}',
  );

  final opt = Adam(params, lr: _lr);
  final rng = math.Random(0);

  // 2. Baseline eval — random init.
  final before = _evalGenerate(lv, device, math.Random(100));
  print(
    '\nBEFORE  exact-name accuracy = '
    '${(before.correct / before.total * 100).toStringAsFixed(1)}% '
    '(${before.correct}/${before.total})',
  );
  for (final p in before.preds) print('        $p');

  // 3. Train.
  print(
    '\ntraining $_steps steps (lr=$_lr, cross-entropy on text positions '
    'only)...',
  );
  final sw = Stopwatch()..start();
  double lossSum = 0;
  for (int step = 1; step <= _steps; step++) {
    opt.zeroGrad();
    final c = rng.nextInt(_numClasses);
    final s = _mkSample(c, rng, device: device);
    final full = lv(s.imagePatches, s.decIn);
    final tgtLen = s.decIn.shape[0];
    final textLogits = _textLogits(full, _numImageTokens, tgtLen);
    final loss = textLogits.crossEntropy(s.decTgt).mean();
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

  // 4. Final eval.
  lv.eval();
  final after = _evalGenerate(lv, device, math.Random(200));
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
    print('✅ LlamaVision generates the correct class name.');
  } else if (after.correct > before.correct) {
    print('⚠️  training improved but did not reach 75% exact match.');
  } else {
    print('❌ training did not improve exact-match accuracy.');
  }
}
