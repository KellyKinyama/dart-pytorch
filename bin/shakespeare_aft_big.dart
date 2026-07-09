/// Bigger tiny-Shakespeare AFT demo — attention-free transformer at
/// a size where a 6 GB GPU comfortably beats CPU.
///
/// AFT-Full is O(T*D) per layer (no explicit T x T attention
/// matrix), so it scales well to long contexts. The default
/// `blockSize = 512` here is 16x the small demo's context and is
/// what really lets the GPU pull ahead.
///
/// Two GPU-friendly habits vs. the small `shakespeare_aft.dart`:
///
///   * Read `loss.toList()[0]` only at the print interval.
///   * Skip `clipGradNorm` — its per-parameter D2H reads drain
///     throughput; Adam is scale-invariant so it's not required.
///
/// Run from the repository root:
///
///     dart run bin/shakespeare_aft_big.dart              # GPU (default)
///     dart run bin/shakespeare_aft_big.dart --cpu        # CPU (slow)
///     dart run bin/shakespeare_aft_big.dart --steps 200
library;

import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

import 'shakespeare_util.dart';

// -----------------------------------------------------------------------------
// Sizing defaults.
//
// Params:
//   embed(V,D) + L * (3 * D^2 (Q,K,V) + T^2 (posBias) + 2 * D * (4 D) ffn)
//   = 65*384 + 6 * (3*384^2 + 512^2 + 8*384^2)
//   = 25k + 6 * (442k + 262k + 1.18M)
//   ~ 11.3 M scalars, ~45 MB fp32; ~135 MB with Adam moments.
// -----------------------------------------------------------------------------

const int _corpusChars = 300000;
const int _blockSize = 512;
const int _embedDim = 384;
const int _numLayers = 6;
const int _defaultSteps = 400;
const double _lr = 3e-4;
const int _printEvery = 20;
const int _sampleTokens = 300;

int? _intFlag(List<String> args, String name) {
  final idx = args.indexOf(name);
  if (idx < 0 || idx + 1 >= args.length) return null;
  return int.tryParse(args[idx + 1]);
}

void main(List<String> args) {
  final useCpu = args.contains('--cpu');
  final device = useCpu ? Device.CPU : Device.GPU;
  final trainSteps = _intFlag(args, '--steps') ?? _defaultSteps;

  print('=== shakespeare_aft_big : bigger AFT demo (${device.name}) ===');
  if (useCpu) {
    print(
      'note: --cpu selected; a big AFT model on CPU is slow. Consider '
      '`dart run bin/shakespeare_aft.dart` for the fast small demo.',
    );
  }
  final text = loadCorpus(maxChars: _corpusChars);
  final tok = CharTokenizer.fromText(text);
  final ids = tok.encode(text);
  print(
    'corpus: ${text.length} chars, vocab: ${tok.vocabSize}, '
    'tokens: ${ids.length}',
  );

  final lm = AFTLanguageModel(
    vocabSize: tok.vocabSize,
    embedDim: _embedDim,
    numLayers: _numLayers,
    maxLen: _blockSize,
    device: device,
    seed: 1,
  );
  final params = lm.parameters();
  print(
    'model : AFTLanguageModel ${_numLayers}L x ${_embedDim}d  '
    'block=$_blockSize  ${paramScalarCount(params)} scalars',
  );

  final opt = Adam(params, lr: _lr);
  final rng = math.Random(0);
  final sw = Stopwatch()..start();
  print('training $trainSteps steps (lr=$_lr, print every $_printEvery)...');
  int printedSteps = 0;
  int printedMs = 0;
  for (int step = 1; step <= trainSteps; step++) {
    opt.zeroGrad();
    final (x, y) = getWindow(ids, _blockSize, rng, device: device);
    final loss = lm(x).crossEntropy(y).mean();
    loss.backward();
    // clipGradNorm intentionally omitted (see file header).
    opt.step();
    final atInterval = step == 1 || step % _printEvery == 0 || step == trainSteps;
    if (atInterval) {
      final l = loss.toList()[0];
      final segMs = sw.elapsedMilliseconds - printedMs;
      final segSteps = step - printedSteps;
      print(
        '  step ${step.toString().padLeft(5)}  '
        'loss=${l.toStringAsFixed(4)}  '
        '${(segMs / segSteps).toStringAsFixed(1)} ms/step',
      );
      printedSteps = step;
      printedMs = sw.elapsedMilliseconds;
    }
  }
  sw.stop();
  print(
    'training done in ${sw.elapsedMilliseconds} ms '
    '(${(sw.elapsedMilliseconds / trainSteps).toStringAsFixed(1)} ms/step)',
  );

  const promptStr = 'ROMEO:';
  final prompt = tok.encode(promptStr);
  print('\nprompt: "$promptStr"');
  lm.eval();

  List<double> stepFn(List<double> ctx) {
    final x = Tensor.fromList([ctx.length], ctx, device: device);
    return lastRowLogits(lm(x), tok.vocabSize);
  }

  final sampled = generateText(
    prompt,
    maxNewTokens: _sampleTokens,
    maxCtx: _blockSize,
    vocabSize: tok.vocabSize,
    stepFn: stepFn,
    temperature: 0.8,
    topK: 20,
    rng: math.Random(1),
  );
  print('--- T=0.8 top-k=20 ---');
  print(tok.decode(sampled));
}
