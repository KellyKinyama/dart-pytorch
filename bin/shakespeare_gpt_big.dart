/// Bigger tiny-Shakespeare GPT demo — sized to make a modest GPU
/// (~6 GB) clearly beat CPU. Also demonstrates two GPU-friendly
/// habits that the smaller `shakespeare_gpt.dart` demo doesn't
/// bother with:
///
///   * Read `loss.toList()[0]` only at the print interval, not every
///     step. Reading it every step forces a `cudaMemcpy D2H` + full
///     sync per step and is one of the main reasons GPU appears
///     slow at demo sizes.
///   * Skip `clipGradNorm` — its per-parameter `g.toList()` reads
///     also drain host↔device throughput. Adam is scale-invariant
///     so gradient clipping is not required for stability at these
///     sizes.
///
/// Defaults are tuned to fit comfortably in ~1 GB of VRAM with Adam
/// moment buffers included, leaving plenty of headroom on a 6 GB
/// card. Passing `--cpu` still works but training will be slow.
///
/// Run from the repository root:
///
///     dart run bin/shakespeare_gpt_big.dart              # GPU (default)
///     dart run bin/shakespeare_gpt_big.dart --cpu        # CPU (slow)
///     dart run bin/shakespeare_gpt_big.dart --steps 200  # shorter
library;

import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

import 'shakespeare_util.dart';

// -----------------------------------------------------------------------------
// Sizing defaults.
//
// Params (weight-tied GPT):
//   embed(V,D) + pos(T,D) + L * (4 * D^2 attention + 2 * D * (4 D) ffn)
//   = 65*384 + 192*384 + 6 * (4*384^2 + 8*384^2)
//   = ~11 M scalars, ~44 MB fp32; ~132 MB with Adam moments.
// Attention scratch per step:
//   T^2 * numHeads * L = 192^2 * 6 * 6 ~ 1.3 M scalars = ~5 MB. Tiny.
// -----------------------------------------------------------------------------

const int _corpusChars = 200000;
const int _blockSize = 192;
const int _embedDim = 384;
const int _numLayers = 6;
const int _numHeads = 6;
const int _defaultSteps = 500;
const double _lr = 3e-4;
const int _printEvery = 25;
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

  print('=== shakespeare_gpt_big : bigger GPT demo (${device.name}) ===');
  if (useCpu) {
    print(
      'note: --cpu selected; a big model on CPU is slow. Consider '
      '`dart run bin/shakespeare_gpt.dart` for the fast small demo.',
    );
  }
  final text = loadCorpus(maxChars: _corpusChars);
  final tok = CharTokenizer.fromText(text);
  final ids = tok.encode(text);
  print(
    'corpus: ${text.length} chars, vocab: ${tok.vocabSize}, '
    'tokens: ${ids.length}',
  );

  final gpt = GPT(
    GPTConfig(
      vocabSize: tok.vocabSize,
      maxCtx: _blockSize,
      embedDim: _embedDim,
      numLayers: _numLayers,
      numHeads: _numHeads,
      dropoutP: 0.0,
      tieWeights: true,
      device: device,
      seed: 1,
    ),
  );
  final params = gpt.parameters();
  print(
    'model : GPT (weight-tied) '
    '${_numLayers}L x ${_embedDim}d x ${_numHeads}h  '
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
    final loss = gpt(x).crossEntropy(y).mean();
    loss.backward();
    // clipGradNorm intentionally omitted — its per-parameter D2H
    // reads dominate wall-clock at these sizes and Adam is
    // scale-invariant.
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
  final prompt = tok.encode(promptStr).map((i) => i.toDouble()).toList();
  print('\nprompt: "$promptStr"');
  gpt.eval();
  final sampled = gpt.generate(
    prompt,
    maxNewTokens: _sampleTokens,
    temperature: 0.8,
    topK: 20,
    rng: math.Random(1),
    useCache: true,
  );
  print('--- T=0.8 top-k=20 ---');
  print(tok.decode(sampled.map((v) => v.toInt()).toList()));
}
