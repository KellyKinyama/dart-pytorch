/// Bigger tiny-Shakespeare MoE demo — mixture-of-experts sized to
/// let a 6 GB GPU beat CPU while showcasing DeepSeek-V3-style
/// routing (sigmoid gate + top-K renorm + sign-based bias
/// balancer). Sparse execution is enabled: only the top-K experts
/// are materialised per token via `Tensor.embedding` +
/// `Tensor.scatterRowsAdd`, so total FLOPs stay bounded even as the
/// expert count grows.
///
/// Two GPU-friendly habits vs. the small `shakespeare_moe.dart`:
///
///   * Read `loss.toList()[0]` only at the print interval.
///   * Skip `clipGradNorm` — its per-parameter D2H reads drain
///     throughput; Adam is scale-invariant so it's not required.
///
/// Run from the repository root:
///
///     dart run bin/shakespeare_moe_big.dart              # GPU (default)
///     dart run bin/shakespeare_moe_big.dart --cpu        # CPU (slow)
///     dart run bin/shakespeare_moe_big.dart --steps 200
library;

import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

import 'shakespeare_util.dart';

// -----------------------------------------------------------------------------
// Sizing defaults.
//
// Params (per routed+shared MLP expert): 2 * D * H  where D=256, H=1024
//   -> ~525 k per expert; 8 routed + 1 shared = ~4.7 M per block.
// Attention (4 heads x D=256): 4 * D^2 = ~262 k per block.
// Total per block ~ 5.0 M; x 4 blocks = ~20 M scalars.
// Plus embedding + final ln + head:
//   65*256 + T*256 (posEnc is sinusoidal, no params) + 256 * 65
//   = ~50 k. Grand total ~20 M scalars = 80 MB fp32;
//   ~240 MB with Adam moments.
// -----------------------------------------------------------------------------

const int _corpusChars = 200000;
const int _blockSize = 128;
const int _embedDim = 256;
const int _numLayers = 4;
const int _numHeads = 4;
const int _numRoutedExperts = 8;
const int _numSharedExperts = 1;
const int _topK = 2;
const int _expertHiddenDim = 1024;
const int _defaultSteps = 400;
const int _biasUpdateEvery = 50;
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
  final activation = useCpu ? ExpertActivation.relu : ExpertActivation.silu;

  print('=== shakespeare_moe_big : bigger MoE LM (${device.name}) ===');
  if (useCpu) {
    print(
      'note: --cpu selected; a big MoE model on CPU is slow. Consider '
      '`dart run bin/shakespeare_moe.dart` for the fast small demo.',
    );
  }
  final text = loadCorpus(maxChars: _corpusChars);
  final tok = CharTokenizer.fromText(text);
  final ids = tok.encode(text);
  print(
    'corpus: ${text.length} chars, vocab: ${tok.vocabSize}, '
    'tokens: ${ids.length}',
  );

  final lm = MoELanguageModel(
    vocabSize: tok.vocabSize,
    embedDim: _embedDim,
    numLayers: _numLayers,
    numHeads: _numHeads,
    maxLen: _blockSize,
    numRoutedExperts: _numRoutedExperts,
    numSharedExperts: _numSharedExperts,
    topK: _topK,
    expertHiddenDim: _expertHiddenDim,
    biasUpdateRate: 0.01,
    device: device,
    activation: activation,
    expertVariant: ExpertVariant.swiGlu,
    gateFunction: GateFunction.sigmoid,
    biasUpdateRule: BiasUpdateRule.sign,
    routeScale: 2.5,
    sparseExecution: true,
    seed: 1,
  );
  final params = lm.parameters();
  print(
    'model : MoE LM ${_numLayers}L x ${_embedDim}d x ${_numHeads}h  '
    '(routed=$_numRoutedExperts shared=$_numSharedExperts topK=$_topK '
    'hidden=$_expertHiddenDim swiGlu+sigmoid+sparse)  '
    '${paramScalarCount(params)} scalars',
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
    if (step % _biasUpdateEvery == 0) {
      lm.updateRoutingBias();
    }
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

  print('\nfinal routing bias per block:');
  for (int b = 0; b < lm.blocks.length; b++) {
    final biases = lm.blocks[b].moe.routingBias;
    final formatted = biases.map((v) => v.toStringAsFixed(3)).join(', ');
    print('  block $b: [$formatted]');
  }

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
