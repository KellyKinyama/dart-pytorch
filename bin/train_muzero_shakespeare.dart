/// MuZero-LM on tiny-Shakespeare — character-level language modelling
/// framed as a Markov decision process and trained through MuZero's
/// K-step latent unroll.
///
/// MDP formulation:
///
///   * Observation `o`   — the last `_obsLen` chars of context.
///   * State       `s`   — the latent encoding of `o`
///                         (produced by [MuZeroRepresentation]).
///   * Action set  `A`   — the full char vocabulary
///                         (one action per possible next char).
///   * Transition        — deterministic, "append action's char to
///                         context"; the *latent* transition is
///                         learned by [MuZeroDynamics].
///   * Reward     `r`    — `+1` if the emitted char matches the
///                         corpus at that position, else `0`. Under
///                         teacher forcing during training, this is
///                         always `+1`.
///   * Value       `v`   — bootstrapped discounted future rewards.
///                         Under teacher forcing this has the closed
///                         form `v(s^k) = (1 - γ^{K-k}) / (1 - γ)`.
///   * Policy      `π`   — categorical over vocab; target is the
///                         actual next char in the corpus (i.e.
///                         behavioural cloning of the corpus).
///
/// Training step (teacher forcing, `_accumBatch` windows per Adam
/// update):
///
///   1. Sample a random window `[start, start + _obsLen + _unrollK]`
///      from the training half of the corpus.
///   2. `obsTokens = corpus[start .. start + _obsLen)`.
///   3. `actions[k] = corpus[start + _obsLen + k]`  for k ∈ [0, K).
///   4. `policy^k target = corpus[start + _obsLen + k]`  for k ∈ [0, K].
///   5. `reward^k target = 1`, `value^k target = closed-form above`.
///   6. Loss = Σ MSE(reward) + Σ MSE(value) + Σ CE(policy).
///
/// Because reward and value are essentially constants under teacher
/// forcing, they act as regularisers on the latent state; the
/// policy-CE term does the real heavy lifting. What makes this a
/// MuZero demo (and not just a small GPT) is that we can generate
/// text in *two* modes:
///
///   * **Re-encode**   — every step, run `h_θ` on the full latest
///                       context. Standard autoregressive LM. Serves
///                       as an upper bound on generation quality.
///   * **Latent-only** — run `h_θ` ONCE on the seed, then unroll
///                       purely through `g_θ` (dynamics) + `f_θ`
///                       (policy head). The LM never sees the emitted
///                       chars — it advances entirely in latent
///                       space, MuZero-style. This exercises the
///                       dynamics function's fidelity.
///
/// The eval prints per-step accuracies of the latent unroll against
/// the corpus, so you can see how far the learned dynamics can carry
/// the state before drifting away from the true trajectory.
///
/// Run from the repository root:
///
///     dart run bin/train_muzero_shakespeare.dart          # CPU
///     dart run bin/train_muzero_shakespeare.dart --gpu    # GPU
library;

import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

import 'shakespeare_util.dart' show loadCorpus, paramScalarCount;

// -------------------------------------------------------------------
// Hyperparameters
// -------------------------------------------------------------------
const int _corpusChars = 40000; // ~4% of the full corpus — fast demo
const double _trainFrac = 0.9;

const int _obsLen = 32; // context length seen by representation
const int _unrollK = 4; // K-step latent unroll
const double _gamma = 0.9;

const int _embedDim = 96;
const int _numLayers = 2;
const int _numHeads = 4;

const int _steps = 3000;
const int _accumBatch = 4;
const int _logEvery = 200;
const double _lr = 3e-3;
const int _evalEvery = 500;

const int _valWindows = 128;
const int _genLen = 200;

// -------------------------------------------------------------------
// Loss helpers
// -------------------------------------------------------------------
Tensor _mse(Tensor pred, double target, Device device) {
  final t = Tensor.fromList([1, 1], [target], device: device);
  final d = pred - t;
  return d * d;
}

Tensor _ce(Tensor logits, int label, Device device) {
  final tgt = Tensor.fromList([1], [label.toDouble()], device: device);
  return logits.crossEntropy(tgt);
}

/// Value target at unroll position k under teacher-forced +1 rewards
/// (v(s^K) = 0):  `(1 - γ^{K-k}) / (1 - γ)`.
double _valueTarget(int k) {
  final rem = _unrollK - k;
  if (rem <= 0) return 0.0;
  return (1.0 - math.pow(_gamma, rem).toDouble()) / (1.0 - _gamma);
}

Tensor _obsTensor(List<int> ids, int start, Device device) {
  final data = List<double>.generate(_obsLen, (i) => ids[start + i].toDouble());
  return Tensor.fromList([_obsLen], data, device: device);
}

int _argmax(List<double> row) {
  int best = 0;
  double bestV = row[0];
  for (int j = 1; j < row.length; j++) {
    if (row[j] > bestV) {
      bestV = row[j];
      best = j;
    }
  }
  return best;
}

// -------------------------------------------------------------------
// Eval — val CE at s^0, plus per-step accuracy of the teacher-forced
// latent unroll (does the dynamics correctly advance the state?).
// -------------------------------------------------------------------
({double reEncodeAcc, double valCE, List<double> unrollAcc}) _evalMuZero(
  MuZeroLM model,
  List<int> valIds,
  Device device,
  math.Random rng, {
  int n = _valWindows,
}) {
  final unrollCorrect = List<int>.filled(_unrollK + 1, 0);
  int reCorrect = 0;
  int total = 0;
  double lossSum = 0;
  final maxStart = valIds.length - _obsLen - _unrollK - 1;
  if (maxStart <= 0) {
    throw StateError(
      'val split too short: need >${_obsLen + _unrollK + 1} tokens',
    );
  }
  for (int w = 0; w < n; w++) {
    final start = rng.nextInt(maxStart);
    final obs = _obsTensor(valIds, start, device);

    // Val next-char CE at s^0.
    final tgt = Tensor.fromList(
      [1],
      [valIds[start + _obsLen].toDouble()],
      device: device,
    );
    final init = model.initialInference(obs);
    lossSum += init.policy.crossEntropy(tgt).toList()[0];

    // Re-encode next-char argmax accuracy at s^0.
    if (_argmax(init.policy.toList()) == valIds[start + _obsLen]) {
      reCorrect++;
    }

    // Teacher-forced K-step latent unroll — accuracy at each s^k
    // measures how faithful the dynamics are.
    final actions = List<int>.generate(
      _unrollK,
      (k) => valIds[start + _obsLen + k],
    );
    final u = model.unroll(obs, actions);
    for (int k = 0; k <= _unrollK; k++) {
      if (_argmax(u.policies[k].toList()) == valIds[start + _obsLen + k]) {
        unrollCorrect[k]++;
      }
    }
    total++;
  }
  return (
    reEncodeAcc: reCorrect / total,
    valCE: lossSum / total,
    unrollAcc: unrollCorrect.map((c) => c / total).toList(),
  );
}

// -------------------------------------------------------------------
// Generation — two modes.
// -------------------------------------------------------------------

/// Standard AR: at every step re-encode the whole context with
/// `h_θ` and pick argmax of the initial-inference policy.
String _generateReencode(
  MuZeroLM model,
  CharTokenizer tok,
  String seed,
  int maxNewTokens,
  Device device,
) {
  final ids = tok.encode(seed);
  final out = List<int>.of(ids);
  for (int step = 0; step < maxNewTokens; step++) {
    final start = out.length > _obsLen ? out.length - _obsLen : 0;
    final ctxLen = out.length - start;
    final data = List<double>.generate(
      ctxLen,
      (i) => out[start + i].toDouble(),
    );
    final obs = Tensor.fromList([ctxLen], data, device: device);
    final init = model.initialInference(obs);
    out.add(_argmax(init.policy.toList()));
  }
  return tok.decode(out);
}

/// MuZero-style: run `h_θ` ONCE on the seed, then advance purely
/// through `g_θ` (dynamics) + `f_θ` (policy head). The model never
/// re-encodes the emitted chars — the latent state carries all the
/// information about what has been generated so far.
String _generateLatent(
  MuZeroLM model,
  CharTokenizer tok,
  String seed,
  int maxNewTokens,
  Device device,
) {
  final ids = tok.encode(seed);
  final ctxLen = ids.length > _obsLen ? _obsLen : ids.length;
  final ctxStart = ids.length - ctxLen;
  final data = List<double>.generate(
    ctxLen,
    (i) => ids[ctxStart + i].toDouble(),
  );
  final obs = Tensor.fromList([ctxLen], data, device: device);
  final init = model.initialInference(obs);
  var state = init.state;
  var policy = init.policy;
  final out = List<int>.of(ids);
  for (int step = 0; step < maxNewTokens; step++) {
    final next = _argmax(policy.toList());
    out.add(next);
    final rec = model.recurrentInference(state, next);
    state = rec.nextState;
    policy = rec.policy;
  }
  return tok.decode(out);
}

// -------------------------------------------------------------------
// Main
// -------------------------------------------------------------------
void main(List<String> args) {
  final device = args.contains('--gpu') ? Device.GPU : Device.CPU;
  print('=== train_muzero_shakespeare (${device.name}) ===');

  final text = loadCorpus(maxChars: _corpusChars);
  final tok = CharTokenizer.fromText(text);
  final ids = tok.encode(text);
  final splitIdx = (ids.length * _trainFrac).round();
  final trainIds = ids.sublist(0, splitIdx);
  final valIds = ids.sublist(splitIdx);
  print('corpus:    ${text.length} chars, vocab: ${tok.vocabSize}');
  print('train/val: ${trainIds.length} / ${valIds.length} tokens');
  print('unroll:    obsLen=$_obsLen  K=$_unrollK  γ=$_gamma');
  print(
    'model:     MuZeroLM  embedDim=$_embedDim  '
    'numLayers=$_numLayers  numHeads=$_numHeads',
  );

  final model = MuZeroLM(
    vocabSize: tok.vocabSize,
    maxObsLen: _obsLen,
    embedDim: _embedDim,
    numActions: tok.vocabSize,
    numLayers: _numLayers,
    numHeads: _numHeads,
    device: device,
    seed: 0,
  );
  final params = model.parameters();
  print('params:    ${paramScalarCount(params)} scalars');

  final opt = Adam(params, lr: _lr);

  // Value-target lookup table (constants under teacher forcing).
  final valueTargets = List<double>.generate(_unrollK + 1, _valueTarget);

  // Baseline eval — untrained.
  model.eval();
  final before = _evalMuZero(model, valIds, device, math.Random(100));
  print(
    '\nBEFORE  val CE = ${before.valCE.toStringAsFixed(4)}  '
    '(uniform ≈ ${math.log(tok.vocabSize).toStringAsFixed(4)})',
  );
  print(
    '        re-encode next-char acc = '
    '${(before.reEncodeAcc * 100).toStringAsFixed(1)}%',
  );
  print(
    '        latent unroll acc @ k = 0..$_unrollK: '
    '${before.unrollAcc.map((a) => '${(a * 100).toStringAsFixed(1)}%').join(', ')}',
  );

  // Training.
  model.train();
  print(
    '\ntraining $_steps steps '
    '(lr=$_lr, batch=$_accumBatch, unroll K=$_unrollK)…',
  );
  final rng = math.Random(0);
  final maxStart = trainIds.length - _obsLen - _unrollK - 1;
  final sw = Stopwatch()..start();
  double lossSum = 0;
  for (int step = 1; step <= _steps; step++) {
    opt.zeroGrad();
    Tensor? batchLoss;
    for (int b = 0; b < _accumBatch; b++) {
      final start = rng.nextInt(maxStart);
      final obs = _obsTensor(trainIds, start, device);
      final actions = List<int>.generate(
        _unrollK,
        (k) => trainIds[start + _obsLen + k],
      );
      final u = model.unroll(obs, actions);

      // s^0 : value + policy only (no reward at position 0).
      var loss = _mse(u.values[0], valueTargets[0], device);
      loss = loss + _ce(u.policies[0], trainIds[start + _obsLen], device);
      // s^{k+1} : reward for the just-taken action + value + policy.
      for (int k = 0; k < _unrollK; k++) {
        loss = loss + _mse(u.rewards[k], 1.0, device);
        loss = loss + _mse(u.values[k + 1], valueTargets[k + 1], device);
        loss =
            loss +
            _ce(u.policies[k + 1], trainIds[start + _obsLen + k + 1], device);
      }
      batchLoss = batchLoss == null ? loss : batchLoss + loss;
    }
    final l = batchLoss!.mean();
    l.backward();
    clipGradNorm(params, 1.0);
    opt.step();

    final lVal = l.toList()[0];
    lossSum += lVal;
    if (step == 1 || step % _logEvery == 0 || step == _steps) {
      final ms = sw.elapsedMilliseconds / step;
      print(
        '  step ${step.toString().padLeft(4)}  '
        'loss=${lVal.toStringAsFixed(4)}  '
        'avg=${(lossSum / step).toStringAsFixed(4)}  '
        '(${ms.toStringAsFixed(1)} ms/step)',
      );
    }
    if (step % _evalEvery == 0 && step != _steps) {
      model.eval();
      final e = _evalMuZero(model, valIds, device, math.Random(100), n: 64);
      print(
        '            val CE=${e.valCE.toStringAsFixed(4)}  '
        're-encode-acc=${(e.reEncodeAcc * 100).toStringAsFixed(1)}%  '
        'unroll@K=${(e.unrollAcc.last * 100).toStringAsFixed(1)}%',
      );
      model.train();
    }
  }
  sw.stop();

  // Final eval.
  model.eval();
  final after = _evalMuZero(model, valIds, device, math.Random(200));
  print(
    '\nAFTER   val CE = ${after.valCE.toStringAsFixed(4)}  '
    '(before ${before.valCE.toStringAsFixed(4)})',
  );
  print(
    '        re-encode next-char acc = '
    '${(after.reEncodeAcc * 100).toStringAsFixed(1)}% '
    '(before ${(before.reEncodeAcc * 100).toStringAsFixed(1)}%)',
  );
  print(
    '        latent unroll acc @ k = 0..$_unrollK: '
    '${after.unrollAcc.map((a) => '${(a * 100).toStringAsFixed(1)}%').join(', ')}',
  );

  // Generation.
  const seed = 'ROMEO:';
  print('\n--- generation seed: ${seed.replaceAll("\n", "\\n")} ---');
  print(
    '\n[re-encode mode: h(context) every step]\n'
    '${_generateReencode(model, tok, seed, _genLen, device)}',
  );
  print(
    '\n[latent-only mode: h(seed) once, then dynamics only]\n'
    '${_generateLatent(model, tok, seed, _genLen, device)}',
  );

  final ceDrop = before.valCE - after.valCE;
  if (ceDrop > 1.0 && after.reEncodeAcc > 0.20) {
    print(
      '\n✅ MuZero-LM learned Shakespeare: '
      'val CE dropped ${ceDrop.toStringAsFixed(2)}, '
      'next-char acc ${(after.reEncodeAcc * 100).toStringAsFixed(1)}% '
      '(uniform baseline ${(100.0 / tok.vocabSize).toStringAsFixed(1)}%).',
    );
  } else if (ceDrop > 0) {
    print(
      '\n⚠️  training improved but did not clear the strong threshold '
      '(val CE −${ceDrop.toStringAsFixed(2)}).',
    );
  } else {
    print('\n❌ training did not improve val CE.');
  }
}
