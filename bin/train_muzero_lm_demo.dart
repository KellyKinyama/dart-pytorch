/// MuZero-LM demo: supervised K-step latent-dynamics training on a
/// tiny toy environment.
///
/// Task ("hot-digit"):
///
///   * State: 4 digits `[a₀, a₁, a₂, a₃]`, each in `[0..3]`. Vocab
///     size = 4. This is the observation the network sees (encoded
///     into a latent state by the language-model representation
///     function).
///   * Actions: 4 discrete ids. Action `i` increments digit `aᵢ`
///     modulo 4.
///   * Reward: on action `i` at state `s`, if `s[i] == 2` (bump makes
///     it `3`) the reward is `+1`; if `s[i] == 3` (bumps and wraps to
///     `0`) the reward is `-1`; else `0`.
///   * "Optimal action" at any state = the index of a digit that
///     equals `2` (breaking ties by lowest index). If no digit is
///     hot, the optimal action defaults to `0`.
///
/// Data generation (per training step):
///
///   * Sample a random 4-digit start state → `obsTokens`.
///   * Sample K random actions.
///   * Simulate the environment: collect true `rewards[1..K]` from
///     that random rollout.
///   * Compute `valueTargets[0..K]` by bootstrapping `z_K = 0` and
///     back-propagating `z_k = r_{k+1} + γ · z_{k+1}`.
///   * Compute `policyTargets[0..K]` as the optimal-action int at
///     each *true* environment state.
///
/// Loss (following MuZero, Schrittwieser et al. 2020):
///
///   ```
///   L = MSE(v_pred[0], z[0]) + CE(p_pred[0], π*[0])
///     + Σ_{k=1..K} [ MSE(r_pred[k], u[k])
///                  + MSE(v_pred[k], z[k])
///                  + CE (p_pred[k], π*[k]) ]
///   ```
///
/// The network:
///
///   * Representation (`MuZeroRepresentation`) — Embedding + learned
///     positional embedding + 2-layer TransformerEncoder + mean-pool.
///     THIS is the "language model that encodes the states".
///   * Dynamics (`MuZeroDynamics`) — residual MLP that takes the
///     latent state + an action embedding and predicts both the next
///     latent state and a scalar reward.
///   * Prediction (`MuZeroPrediction`) — two linear heads that read
///     the latent state to produce policy logits and a scalar value.
///
/// Eval after training: for a batch of fresh random states, we check
/// (a) does the initial-inference policy pick the optimal action, and
/// (b) do the unrolled reward/value predictions track the true
/// discounted returns from a fresh random rollout?
///
/// Run:
///     dart run bin/train_muzero_lm_demo.dart           # CPU
///     dart run bin/train_muzero_lm_demo.dart --gpu     # GPU
library;

import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

import 'shakespeare_util.dart' show paramScalarCount;

// -------------------------------------------------------------------
// Environment
// -------------------------------------------------------------------
const int _stateLen = 4;
const int _vocabSize = 4; // digit values 0..3
const int _numActions = 4; // one per digit position
const double _gamma = 0.9;

/// Reward for taking action [a] in state [s]:
///   +1 if s[a] == 2 (bump → 3)
///   -1 if s[a] == 3 (bump → 0)
///    0 otherwise
double _reward(List<int> s, int a) {
  final v = s[a];
  if (v == 2) return 1.0;
  if (v == 3) return -1.0;
  return 0.0;
}

/// In-place environment step.
List<int> _step(List<int> s, int a) {
  final next = List<int>.of(s);
  next[a] = (next[a] + 1) % _vocabSize;
  return next;
}

/// Optimal-action rule used as the policy target: index of the
/// lowest-index digit equal to 2 (which yields immediate +1). If
/// none, default to 0 (a harmless choice).
int _optimalAction(List<int> s) {
  for (int i = 0; i < _stateLen; i++) {
    if (s[i] == 2) return i;
  }
  return 0;
}

// -------------------------------------------------------------------
// Model hyperparameters
// -------------------------------------------------------------------
const int _embedDim = 32;
const int _numLayers = 2;
const int _numHeads = 4;

const int _unrollK = 4; // K-step unroll length
const int _steps = 2500;
const int _accumBatch = 4; // trajectories per parameter update (mini-batch)
const int _logEvery = 100;
const double _lr = 3e-3;

// -------------------------------------------------------------------
// Loss helpers
// -------------------------------------------------------------------

/// Scalar mean-squared-error between a `[1, 1]` prediction and a
/// scalar target. Returned as a `[1, 1]` tensor for easy summing.
Tensor _mseScalar(Tensor pred, double target, Device device) {
  final t = Tensor.fromList([1, 1], [target], device: device);
  final d = pred - t;
  return d * d;
}

/// Cross-entropy of `[1, C]` logits against integer class label.
/// Returns `[1, 1]`.
Tensor _ceScalar(Tensor logits, int label, Device device) {
  final tgt = Tensor.fromList([1], [label.toDouble()], device: device);
  return logits.crossEntropy(tgt);
}

// -------------------------------------------------------------------
// Trajectory generation
// -------------------------------------------------------------------
class _Trajectory {
  final List<int> obs; // initial state s0 as int digits
  final List<int> actions; // K random actions
  final List<double> rewards; // length K:  reward received on step k
  final List<double> values; // length K+1: bootstrapped return z_k
  final List<int> policies; // length K+1: optimal-action target π*_k

  _Trajectory(this.obs, this.actions, this.rewards, this.values, this.policies);
}

_Trajectory _sampleTraj(math.Random rng) {
  final s0 = List<int>.generate(_stateLen, (_) => rng.nextInt(_vocabSize));
  final actions = List<int>.generate(_unrollK, (_) => rng.nextInt(_numActions));

  final trueStates = <List<int>>[s0];
  final rewards = <double>[];
  var cur = s0;
  for (final a in actions) {
    rewards.add(_reward(cur, a));
    cur = _step(cur, a);
    trueStates.add(cur);
  }

  // Bootstrap z_K = 0, back-fill.
  final values = List<double>.filled(_unrollK + 1, 0.0);
  for (int k = _unrollK - 1; k >= 0; k--) {
    values[k] = rewards[k] + _gamma * values[k + 1];
  }

  final policies = [for (final s in trueStates) _optimalAction(s)];
  return _Trajectory(s0, actions, rewards, values, policies);
}

Tensor _obsTensor(List<int> s, Device device) => Tensor.fromList(
  [s.length],
  s.map((d) => d.toDouble()).toList(),
  device: device,
);

// -------------------------------------------------------------------
// Eval — argmax(initial-inference policy) matches optimal action, and
// unrolled reward/value predictions on a fresh random rollout.
// -------------------------------------------------------------------
({int policyCorrect, int policyTotal, double rewardMae, double valueMae})
_evalMuZero(MuZeroLM model, Device device, math.Random rng, {int n = 32}) {
  int policyCorrect = 0;
  int policyTotal = 0;
  double rewardMae = 0;
  double valueMae = 0;
  int rewardCount = 0;
  int valueCount = 0;
  for (int i = 0; i < n; i++) {
    final tr = _sampleTraj(rng);
    final u = model.unroll(_obsTensor(tr.obs, device), tr.actions);

    for (int k = 0; k < tr.policies.length; k++) {
      final logits = u.policies[k].toList();
      int best = 0;
      double bestVal = logits[0];
      for (int j = 1; j < logits.length; j++) {
        if (logits[j] > bestVal) {
          bestVal = logits[j];
          best = j;
        }
      }
      if (best == tr.policies[k]) policyCorrect++;
      policyTotal++;
    }
    for (int k = 0; k < tr.rewards.length; k++) {
      final p = u.rewards[k].toList()[0];
      rewardMae += (p - tr.rewards[k]).abs();
      rewardCount++;
    }
    for (int k = 0; k < tr.values.length; k++) {
      final p = u.values[k].toList()[0];
      valueMae += (p - tr.values[k]).abs();
      valueCount++;
    }
  }
  return (
    policyCorrect: policyCorrect,
    policyTotal: policyTotal,
    rewardMae: rewardMae / rewardCount,
    valueMae: valueMae / valueCount,
  );
}

// -------------------------------------------------------------------
// Main
// -------------------------------------------------------------------
void main(List<String> args) {
  final device = args.contains('--gpu') ? Device.GPU : Device.CPU;
  print('=== train_muzero_lm_demo (${device.name}) ===');
  print('arch:      MuZeroLM — language-model representation + latent');
  print('           dynamics + policy/value/reward prediction heads.');
  print(
    'task:      "hot-digit" — 4-digit state (vocab=$_vocabSize), '
    '$_numActions actions,',
  );
  print('           reward +1 on bumping a 2, −1 on wrapping a 3.');
  print('unroll:    K=$_unrollK, γ=$_gamma');
  print(
    'model:     embedDim=$_embedDim  numLayers=$_numLayers  '
    'numHeads=$_numHeads',
  );

  final model = MuZeroLM(
    vocabSize: _vocabSize,
    maxObsLen: _stateLen,
    embedDim: _embedDim,
    numActions: _numActions,
    numLayers: _numLayers,
    numHeads: _numHeads,
    device: device,
    seed: 0,
  );
  final params = model.parameters();
  print('params:    ${paramScalarCount(params)} scalars');

  final opt = Adam(params, lr: _lr);

  // Baseline eval — untrained.
  model.eval();
  final before = _evalMuZero(model, device, math.Random(100));
  print(
    '\nBEFORE  policy-acc = '
    '${(before.policyCorrect / before.policyTotal * 100).toStringAsFixed(1)}% '
    '(${before.policyCorrect}/${before.policyTotal})  '
    'reward-MAE=${before.rewardMae.toStringAsFixed(3)}  '
    'value-MAE=${before.valueMae.toStringAsFixed(3)}',
  );

  // Training.
  model.train();
  print(
    '\ntraining $_steps steps '
    '(lr=$_lr, MuZero unroll K=$_unrollK, MSE reward + MSE value + CE policy)…',
  );
  final rng = math.Random(0);
  final sw = Stopwatch()..start();
  double lossSum = 0;
  for (int step = 1; step <= _steps; step++) {
    opt.zeroGrad();
    // Mini-batch of _accumBatch trajectories: sum their losses before
    // one Adam step. Reduces gradient variance vs. single-trajectory
    // updates and helps the reward/value MSE and policy CE converge.
    Tensor? batchLoss;
    for (int b = 0; b < _accumBatch; b++) {
      final tr = _sampleTraj(rng);
      final u = model.unroll(_obsTensor(tr.obs, device), tr.actions);
      // k = 0 : only value + policy (no reward before step 0).
      var loss = _mseScalar(u.values[0], tr.values[0], device);
      loss = loss + _ceScalar(u.policies[0], tr.policies[0], device);
      // k = 1..K : reward for the k-th transition, plus value/policy
      // at the resulting state.
      for (int k = 0; k < _unrollK; k++) {
        loss = loss + _mseScalar(u.rewards[k], tr.rewards[k], device);
        loss = loss + _mseScalar(u.values[k + 1], tr.values[k + 1], device);
        loss = loss + _ceScalar(u.policies[k + 1], tr.policies[k + 1], device);
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
  }
  sw.stop();

  // Final eval.
  model.eval();
  final after = _evalMuZero(model, device, math.Random(200));
  print(
    '\nAFTER   policy-acc = '
    '${(after.policyCorrect / after.policyTotal * 100).toStringAsFixed(1)}% '
    '(${after.policyCorrect}/${after.policyTotal})  '
    'reward-MAE=${after.rewardMae.toStringAsFixed(3)}  '
    'value-MAE=${after.valueMae.toStringAsFixed(3)}',
  );

  final polDelta =
      (after.policyCorrect - before.policyCorrect) / before.policyTotal * 100;
  print(
    '\npolicy-acc change: ${polDelta >= 0 ? '+' : ''}'
    '${polDelta.toStringAsFixed(1)}%   '
    'reward-MAE ${before.rewardMae.toStringAsFixed(3)} → '
    '${after.rewardMae.toStringAsFixed(3)}   '
    'value-MAE ${before.valueMae.toStringAsFixed(3)} → '
    '${after.valueMae.toStringAsFixed(3)}',
  );
  if (after.policyCorrect >= (0.75 * after.policyTotal).round() &&
      after.rewardMae < 0.25) {
    print('✅ MuZero-LM learned latent dynamics and optimal policy.');
  } else if (after.policyCorrect > before.policyCorrect &&
      after.rewardMae < before.rewardMae) {
    print('⚠️  training improved but did not clear the strong threshold.');
  } else {
    print('❌ training did not improve.');
  }
}
