// Smoke tests for the MuZero-style language-model dynamics network.
//
// Covers:
//   * MuZeroRepresentation encodes a [T] token sequence to a
//     [1, embedDim] latent state with finite values.
//   * MuZeroDynamics maps (state, action) to next-state [1, D] and
//     reward [1, 1] with finite values.
//   * MuZeroPrediction produces policy [1, A] and value [1, 1].
//   * MuZeroLM.initialInference / recurrentInference / unroll shapes
//     agree with the paper (K+1 states/policies/values, K rewards).
//   * A few Adam steps on a fabricated (obs, actions, rewards, values,
//     policy) tuple reduce the MuZero K-step loss.

import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

Tensor _tokens(List<int> ids, {required Device device}) => Tensor.fromList(
  [ids.length],
  ids.map((i) => i.toDouble()).toList(),
  device: device,
);

MuZeroLM _build({
  Device device = Device.CPU,
  int vocab = 8,
  int maxLen = 6,
  int embed = 16,
  int actions = 4,
}) => MuZeroLM(
  vocabSize: vocab,
  maxObsLen: maxLen,
  embedDim: embed,
  numActions: actions,
  numLayers: 1,
  numHeads: 4,
  device: device,
  seed: 0,
);

void main() {
  group('MuZeroRepresentation', () {
    test('encodes [T] tokens → [1, embedDim] with finite values', () {
      final rep = MuZeroRepresentation(
        vocabSize: 8,
        maxObsLen: 6,
        embedDim: 16,
        numLayers: 1,
        numHeads: 4,
        seed: 1,
      );
      final obs = _tokens([1, 2, 3, 4], device: Device.CPU);
      final s = rep(obs);
      expect(s.shape, equals([1, 16]));
      for (final v in s.toList()) expect(v.isFinite, isTrue);
    });

    test('rejects obs longer than maxObsLen', () {
      final rep = MuZeroRepresentation(
        vocabSize: 4,
        maxObsLen: 3,
        embedDim: 8,
        numLayers: 1,
        numHeads: 2,
        seed: 2,
      );
      expect(
        () => rep(_tokens([0, 1, 2, 3], device: Device.CPU)),
        throwsArgumentError,
      );
    });
  });

  group('MuZeroDynamics', () {
    test('(state, action) → nextState [1, D] and reward [1, 1]', () {
      final dyn = MuZeroDynamics(embedDim: 12, numActions: 3, seed: 5);
      final s = Tensor.fromList(
        [1, 12],
        List.generate(12, (i) => i * 0.05),
        requiresGrad: true,
      );
      final out = dyn(s, 2);
      expect(out.nextState.shape, equals([1, 12]));
      expect(out.reward.shape, equals([1, 1]));
      for (final v in out.nextState.toList()) expect(v.isFinite, isTrue);
      for (final v in out.reward.toList()) expect(v.isFinite, isTrue);
    });

    test('rejects out-of-range action', () {
      final dyn = MuZeroDynamics(embedDim: 8, numActions: 2, seed: 5);
      final s = Tensor.fromList([1, 8], List<double>.filled(8, 0));
      expect(() => dyn(s, 2), throwsArgumentError);
      expect(() => dyn(s, -1), throwsArgumentError);
    });
  });

  group('MuZeroPrediction', () {
    test('state → policy [1, A] and value [1, 1]', () {
      final pred = MuZeroPrediction(embedDim: 10, numActions: 5, seed: 7);
      final s = Tensor.fromList([
        1,
        10,
      ], List.generate(10, (i) => math.sin(i.toDouble())));
      final out = pred(s);
      expect(out.policy.shape, equals([1, 5]));
      expect(out.value.shape, equals([1, 1]));
    });
  });

  group('MuZeroLM', () {
    test('initialInference and recurrentInference shapes agree', () {
      final m = _build();
      final obs = _tokens([1, 2, 3], device: Device.CPU);
      final init = m.initialInference(obs);
      expect(init.state.shape, equals([1, 16]));
      expect(init.policy.shape, equals([1, 4]));
      expect(init.value.shape, equals([1, 1]));

      final rec = m.recurrentInference(init.state, 1);
      expect(rec.nextState.shape, equals([1, 16]));
      expect(rec.reward.shape, equals([1, 1]));
      expect(rec.policy.shape, equals([1, 4]));
      expect(rec.value.shape, equals([1, 1]));
    });

    test('K-step unroll returns K+1 states / K rewards', () {
      final m = _build();
      final obs = _tokens([1, 2, 3, 0], device: Device.CPU);
      final u = m.unroll(obs, [0, 1, 2]);
      expect(u.states.length, equals(4)); // K+1
      expect(u.policies.length, equals(4));
      expect(u.values.length, equals(4));
      expect(u.rewards.length, equals(3)); // K
      for (final s in u.states) expect(s.shape, equals([1, 16]));
      for (final p in u.policies) expect(p.shape, equals([1, 4]));
      for (final v in u.values) expect(v.shape, equals([1, 1]));
      for (final r in u.rewards) expect(r.shape, equals([1, 1]));
    });

    test('a few Adam steps reduce the K-step MuZero loss', () {
      final m = _build();
      final params = m.parameters();
      final opt = Adam(params, lr: 1e-2);

      // Fabricated fixed target trajectory.
      final obs = _tokens([2, 1, 0, 3], device: Device.CPU);
      final actions = [1, 2, 0];
      final rewardT = [1.0, 0.0, -1.0];
      final valueT = [0.5, 0.2, -0.4, 0.0];
      final policyT = [2, 0, 1, 3];

      Tensor mseS(Tensor pred, double t) {
        final tgt = Tensor.fromList([1, 1], [t]);
        final d = pred - tgt;
        return d * d;
      }

      Tensor ceS(Tensor logits, int label) {
        final tgt = Tensor.fromList([1], [label.toDouble()]);
        return logits.crossEntropy(tgt);
      }

      Tensor totalLoss() {
        final u = m.unroll(obs, actions);
        var l = mseS(u.values[0], valueT[0]) + ceS(u.policies[0], policyT[0]);
        for (int k = 0; k < actions.length; k++) {
          l = l + mseS(u.rewards[k], rewardT[k]);
          l = l + mseS(u.values[k + 1], valueT[k + 1]);
          l = l + ceS(u.policies[k + 1], policyT[k + 1]);
        }
        return l.mean();
      }

      final before = Tensor.noGrad(() => totalLoss().toList()[0]);
      for (int i = 0; i < 20; i++) {
        opt.zeroGrad();
        final l = totalLoss();
        l.backward();
        opt.step();
      }
      final after = Tensor.noGrad(() => totalLoss().toList()[0]);
      expect(
        after < before,
        isTrue,
        reason: 'expected loss to decrease; before=$before after=$after',
      );
    });

    test('parameters include representation + dynamics + prediction', () {
      final m = _build();
      final total = m.parameters().length;
      final r = m.representation.parameters().length;
      final d = m.dynamics.parameters().length;
      final p = m.prediction.parameters().length;
      expect(total, equals(r + d + p));
      expect(total, greaterThan(0));
    });
  });
}
