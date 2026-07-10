/// MuZero-style latent-dynamics network with a *language-model
/// representation function*.
///
/// Follows the three-network decomposition of Schrittwieser et al.
/// (2020), "Mastering Atari, Go, Chess and Shogi by Planning with a
/// Learned Model" (a.k.a. the MuZero paper):
///
///   1. Representation `h_θ : obs → s⁰`
///        Here [MuZeroRepresentation] treats the observation as a
///        sequence of discrete tokens (a "language description of the
///        state") and encodes it with an [Embedding] + positional
///        embedding + [TransformerEncoder] + mean-pool to a single
///        latent vector `[1, embedDim]`.
///
///   2. Dynamics `g_θ : (sᵏ, aᵏ⁺¹) → (sᵏ⁺¹, rᵏ⁺¹)`
///        [MuZeroDynamics] embeds the discrete action, concatenates
///        with the current state, runs a small residual MLP, and
///        splits into a next-state head and a scalar reward head.
///
///   3. Prediction `f_θ : sᵏ → (pᵏ, vᵏ)`
///        [MuZeroPrediction] projects the latent state to policy
///        logits over the action set and a scalar value.
///
/// [MuZeroLM] bundles the three networks and provides:
///
///   * [MuZeroLM.initialInference]   — obs tokens → (s⁰, p⁰, v⁰)
///   * [MuZeroLM.recurrentInference] — (sᵏ, aᵏ⁺¹) → (sᵏ⁺¹, rᵏ⁺¹, pᵏ⁺¹, vᵏ⁺¹)
///   * [MuZeroLM.unroll]             — K-step planning unroll used
///                                     during training.
///
/// The K-step supervised training loss from the paper
///
///   L = Σₖ  l_r(uᵏ⁺¹, rᵏ⁺¹) + l_v(zᵏ, vᵏ) + l_p(πᵏ, pᵏ)
///
/// is expressed in Dart / this API by e.g.
///
/// ```dart
/// final u = model.unroll(obsTokens, actions);
/// var loss = _mse(u.values[0], zTargets[0]) + _crossEntropy(u.policies[0], piTargets[0]);
/// for (var k = 0; k < actions.length; k++) {
///   loss += _mse(u.rewards[k],  uTargets[k]);
///   loss += _mse(u.values[k+1], zTargets[k+1]);
///   loss += _crossEntropy(u.policies[k+1], piTargets[k+1]);
/// }
/// loss.backward();
/// ```
///
/// The paper's core insight — that `g_θ` need not reconstruct the
/// observation, only predict rewards, values and policies useful for
/// planning — is preserved: the dynamics module has no
/// observation-decoder head, and the representation module is only
/// ever invoked on the very first observation.
library;

import '../tensor/tensor.dart';
import 'embedding.dart';
import 'linear.dart';
import 'modalities/audio_transformer.dart' show meanRows;
import 'module.dart';
import 'positional.dart';
import 'transformer_encoder.dart';

// -------------------------------------------------------------------
// Representation:  obs tokens  →  latent state  [1, embedDim]
// -------------------------------------------------------------------

/// Language-model style representation function `h_θ`.
///
/// Encodes a sequence of `[T]` observation tokens into a single
/// `[1, embedDim]` latent state via token embedding, learned
/// positional embedding, transformer encoder, and mean-pool.
class MuZeroRepresentation extends Module {
  final int vocabSize;
  final int maxObsLen;
  final int embedDim;
  final int numLayers;
  final int numHeads;

  final Embedding tokenEmbed;
  final LearnedPositionalEmbedding posEmbed;
  final TransformerEncoder encoder;

  MuZeroRepresentation({
    required this.vocabSize,
    required this.maxObsLen,
    required this.embedDim,
    this.numLayers = 2,
    this.numHeads = 4,
    double dropoutP = 0.0,
    Device device = Device.CPU,
    int seed = 0,
  }) : tokenEmbed = Embedding(vocabSize, embedDim, device: device, seed: seed),
       posEmbed = LearnedPositionalEmbedding(
         maxObsLen,
         embedDim,
         device: device,
         seed: seed + 1,
       ),
       encoder = TransformerEncoder(
         numLayers,
         embedDim,
         numHeads,
         dropoutP: dropoutP,
         device: device,
         seed: seed + 100,
       );

  /// `obsTokens`: `[T]` int-in-`[0, vocabSize)` observation ids
  /// (encoded as float32). Returns latent state `[1, embedDim]`.
  Tensor call(Tensor obsTokens) {
    if (obsTokens.shape.length != 1) {
      throw ArgumentError(
        'MuZeroRepresentation: expected [T] token ids; got '
        '${obsTokens.shape}',
      );
    }
    if (obsTokens.shape[0] > maxObsLen) {
      throw ArgumentError(
        'MuZeroRepresentation: obs length ${obsTokens.shape[0]} '
        'exceeds maxObsLen $maxObsLen',
      );
    }
    final e = tokenEmbed(obsTokens); // [T, D]
    final ep = posEmbed(e); // [T, D]
    final h = encoder(ep); // [T, D]
    return meanRows(h); // [1, D]
  }

  @override
  List<Tensor> parameters() => [
    ...tokenEmbed.parameters(),
    ...posEmbed.parameters(),
    ...encoder.parameters(),
  ];

  @override
  List<Module> submodules() => [tokenEmbed, posEmbed, encoder];
}

// -------------------------------------------------------------------
// Dynamics:  (state, action) → (next state, reward)
// -------------------------------------------------------------------

/// Dynamics function `g_θ`.
///
/// Given a latent state `[1, embedDim]` and a discrete action id,
/// predicts the next latent state and the *scalar* reward received on
/// that transition. Uses a small residual MLP:
///
/// ```
///   z    = concat(state, actionEmbed(action))       # [1, 2D]
///   h    = ReLU(W1 z)                                # [1, hidden]
///   next = state + W_next h                          # residual state
///   r    = W_reward h                                # scalar in [1, 1]
/// ```
///
/// The residual connection on the state is not required by the paper
/// but stabilises the K-step unroll considerably: gradients flow back
/// through many recurrent steps without vanishing.
class MuZeroDynamics extends Module {
  final int embedDim;
  final int numActions;
  final int hiddenDim;

  final Embedding actionEmbed;
  final Linear fuse;
  final Linear stateHead;
  final Linear rewardHead;

  MuZeroDynamics({
    required this.embedDim,
    required this.numActions,
    int? hiddenDim,
    Device device = Device.CPU,
    int seed = 0,
  }) : hiddenDim = hiddenDim ?? (embedDim * 2),
       actionEmbed = Embedding(
         numActions,
         embedDim,
         device: device,
         seed: seed,
       ),
       fuse = Linear(
         embedDim * 2,
         hiddenDim ?? (embedDim * 2),
         bias: true,
         device: device,
         seed: seed + 1,
       ),
       stateHead = Linear(
         hiddenDim ?? (embedDim * 2),
         embedDim,
         bias: true,
         device: device,
         seed: seed + 2,
       ),
       rewardHead = Linear(
         hiddenDim ?? (embedDim * 2),
         1,
         bias: true,
         device: device,
         seed: seed + 3,
       );

  /// `state`: `[1, embedDim]`. `actionId`: int in `[0, numActions)`.
  /// Returns `nextState [1, embedDim]` and `reward [1, 1]`.
  ({Tensor nextState, Tensor reward}) call(Tensor state, int actionId) {
    if (state.shape.length != 2 ||
        state.shape[0] != 1 ||
        state.shape[1] != embedDim) {
      throw ArgumentError(
        'MuZeroDynamics: expected state [1, $embedDim]; got ${state.shape}',
      );
    }
    if (actionId < 0 || actionId >= numActions) {
      throw ArgumentError(
        'MuZeroDynamics: action $actionId out of [0, $numActions)',
      );
    }
    final aIdx = Tensor.fromList(
      [1],
      [actionId.toDouble()],
      device: state.device,
    );
    final aEmb = actionEmbed(aIdx); // [1, D]
    final joint = TensorConcat.concat([state, aEmb], axis: 1); // [1, 2D]
    final h = fuse(joint).relu(); // [1, hidden]
    final delta = stateHead(h); // [1, D]
    final next = state + delta; // residual  [1, D]
    final reward = rewardHead(h); // [1, 1]
    return (nextState: next, reward: reward);
  }

  @override
  List<Tensor> parameters() => [
    ...actionEmbed.parameters(),
    ...fuse.parameters(),
    ...stateHead.parameters(),
    ...rewardHead.parameters(),
  ];

  @override
  List<Module> submodules() => [actionEmbed, fuse, stateHead, rewardHead];
}

// -------------------------------------------------------------------
// Prediction:  state → (policy logits, value)
// -------------------------------------------------------------------

/// Prediction function `f_θ`.
///
/// Two independent linear heads read the latent state: one produces
/// policy logits over the action set, the other a scalar value.
class MuZeroPrediction extends Module {
  final int embedDim;
  final int numActions;

  final Linear policyHead;
  final Linear valueHead;

  MuZeroPrediction({
    required this.embedDim,
    required this.numActions,
    Device device = Device.CPU,
    int seed = 0,
  }) : policyHead = Linear(
         embedDim,
         numActions,
         bias: true,
         device: device,
         seed: seed,
       ),
       valueHead = Linear(
         embedDim,
         1,
         bias: true,
         device: device,
         seed: seed + 1,
       );

  /// `state`: `[1, embedDim]`. Returns `policy [1, numActions]` and
  /// `value [1, 1]`.
  ({Tensor policy, Tensor value}) call(Tensor state) {
    if (state.shape.length != 2 ||
        state.shape[0] != 1 ||
        state.shape[1] != embedDim) {
      throw ArgumentError(
        'MuZeroPrediction: expected state [1, $embedDim]; got ${state.shape}',
      );
    }
    return (policy: policyHead(state), value: valueHead(state));
  }

  @override
  List<Tensor> parameters() => [
    ...policyHead.parameters(),
    ...valueHead.parameters(),
  ];

  @override
  List<Module> submodules() => [policyHead, valueHead];
}

// -------------------------------------------------------------------
// The full MuZero-LM
// -------------------------------------------------------------------

/// K-step unroll of the model, used as the training signal:
///
///   * `states[0]  = h(obs)`,  `states[k+1] = g_state(states[k], actions[k])`
///   * `policies[k], values[k]` = `f(states[k])`
///   * `rewards[k]` = `g_reward(states[k], actions[k])`, for
///     `k ∈ [0, actions.length)`.
class MuZeroUnroll {
  final List<Tensor> states;
  final List<Tensor> policies;
  final List<Tensor> values;
  final List<Tensor> rewards;

  MuZeroUnroll({
    required this.states,
    required this.policies,
    required this.values,
    required this.rewards,
  });
}

/// Full MuZero network with a language-model representation function.
///
/// Bundles [MuZeroRepresentation], [MuZeroDynamics], [MuZeroPrediction]
/// and offers the standard [initialInference], [recurrentInference],
/// and [unroll] entry points from the paper's Algorithm 1.
class MuZeroLM extends Module {
  final int vocabSize;
  final int maxObsLen;
  final int embedDim;
  final int numActions;

  final MuZeroRepresentation representation;
  final MuZeroDynamics dynamics;
  final MuZeroPrediction prediction;

  MuZeroLM({
    required this.vocabSize,
    required this.maxObsLen,
    required this.embedDim,
    required this.numActions,
    int numLayers = 2,
    int numHeads = 4,
    int? dynamicsHiddenDim,
    double dropoutP = 0.0,
    Device device = Device.CPU,
    int seed = 0,
  }) : representation = MuZeroRepresentation(
         vocabSize: vocabSize,
         maxObsLen: maxObsLen,
         embedDim: embedDim,
         numLayers: numLayers,
         numHeads: numHeads,
         dropoutP: dropoutP,
         device: device,
         seed: seed + 1000,
       ),
       dynamics = MuZeroDynamics(
         embedDim: embedDim,
         numActions: numActions,
         hiddenDim: dynamicsHiddenDim,
         device: device,
         seed: seed + 2000,
       ),
       prediction = MuZeroPrediction(
         embedDim: embedDim,
         numActions: numActions,
         device: device,
         seed: seed + 3000,
       );

  /// Encode a fresh observation → `(s⁰, p⁰, v⁰)`.
  ({Tensor state, Tensor policy, Tensor value}) initialInference(
    Tensor obsTokens,
  ) {
    final state = representation(obsTokens);
    final pv = prediction(state);
    return (state: state, policy: pv.policy, value: pv.value);
  }

  /// Apply the learned dynamics one step and re-read policy/value:
  /// `(sᵏ, aᵏ⁺¹) → (sᵏ⁺¹, rᵏ⁺¹, pᵏ⁺¹, vᵏ⁺¹)`.
  ({Tensor nextState, Tensor reward, Tensor policy, Tensor value})
  recurrentInference(Tensor state, int action) {
    final d = dynamics(state, action);
    final pv = prediction(d.nextState);
    return (
      nextState: d.nextState,
      reward: d.reward,
      policy: pv.policy,
      value: pv.value,
    );
  }

  /// Full K-step unroll: encode `obsTokens` to `s⁰`, then apply the
  /// learned dynamics for each id in `actions`, running the
  /// prediction head at every state (including `s⁰`).
  ///
  /// Returned lists have the following lengths (K = `actions.length`):
  ///
  /// | field    | length | index range |
  /// |----------|:------:|:-----------:|
  /// | states   | K+1    | 0..K        |
  /// | policies | K+1    | 0..K        |
  /// | values   | K+1    | 0..K        |
  /// | rewards  | K      | 0..K-1      |
  MuZeroUnroll unroll(Tensor obsTokens, List<int> actions) {
    final init = initialInference(obsTokens);
    final states = <Tensor>[init.state];
    final policies = <Tensor>[init.policy];
    final values = <Tensor>[init.value];
    final rewards = <Tensor>[];
    for (final a in actions) {
      final r = recurrentInference(states.last, a);
      states.add(r.nextState);
      policies.add(r.policy);
      values.add(r.value);
      rewards.add(r.reward);
    }
    return MuZeroUnroll(
      states: states,
      policies: policies,
      values: values,
      rewards: rewards,
    );
  }

  @override
  List<Tensor> parameters() => [
    ...representation.parameters(),
    ...dynamics.parameters(),
    ...prediction.parameters(),
  ];

  @override
  List<Module> submodules() => [representation, dynamics, prediction];
}
