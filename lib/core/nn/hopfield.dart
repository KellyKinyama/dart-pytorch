/// Binary and continuous Hopfield associative memory (MacKay ch. 42).
///
/// A [HopfieldNetwork] with `I` neurons stores a list of `±1` binary
/// patterns via the Hebb rule (MacKay eq. 42.5):
///
///     w_ij = sum_n x_i^(n) x_j^(n),   w_ii = 0,   W = W^T
///
/// and lets the caller retrieve stored memories from noisy cues by
/// running discrete or continuous dynamics that decrease the Lyapunov
/// energy `E(x) = -1/2 sum_ij w_ij x_i x_j`.
///
/// The network is intentionally tiny (dozens to a few hundred neurons)
/// and lives entirely in `Float32List` / `Int8List` — no autograd tape,
/// no GPU. This mirrors MacKay's pedagogical examples (fig. 42.4).
library;

import 'dart:math' as math;
import 'dart:typed_data';

/// Convergence style for [HopfieldNetwork.recallDiscrete].
enum HopfieldUpdate {
  /// Update every neuron simultaneously using the previous full state
  /// (Little dynamics). Can oscillate between two states.
  synchronous,

  /// Update one neuron at a time (Glauber / Hopfield dynamics). Every
  /// flip is guaranteed to weakly decrease the energy (MacKay eq. 42.14).
  asynchronous,
}

/// Result of a discrete recall run.
class HopfieldRecall {
  HopfieldRecall({
    required this.state,
    required this.sweeps,
    required this.converged,
    required this.finalEnergy,
  });

  /// Final `±1` state, length `I`.
  final Int8List state;

  /// Number of full sweeps performed.
  final int sweeps;

  /// True if the state stopped changing before [maxSweeps].
  final bool converged;

  /// Energy `E(x)` at the final state.
  final double finalEnergy;
}

/// Symmetric zero-diagonal Hopfield associative memory.
class HopfieldNetwork {
  HopfieldNetwork(this.numNeurons)
    : weights = Float32List(numNeurons * numNeurons);

  /// Number of neurons `I`.
  final int numNeurons;

  /// Row-major `[I, I]` weight matrix. Symmetric with zero diagonal
  /// after any of the `store*` methods runs.
  final Float32List weights;

  /// Set all weights to zero.
  void reset() {
    for (int i = 0; i < weights.length; i++) {
      weights[i] = 0.0;
    }
  }

  /// Store [patterns] using the Hebb rule (MacKay eq. 42.5). Each
  /// pattern must be length `I` with entries in `{-1, +1}`. Overwrites
  /// any existing weights.
  void storeHebb(List<Int8List> patterns) {
    reset();
    addHebb(patterns);
  }

  /// Accumulate the Hebbian outer product of each pattern into the
  /// existing weights (does not reset). Useful for incrementally adding
  /// memories to a network.
  void addHebb(List<Int8List> patterns) {
    final I = numNeurons;
    for (final x in patterns) {
      if (x.length != I) {
        throw ArgumentError('pattern length ${x.length} != numNeurons $I');
      }
      for (int i = 0; i < I; i++) {
        final xi = x[i];
        if (xi != 1 && xi != -1) {
          throw ArgumentError('pattern entries must be ±1, got $xi');
        }
        final rowBase = i * I;
        for (int j = 0; j < I; j++) {
          if (i == j) continue;
          weights[rowBase + j] += (xi * x[j]).toDouble();
        }
      }
    }
  }

  /// Activations `a_i = sum_j w_ij x_j` for a bipolar input.
  Float32List activations(Int8List x) {
    final I = numNeurons;
    if (x.length != I) {
      throw ArgumentError('state length ${x.length} != numNeurons $I');
    }
    final a = Float32List(I);
    for (int i = 0; i < I; i++) {
      final rowBase = i * I;
      double s = 0.0;
      for (int j = 0; j < I; j++) {
        s += weights[rowBase + j] * x[j];
      }
      a[i] = s;
    }
    return a;
  }

  /// Energy `E(x) = -1/2 sum_ij w_ij x_i x_j` (MacKay eq. 42.7).
  double energy(Int8List x) {
    final a = activations(x);
    double s = 0.0;
    for (int i = 0; i < numNeurons; i++) {
      s += a[i] * x[i];
    }
    return -0.5 * s;
  }

  /// Continuous energy `E(x) = -1/2 x^T W x` for a real-valued state,
  /// used with the continuous dynamics from §42.6.
  double energyContinuous(Float32List x) {
    final I = numNeurons;
    if (x.length != I) {
      throw ArgumentError('state length ${x.length} != numNeurons $I');
    }
    double s = 0.0;
    for (int i = 0; i < I; i++) {
      final rowBase = i * I;
      double a = 0.0;
      for (int j = 0; j < I; j++) {
        a += weights[rowBase + j] * x[j];
      }
      s += a * x[i];
    }
    return -0.5 * s;
  }

  /// Run discrete recall starting from [initial], returning the final
  /// state and diagnostics.
  ///
  /// Neurons update via `x_i ← sign(a_i)`; ties (`a_i == 0`) keep the
  /// current value, matching MacKay's convention. Asynchronous updates
  /// visit neurons in a random permutation per sweep.
  HopfieldRecall recallDiscrete(
    Int8List initial, {
    int maxSweeps = 32,
    HopfieldUpdate update = HopfieldUpdate.asynchronous,
    math.Random? rng,
  }) {
    final I = numNeurons;
    if (initial.length != I) {
      throw ArgumentError('initial length ${initial.length} != numNeurons $I');
    }
    final x = Int8List.fromList(initial);
    final gen = rng ?? math.Random(0);

    int sweep = 0;
    bool converged = false;
    for (; sweep < maxSweeps; sweep++) {
      bool anyChange = false;
      if (update == HopfieldUpdate.synchronous) {
        final a = activations(x);
        for (int i = 0; i < I; i++) {
          final ai = a[i];
          if (ai > 0 && x[i] != 1) {
            x[i] = 1;
            anyChange = true;
          } else if (ai < 0 && x[i] != -1) {
            x[i] = -1;
            anyChange = true;
          }
        }
      } else {
        final order = List<int>.generate(I, (k) => k);
        for (int k = I - 1; k > 0; k--) {
          final j = gen.nextInt(k + 1);
          final t = order[k];
          order[k] = order[j];
          order[j] = t;
        }
        for (final i in order) {
          final rowBase = i * I;
          double ai = 0.0;
          for (int j = 0; j < I; j++) {
            ai += weights[rowBase + j] * x[j];
          }
          if (ai > 0 && x[i] != 1) {
            x[i] = 1;
            anyChange = true;
          } else if (ai < 0 && x[i] != -1) {
            x[i] = -1;
            anyChange = true;
          }
        }
      }
      if (!anyChange) {
        converged = true;
        sweep++;
        break;
      }
    }
    return HopfieldRecall(
      state: x,
      sweeps: sweep,
      converged: converged,
      finalEnergy: energy(x),
    );
  }

  /// Cross-entropy loss from the most recent [trainPerceptron] call.
  double? _lastTrainLoss;

  /// Improve the weights beyond what the Hebb rule can store, using the
  /// logistic-perceptron rule from MacKay §42.8 / algorithm 42.9.
  ///
  /// For every neuron `i` and pattern `x^(n)`, the rule pushes the
  /// activation `a_i^(n) = sum_j w_ij x_j^(n)` toward the sign of
  /// `x_i^(n)` by descending the cross-entropy
  ///
  ///     G(W) = -sum_{i,n} t_i^(n) log y_i^(n)
  ///                     + (1 - t_i^(n)) log (1 - y_i^(n))
  ///
  /// with `t = (x + 1) / 2` and `y = sigmoid(a)`, symmetrising the
  /// gradient (`w_ij = w_ji`) and applying weight decay `alpha` and
  /// learning rate `eta`. Self-weights are pinned to zero after each
  /// step. Unlike Hebb this can memorise up to ~2 bits/weight on random
  /// patterns and copes with correlated memories that Hebb cannot store
  /// (MacKay fig. 42.5).
  ///
  /// If [initHebb] is true (default), the weights are first replaced by
  /// the Hebb rule outer product; otherwise the existing weights are
  /// used as the starting point.
  ///
  /// Returns the number of patterns that are stable one-step fixed
  /// points (0..N) after the last training step. A return value equal
  /// to `patterns.length` means every memory is at least a synchronous
  /// fixed point.
  int trainPerceptron(
    List<Int8List> patterns, {
    int steps = 500,
    double eta = 0.01,
    double alpha = 0.0,
    bool initHebb = true,
  }) {
    final I = numNeurons;
    final N = patterns.length;
    if (N == 0) return 0;
    for (final p in patterns) {
      if (p.length != I) {
        throw ArgumentError('pattern length ${p.length} != numNeurons $I');
      }
    }

    if (initHebb) {
      storeHebb(patterns);
    }

    final X = Float32List(N * I);
    final T = Float32List(N * I);
    for (int n = 0; n < N; n++) {
      final row = n * I;
      final p = patterns[n];
      for (int i = 0; i < I; i++) {
        final xi = p[i].toDouble();
        X[row + i] = xi;
        T[row + i] = xi > 0 ? 1.0 : 0.0;
      }
    }

    final A = Float32List(N * I);
    final Y = Float32List(N * I);
    final E = Float32List(N * I);
    final G = Float32List(I * I);

    for (int step = 0; step < steps; step++) {
      for (int n = 0; n < N; n++) {
        final rowA = n * I;
        for (int i = 0; i < I; i++) {
          final rowW = i * I;
          double s = 0.0;
          for (int j = 0; j < I; j++) {
            s += weights[rowW + j] * X[rowA + j];
          }
          A[rowA + i] = s;
        }
      }
      for (int k = 0; k < N * I; k++) {
        Y[k] = 1.0 / (1.0 + math.exp(-A[k]));
        E[k] = T[k] - Y[k];
      }
      for (int i = 0; i < I; i++) {
        for (int j = 0; j < I; j++) {
          double s = 0.0;
          for (int n = 0; n < N; n++) {
            s += X[n * I + i] * E[n * I + j];
          }
          G[i * I + j] = s;
        }
      }
      for (int i = 0; i < I; i++) {
        for (int j = i; j < I; j++) {
          final g = G[i * I + j] + G[j * I + i];
          final gij = i == j ? 0.0 : g;
          final w = weights[i * I + j];
          final wNew = w + eta * (gij - alpha * w);
          weights[i * I + j] = wNew;
          weights[j * I + i] = wNew;
        }
      }
      for (int i = 0; i < I; i++) {
        weights[i * I + i] = 0.0;
      }
    }

    double loss = 0.0;
    for (int k = 0; k < N * I; k++) {
      final y = 1.0 / (1.0 + math.exp(-A[k]));
      final t = T[k];
      final yc = y.clamp(1e-9, 1 - 1e-9);
      loss += -(t * math.log(yc) + (1 - t) * math.log(1 - yc));
    }
    _lastTrainLoss = loss;

    return countStableSynchronous(patterns);
  }

  /// Loss from the most recent [trainPerceptron] call, or null if never
  /// trained.
  double? get lastTrainLoss => _lastTrainLoss;

  /// Denoising variant of [trainPerceptron]: minimise the cross-entropy
  /// between `sigmoid(W x_input)` and `target = (x_target + 1) / 2`.
  ///
  /// Unlike [trainPerceptron], input and target may differ, which lets
  /// you train the network to map noisy or partial cues onto clean
  /// memories in a single sigmoid step. Presenting each memory together
  /// with several noise-corrupted copies (all sharing the clean target)
  /// widens the basins of attraction — useful when the raw perceptron
  /// rule makes the memories stable but with only 1–2 bit basins.
  ///
  /// [inputs] and [targets] must have the same length and each pattern
  /// must be length [numNeurons] with entries in `{-1, +1}`.
  void trainDenoise(
    List<Int8List> inputs,
    List<Int8List> targets, {
    int steps = 500,
    double eta = 0.01,
    double alpha = 0.0,
  }) {
    if (inputs.length != targets.length) {
      throw ArgumentError(
        'inputs (${inputs.length}) and targets (${targets.length}) '
        'must have equal length',
      );
    }
    final I = numNeurons;
    final N = inputs.length;
    if (N == 0) return;
    for (int n = 0; n < N; n++) {
      if (inputs[n].length != I || targets[n].length != I) {
        throw ArgumentError('pattern length != numNeurons $I at index $n');
      }
    }

    final X = Float32List(N * I);
    final T = Float32List(N * I);
    for (int n = 0; n < N; n++) {
      final row = n * I;
      final xin = inputs[n];
      final xtg = targets[n];
      for (int i = 0; i < I; i++) {
        X[row + i] = xin[i].toDouble();
        T[row + i] = xtg[i] > 0 ? 1.0 : 0.0;
      }
    }

    final A = Float32List(N * I);
    final Y = Float32List(N * I);
    final E = Float32List(N * I);
    final G = Float32List(I * I);

    for (int step = 0; step < steps; step++) {
      for (int n = 0; n < N; n++) {
        final rowA = n * I;
        for (int i = 0; i < I; i++) {
          final rowW = i * I;
          double s = 0.0;
          for (int j = 0; j < I; j++) {
            s += weights[rowW + j] * X[rowA + j];
          }
          A[rowA + i] = s;
        }
      }
      for (int k = 0; k < N * I; k++) {
        Y[k] = 1.0 / (1.0 + math.exp(-A[k]));
        E[k] = T[k] - Y[k];
      }
      for (int i = 0; i < I; i++) {
        for (int j = 0; j < I; j++) {
          double s = 0.0;
          for (int n = 0; n < N; n++) {
            s += X[n * I + i] * E[n * I + j];
          }
          G[i * I + j] = s;
        }
      }
      for (int i = 0; i < I; i++) {
        for (int j = i; j < I; j++) {
          final g = G[i * I + j] + G[j * I + i];
          final gij = i == j ? 0.0 : g;
          final w = weights[i * I + j];
          final wNew = w + eta * (gij - alpha * w);
          weights[i * I + j] = wNew;
          weights[j * I + i] = wNew;
        }
      }
      for (int i = 0; i < I; i++) {
        weights[i * I + i] = 0.0;
      }
    }

    double loss = 0.0;
    for (int k = 0; k < N * I; k++) {
      final y = 1.0 / (1.0 + math.exp(-A[k]));
      final t = T[k];
      final yc = y.clamp(1e-9, 1 - 1e-9);
      loss += -(t * math.log(yc) + (1 - t) * math.log(1 - yc));
    }
    _lastTrainLoss = loss;
  }

  /// Number of [patterns] that are exact fixed points under a single
  /// synchronous `sign(W x)` update. Useful for scoring capacity.
  int countStableSynchronous(List<Int8List> patterns) {
    int ok = 0;
    for (final p in patterns) {
      final a = activations(p);
      bool stable = true;
      for (int i = 0; i < numNeurons; i++) {
        final want = p[i] > 0;
        final got = a[i] > 0;
        if (want != got) {
          stable = false;
          break;
        }
      }
      if (stable) ok++;
    }
    return ok;
  }

  /// Continuous-time relaxation (MacKay eq. 42.17):
  ///
  ///     tau dx_i/dt = tanh(beta * a_i) - x_i
  ///
  /// Integrated with forward Euler for [steps] steps of size [dt].
  /// Returns the final real-valued state (each component in `(-1, 1)`).
  /// The signed version of the final state approximates the discrete
  /// attractor.
  Float32List relaxContinuous(
    Float32List initial, {
    double beta = 1.0,
    double tau = 1.0,
    double dt = 0.1,
    int steps = 200,
  }) {
    final I = numNeurons;
    if (initial.length != I) {
      throw ArgumentError('initial length ${initial.length} != numNeurons $I');
    }
    final x = Float32List.fromList(initial);
    final a = Float32List(I);
    final coeff = dt / tau;
    for (int t = 0; t < steps; t++) {
      for (int i = 0; i < I; i++) {
        final rowBase = i * I;
        double s = 0.0;
        for (int j = 0; j < I; j++) {
          s += weights[rowBase + j] * x[j];
        }
        a[i] = s;
      }
      for (int i = 0; i < I; i++) {
        final target = _tanh(beta * a[i]);
        x[i] += coeff * (target - x[i]);
      }
    }
    return x;
  }
}

double _tanh(double z) {
  if (z > 20) return 1.0;
  if (z < -20) return -1.0;
  final e2 = math.exp(2 * z);
  return (e2 - 1) / (e2 + 1);
}

/// Utility: convert a `±1` bipolar pattern to `0/1` bits.
Uint8List bipolarToBits(Int8List x) {
  final out = Uint8List(x.length);
  for (int i = 0; i < x.length; i++) {
    out[i] = x[i] > 0 ? 1 : 0;
  }
  return out;
}

/// Utility: convert `0/1` bits to a `±1` bipolar pattern.
Int8List bitsToBipolar(Uint8List bits) {
  final out = Int8List(bits.length);
  for (int i = 0; i < bits.length; i++) {
    out[i] = bits[i] != 0 ? 1 : -1;
  }
  return out;
}

/// Flip [nFlips] uniformly-random distinct bits of [x], returning a new
/// bipolar pattern. Useful for building noisy cues in demos.
Int8List flipRandomBits(Int8List x, int nFlips, {math.Random? rng}) {
  final gen = rng ?? math.Random();
  final n = x.length;
  if (nFlips < 0 || nFlips > n) {
    throw ArgumentError('nFlips $nFlips out of range [0, $n]');
  }
  final indices = List<int>.generate(n, (i) => i);
  for (int k = n - 1; k > 0; k--) {
    final j = gen.nextInt(k + 1);
    final t = indices[k];
    indices[k] = indices[j];
    indices[j] = t;
  }
  final out = Int8List.fromList(x);
  for (int k = 0; k < nFlips; k++) {
    final i = indices[k];
    out[i] = (-out[i]).toInt();
  }
  return out;
}

/// Hamming distance between two bipolar `±1` patterns.
int bipolarHamming(Int8List a, Int8List b) {
  if (a.length != b.length) {
    throw ArgumentError('length mismatch ${a.length} vs ${b.length}');
  }
  int d = 0;
  for (int i = 0; i < a.length; i++) {
    if (a[i] != b[i]) d++;
  }
  return d;
}

/// Overlap `sum_i a_i b_i / I` — 1.0 for identical bipolar patterns,
/// 0.0 for uncorrelated ones (MacKay fig. 42.8).
double bipolarOverlap(Int8List a, Int8List b) {
  if (a.length != b.length) {
    throw ArgumentError('length mismatch ${a.length} vs ${b.length}');
  }
  int s = 0;
  for (int i = 0; i < a.length; i++) {
    s += a[i] * b[i];
  }
  return s / a.length;
}
