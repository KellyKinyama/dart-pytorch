/// F5-TTS DiT (Diffusion Transformer) primitives.
///
/// F5-TTS (Chen et al. 2024) generates mel spectrograms from text via a
/// **Conditional Flow Matching** transformer. This file ships the two
/// pieces that make the F5-TTS DiT tick and aren't part of the plain
/// Transformer stack:
///
///   * [SinusoidalTimestepEmbedding] — maps a scalar flow-matching
///     timestep `t ∈ [0, 1]` into a `[freqDim]` positional vector
///     using the standard "sinusoidal freq schedule + Linear + SiLU +
///     Linear" recipe from DiT (Peebles & Xie 2022). The output is
///     the conditioning vector that drives every block's AdaLN.
///   * [AdaLNZero] — adaptive LayerNorm modulated by the timestep
///     embedding. Produces per-block `(scale_msa, shift_msa,
///     gate_msa, scale_mlp, shift_mlp, gate_mlp)` from a single
///     linear projection whose weights are **initialised to zero**
///     (hence "AdaLN-Zero"). At init the transformer is an identity
///     residual stack; only weight updates give the model expressive
///     power. Matches `facebookresearch/DiT` line-for-line.
///   * [F5DiTBlock] — one Transformer block wired up the DiT way:
///
///         h = x + gate_msa · MHA( AdaLN(x; scale_msa, shift_msa) )
///         y = h + gate_mlp · MLP( AdaLN(h; scale_mlp, shift_mlp) )
///
///     Attention uses [MultiHeadAttention] under the hood; MLP is a
///     two-layer bias-carrying block with GELU (`gelu_new`).
///
/// Not yet ported: the F5-TTS text encoder (a ConvNeXt-V2-lite
/// stack), the full DiT stack that consumes text + audio reference +
/// timestep, and the flow-matching ODE sampler. Those bind together
/// in a follow-up session.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../tensor/tensor.dart';
import 'attention/multi_head_attention.dart';
import 'layer_norm.dart';
import 'linear.dart';
import 'module.dart';

// ---------------------------------------------------------------------------
// SinusoidalTimestepEmbedding
// ---------------------------------------------------------------------------

/// Sinusoidal timestep embedding + a two-layer MLP, standard DiT
/// recipe. Given a scalar `t` returns a `[embedDim]` vector.
///
/// The frequency schedule follows Ho-Jain-Abbeel: for `i ∈ [0,
/// freqDim/2)` a frequency `1 / (10000^(2i/freqDim))` maps to a
/// `(cos, sin)` pair. This gives a `[freqDim]` sinusoidal vector,
/// which is then passed through `Linear(freqDim, embedDim) → SiLU →
/// Linear(embedDim, embedDim)` to produce the final conditioning
/// vector.
class SinusoidalTimestepEmbedding extends Module {
  final int freqDim;
  final int embedDim;
  final Linear proj1;
  final Linear proj2;

  SinusoidalTimestepEmbedding({
    required this.freqDim,
    required this.embedDim,
    Device device = Device.CPU,
    int seed = 0,
  }) : proj1 = Linear(
         freqDim,
         embedDim,
         bias: true,
         device: device,
         seed: seed,
       ),
       proj2 = Linear(
         embedDim,
         embedDim,
         bias: true,
         device: device,
         seed: seed + 1,
       ) {
    if (freqDim.isOdd) {
      throw ArgumentError(
        'SinusoidalTimestepEmbedding: freqDim ($freqDim) must be even',
      );
    }
  }

  /// Forward pass. `t` is a `[1]` scalar tensor holding a timestep.
  /// Returns a `[embedDim]` conditioning vector.
  Tensor call(Tensor t) {
    if (t.shape.length != 1 || t.shape[0] != 1) {
      throw ArgumentError(
        'SinusoidalTimestepEmbedding: expected [1] scalar; got ${t.shape}',
      );
    }
    final tVal = t.toList()[0];
    final half = freqDim ~/ 2;
    final buf = Float32List(freqDim);
    for (int i = 0; i < half; i++) {
      final freq = math.exp(-math.log(10000.0) * i / half);
      final angle = tVal * freq;
      buf[i] = math.cos(angle);
      buf[i + half] = math.sin(angle);
    }
    final sin = Tensor.fromFloat32List([1, freqDim], buf, device: t.device);
    // MLP: Linear → SiLU → Linear.
    final h = proj1(sin);
    final silu = h * h.sigmoid();
    final out = proj2(silu);
    // Return as [embedDim] rank-1.
    final flat = out.toList();
    return Tensor.fromList([embedDim], flat, device: t.device);
  }

  @override
  List<Tensor> parameters() => [...proj1.parameters(), ...proj2.parameters()];

  @override
  List<Module> submodules() => [proj1, proj2];
}

// ---------------------------------------------------------------------------
// AdaLNZero
// ---------------------------------------------------------------------------

/// Adaptive-LN with zero-init modulation. A single Linear
/// `Linear(embedDim, 6 · embedDim)` (weights = 0, bias = 0 at init)
/// produces six per-channel modulation vectors from the timestep
/// conditioning vector: `(scale_msa, shift_msa, gate_msa, scale_mlp,
/// shift_mlp, gate_mlp)`. Because the projection starts at zero the
/// DiT block is an identity residual at init; training grows the
/// modulation from there.
class AdaLNZero extends Module {
  final int embedDim;
  final Linear modulation;

  AdaLNZero({required this.embedDim, Device device = Device.CPU, int seed = 0})
    : modulation = Linear(
        embedDim,
        6 * embedDim,
        bias: true,
        device: device,
        seed: seed,
      ) {
    // Zero-initialise the modulation projection so each DiT block
    // starts as an identity residual (per Peebles & Xie 2022).
    modulation.weight.assign(
      Tensor.fromList(
        modulation.weight.shape,
        List<double>.filled(modulation.weight.length, 0.0),
        device: modulation.weight.device,
      ),
    );
    modulation.bias!.assign(
      Tensor.fromList(
        modulation.bias!.shape,
        List<double>.filled(modulation.bias!.length, 0.0),
        device: modulation.bias!.device,
      ),
    );
  }

  /// Produce the six per-block modulation vectors from the
  /// conditioning vector `c` `[embedDim]`. Returned as
  /// `[6, embedDim]` — indexed as `[i, :]` for i ∈ [0..6).
  Tensor call(Tensor c) {
    if (c.shape.length != 1 || c.shape[0] != embedDim) {
      throw ArgumentError(
        'AdaLNZero: expected [embedDim=$embedDim]; got ${c.shape}',
      );
    }
    // SiLU(c), then Linear to 6·embedDim, reshape to [6, embedDim].
    final cRow = c.reshape([1, embedDim]);
    final silu = cRow * cRow.sigmoid();
    final mod = modulation(silu); // [1, 6·embedDim]
    return mod.reshape([6, embedDim]);
  }

  @override
  List<Tensor> parameters() => modulation.parameters();

  @override
  List<Module> submodules() => [modulation];
}

/// Apply an AdaLN modulation to a LayerNorm output. Given `x ∈ [N,
/// embedDim]`, layer-normed to `hn`, and modulation vectors `scale`,
/// `shift` (each `[embedDim]`), returns `hn * (1 + scale) + shift`
/// with broadcast over the row axis.
Tensor adaLNModulate(Tensor xn, Tensor scale, Tensor shift) {
  if (xn.shape.length != 2 ||
      scale.shape.length != 1 ||
      shift.shape.length != 1 ||
      scale.shape[0] != xn.shape[1] ||
      shift.shape[0] != xn.shape[1]) {
    throw ArgumentError(
      'adaLNModulate: expected xn=[N,D], scale=[D], shift=[D]; got '
      'xn=${xn.shape}, scale=${scale.shape}, shift=${shift.shape}',
    );
  }
  final n = xn.shape[0];
  final d = xn.shape[1];
  final xData = xn.toFloat32List();
  final sData = scale.toFloat32List();
  final shData = shift.toFloat32List();
  final out = Float32List(n * d);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {
      out[i * d + j] = xData[i * d + j] * (1.0 + sData[j]) + shData[j];
    }
  }
  return Tensor.fromFloat32List([n, d], out, device: xn.device);
}

/// Broadcast a `[embedDim]` gate vector against a `[N, embedDim]`
/// residual — element-wise multiply, row-wise broadcast.
Tensor adaLNGate(Tensor h, Tensor gate) {
  if (h.shape.length != 2 ||
      gate.shape.length != 1 ||
      gate.shape[0] != h.shape[1]) {
    throw ArgumentError(
      'adaLNGate: expected h=[N,D], gate=[D]; got h=${h.shape}, '
      'gate=${gate.shape}',
    );
  }
  final n = h.shape[0];
  final d = h.shape[1];
  final hData = h.toFloat32List();
  final gData = gate.toFloat32List();
  final out = Float32List(n * d);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {
      out[i * d + j] = hData[i * d + j] * gData[j];
    }
  }
  return Tensor.fromFloat32List([n, d], out, device: h.device);
}

// ---------------------------------------------------------------------------
// F5DiTBlock
// ---------------------------------------------------------------------------

/// One DiT block, F5-TTS flavour. Pre-LN attention + pre-LN MLP with
/// AdaLN modulation and residual gating from a shared timestep vector.
///
///     mod = adaLN(c)                                              # [6, D]
///     xn  = adaLNModulate( LayerNorm(x), mod[0], mod[1] )
///     h   = x + adaLNGate( MHA(xn), mod[2] )
///     hn  = adaLNModulate( LayerNorm(h), mod[3], mod[4] )
///     y   = h + adaLNGate( MLP(hn), mod[5] )
class F5DiTBlock extends Module {
  final int embedDim;
  final int numHeads;
  final LayerNorm norm1;
  final MultiHeadAttention attn;
  final LayerNorm norm2;
  final Linear fc1;
  final Linear fc2;
  final AdaLNZero adaLn;

  F5DiTBlock({
    required this.embedDim,
    required this.numHeads,
    required int mlpDim,
    Device device = Device.CPU,
    int seed = 0,
  }) : norm1 = LayerNorm(embedDim, eps: 1e-6, device: device),
       attn = MultiHeadAttention(
         embedDim,
         numHeads,
         bias: true,
         device: device,
         seed: seed,
       ),
       norm2 = LayerNorm(embedDim, eps: 1e-6, device: device),
       fc1 = Linear(
         embedDim,
         mlpDim,
         bias: true,
         device: device,
         seed: seed + 10_000,
       ),
       fc2 = Linear(
         mlpDim,
         embedDim,
         bias: true,
         device: device,
         seed: seed + 20_000,
       ),
       adaLn = AdaLNZero(
         embedDim: embedDim,
         device: device,
         seed: seed + 30_000,
       );

  /// Forward pass. `x` is `[N, embedDim]` and `c` is the timestep
  /// conditioning vector `[embedDim]`.
  Tensor call(Tensor x, Tensor c) {
    if (x.shape.length != 2 || x.shape[1] != embedDim) {
      throw ArgumentError(
        'F5DiTBlock: expected x=[N, $embedDim]; got ${x.shape}',
      );
    }
    final mod = adaLn(c); // [6, D]
    final scaleMsa = mod.sliceRows(0, 1).reshape([embedDim]);
    final shiftMsa = mod.sliceRows(1, 2).reshape([embedDim]);
    final gateMsa = mod.sliceRows(2, 3).reshape([embedDim]);
    final scaleMlp = mod.sliceRows(3, 4).reshape([embedDim]);
    final shiftMlp = mod.sliceRows(4, 5).reshape([embedDim]);
    final gateMlp = mod.sliceRows(5, 6).reshape([embedDim]);

    final xn = adaLNModulate(norm1(x), scaleMsa, shiftMsa);
    final h = x + adaLNGate(attn(xn), gateMsa);
    final hn = adaLNModulate(norm2(h), scaleMlp, shiftMlp);
    return h + adaLNGate(fc2(_gelu(fc1(hn))), gateMlp);
  }

  static Tensor _gelu(Tensor x) {
    const invSqrt2 = 0.7071067811865475;
    final data = x.toFloat32List();
    final out = Float32List(data.length);
    for (int i = 0; i < data.length; i++) {
      out[i] = 0.5 * data[i] * (1.0 + _erf(data[i] * invSqrt2));
    }
    return Tensor.fromFloat32List(x.shape, out, device: x.device);
  }

  static double _erf(double x) {
    final sign = x < 0 ? -1.0 : 1.0;
    x = x.abs();
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;
    final t = 1.0 / (1.0 + p * x);
    final y =
        1.0 -
        (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x);
    return sign * y;
  }

  @override
  List<Tensor> parameters() => [
    ...norm1.parameters(),
    ...attn.parameters(),
    ...norm2.parameters(),
    ...fc1.parameters(),
    ...fc2.parameters(),
    ...adaLn.parameters(),
  ];

  @override
  List<Module> submodules() => [norm1, attn, norm2, fc1, fc2, adaLn];
}

// ---------------------------------------------------------------------------
// FlowMatchingSampler — ODE integration from noise to mel.
// ---------------------------------------------------------------------------

/// A velocity-predicting model: given the current sample `x` and
/// timestep `t ∈ [0, 1]`, return `dx/dt = v(x, t)`. Used by
/// [FlowMatchingSampler] to integrate the ODE from `t=0` (noise) to
/// `t=1` (sample). Shape contract: `v(x, t)` must return a tensor
/// with the same shape as `x`.
typedef VelocityField = Tensor Function(Tensor x, Tensor t);

/// Integration scheme used by [FlowMatchingSampler].
///
///   * [FlowSolver.euler] — one velocity evaluation per step.
///     `x_{n+1} = x_n + Δt · v(x_n, t_n)`. Fast, adequate at ≥ 32
///     steps for F5-TTS-style flow-matching mel generation.
///   * [FlowSolver.midpoint] — two velocity evaluations per step.
///     Second-order accurate:
///     `x_{n+1} = x_n + Δt · v(x_n + Δt/2 · v(x_n, t_n),
///                              t_n + Δt/2)`.
///     ~2× the compute of Euler at each step; typically halves
///     the number of steps needed for the same quality.
enum FlowSolver { euler, midpoint }

/// Conditional Flow Matching ODE sampler. Applies to any velocity
/// field — F5-TTS's DiT is the intended target, but the sampler
/// itself is model-agnostic.
///
///     x(0) = z ~ N(0, I)
///     dx/dt = v(x, t)
///     x(1) = generated sample
class FlowMatchingSampler {
  final int numSteps;
  final FlowSolver solver;

  const FlowMatchingSampler({
    required this.numSteps,
    this.solver = FlowSolver.euler,
  });

  /// Sample by integrating the ODE from `t=0` (starting at
  /// [initialNoise]) to `t=1`. [velocityField] is called once per
  /// Euler step, or twice per midpoint step. Returns the final
  /// `x(1)` on the same device as [initialNoise].
  Tensor sample({
    required Tensor initialNoise,
    required VelocityField velocityField,
  }) {
    var x = initialNoise;
    final dt = 1.0 / numSteps;
    for (int step = 0; step < numSteps; step++) {
      final tStart = step * dt;
      final tScalar = Tensor.fromList([1], [tStart], device: x.device);
      switch (solver) {
        case FlowSolver.euler:
          final v = velocityField(x, tScalar);
          _requireMatchingShape(x, v, 'euler');
          x = x + v * dt;
          break;
        case FlowSolver.midpoint:
          final v1 = velocityField(x, tScalar);
          _requireMatchingShape(x, v1, 'midpoint step 1');
          final xMid = x + v1 * (dt / 2);
          final tMid = Tensor.fromList(
            [1],
            [tStart + dt / 2],
            device: x.device,
          );
          final v2 = velocityField(xMid, tMid);
          _requireMatchingShape(x, v2, 'midpoint step 2');
          x = x + v2 * dt;
          break;
      }
    }
    return x;
  }

  static void _requireMatchingShape(Tensor x, Tensor v, String stage) {
    if (v.shape.length != x.shape.length) {
      throw ArgumentError(
        'FlowMatchingSampler ($stage): velocity shape ${v.shape} does '
        'not match x shape ${x.shape}',
      );
    }
    for (int i = 0; i < x.shape.length; i++) {
      if (v.shape[i] != x.shape[i]) {
        throw ArgumentError(
          'FlowMatchingSampler ($stage): velocity shape ${v.shape} does '
          'not match x shape ${x.shape}',
        );
      }
    }
  }
}

/// Draw a Gaussian noise tensor of the given shape (Box-Muller).
/// Convenience helper for the sampler — you can also pass your own
/// noise into [FlowMatchingSampler.sample].
Tensor gaussianNoise(List<int> shape, {int? seed, Device device = Device.CPU}) {
  var n = 1;
  for (final d in shape) {
    n *= d;
  }
  final rng = seed == null ? math.Random() : math.Random(seed);
  final v = Float32List(n);
  int i = 0;
  while (i < n) {
    final u1 = rng.nextDouble().clamp(1e-12, 1.0);
    final u2 = rng.nextDouble();
    final r = math.sqrt(-2.0 * math.log(u1));
    final theta = 2 * math.pi * u2;
    v[i] = r * math.cos(theta);
    if (i + 1 < n) v[i + 1] = r * math.sin(theta);
    i += 2;
  }
  return Tensor.fromFloat32List(shape, v, device: device);
}
