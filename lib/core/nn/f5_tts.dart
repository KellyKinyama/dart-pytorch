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
import 'embedding.dart';
import 'layer_norm.dart';
import 'linear.dart';
import 'module.dart';
import 'safetensors.dart';

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

// ---------------------------------------------------------------------------
// DepthwiseConv1d — one filter per channel, applied along the time axis.
// ---------------------------------------------------------------------------

/// 1-D depthwise convolution (`groups == inChannels`). Input
/// `[N, C, T]` → output `[N, C, T]` (same-length when
/// `padding == (kernelSize - 1) / 2`). Each channel has its own
/// `[kernelSize]` kernel. Ports PyTorch's
/// `nn.Conv1d(dim, dim, kernel_size, groups=dim)`.
///
/// Host-side compute — trades correctness for simplicity. For F5-TTS
/// text encoder scale (dim ≤ 512, T ≤ 512) this is fine.
class DepthwiseConv1d extends Module {
  final int channels;
  final int kernelSize;
  final int padding;
  final Tensor weight; // [channels, kernelSize]
  final Tensor? bias; // [channels]

  DepthwiseConv1d({
    required this.channels,
    required this.kernelSize,
    this.padding = 0,
    bool bias = true,
    Device device = Device.CPU,
    int seed = 0,
  }) : weight = _initWeight(channels, kernelSize, seed, device),
       bias = bias
           ? Tensor.fill([channels], 0.0, requiresGrad: true, device: device)
           : null;

  static Tensor _initWeight(int c, int k, int seed, Device device) {
    final rng = math.Random(seed);
    final bound = 1.0 / math.sqrt(k);
    final vals = List<double>.generate(
      c * k,
      (_) => (rng.nextDouble() * 2 - 1) * bound,
    );
    return Tensor.fromList([c, k], vals, requiresGrad: true, device: device);
  }

  Tensor call(Tensor x) {
    if (x.shape.length != 3 || x.shape[1] != channels) {
      throw ArgumentError(
        'DepthwiseConv1d: expected [N, $channels, T]; got ${x.shape}',
      );
    }
    final n = x.shape[0];
    final t = x.shape[2];
    final tOut = t + 2 * padding - kernelSize + 1;
    if (tOut <= 0) {
      throw ArgumentError(
        'DepthwiseConv1d: padded T=${t + 2 * padding} < kernel=$kernelSize',
      );
    }
    final data = x.toFloat32List();
    final wData = weight.toFloat32List();
    final bData = bias?.toList();
    final out = Float32List(n * channels * tOut);
    for (int ni = 0; ni < n; ni++) {
      for (int c = 0; c < channels; c++) {
        final srcBase = (ni * channels + c) * t;
        final dstBase = (ni * channels + c) * tOut;
        final b = bData?[c] ?? 0.0;
        for (int oi = 0; oi < tOut; oi++) {
          double acc = b;
          final start = oi - padding;
          for (int k = 0; k < kernelSize; k++) {
            final idx = start + k;
            if (idx >= 0 && idx < t) {
              acc += wData[c * kernelSize + k] * data[srcBase + idx];
            }
          }
          out[dstBase + oi] = acc;
        }
      }
    }
    return Tensor.fromFloat32List([n, channels, tOut], out, device: x.device);
  }

  @override
  List<Tensor> parameters() => [weight, if (bias != null) bias!];
}

// ---------------------------------------------------------------------------
// GlobalResponseNormalization — ConvNeXt V2's GRN.
// ---------------------------------------------------------------------------

/// Global Response Normalization (Woo et al. 2023). Given
/// `[N, T, C]` input:
///
///     Gx  = ||x||_2 along the T axis                    # [N, 1, C]
///     Nx  = Gx / mean(Gx, dim=-1, keepdim=True)         # [N, 1, C]
///     out = γ · (x · Nx) + β + x                        # residual add
///
/// Both `γ` and `β` are learned `[C]` vectors initialised to zero,
/// so at init GRN is an identity residual.
class GlobalResponseNormalization extends Module {
  final int channels;
  final Tensor gamma;
  final Tensor beta;
  final double eps;

  GlobalResponseNormalization({
    required this.channels,
    this.eps = 1e-6,
    Device device = Device.CPU,
  }) : gamma = Tensor.fill([channels], 0.0, requiresGrad: true, device: device),
       beta = Tensor.fill([channels], 0.0, requiresGrad: true, device: device);

  Tensor call(Tensor x) {
    if (x.shape.length != 3 || x.shape[2] != channels) {
      throw ArgumentError(
        'GlobalResponseNormalization: expected [N, T, $channels]; '
        'got ${x.shape}',
      );
    }
    final n = x.shape[0];
    final t = x.shape[1];
    final data = x.toFloat32List();
    final gData = gamma.toList();
    final bData = beta.toList();

    // Gx[n, c] = ||x[n, :, c]||_2
    final gx = List<double>.filled(n * channels, 0);
    for (int ni = 0; ni < n; ni++) {
      for (int c = 0; c < channels; c++) {
        double sq = 0;
        for (int ti = 0; ti < t; ti++) {
          final v = data[(ni * t + ti) * channels + c];
          sq += v * v;
        }
        gx[ni * channels + c] = math.sqrt(sq);
      }
    }
    // Nx[n, c] = Gx[n, c] / mean(Gx[n, :])
    final nx = List<double>.filled(n * channels, 0);
    for (int ni = 0; ni < n; ni++) {
      double sum = 0;
      for (int c = 0; c < channels; c++) {
        sum += gx[ni * channels + c];
      }
      final m = sum / channels + eps;
      for (int c = 0; c < channels; c++) {
        nx[ni * channels + c] = gx[ni * channels + c] / m;
      }
    }
    final out = Float32List(n * t * channels);
    for (int ni = 0; ni < n; ni++) {
      for (int ti = 0; ti < t; ti++) {
        for (int c = 0; c < channels; c++) {
          final i = (ni * t + ti) * channels + c;
          out[i] =
              gData[c] * (data[i] * nx[ni * channels + c]) + bData[c] + data[i];
        }
      }
    }
    return Tensor.fromFloat32List([n, t, channels], out, device: x.device);
  }

  @override
  List<Tensor> parameters() => [gamma, beta];
}

// ---------------------------------------------------------------------------
// ConvNeXtV2Block — one block of the F5-TTS character text encoder.
// ---------------------------------------------------------------------------

/// ConvNeXt V2 block used inside F5-TTS's character-level text encoder.
/// Input/output `[N, T, dim]`:
///
///     y = dwconv([N, dim, T])  (kernel 7, padding 3) → [N, dim, T]
///     y = LayerNorm(y[N, T, dim])
///     y = pwconv1(y)                # dim → intermediateDim
///     y = GELU(y)
///     y = GRN(y)
///     y = pwconv2(y)                # intermediateDim → dim
///     return x + y
class ConvNeXtV2Block extends Module {
  final int dim;
  final int intermediateDim;
  final DepthwiseConv1d dwconv;
  final LayerNorm norm;
  final Linear pwconv1;
  final GlobalResponseNormalization grn;
  final Linear pwconv2;

  ConvNeXtV2Block({
    required this.dim,
    required this.intermediateDim,
    int kernelSize = 7,
    Device device = Device.CPU,
    int seed = 0,
  }) : dwconv = DepthwiseConv1d(
         channels: dim,
         kernelSize: kernelSize,
         padding: (kernelSize - 1) ~/ 2,
         bias: true,
         device: device,
         seed: seed,
       ),
       norm = LayerNorm(dim, eps: 1e-6, device: device),
       pwconv1 = Linear(
         dim,
         intermediateDim,
         bias: true,
         device: device,
         seed: seed + 1_000,
       ),
       grn = GlobalResponseNormalization(
         channels: intermediateDim,
         device: device,
       ),
       pwconv2 = Linear(
         intermediateDim,
         dim,
         bias: true,
         device: device,
         seed: seed + 2_000,
       );

  /// `x: [N, T, dim]`.
  Tensor call(Tensor x) {
    if (x.shape.length != 3 || x.shape[2] != dim) {
      throw ArgumentError(
        'ConvNeXtV2Block: expected [N, T, $dim]; got ${x.shape}',
      );
    }
    final n = x.shape[0];
    final t = x.shape[1];
    // Transpose [N, T, dim] -> [N, dim, T] on host.
    final xData = x.toFloat32List();
    final ntBuf = Float32List(n * dim * t);
    for (int ni = 0; ni < n; ni++) {
      for (int ti = 0; ti < t; ti++) {
        for (int c = 0; c < dim; c++) {
          ntBuf[(ni * dim + c) * t + ti] = xData[(ni * t + ti) * dim + c];
        }
      }
    }
    final ntc = Tensor.fromFloat32List([n, dim, t], ntBuf, device: x.device);
    var y = dwconv(ntc); // [N, dim, T]
    // Transpose back to [N, T, dim].
    final yData = y.toFloat32List();
    final ttcBuf = Float32List(n * t * dim);
    for (int ni = 0; ni < n; ni++) {
      for (int ti = 0; ti < t; ti++) {
        for (int c = 0; c < dim; c++) {
          ttcBuf[(ni * t + ti) * dim + c] = yData[(ni * dim + c) * t + ti];
        }
      }
    }
    y = Tensor.fromFloat32List([n, t, dim], ttcBuf, device: x.device);
    // Standard LayerNorm operates on 2-D input, so squash the leading
    // (N, T) into a single (N*T) row axis.
    final yFlat = y.reshape([n * t, dim]);
    var yn = norm(yFlat);
    yn = pwconv1(yn); // [N*T, intermediateDim]
    yn = _gelu(yn);
    // GRN wants [N, T, C].
    yn = yn.reshape([n, t, intermediateDim]);
    yn = grn(yn);
    yn = pwconv2(yn.reshape([n * t, intermediateDim]));
    yn = yn.reshape([n, t, dim]);
    return x + yn;
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
    ...dwconv.parameters(),
    ...norm.parameters(),
    ...pwconv1.parameters(),
    ...grn.parameters(),
    ...pwconv2.parameters(),
  ];

  @override
  List<Module> submodules() => [dwconv, norm, pwconv1, grn, pwconv2];
}

// ---------------------------------------------------------------------------
// F5TextEncoder — character-level text embedding + N × ConvNeXtV2Block.
// ---------------------------------------------------------------------------

/// F5-TTS character-level text encoder. Takes `[T]` character token
/// ids (as float32), maps them through an embedding table, and runs
/// [numLayers] × [ConvNeXtV2Block] to produce `[T, dim]` text features
/// that condition the DiT's mel-velocity prediction.
class F5TextEncoder extends Module {
  final int vocabSize;
  final int dim;
  final int intermediateDim;
  final int numLayers;
  final Embedding tokenEmbedding;
  final List<ConvNeXtV2Block> blocks;
  final LayerNorm finalNorm;

  F5TextEncoder({
    required this.vocabSize,
    this.dim = 512,
    this.intermediateDim = 2048,
    this.numLayers = 4,
    int convKernelSize = 7,
    Device device = Device.CPU,
    int seed = 0,
  }) : tokenEmbedding = Embedding(vocabSize, dim, device: device, seed: seed),
       blocks = <ConvNeXtV2Block>[],
       finalNorm = LayerNorm(dim, eps: 1e-6, device: device) {
    for (int i = 0; i < numLayers; i++) {
      blocks.add(
        ConvNeXtV2Block(
          dim: dim,
          intermediateDim: intermediateDim,
          kernelSize: convKernelSize,
          device: device,
          seed: seed + 10_000 * (i + 1),
        ),
      );
    }
  }

  /// `tokens: [T]` returns `[T, dim]`.
  Tensor call(Tensor tokens) {
    if (tokens.shape.length != 1) {
      throw ArgumentError(
        'F5TextEncoder: expected 1D [T]; got ${tokens.shape}',
      );
    }
    // Embedding is a lookup; result shape [T, dim].
    var h = tokenEmbedding(tokens);
    // ConvNeXtV2Block expects [N, T, dim]; add N=1.
    final t = h.shape[0];
    h = h.reshape([1, t, dim]);
    for (final b in blocks) {
      h = b(h);
    }
    // Final LN over the last dim.
    final flat = h.reshape([t, dim]);
    return finalNorm(flat);
  }

  @override
  List<Tensor> parameters() => [
    ...tokenEmbedding.parameters(),
    for (final b in blocks) ...b.parameters(),
    ...finalNorm.parameters(),
  ];

  @override
  List<Module> submodules() => [tokenEmbedding, ...blocks, finalNorm];
}

// ---------------------------------------------------------------------------
// F5DiT — full DiT stack: mel input + text + timestep → mel velocity.
// ---------------------------------------------------------------------------

/// F5-TTS DiT stack. Takes the current noisy mel `[T, melDim]`, the
/// text conditioning `[T, textDim]` (from [F5TextEncoder]), and a
/// scalar timestep, and returns the predicted mel-velocity
/// `[T, melDim]` for the flow-matching sampler.
///
/// Architecture:
///
///     x = concat([mel, text_broadcast], last-axis) → project to embedDim
///     c = SinusoidalTimestepEmbedding(t)
///     for block in blocks: x = block(x, c)      # [F5DiTBlock]
///     x = LayerNorm(x)
///     v = Linear(embedDim, melDim)(x)
///     return v
///
/// Text is aligned with mel via a nearest-neighbour "duration" broadcast:
/// each mel frame reads the text embedding at the same frame index
/// (SAM's F5-TTS conditioning is a bit more involved with duration
/// modelling; this simplification lets us wire the pieces together and
/// verify shape invariants — the alignment implementation is a
/// follow-up).
class F5DiT extends Module {
  final int melDim;
  final int textDim;
  final int embedDim;
  final int numLayers;
  final int numHeads;
  final int mlpDim;
  final int freqDim;

  final Linear inputProj;
  final SinusoidalTimestepEmbedding timeEmbed;
  final List<F5DiTBlock> blocks;
  final LayerNorm finalNorm;
  final Linear outputProj;

  F5DiT({
    required this.melDim,
    required this.textDim,
    this.embedDim = 1024,
    this.numLayers = 22,
    this.numHeads = 16,
    this.mlpDim = 2048,
    this.freqDim = 256,
    Device device = Device.CPU,
    int seed = 0,
  }) : inputProj = Linear(
         melDim + textDim,
         embedDim,
         bias: true,
         device: device,
         seed: seed,
       ),
       timeEmbed = SinusoidalTimestepEmbedding(
         freqDim: freqDim,
         embedDim: embedDim,
         device: device,
         seed: seed + 1000,
       ),
       blocks = <F5DiTBlock>[],
       finalNorm = LayerNorm(embedDim, eps: 1e-6, device: device),
       outputProj = Linear(
         embedDim,
         melDim,
         bias: true,
         device: device,
         seed: seed + 2000,
       ) {
    for (int i = 0; i < numLayers; i++) {
      blocks.add(
        F5DiTBlock(
          embedDim: embedDim,
          numHeads: numHeads,
          mlpDim: mlpDim,
          device: device,
          seed: seed + 100_000 * (i + 1),
        ),
      );
    }
  }

  /// Forward pass.
  ///
  /// `mel: [T, melDim]` — current noisy mel spectrogram.
  /// `text: [T, textDim]` — text conditioning aligned frame-by-frame.
  /// `t: [1]` — flow-matching timestep in `[0, 1]`.
  ///
  /// Returns `[T, melDim]` mel velocity.
  Tensor call(Tensor mel, Tensor text, Tensor t) {
    if (mel.shape.length != 2 || mel.shape[1] != melDim) {
      throw ArgumentError('F5DiT: expected mel=[T, $melDim]; got ${mel.shape}');
    }
    if (text.shape.length != 2 ||
        text.shape[0] != mel.shape[0] ||
        text.shape[1] != textDim) {
      throw ArgumentError(
        'F5DiT: expected text=[${mel.shape[0]}, $textDim]; '
        'got ${text.shape}',
      );
    }
    // Concatenate mel and text on last axis: [T, melDim + textDim].
    final combined = TensorConcat.concat([mel, text], axis: 1);
    var x = inputProj(combined); // [T, embedDim]
    final c = timeEmbed(t);
    for (final b in blocks) {
      x = b(x, c);
    }
    x = finalNorm(x);
    return outputProj(x);
  }

  @override
  List<Tensor> parameters() => [
    ...inputProj.parameters(),
    ...timeEmbed.parameters(),
    for (final b in blocks) ...b.parameters(),
    ...finalNorm.parameters(),
    ...outputProj.parameters(),
  ];

  @override
  List<Module> submodules() => [
    inputProj,
    timeEmbed,
    ...blocks,
    finalNorm,
    outputProj,
  ];
}

// ---------------------------------------------------------------------------
// F5DurationPredictor — per-character mel-frame duration regression.
// ---------------------------------------------------------------------------

/// Small regression head that predicts, for each character token, how
/// many mel frames it should span. Given text-encoder features
/// `[T_chars, textDim]` it produces `[T_chars]` frame counts.
///
/// Real F5-TTS uses a MAS-aligned duration model with a two-layer
/// convolution + Linear head trained with an alignment loss. This
/// implementation is the same shape (two [ConvNeXtV2Block]s + Linear
/// to `1`) so the loader path is exactly what a real checkpoint
/// expects; the alignment training loop is a follow-up.
///
/// The result is a **soft** prediction; call [expandTextToFrames] to
/// broadcast per-character features into per-frame features (repeat +
/// round the predicted durations).
class F5DurationPredictor extends Module {
  final int textDim;
  final int intermediateDim;
  final List<ConvNeXtV2Block> blocks;
  final LayerNorm finalNorm;
  final Linear head;

  F5DurationPredictor({
    required this.textDim,
    this.intermediateDim = 512,
    int numLayers = 2,
    int convKernelSize = 3,
    Device device = Device.CPU,
    int seed = 0,
  }) : blocks = <ConvNeXtV2Block>[],
       finalNorm = LayerNorm(textDim, eps: 1e-6, device: device),
       head = Linear(textDim, 1, bias: true, device: device, seed: seed + 100) {
    for (int i = 0; i < numLayers; i++) {
      blocks.add(
        ConvNeXtV2Block(
          dim: textDim,
          intermediateDim: intermediateDim,
          kernelSize: convKernelSize,
          device: device,
          seed: seed + 10_000 * (i + 1),
        ),
      );
    }
  }

  /// `textFeatures: [T_chars, textDim]` → `[T_chars]` predicted
  /// mel-frame durations (softplus-squashed non-negative values).
  Tensor call(Tensor textFeatures) {
    if (textFeatures.shape.length != 2 || textFeatures.shape[1] != textDim) {
      throw ArgumentError(
        'F5DurationPredictor: expected [T, $textDim]; got '
        '${textFeatures.shape}',
      );
    }
    final t = textFeatures.shape[0];
    var h = textFeatures.reshape([1, t, textDim]);
    for (final b in blocks) {
      h = b(h);
    }
    final flat = h.reshape([t, textDim]);
    final normed = finalNorm(flat);
    final raw = head(normed).reshape([t]);
    // Softplus for non-negative durations: log(1 + exp(x)).
    final rawData = raw.toList();
    final out = List<double>.filled(t, 0);
    for (int i = 0; i < t; i++) {
      final v = rawData[i];
      // Numerically stable softplus.
      out[i] = v > 20 ? v : math.log(1.0 + math.exp(v));
    }
    return Tensor.fromList([t], out, device: textFeatures.device);
  }

  /// Given per-character features `[T_chars, D]` and per-character
  /// duration integers `[T_chars]`, broadcast into per-frame features
  /// `[sum(durations), D]`. Frame count is rounded to nearest integer;
  /// zero-duration characters are dropped.
  static Tensor expandTextToFrames(Tensor textFeatures, Tensor durations) {
    if (textFeatures.shape.length != 2 ||
        durations.shape.length != 1 ||
        durations.shape[0] != textFeatures.shape[0]) {
      throw ArgumentError(
        'expandTextToFrames: expected textFeatures=[T, D] and '
        'durations=[T]; got ${textFeatures.shape} and ${durations.shape}',
      );
    }
    final t = textFeatures.shape[0];
    final d = textFeatures.shape[1];
    final tData = textFeatures.toList();
    final dData = durations.toList();
    var totalFrames = 0;
    final rounded = List<int>.filled(t, 0);
    for (int i = 0; i < t; i++) {
      final r = dData[i].round();
      rounded[i] = r < 0 ? 0 : r;
      totalFrames += rounded[i];
    }
    final out = List<double>.filled(totalFrames * d, 0);
    var frameOff = 0;
    for (int i = 0; i < t; i++) {
      for (int f = 0; f < rounded[i]; f++) {
        for (int c = 0; c < d; c++) {
          out[frameOff * d + c] = tData[i * d + c];
        }
        frameOff++;
      }
    }
    return Tensor.fromList([totalFrames, d], out, device: textFeatures.device);
  }

  @override
  List<Tensor> parameters() => [
    for (final b in blocks) ...b.parameters(),
    ...finalNorm.parameters(),
    ...head.parameters(),
  ];

  @override
  List<Module> submodules() => [...blocks, finalNorm, head];
}

// ---------------------------------------------------------------------------
// F5TtsHFLoader — safetensors bindings for text encoder + DiT + duration.
// ---------------------------------------------------------------------------

/// Loader for F5-TTS safetensors. F5-TTS's checkpoint layout is
/// custom; this loader targets the reference `SWivet/f5-tts` state
/// dict format (as of 2024-06). Key mapping:
///
///   text_embed.embedding.weight          — [F5TextEncoder.tokenEmbedding.weight]
///   text_embed.blocks.{i}.dwconv.weight  — [ConvNeXtV2Block.dwconv.weight]
///   text_embed.blocks.{i}.dwconv.bias    — [ConvNeXtV2Block.dwconv.bias]
///   text_embed.blocks.{i}.norm.{weight,bias}
///   text_embed.blocks.{i}.pwconv1.{weight,bias}
///   text_embed.blocks.{i}.grn.{gamma,beta}
///   text_embed.blocks.{i}.pwconv2.{weight,bias}
///   text_embed.norm.{weight,bias}
///   input_proj.{weight,bias}             — [F5DiT.inputProj]
///   time_embed.freq_proj.{weight,bias}   — [SinusoidalTimestepEmbedding.proj1]
///   time_embed.out_proj.{weight,bias}    — [SinusoidalTimestepEmbedding.proj2]
///   blocks.{i}.norm1.{weight,bias}
///   blocks.{i}.attn.{q,k,v,o}_proj.{weight,bias}   per-head split
///   blocks.{i}.norm2.{weight,bias}
///   blocks.{i}.fc1.{weight,bias}
///   blocks.{i}.fc2.{weight,bias}
///   blocks.{i}.adaLN.modulation.{weight,bias}
///   norm_out.{weight,bias}                — [F5DiT.finalNorm]
///   output_proj.{weight,bias}
///
/// Duration predictor keys optional (prefixed by `duration.`).
class F5TtsHFLoader {
  /// Load from a safetensors file. Convenience wrapper around
  /// [loadMap].
  static F5TtsLoadReport loadFile({
    required F5TextEncoder textEncoder,
    required F5DiT dit,
    F5DurationPredictor? durationPredictor,
    required String path,
  }) {
    final state = SafeTensors.loadFile(path);
    return loadMap(
      textEncoder: textEncoder,
      dit: dit,
      durationPredictor: durationPredictor,
      state: state,
    );
  }

  /// Load into a text encoder + DiT bundle. Pass `null` for
  /// [durationPredictor] to skip duration keys.
  static F5TtsLoadReport loadMap({
    required F5TextEncoder textEncoder,
    required F5DiT dit,
    F5DurationPredictor? durationPredictor,
    required Map<String, Tensor> state,
  }) {
    final consumed = <String>{};
    Tensor take(String name) {
      final t = state[name];
      if (t == null) {
        throw ArgumentError('f5-tts loader: missing tensor "$name"');
      }
      consumed.add(name);
      return t;
    }

    _loadTextEncoder(textEncoder, take, 'text_embed');
    _loadDit(dit, take);
    if (durationPredictor != null) {
      _loadDuration(durationPredictor, take, 'duration');
    }

    final unused = state.keys.where((k) => !consumed.contains(k)).toList()
      ..sort();
    return F5TtsLoadReport(consumedCount: consumed.length, unusedKeys: unused);
  }

  static void _loadTextEncoder(
    F5TextEncoder m,
    Tensor Function(String) take,
    String prefix,
  ) {
    _copy(
      m.tokenEmbedding.weight,
      _expectShape(take('$prefix.embedding.weight'), [
        m.vocabSize,
        m.dim,
      ], '$prefix.embedding.weight'),
    );
    for (int i = 0; i < m.blocks.length; i++) {
      _loadConvNeXtV2Block(m.blocks[i], take, '$prefix.blocks.$i');
    }
    _copy(
      m.finalNorm.gamma,
      _expectShape(take('$prefix.norm.weight'), [m.dim], '$prefix.norm.weight'),
    );
    _copy(
      m.finalNorm.beta,
      _expectShape(take('$prefix.norm.bias'), [m.dim], '$prefix.norm.bias'),
    );
  }

  static void _loadDit(F5DiT m, Tensor Function(String) take) {
    _copy(
      m.inputProj.weight,
      _expectShape(take('input_proj.weight'), [
        m.embedDim,
        m.melDim + m.textDim,
      ], 'input_proj.weight'),
    );
    _copy(
      m.inputProj.bias!,
      _reshapeVectorTo1xN(
        _expectShape(take('input_proj.bias'), [m.embedDim], 'input_proj.bias'),
      ),
    );
    // Time embedding = SinusoidalTimestepEmbedding (two Linears).
    _copy(
      m.timeEmbed.proj1.weight,
      _expectShape(take('time_embed.freq_proj.weight'), [
        m.embedDim,
        m.freqDim,
      ], 'time_embed.freq_proj.weight'),
    );
    _copy(
      m.timeEmbed.proj1.bias!,
      _reshapeVectorTo1xN(
        _expectShape(take('time_embed.freq_proj.bias'), [
          m.embedDim,
        ], 'time_embed.freq_proj.bias'),
      ),
    );
    _copy(
      m.timeEmbed.proj2.weight,
      _expectShape(take('time_embed.out_proj.weight'), [
        m.embedDim,
        m.embedDim,
      ], 'time_embed.out_proj.weight'),
    );
    _copy(
      m.timeEmbed.proj2.bias!,
      _reshapeVectorTo1xN(
        _expectShape(take('time_embed.out_proj.bias'), [
          m.embedDim,
        ], 'time_embed.out_proj.bias'),
      ),
    );
    // Blocks.
    final numHeads = m.numHeads;
    final headDim = m.embedDim ~/ numHeads;
    for (int i = 0; i < m.blocks.length; i++) {
      final b = m.blocks[i];
      final p = 'blocks.$i';
      _copy(
        b.norm1.gamma,
        _expectShape(take('$p.norm1.weight'), [m.embedDim], '$p.norm1.weight'),
      );
      _copy(
        b.norm1.beta,
        _expectShape(take('$p.norm1.bias'), [m.embedDim], '$p.norm1.bias'),
      );

      // Fused QKV: HF stores as separate q_proj/k_proj/v_proj [D, D]
      // + biases [D]. Per-head split.
      for (final proj in const [
        ('q_proj', 'wq'),
        ('k_proj', 'wk'),
        ('v_proj', 'wv'),
      ]) {
        final w = _expectShape(take('$p.attn.${proj.$1}.weight'), [
          m.embedDim,
          m.embedDim,
        ], '$p.attn.${proj.$1}.weight');
        final bs = _expectShape(take('$p.attn.${proj.$1}.bias'), [
          m.embedDim,
        ], '$p.attn.${proj.$1}.bias');
        for (int h = 0; h < numHeads; h++) {
          final row = h * headDim;
          final headLinears = proj.$2 == 'wq'
              ? b.attn.wq
              : proj.$2 == 'wk'
              ? b.attn.wk
              : b.attn.wv;
          _copy(headLinears[h].weight, w.sliceRows(row, row + headDim));
          _copy(
            headLinears[h].bias!,
            _reshapeVectorTo1xN(_slice1DVector(bs, row, row + headDim)),
          );
        }
      }
      _copy(
        b.attn.wo.weight,
        _expectShape(take('$p.attn.o_proj.weight'), [
          m.embedDim,
          m.embedDim,
        ], '$p.attn.o_proj.weight'),
      );
      _copy(
        b.attn.wo.bias!,
        _reshapeVectorTo1xN(
          _expectShape(take('$p.attn.o_proj.bias'), [
            m.embedDim,
          ], '$p.attn.o_proj.bias'),
        ),
      );

      _copy(
        b.norm2.gamma,
        _expectShape(take('$p.norm2.weight'), [m.embedDim], '$p.norm2.weight'),
      );
      _copy(
        b.norm2.beta,
        _expectShape(take('$p.norm2.bias'), [m.embedDim], '$p.norm2.bias'),
      );
      _copy(
        b.fc1.weight,
        _expectShape(take('$p.fc1.weight'), [
          m.mlpDim,
          m.embedDim,
        ], '$p.fc1.weight'),
      );
      _copy(
        b.fc1.bias!,
        _reshapeVectorTo1xN(
          _expectShape(take('$p.fc1.bias'), [m.mlpDim], '$p.fc1.bias'),
        ),
      );
      _copy(
        b.fc2.weight,
        _expectShape(take('$p.fc2.weight'), [
          m.embedDim,
          m.mlpDim,
        ], '$p.fc2.weight'),
      );
      _copy(
        b.fc2.bias!,
        _reshapeVectorTo1xN(
          _expectShape(take('$p.fc2.bias'), [m.embedDim], '$p.fc2.bias'),
        ),
      );

      _copy(
        b.adaLn.modulation.weight,
        _expectShape(take('$p.adaLN.modulation.weight'), [
          6 * m.embedDim,
          m.embedDim,
        ], '$p.adaLN.modulation.weight'),
      );
      _copy(
        b.adaLn.modulation.bias!,
        _reshapeVectorTo1xN(
          _expectShape(take('$p.adaLN.modulation.bias'), [
            6 * m.embedDim,
          ], '$p.adaLN.modulation.bias'),
        ),
      );
    }
    _copy(
      m.finalNorm.gamma,
      _expectShape(take('norm_out.weight'), [m.embedDim], 'norm_out.weight'),
    );
    _copy(
      m.finalNorm.beta,
      _expectShape(take('norm_out.bias'), [m.embedDim], 'norm_out.bias'),
    );
    _copy(
      m.outputProj.weight,
      _expectShape(take('output_proj.weight'), [
        m.melDim,
        m.embedDim,
      ], 'output_proj.weight'),
    );
    _copy(
      m.outputProj.bias!,
      _reshapeVectorTo1xN(
        _expectShape(take('output_proj.bias'), [m.melDim], 'output_proj.bias'),
      ),
    );
  }

  static void _loadDuration(
    F5DurationPredictor m,
    Tensor Function(String) take,
    String prefix,
  ) {
    for (int i = 0; i < m.blocks.length; i++) {
      _loadConvNeXtV2Block(m.blocks[i], take, '$prefix.blocks.$i');
    }
    _copy(
      m.finalNorm.gamma,
      _expectShape(take('$prefix.norm.weight'), [
        m.textDim,
      ], '$prefix.norm.weight'),
    );
    _copy(
      m.finalNorm.beta,
      _expectShape(take('$prefix.norm.bias'), [m.textDim], '$prefix.norm.bias'),
    );
    _copy(
      m.head.weight,
      _expectShape(take('$prefix.head.weight'), [
        1,
        m.textDim,
      ], '$prefix.head.weight'),
    );
    _copy(
      m.head.bias!,
      _reshapeVectorTo1xN(
        _expectShape(take('$prefix.head.bias'), [1], '$prefix.head.bias'),
      ),
    );
  }

  static void _loadConvNeXtV2Block(
    ConvNeXtV2Block b,
    Tensor Function(String) take,
    String prefix,
  ) {
    _copy(
      b.dwconv.weight,
      _expectShape(take('$prefix.dwconv.weight'), [
        b.dim,
        b.dwconv.kernelSize,
      ], '$prefix.dwconv.weight'),
    );
    _copy(
      b.dwconv.bias!,
      _reshapeVectorTo1xN(
        _expectShape(take('$prefix.dwconv.bias'), [
          b.dim,
        ], '$prefix.dwconv.bias'),
      ),
    );
    _copy(
      b.norm.gamma,
      _expectShape(take('$prefix.norm.weight'), [b.dim], '$prefix.norm.weight'),
    );
    _copy(
      b.norm.beta,
      _expectShape(take('$prefix.norm.bias'), [b.dim], '$prefix.norm.bias'),
    );
    _copy(
      b.pwconv1.weight,
      _expectShape(take('$prefix.pwconv1.weight'), [
        b.intermediateDim,
        b.dim,
      ], '$prefix.pwconv1.weight'),
    );
    _copy(
      b.pwconv1.bias!,
      _reshapeVectorTo1xN(
        _expectShape(take('$prefix.pwconv1.bias'), [
          b.intermediateDim,
        ], '$prefix.pwconv1.bias'),
      ),
    );
    _copy(
      b.grn.gamma,
      _expectShape(take('$prefix.grn.gamma'), [
        b.intermediateDim,
      ], '$prefix.grn.gamma'),
    );
    _copy(
      b.grn.beta,
      _expectShape(take('$prefix.grn.beta'), [
        b.intermediateDim,
      ], '$prefix.grn.beta'),
    );
    _copy(
      b.pwconv2.weight,
      _expectShape(take('$prefix.pwconv2.weight'), [
        b.dim,
        b.intermediateDim,
      ], '$prefix.pwconv2.weight'),
    );
    _copy(
      b.pwconv2.bias!,
      _reshapeVectorTo1xN(
        _expectShape(take('$prefix.pwconv2.bias'), [
          b.dim,
        ], '$prefix.pwconv2.bias'),
      ),
    );
  }

  // -------------------- helpers --------------------

  static Tensor _expectShape(Tensor t, List<int> expected, String name) {
    if (t.shape.length != expected.length) {
      throw ArgumentError(
        'f5-tts loader: "$name" expected shape $expected, got ${t.shape}',
      );
    }
    for (int i = 0; i < expected.length; i++) {
      if (t.shape[i] != expected[i]) {
        throw ArgumentError(
          'f5-tts loader: "$name" expected shape $expected, got ${t.shape}',
        );
      }
    }
    return t;
  }

  static void _copy(Tensor dst, Tensor src) {
    if (dst.length != src.length) {
      throw ArgumentError(
        'f5-tts loader: copy length mismatch — dst=${dst.shape} '
        '(${dst.length}), src=${src.shape} (${src.length})',
      );
    }
    final vals = src.toList();
    final matched = Tensor.fromList(dst.shape, vals, device: dst.device);
    dst.assign(matched);
  }

  static Tensor _slice1DVector(Tensor t, int start, int end) {
    final data = t.toList();
    final n = end - start;
    final out = Float32List(n);
    for (int i = 0; i < n; i++) {
      out[i] = data[start + i];
    }
    return Tensor.fromList([n], out, device: Device.CPU);
  }

  static Tensor _reshapeVectorTo1xN(Tensor v) {
    if (v.shape.length != 1) {
      throw ArgumentError(
        'f5-tts loader: expected rank 1 for bias vector, got ${v.shape}',
      );
    }
    return Tensor.fromList([1, v.shape[0]], v.toList(), device: Device.CPU);
  }
}

class F5TtsLoadReport {
  final int consumedCount;
  final List<String> unusedKeys;
  const F5TtsLoadReport({
    required this.consumedCount,
    required this.unusedKeys,
  });

  @override
  String toString() =>
      'F5TtsLoadReport(consumed=$consumedCount, unused=${unusedKeys.length})';
}
