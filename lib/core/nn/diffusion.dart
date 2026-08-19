/// Denoising Diffusion Probabilistic Models — schedule + tiny U-Net.
///
/// Ho, Jain, Abbeel (2020). What's here:
///
///   * [NoiseSchedule.linear] — linear beta schedule with the paper's
///     defaults (`beta_start = 1e-4`, `beta_end = 0.02`, T = 1000).
///     Precomputes `alpha`, `alpha_bar`, `sqrt(alpha_bar)`,
///     `sqrt(1 - alpha_bar)`, and the posterior variance used in the
///     reverse step.
///   * [NoiseSchedule.forwardDiffuse] — closed-form
///     `q(x_t | x_0) = N(√α̅_t · x_0, (1 - α̅_t) I)`.
///   * [NoiseSchedule.reverseStep] — single Langevin step of the
///     Markov chain given a predicted `ε̂`.
///   * [TinyUNet] — a minimal 2-down / 2-up U-Net whose upsampling
///     path is our fresh [ConvTranspose2d]. Forward-only wiring
///     showcase (no training loop here — that needs Conv2d input-grad
///     support, currently missing). Loads pretrained tiny-DDPM
///     checkpoints or serves as scaffolding for future ports.
///
/// See `test/diffusion_test.dart` for the schedule invariants, exact
/// forward-diffusion means/variances, reverse-step algebra, and the
/// U-Net shape checks.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../tensor/tensor.dart';
import 'conv2d.dart';
import 'conv_transpose_2d.dart';
import 'linear.dart';
import 'module.dart';

class NoiseSchedule {
  final int numTimesteps;
  final Float32List betas;
  final Float32List alphas;
  final Float32List alphaBars;
  final Float32List sqrtAlphaBars;
  final Float32List sqrtOneMinusAlphaBars;
  final Float32List sqrtRecipAlphas;
  final Float32List posteriorVariance;

  NoiseSchedule._({
    required this.numTimesteps,
    required this.betas,
    required this.alphas,
    required this.alphaBars,
    required this.sqrtAlphaBars,
    required this.sqrtOneMinusAlphaBars,
    required this.sqrtRecipAlphas,
    required this.posteriorVariance,
  });

  /// Ho-et-al linear beta schedule.
  factory NoiseSchedule.linear({
    int numTimesteps = 1000,
    double betaStart = 1e-4,
    double betaEnd = 0.02,
  }) {
    if (numTimesteps < 2) {
      throw ArgumentError('NoiseSchedule.linear: numTimesteps must be >= 2');
    }
    final betas = Float32List(numTimesteps);
    final alphas = Float32List(numTimesteps);
    final alphaBars = Float32List(numTimesteps);
    final sqrtAB = Float32List(numTimesteps);
    final sqrtOneMinusAB = Float32List(numTimesteps);
    final sqrtRecipA = Float32List(numTimesteps);
    final postVar = Float32List(numTimesteps);
    double abCum = 1.0;
    for (int t = 0; t < numTimesteps; t++) {
      final b = betaStart + (betaEnd - betaStart) * t / (numTimesteps - 1);
      final a = 1.0 - b;
      abCum *= a;
      betas[t] = b;
      alphas[t] = a;
      alphaBars[t] = abCum;
      sqrtAB[t] = math.sqrt(abCum);
      sqrtOneMinusAB[t] = math.sqrt(1.0 - abCum);
      sqrtRecipA[t] = 1.0 / math.sqrt(a);
    }
    // Posterior variance β̃_t = β_t · (1 - α̅_{t-1}) / (1 - α̅_t).
    // For t = 0 the posterior collapses to a delta — we set it to 0.
    for (int t = 0; t < numTimesteps; t++) {
      if (t == 0) {
        postVar[t] = 0.0;
      } else {
        final abPrev = alphaBars[t - 1];
        postVar[t] = betas[t] * (1.0 - abPrev) / (1.0 - alphaBars[t]);
      }
    }
    return NoiseSchedule._(
      numTimesteps: numTimesteps,
      betas: betas,
      alphas: alphas,
      alphaBars: alphaBars,
      sqrtAlphaBars: sqrtAB,
      sqrtOneMinusAlphaBars: sqrtOneMinusAB,
      sqrtRecipAlphas: sqrtRecipA,
      posteriorVariance: postVar,
    );
  }

  void _checkT(int t) {
    if (t < 0 || t >= numTimesteps) {
      throw ArgumentError(
        'NoiseSchedule: t=$t out of range [0, $numTimesteps)',
      );
    }
  }

  /// Closed-form forward diffusion:
  /// `x_t = √α̅_t · x_0 + √(1 - α̅_t) · ε` with `ε ~ N(0, I)`.
  /// Returns the noised sample and the noise it used (so training
  /// code can regress `ε̂ → ε`). Pass [eps] to make the noise
  /// deterministic; otherwise it is drawn using [seed] (defaults to
  /// the system RNG).
  ({Tensor xT, Tensor eps}) forwardDiffuse(
    Tensor x0,
    int t, {
    int? seed,
    Tensor? eps,
  }) {
    _checkT(t);
    final n = x0.length;
    final noise =
        eps ?? _gaussianTensor(x0.shape, seed: seed, device: x0.device);
    if (noise.length != n || !_shapesEqual(noise.shape, x0.shape)) {
      throw ArgumentError(
        'forwardDiffuse: eps shape ${noise.shape} does not match x0 ${x0.shape}',
      );
    }
    final aScalar = sqrtAlphaBars[t];
    final bScalar = sqrtOneMinusAlphaBars[t];
    return (xT: x0 * aScalar + noise * bScalar, eps: noise);
  }

  /// One Langevin step of the reverse chain given a predicted noise
  /// [epsHat] (same shape as [xt]).
  ///
  ///   μ_t = (1 / √α_t) · (x_t − β_t / √(1 - α̅_t) · ε̂)
  ///   x_{t-1} = μ_t + √β̃_t · z              (z ~ N(0, I), t > 0)
  ///
  /// At `t == 0` the added noise is zero and the mean is returned
  /// directly, so this method safely terminates the chain.
  Tensor reverseStep(Tensor xt, int t, Tensor epsHat, {int? seed}) {
    _checkT(t);
    if (!_shapesEqual(xt.shape, epsHat.shape)) {
      throw ArgumentError(
        'reverseStep: epsHat shape ${epsHat.shape} != xt shape ${xt.shape}',
      );
    }
    final invSqrtA = sqrtRecipAlphas[t];
    final coefEps = betas[t] / sqrtOneMinusAlphaBars[t];
    final mean = (xt - epsHat * coefEps) * invSqrtA;
    if (t == 0) return mean;
    final sigma = math.sqrt(posteriorVariance[t]);
    if (sigma == 0.0) return mean;
    final z = _gaussianTensor(xt.shape, seed: seed, device: xt.device);
    return mean + z * sigma;
  }
}

/// Draw an `N(0, I)` tensor of shape [shape] onto [device] via
/// Box-Muller. `seed == null` uses the system RNG.
Tensor _gaussianTensor(
  List<int> shape, {
  int? seed,
  Device device = Device.CPU,
}) {
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

bool _shapesEqual(List<int> a, List<int> b) {
  if (a.length != b.length) return false;
  for (int i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

/// Minimal U-Net for `[N, 1, H, W]` grayscale images. Two down blocks
/// (stride-2 Conv2d), a mid block, then two up blocks
/// ([ConvTranspose2d], stride-2). A per-image scalar timestep is
/// embedded via a linear projection and broadcast-added to the mid
/// features. Predicts an ε-shape tensor `[N, 1, H, W]`.
class TinyUNet extends Module {
  final int totalTimesteps;

  final Conv2d down1; // 1 → hidden, s=2
  final Conv2d down2; // hidden → 2*hidden, s=2
  final Conv2d mid; // 2*hidden → 2*hidden, s=1
  final Linear timeProj; // 1 → 2*hidden
  final ConvTranspose2d up1; // 2*hidden → hidden, s=2
  final ConvTranspose2d up2; // hidden → 1, s=2

  TinyUNet({
    int hidden = 16,
    this.totalTimesteps = 1000,
    Device device = Device.CPU,
    int seed = 0,
  }) : down1 = Conv2d(
         1,
         hidden,
         kernel: 3,
         stride: 2,
         padding: 1,
         bias: true,
         device: device,
         seed: seed,
       ),
       down2 = Conv2d(
         hidden,
         hidden * 2,
         kernel: 3,
         stride: 2,
         padding: 1,
         bias: true,
         device: device,
         seed: seed + 1,
       ),
       mid = Conv2d(
         hidden * 2,
         hidden * 2,
         kernel: 3,
         stride: 1,
         padding: 1,
         bias: true,
         device: device,
         seed: seed + 2,
       ),
       timeProj = Linear(
         1,
         hidden * 2,
         bias: true,
         device: device,
         seed: seed + 3,
       ),
       up1 = ConvTranspose2d(
         hidden * 2,
         hidden,
         kernel: 4,
         stride: 2,
         padding: 1,
         bias: true,
         device: device,
         seed: seed + 4,
       ),
       up2 = ConvTranspose2d(
         hidden,
         1,
         kernel: 4,
         stride: 2,
         padding: 1,
         bias: true,
         device: device,
         seed: seed + 5,
       );

  Tensor call(Tensor x, int t) {
    if (x.shape.length != 4 || x.shape[1] != 1) {
      throw ArgumentError('TinyUNet: expected [N, 1, H, W]; got ${x.shape}');
    }
    if (t < 0 || t >= totalTimesteps) {
      throw ArgumentError('TinyUNet: t=$t out of range [0, $totalTimesteps)');
    }
    final n = x.shape[0];
    var h = down1(x).relu();
    h = down2(h).relu();
    h = mid(h);
    // Time embedding: broadcast a `[N, 2*hidden]` vector across the
    // spatial axes of the mid feature map.
    final tScalar = t / totalTimesteps;
    final tIn = Tensor.fromList(
      [n, 1],
      List<double>.filled(n, tScalar),
      device: x.device,
    );
    final tEmb = timeProj(tIn); // [N, 2*hidden]
    final ch = tEmb.shape[1];
    final sh = h.shape[2];
    final sw = h.shape[3];
    // Broadcast add: replicate tEmb across [H, W] on host, then re-lift.
    final tHost = Tensor.noGrad(() => tEmb.toList());
    final broadcast = Float32List(n * ch * sh * sw);
    for (int ni = 0; ni < n; ni++) {
      for (int c = 0; c < ch; c++) {
        final val = tHost[ni * ch + c];
        final base = ((ni * ch + c) * sh) * sw;
        for (int i = 0; i < sh * sw; i++) {
          broadcast[base + i] = val;
        }
      }
    }
    final tEmbBroadcast = Tensor.fromFloat32List(
      [n, ch, sh, sw],
      broadcast,
      device: x.device,
    );
    h = (h + tEmbBroadcast).relu();
    h = up1(h).relu();
    return up2(h);
  }

  @override
  List<Tensor> parameters() => [
    ...down1.parameters(),
    ...down2.parameters(),
    ...mid.parameters(),
    ...timeProj.parameters(),
    ...up1.parameters(),
    ...up2.parameters(),
  ];

  @override
  List<Module> submodules() => [down1, down2, mid, timeProj, up1, up2];
}
