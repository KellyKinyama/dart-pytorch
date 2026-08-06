part of 'tensor.dart';

/// LayerNorm along the last axis of a 2D tensor.
///
/// Contract:
///   * `x` shape `[R, C]` — normalization is per-row over `C`.
///   * `gamma` and `beta` have length `C` (shape `[C]` or `[1, C]`).
///   * All three tensors must live on the same device.
///
/// Backward: computes gradients for `x`, `gamma`, and `beta` when any
/// of them has `requiresGrad = true`. Gradient math for CPU is inline;
/// for GPU we call the `layernorm_backward` kernel which atomically
/// accumulates `dGamma`/`dBeta` into the existing grad tensors and
/// writes a fresh `dX` handle.
extension TensorLayerNorm on Tensor {
  Tensor layerNorm(Tensor gamma, Tensor beta, {double eps = 1e-5}) {
    // Any rank >= 2 works: leading dims are treated as batch, and
    // normalization happens per row of the last axis. Composed via
    // reshape so the 2D kernel stays the single implementation.
    if (shape.length > 2) {
      final cols = shape.last;
      final rows = length ~/ cols;
      final original = List<int>.of(shape);
      return reshape([
        rows,
        cols,
      ]).layerNorm(gamma, beta, eps: eps).reshape(original);
    }
    if (shape.length != 2) {
      throw ArgumentError('layerNorm requires rank >= 2 x; got $shape');
    }
    final r = shape[0];
    final c = shape[1];
    if (gamma.length != c || beta.length != c) {
      throw ArgumentError(
        'layerNorm: gamma/beta length must equal $c; '
        'got gamma=${gamma.shape} beta=${beta.shape}',
      );
    }
    if (device != gamma.device || device != beta.device) {
      throw ArgumentError(
        'layerNorm: mixed devices — x=$device, gamma=${gamma.device}, '
        'beta=${beta.device}. Call .to(...) first.',
      );
    }

    final out = device == Device.CPU
        ? _layerNormFwdCpu(gamma, beta, eps, r, c)
        : Tensor._gpu(
            shape,
            engine.layernormForward(
              _handle!,
              gamma._handle!,
              beta._handle!,
              eps,
            ),
          );

    if (requiresGrad || gamma.requiresGrad || beta.requiresGrad) {
      final x = this;
      final g = gamma;
      final b = beta;
      out._setBackward([x, g, b], () {
        if (x.device == Device.CPU) {
          _layerNormBwdCpu(x, g, b, out._grad!, eps, r, c);
        } else {
          _layerNormBwdGpu(x, g, b, out._grad!, eps);
        }
      });
    }
    return out;
  }

  // -------------------------------------------------------------------
  // CPU implementations (correctness reference + test target without GPU).
  // -------------------------------------------------------------------

  Tensor _layerNormFwdCpu(Tensor gamma, Tensor beta, double eps, int r, int c) {
    final xd = _cpuData!;
    final gd = gamma._readAsFp32();
    final bd = beta._readAsFp32();
    final out = Float32List(r * c);
    for (int i = 0; i < r; i++) {
      double s = 0;
      for (int j = 0; j < c; j++) {
        s += xd[i * c + j];
      }
      final mean = s / c;
      double sq = 0;
      for (int j = 0; j < c; j++) {
        final d = xd[i * c + j] - mean;
        sq += d * d;
      }
      final invStd = 1.0 / math.sqrt(sq / c + eps);
      for (int j = 0; j < c; j++) {
        final xh = (xd[i * c + j] - mean) * invStd;
        out[i * c + j] = xh * gd[j] + bd[j];
      }
    }
    return Tensor._cpu(shape, out);
  }

  static void _layerNormBwdCpu(
    Tensor x,
    Tensor gamma,
    Tensor beta,
    Tensor gOut,
    double eps,
    int r,
    int c,
  ) {
    final xd = x._cpuData!;
    final gd = gamma._cpuData!;
    final gOd = gOut._cpuData!;

    final gX = Float32List(r * c);
    final gGamma = Float32List(c);
    final gBeta = Float32List(c);
    final invC = 1.0 / c;

    for (int i = 0; i < r; i++) {
      double s = 0;
      for (int j = 0; j < c; j++) {
        s += xd[i * c + j];
      }
      final mean = s * invC;

      double sq = 0;
      for (int j = 0; j < c; j++) {
        final d = xd[i * c + j] - mean;
        sq += d * d;
      }
      final invStd = 1.0 / math.sqrt(sq * invC + eps);

      double dxhSum = 0;
      double dxhXhSum = 0;
      for (int j = 0; j < c; j++) {
        final xh = (xd[i * c + j] - mean) * invStd;
        final dxh = gOd[i * c + j] * gd[j];
        dxhSum += dxh;
        dxhXhSum += dxh * xh;
        gGamma[j] += gOd[i * c + j] * xh;
        gBeta[j] += gOd[i * c + j];
      }
      for (int j = 0; j < c; j++) {
        final xh = (xd[i * c + j] - mean) * invStd;
        final dxh = gOd[i * c + j] * gd[j];
        gX[i * c + j] = invStd * (dxh - dxhSum * invC - xh * dxhXhSum * invC);
      }
    }

    if (x.requiresGrad) {
      x._accumulateGrad(Tensor._cpu(x.shape, gX));
    }
    if (gamma.requiresGrad) {
      gamma._accumulateGrad(Tensor._cpu(gamma.shape, gGamma));
    }
    if (beta.requiresGrad) {
      beta._accumulateGrad(Tensor._cpu(beta.shape, gBeta));
    }
  }

  // -------------------------------------------------------------------
  // GPU backward: pre-allocate gGamma / gBeta accumulators (zero-filled)
  // if the parents don't already have `.grad`, then hand their handles
  // to the kernel which atomicAdds into them. `dX` comes back as a
  // fresh tensor handle.
  // -------------------------------------------------------------------

  static void _layerNormBwdGpu(
    Tensor x,
    Tensor gamma,
    Tensor beta,
    Tensor gOut,
    double eps,
  ) {
    // Prime gGamma / gBeta so the atomicAdds have somewhere to land.
    // The kernel is additive, so an existing grad from a prior branch
    // gets extended (matching our accumulate-into semantics).
    gamma._grad ??= Tensor.fill(gamma.shape, 0.0, device: Device.GPU);
    beta._grad ??= Tensor.fill(beta.shape, 0.0, device: Device.GPU);

    final gXHandle = engine.layernormBackward(
      x._handle!,
      gamma._handle!,
      gOut._handle!,
      gamma._grad!._handle!,
      beta._grad!._handle!,
      eps,
    );

    if (x.requiresGrad) {
      x._accumulateGrad(Tensor._gpu(x.shape, gXHandle));
    } else {
      // Kernel wrote to a buffer we don't need — dispose it.
      Tensor._gpu(x.shape, gXHandle).dispose();
    }
    // gGamma / gBeta were mutated in place; nothing to accumulate here.
  }
}
