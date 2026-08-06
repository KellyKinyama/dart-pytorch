part of 'tensor.dart';

/// RMSNorm along the last axis of a 2D tensor.
///
/// Contract:
///   * `x` shape `[R, C]` (or any rank >= 2 — leading dims are treated
///     as batch, normalisation happens per row of the last axis).
///   * `gamma` has length `C` (shape `[C]` or `[1, C]`).
///   * All tensors must live on the same device.
///
/// Formula (LayerNorm without mean-subtract and without beta):
///     y_j = (x_j / sqrt(mean_k(x_k^2) + eps)) * gamma_j
///
/// Backward: computes gradients for `x` and `gamma` when either has
/// `requiresGrad = true`. CPU path is inline; GPU path calls
/// `rmsnorm_backward` which atomically accumulates `dGamma` into the
/// existing grad tensor and returns a fresh `dX` handle.
extension TensorRmsNorm on Tensor {
  Tensor rmsNorm(Tensor gamma, {double eps = 1e-6}) {
    if (shape.length > 2) {
      final cols = shape.last;
      final rows = length ~/ cols;
      final original = List<int>.of(shape);
      return reshape([rows, cols]).rmsNorm(gamma, eps: eps).reshape(original);
    }
    if (shape.length != 2) {
      throw ArgumentError('rmsNorm requires rank >= 2 x; got $shape');
    }
    final r = shape[0];
    final c = shape[1];
    if (gamma.length != c) {
      throw ArgumentError(
        'rmsNorm: gamma length must equal $c; got gamma=${gamma.shape}',
      );
    }
    if (device != gamma.device) {
      throw ArgumentError(
        'rmsNorm: mixed devices — x=$device, gamma=${gamma.device}. '
        'Call .to(...) first.',
      );
    }

    final out = device == Device.CPU
        ? _rmsNormFwdCpu(gamma, eps, r, c)
        : Tensor._gpu(
            shape,
            engine.rmsnormForward(_handle!, gamma._handle!, eps),
          );

    if (requiresGrad || gamma.requiresGrad) {
      final x = this;
      final g = gamma;
      out._setBackward([x, g], () {
        if (x.device == Device.CPU) {
          _rmsNormBwdCpu(x, g, out._grad!, eps, r, c);
        } else {
          _rmsNormBwdGpu(x, g, out._grad!, eps);
        }
      });
    }
    return out;
  }

  Tensor _rmsNormFwdCpu(Tensor gamma, double eps, int r, int c) {
    final xd = _cpuData!;
    final gd = gamma._readAsFp32();
    final out = Float32List(r * c);
    final invC = 1.0 / c;
    for (int i = 0; i < r; i++) {
      double sq = 0;
      for (int j = 0; j < c; j++) {
        final v = xd[i * c + j];
        sq += v * v;
      }
      final invRms = 1.0 / math.sqrt(sq * invC + eps);
      for (int j = 0; j < c; j++) {
        out[i * c + j] = xd[i * c + j] * invRms * gd[j];
      }
    }
    return Tensor._cpu(shape, out);
  }

  static void _rmsNormBwdCpu(
    Tensor x,
    Tensor gamma,
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
    final invC = 1.0 / c;

    for (int i = 0; i < r; i++) {
      double sq = 0;
      for (int j = 0; j < c; j++) {
        final v = xd[i * c + j];
        sq += v * v;
      }
      final invRms = 1.0 / math.sqrt(sq * invC + eps);
      final invRms3 = invRms * invRms * invRms;

      double dot = 0; // sum_k (gO_k * gamma_k * x_k)
      for (int j = 0; j < c; j++) {
        final xj = xd[i * c + j];
        final gj = gOd[i * c + j];
        dot += gj * gd[j] * xj;
        gGamma[j] += gj * xj * invRms;
      }

      for (int j = 0; j < c; j++) {
        final xj = xd[i * c + j];
        final gj = gOd[i * c + j];
        gX[i * c + j] = gj * gd[j] * invRms - xj * invRms3 * dot * invC;
      }
    }

    if (x.requiresGrad) {
      x._accumulateGrad(Tensor._cpu(x.shape, gX));
    }
    if (gamma.requiresGrad) {
      gamma._accumulateGrad(Tensor._cpu(gamma.shape, gGamma));
    }
  }

  static void _rmsNormBwdGpu(Tensor x, Tensor gamma, Tensor gOut, double eps) {
    gamma._grad ??= Tensor.fill(gamma.shape, 0.0, device: Device.GPU);

    final gXHandle = engine.rmsnormBackward(
      x._handle!,
      gamma._handle!,
      gOut._handle!,
      gamma._grad!._handle!,
      eps,
    );

    if (x.requiresGrad) {
      x._accumulateGrad(Tensor._gpu(x.shape, gXHandle));
    } else {
      Tensor._gpu(x.shape, gXHandle).dispose();
    }
  }
}
