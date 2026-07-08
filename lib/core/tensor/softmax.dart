part of 'tensor.dart';

/// Row-wise softmax along the last axis of a 2D `[R, C]` tensor.
///
/// Numerically stable: subtracts the row max before exponentiating.
/// Backward composes as `dx[r] = (dO[r] - <dO[r], y[r]>) * y[r]`.
///
/// On CPU the backward is expressed via existing ops; on GPU it calls
/// the `softmax_backward` kernel for one fused pass.
extension TensorSoftmax on Tensor {
  Tensor softmax() {
    if (shape.length != 2) {
      throw ArgumentError('softmax requires 2D input; got $shape');
    }
    final r = shape[0];
    final c = shape[1];

    final out = device == Device.CPU
        ? _softmaxFwdCpu(r, c)
        : Tensor._gpu(shape, engine.softmaxForward(_handle!));

    if (requiresGrad) {
      final x = this;
      final y = out;
      out._setBackward([x], () {
        final gO = out._grad!;
        if (x.device == Device.CPU) {
          _softmaxBwdCpu(x, y, gO, r, c);
        } else {
          final gXHandle = engine.softmaxBackward(y._handle!, gO._handle!);
          x._accumulateGrad(Tensor._gpu(x.shape, gXHandle));
        }
      });
    }
    return out;
  }

  Tensor _softmaxFwdCpu(int r, int c) {
    final xd = _cpuData!;
    final out = Float32List(r * c);
    for (int i = 0; i < r; i++) {
      double m = xd[i * c];
      for (int j = 1; j < c; j++) {
        if (xd[i * c + j] > m) m = xd[i * c + j];
      }
      double s = 0;
      for (int j = 0; j < c; j++) {
        final e = math.exp(xd[i * c + j] - m);
        out[i * c + j] = e;
        s += e;
      }
      final inv = 1.0 / s;
      for (int j = 0; j < c; j++) {
        out[i * c + j] *= inv;
      }
    }
    return Tensor._cpu(shape, out);
  }

  static void _softmaxBwdCpu(Tensor x, Tensor y, Tensor gO, int r, int c) {
    final yd = y._cpuData!;
    final gd = gO._cpuData!;
    final gX = Float32List(r * c);
    for (int i = 0; i < r; i++) {
      double dot = 0;
      for (int j = 0; j < c; j++) {
        dot += gd[i * c + j] * yd[i * c + j];
      }
      for (int j = 0; j < c; j++) {
        gX[i * c + j] = (gd[i * c + j] - dot) * yd[i * c + j];
      }
    }
    x._accumulateGrad(Tensor._cpu(x.shape, gX));
  }
}

/// Fused softmax + negative-log-likelihood cross-entropy with integer
/// class labels.
///
/// Contract:
///   * `logits` shape `[R, C]`.
///   * `targets` shape `[R]` (or `[R, 1]`) holding class indices in
///     `[0, C)`. Stored as float32 and rounded to int in the kernel /
///     CPU path.
///   * Returns per-sample losses shape `[R, 1]` — apply `.mean()` or
///     `.sum()` yourself to get a scalar. Keeping the reduction
///     out-of-op lets autograd handle it via existing ops.
///
/// Backward is computed as `dx = (softmax(logits) - one_hot(targets)) * gLoss`
/// (the classic fused form; numerically stable and single-pass).
extension TensorCrossEntropy on Tensor {
  Tensor crossEntropy(Tensor targets) {
    if (shape.length != 2) {
      throw ArgumentError('crossEntropy requires 2D logits; got $shape');
    }
    final r = shape[0];
    final c = shape[1];
    if (targets.length != r) {
      throw ArgumentError(
        'crossEntropy: targets length must equal $r; got ${targets.shape}',
      );
    }
    if (device != targets.device) {
      throw ArgumentError(
        'crossEntropy: mixed devices — logits=$device, '
        'targets=${targets.device}. Call .to(...) first.',
      );
    }

    final Tensor loss;
    if (device == Device.CPU) {
      loss = _crossEntropyFwdCpu(targets, r, c);
    } else {
      loss = Tensor._gpu([
        r,
        1,
      ], engine.crossEntropyForward(_handle!, targets._handle!));
    }

    if (requiresGrad) {
      final x = this;
      final t = targets;
      loss._setBackward([x], () {
        final gLoss = loss._grad!;
        if (x.device == Device.CPU) {
          _crossEntropyBwdCpu(x, t, gLoss, r, c);
        } else {
          final gXHandle = engine.crossEntropyBackward(
            x._handle!,
            t._handle!,
            gLoss._handle!,
          );
          x._accumulateGrad(Tensor._gpu(x.shape, gXHandle));
        }
      });
    }
    return loss;
  }

  Tensor _crossEntropyFwdCpu(Tensor targets, int r, int c) {
    final xd = _cpuData!;
    final td = targets._cpuData!;
    final loss = Float32List(r);
    for (int i = 0; i < r; i++) {
      double m = xd[i * c];
      for (int j = 1; j < c; j++) {
        if (xd[i * c + j] > m) m = xd[i * c + j];
      }
      double s = 0;
      for (int j = 0; j < c; j++) {
        s += math.exp(xd[i * c + j] - m);
      }
      final lse = m + math.log(s);
      final tgt = (td[i] + 0.5).floor();
      loss[i] = lse - xd[i * c + tgt];
    }
    return Tensor._cpu([r, 1], loss);
  }

  static void _crossEntropyBwdCpu(
    Tensor x,
    Tensor targets,
    Tensor gLoss,
    int r,
    int c,
  ) {
    final xd = x._cpuData!;
    final td = targets._cpuData!;
    final gd = gLoss._cpuData!;
    final gX = Float32List(r * c);
    for (int i = 0; i < r; i++) {
      double m = xd[i * c];
      for (int j = 1; j < c; j++) {
        if (xd[i * c + j] > m) m = xd[i * c + j];
      }
      double s = 0;
      for (int j = 0; j < c; j++) {
        s += math.exp(xd[i * c + j] - m);
      }
      final inv = 1.0 / s;
      final tgt = (td[i] + 0.5).floor();
      final g = gd[i];
      for (int j = 0; j < c; j++) {
        final p = math.exp(xd[i * c + j] - m) * inv;
        gX[i * c + j] = (p - (j == tgt ? 1.0 : 0.0)) * g;
      }
    }
    x._accumulateGrad(Tensor._cpu(x.shape, gX));
  }
}
