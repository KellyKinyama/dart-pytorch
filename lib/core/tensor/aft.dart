part of 'tensor.dart';

/// Attention-Free Transformer (AFT) — full variant.
///
/// Implements the operator described in Zhai et al. "An Attention Free
/// Transformer" (2021). Given inputs of shape `[T, D]`:
///
///     out[t, d] = sigmoid(Q[t, d]) *
///                 ( sum_{t'} exp(K[t', d] + W[t, t']) * V[t', d] )
///                 ---------------------------------------------------
///                 (   sum_{t'} exp(K[t', d] + W[t, t'])              )
///
/// The inner ratio is a numerically-stable per-`d` softmax over `t'`
/// of `K[:, d] + W[t, :]`, weighted-summed against `V[:, d]`. The
/// gating `sigmoid(Q)` is elementwise. There is no `Q @ K.T`, so the
/// per-step cost is `O(T * T * D)` in FLOPs but `O(T * D)` in memory
/// for activations — attention pairwise scores never materialise.
///
/// Inputs:
///   * `q`, `k`, `v`   — 2D `[T, D]` (all same shape, all CPU).
///   * `w`             — 2D `[T, T]` position bias.
///   * `masked`        — when `true`, restricts the inner sum to
///                       `t' <= t` (causal / decoder-only).
///
/// Output is `[T, D]` and shares device with the inputs.
///
/// Autograd (custom, single-node): computes analytical gradients for
/// `q`, `k`, `v`, `w` in one backward pass, closing over the cached
/// softmax weights `[T, T, D]`.
///
/// Runs on CPU or GPU (matches the device of the inputs — all four
/// must share device). GPU calls into the `aft_full_forward` /
/// `aft_full_backward` kernels; CPU is the plain-Dart triple loop.
extension TensorAft on Tensor {
  static Tensor aftFull(
    Tensor q,
    Tensor k,
    Tensor v,
    Tensor w, {
    bool masked = false,
  }) {
    if (q.shape.length != 2 || k.shape.length != 2 || v.shape.length != 2) {
      throw ArgumentError(
        'aftFull: q/k/v must be 2D [T, D]; got q=${q.shape} k=${k.shape} '
        'v=${v.shape}',
      );
    }
    final t = q.shape[0];
    final d = q.shape[1];
    if (k.shape[0] != t ||
        k.shape[1] != d ||
        v.shape[0] != t ||
        v.shape[1] != d) {
      throw ArgumentError(
        'aftFull: q, k, v must share shape [T, D]; got q=${q.shape} '
        'k=${k.shape} v=${v.shape}',
      );
    }
    if (w.shape.length != 2 || w.shape[0] != t || w.shape[1] != t) {
      throw ArgumentError('aftFull: w must be [T, T]=[$t, $t]; got ${w.shape}');
    }
    if (q.device != k.device || q.device != v.device || q.device != w.device) {
      throw ArgumentError(
        'aftFull: mixed devices (q=${q.device} k=${k.device} v=${v.device} '
        'w=${w.device}); move all operands to the same device first.',
      );
    }
    if (q.device == Device.GPU) {
      return _aftFullGpu(q, k, v, w, masked: masked);
    }
    return _aftFullCpu(q, k, v, w, masked: masked);
  }

  static Tensor _aftFullGpu(
    Tensor q,
    Tensor k,
    Tensor v,
    Tensor w, {
    required bool masked,
  }) {
    final t = q.shape[0];
    final d = q.shape[1];
    final outHandle = engine.aftFullForward(
      q._handle!,
      k._handle!,
      v._handle!,
      w._handle!,
      masked ? 1 : 0,
    );
    final out = Tensor._gpu([t, d], outHandle);
    if (q.requiresGrad || k.requiresGrad || v.requiresGrad || w.requiresGrad) {
      out._setBackward([q, k, v, w], () {
        // Fresh zero-init grad tensors for the kernel to atomicAdd into.
        final gQ = Tensor.fill(q.shape, 0.0, device: Device.GPU);
        final gK = Tensor.fill(k.shape, 0.0, device: Device.GPU);
        final gV = Tensor.fill(v.shape, 0.0, device: Device.GPU);
        final gW = Tensor.fill(w.shape, 0.0, device: Device.GPU);
        engine.aftFullBackward(
          q._handle!,
          k._handle!,
          v._handle!,
          w._handle!,
          out._grad!._handle!,
          masked ? 1 : 0,
          gQ._handle!,
          gK._handle!,
          gV._handle!,
          gW._handle!,
        );
        if (q.requiresGrad) {
          q._accumulateGrad(gQ);
        } else {
          gQ.dispose();
        }
        if (k.requiresGrad) {
          k._accumulateGrad(gK);
        } else {
          gK.dispose();
        }
        if (v.requiresGrad) {
          v._accumulateGrad(gV);
        } else {
          gV.dispose();
        }
        if (w.requiresGrad) {
          w._accumulateGrad(gW);
        } else {
          gW.dispose();
        }
      });
    }
    return out;
  }

  static Tensor _aftFullCpu(
    Tensor q,
    Tensor k,
    Tensor v,
    Tensor w, {
    required bool masked,
  }) {
    final t = q.shape[0];
    final d = q.shape[1];

    final qd = q._cpuData!;
    final kd = k._cpuData!;
    final vd = v._cpuData!;
    final wd = w._cpuData!;

    final weights = Float32List(t * t * d); // [t_out, t_prime, d]
    final wv = Float32List(t * d); // weighted V per output pos
    final sigQ = Float32List(t * d);
    final outData = Float32List(t * d);

    for (int i = 0; i < t; i++) {
      final limit = masked ? i + 1 : t;
      for (int dd = 0; dd < d; dd++) {
        double maxVal = -1e30;
        for (int tp = 0; tp < limit; tp++) {
          final s = kd[tp * d + dd] + wd[i * t + tp];
          if (s > maxVal) maxVal = s;
        }
        double denom = 0.0;
        for (int tp = 0; tp < limit; tp++) {
          final e = math.exp(kd[tp * d + dd] + wd[i * t + tp] - maxVal);
          weights[(i * t + tp) * d + dd] = e;
          denom += e;
        }
        final invDenom = 1.0 / (denom + 1e-6);
        double num = 0.0;
        for (int tp = 0; tp < limit; tp++) {
          final wt = weights[(i * t + tp) * d + dd] * invDenom;
          weights[(i * t + tp) * d + dd] = wt;
          num += wt * vd[tp * d + dd];
        }
        wv[i * d + dd] = num;
        final sq = 1.0 / (1.0 + math.exp(-qd[i * d + dd]));
        sigQ[i * d + dd] = sq;
        outData[i * d + dd] = sq * num;
      }
    }

    final out = Tensor._cpu([t, d], outData);
    if (q.requiresGrad || k.requiresGrad || v.requiresGrad || w.requiresGrad) {
      out._setBackward([q, k, v, w], () {
        final gO = out._grad!._cpuData!;
        final gQ = Float32List(t * d);
        final gK = Float32List(t * d);
        final gV = Float32List(t * d);
        final gW = Float32List(t * t);

        for (int i = 0; i < t; i++) {
          final limit = masked ? i + 1 : t;
          for (int dd = 0; dd < d; dd++) {
            final gy = gO[i * d + dd];
            if (gy == 0.0) continue;
            final sq = sigQ[i * d + dd];
            final wvv = wv[i * d + dd];

            // out = sq * wv
            gQ[i * d + dd] += gy * sq * (1.0 - sq) * wvv;
            final dWv = gy * sq;

            // wv = sum_{t'} weights[i, t', dd] * V[t', dd]
            // dweights[i, t', dd] = dWv * V[t', dd]
            // dV[t', dd] += dWv * weights[i, t', dd]
            //
            // weights[i, t', dd] is softmax over t' (per (i, dd)) of
            // s[t', dd] = K[t', dd] + W[i, t'].
            // ds[t', dd] = (dweights[i, t', dd] - dot) * weights[i, t', dd]
            // where dot = sum_{t'} dweights[i, t', dd] * weights[i, t', dd]
            //
            // dK[t', dd] += ds[t', dd]
            // dW[i, t']  += ds[t', dd]    (summed over dd)
            double dot = 0.0;
            for (int tp = 0; tp < limit; tp++) {
              final wt = weights[(i * t + tp) * d + dd];
              final dwt = dWv * vd[tp * d + dd];
              dot += dwt * wt;
            }
            for (int tp = 0; tp < limit; tp++) {
              final wt = weights[(i * t + tp) * d + dd];
              final dwt = dWv * vd[tp * d + dd];
              final ds = (dwt - dot) * wt;
              gK[tp * d + dd] += ds;
              gW[i * t + tp] += ds;
              gV[tp * d + dd] += dWv * wt;
            }
          }
        }

        if (q.requiresGrad) q._accumulateGrad(Tensor._cpu(q.shape, gQ));
        if (k.requiresGrad) k._accumulateGrad(Tensor._cpu(k.shape, gK));
        if (v.requiresGrad) v._accumulateGrad(Tensor._cpu(v.shape, gV));
        if (w.requiresGrad) w._accumulateGrad(Tensor._cpu(w.shape, gW));
      });
    }
    return out;
  }

  /// Return the top-left `[rows, cols]` submatrix of a 2D `[R, C]`
  /// tensor. Autograd scatters the slice's gradient back to the same
  /// top-left region of the source (rest gets zero).
  ///
  /// Runs on CPU or GPU (matches the input's device).
  static Tensor sliceTopLeft(Tensor t, int rows, int cols) {
    if (t.shape.length != 2) {
      throw ArgumentError('sliceTopLeft: expected 2D input; got ${t.shape}');
    }
    final r = t.shape[0];
    final c = t.shape[1];
    if (rows > r || cols > c) {
      throw ArgumentError(
        'sliceTopLeft: requested [$rows, $cols] exceeds source [$r, $c]',
      );
    }
    if (t.device == Device.GPU) {
      final outHandle = engine.sliceTopLeftForward(t._handle!, rows, cols);
      final sliced = Tensor._gpu([rows, cols], outHandle);
      if (t.requiresGrad) {
        sliced._setBackward([t], () {
          final gHandle = engine.sliceTopLeftBackward(
            sliced._grad!._handle!,
            r,
            c,
          );
          t._accumulateGrad(Tensor._gpu(t.shape, gHandle));
        });
      }
      return sliced;
    }
    final src = t._cpuData!;
    final out = Float32List(rows * cols);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        out[i * cols + j] = src[i * c + j];
      }
    }
    final sliced = Tensor._cpu([rows, cols], out);
    if (t.requiresGrad) {
      sliced._setBackward([t], () {
        final gs = sliced._grad!._cpuData!;
        final gFull = Float32List(r * c);
        for (int i = 0; i < rows; i++) {
          for (int j = 0; j < cols; j++) {
            gFull[i * c + j] = gs[i * cols + j];
          }
        }
        t._accumulateGrad(Tensor._cpu(t.shape, gFull));
      });
    }
    return sliced;
  }
}
