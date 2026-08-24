part of 'tensor.dart';

/// Matrix multiplication extension for [Tensor].
///
/// Placement: **respects the input device.** Both CPU (naive triple
/// loop) and GPU (tiled 32x32 kernel via `engine.matmulTensors`) paths
/// are implemented. Mixed-device inputs throw — call `.to(...)`
/// explicitly if that is really what you want.
///
/// Assumes 2D shapes: `this` is `[M, K]` and `o` is `[K, N]`; result is
/// `[M, N]` on the same device as the inputs.
extension TensorMatMul on Tensor {
  Tensor matmul(Tensor o) {
    // Batched left operand: `[..., K] @ [K, N] -> [..., N]`. The right
    // operand must be a plain 2D weight. Composed via reshape so the
    // 2D kernel below is still the single implementation.
    if (shape.length > 2 && o.shape.length == 2) {
      final k = shape.last;
      if (o.shape[0] != k) {
        throw ArgumentError(
          'matmul: inner dims mismatch — got $shape @ ${o.shape}',
        );
      }
      final n = o.shape[1];
      final rows = length ~/ k;
      final outShape = [...shape.sublist(0, shape.length - 1), n];
      return reshape([rows, k]).matmul(o).reshape(outShape);
    }
    if (shape.length != 2 || o.shape.length != 2) {
      throw ArgumentError(
        'matmul requires 2D tensors; got $shape @ ${o.shape}',
      );
    }
    final m = shape[0];
    final k = shape[1];
    final k2 = o.shape[0];
    final n = o.shape[1];
    if (k != k2) {
      throw ArgumentError(
        'matmul: inner dims mismatch: A cols=$k vs B rows=$k2 '
        '(shapes $shape @ ${o.shape})',
      );
    }
    if (device != o.device) {
      throw ArgumentError(
        'matmul: mixed devices ($device @ ${o.device}). '
        'Call .to(Device.CPU) or .to(Device.GPU) on one operand first.',
      );
    }

    final out = device == Device.CPU
        ? _matmulCpu(o, m, k, n)
        : Tensor._gpu([m, n], engine.matmulTensors(_handle!, o._handle!));

    // Autograd: dA = dOut @ B^T,  dB = A^T @ dOut.
    if (requiresGrad || o.requiresGrad) {
      final a = this;
      final b = o;
      out._setBackward([a, b], () {
        final gOut = out._grad!;
        if (a.requiresGrad) {
          final bT = b.transpose();
          final contrib = gOut.matmul(bT);
          bT.dispose();
          a._accumulateGrad(contrib);
        }
        if (b.requiresGrad) {
          final aT = a.transpose();
          final contrib = aT.matmul(gOut);
          aT.dispose();
          b._accumulateGrad(contrib);
        }
      });
    }
    return out;
  }

  Tensor _matmulCpu(Tensor o, int m, int k, int n) {
    final a = _readAsFp32();
    final b = o._readAsFp32();
    final out = Float32List(m * n);
    // i,p,j loop order — `b[bRow + j]` and `out[outRow + j]` sweep
    // sequentially on the inner axis, `a[aRow + p]` is invariant.
    // ~5× faster than the i,j,p order that strides through `b` by n
    // in the inner loop.
    for (int i = 0; i < m; i++) {
      final aRow = i * k;
      final outRow = i * n;
      for (int p = 0; p < k; p++) {
        final aip = a[aRow + p];
        if (aip == 0.0) continue;
        final bRow = p * n;
        for (int j = 0; j < n; j++) {
          out[outRow + j] += aip * b[bRow + j];
        }
      }
    }
    return Tensor._cpu([m, n], out);
  }
}
