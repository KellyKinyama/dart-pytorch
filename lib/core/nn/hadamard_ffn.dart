/// Hadamard feed-forward network — a near-parameter-free channel mixer that
/// replaces the standard two-matrix FFN, after Cactus Needle's Hadamard MLP
/// (https://cactuscompute.com/blog/hadamard-mlp).
///
/// A normal FFN spends `2 * d * (4d)` ≈ `8d²` parameters per layer. This
/// mixes channels with a **fixed** orthonormal Walsh-Hadamard matrix `H`
/// (stored as nothing but the rule, applied here as a constant matmul) and
/// learns only three per-channel vectors:
///
///   FFN(x) = D2 ⊙ H · silu(D1 ⊙ (H · x) + b)
///
/// so `3d` parameters per layer instead of `8d²` — a big win when the cost
/// is moving weights (CPU, tiny devices). `dim` must be a power of two.
///
/// This is the fixed-Walsh form; Needle's full version makes the rotation
/// learnable via Monarch factors (needs a batched tile transpose not yet in
/// the autograd core), which the ablation shows is worth only ~1-2%.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../tensor/tensor.dart';
import 'module.dart';

class HadamardFFN extends Module {
  final int dim;

  /// Constant `[dim, dim]` orthonormal Walsh-Hadamard matrix (non-trainable).
  final Tensor _h;

  /// Learnable per-channel scales/bias, shape `[1, dim]` (row-broadcast).
  final Tensor d1, b, d2;

  HadamardFFN(this.dim, {Device device = Device.CPU, int seed = 0})
      : _h = _walsh(dim, device),
        d1 = Tensor.fill([1, dim], 1.0, requiresGrad: true, device: device),
        b = Tensor.fill([1, dim], 0.0, requiresGrad: true, device: device),
        // small output scale → gentle residual write, as in Needle's d4
        d2 = Tensor.fill([1, dim], 0.1, requiresGrad: true, device: device) {
    if (dim < 2 || (dim & (dim - 1)) != 0) {
      throw ArgumentError('HadamardFFN: dim must be a power of two; got $dim');
    }
  }

  /// `x` is `[N, dim]`; returns `[N, dim]`.
  Tensor call(Tensor x) {
    final hx = x.matmul(_h); // H·x
    final pre = hx * d1 + b; // broadcast [N,dim] op [1,dim]
    final act = pre * pre.sigmoid(); // SiLU
    return act.matmul(_h) * d2; // (H·act) ⊙ D2
  }

  @override
  List<Tensor> parameters() => [d1, b, d2];
}

/// Build the `n×n` orthonormal Walsh-Hadamard matrix as a constant tensor.
Tensor _walsh(int n, Device device) {
  final inv = 1.0 / math.sqrt(n);
  final data = Float32List(n * n);
  for (var i = 0; i < n; i++) {
    for (var j = 0; j < n; j++) {
      var bits = i & j;
      var c = 0;
      while (bits != 0) {
        bits &= bits - 1;
        c++;
      }
      data[i * n + j] = (c & 1) == 0 ? inv : -inv;
    }
  }
  return Tensor.fromList([n, n], data, device: device);
}
