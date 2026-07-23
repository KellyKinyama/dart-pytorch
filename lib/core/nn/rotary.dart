/// Rotary Positional Embeddings (RoPE) — see
/// https://arxiv.org/abs/2104.09864.
///
/// This is the "GPT-NeoX / LLaMA" flavour of RoPE (also called
/// `rotate_half`): given a tensor of shape `[N, headDim]`, split the
/// last axis in half and rotate:
///
/// ```text
///     q_out = q * cos + rotate_half(q) * sin
///   where rotate_half([a | b]) = [-b | a]
/// ```
///
/// The cos/sin tables of shape `[maxCtx, headDim]` are precomputed
/// once. `rotate_half` is implemented as a `headDim x headDim`
/// permutation-with-sign matrix `P` so that
/// `rotate_half(q) == q @ P` — this lets us stay inside the existing
/// tensor op surface (matmul + elementwise mul/add) with no new
/// kernels.
///
/// Used by GPT-NeoX / Pythia / LLaMA / Mistral / Falcon etc.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../tensor/tensor.dart';

class RopeCache {
  RopeCache._(
    this.maxCtx,
    this.headDim,
    this.base,
    this._cosPerPos,
    this._sinPerPos,
    this._rotateHalfP,
    this.device,
  );

  final int maxCtx;
  final int headDim;
  final double base;
  final Device device;

  /// `_cosPerPos[i]` is a shape-`[1, headDim]` tensor holding the
  /// cos values for position `i`. Keeping one tensor per position
  /// lets us build the `[N, headDim]` broadcast for arbitrary
  /// contiguous windows `[startPos, startPos+N)` at forward time
  /// via `TensorConcat.concat(axis: 0)`.
  final List<Tensor> _cosPerPos;
  final List<Tensor> _sinPerPos;

  /// Permutation-with-sign matrix P of shape `[headDim, headDim]`
  /// such that `q @ P == rotate_half(q)`.
  final Tensor _rotateHalfP;

  /// Build tables for positions `[0, maxCtx)` and head-dim `headDim`
  /// (must be even). `base` is the geometric-progression base for the
  /// frequency schedule (10000 for GPT-NeoX / Pythia; the same in
  /// LLaMA/Mistral).
  factory RopeCache({
    required int maxCtx,
    required int headDim,
    double base = 10000.0,
    Device device = Device.CPU,
  }) {
    if (headDim.isOdd) {
      throw ArgumentError('RopeCache: headDim ($headDim) must be even');
    }
    final halfD = headDim ~/ 2;

    // Inverse frequencies: shape [halfD].
    final invFreq = List<double>.generate(
      halfD,
      (i) => 1.0 / math.pow(base, 2.0 * i / headDim),
    );

    // For each position m, build a [headDim] vector of cos/sin values
    // laid out as [cos(m*f_0)..cos(m*f_{halfD-1}) | cos(m*f_0)..cos(m*f_{halfD-1})]
    // (i.e. the halves are duplicated, matching GPT-NeoX's einsum +
    // torch.cat convention).
    final cosPerPos = <Tensor>[];
    final sinPerPos = <Tensor>[];
    for (int m = 0; m < maxCtx; m++) {
      final cRow = Float64List(headDim);
      final sRow = Float64List(headDim);
      for (int j = 0; j < halfD; j++) {
        final ang = m * invFreq[j];
        final c = math.cos(ang);
        final s = math.sin(ang);
        cRow[j] = c;
        cRow[j + halfD] = c;
        sRow[j] = s;
        sRow[j + halfD] = s;
      }
      cosPerPos.add(
        Tensor.fromList([1, headDim], cRow.toList(), device: device),
      );
      sinPerPos.add(
        Tensor.fromList([1, headDim], sRow.toList(), device: device),
      );
    }

    // rotate_half([a | b]) = [-b | a]. Encode as a matmul:
    // P has shape [headDim, headDim] where P[i, j] is 1 or -1 iff the
    // output column j sources input column i.
    //   * for j in [0, halfD):   out_j = -in_{j + halfD}
    //     -> P[j + halfD, j] = -1
    //   * for j in [halfD, headDim): out_j = in_{j - halfD}
    //     -> P[j - halfD, j] = 1
    final pData = Float64List(headDim * headDim);
    for (int j = 0; j < halfD; j++) {
      pData[(j + halfD) * headDim + j] = -1.0;
    }
    for (int j = halfD; j < headDim; j++) {
      pData[(j - halfD) * headDim + j] = 1.0;
    }
    final rotateHalfP = Tensor.fromList(
      [headDim, headDim],
      pData.toList(),
      device: device,
    );

    return RopeCache._(
      maxCtx,
      headDim,
      base,
      cosPerPos,
      sinPerPos,
      rotateHalfP,
      device,
    );
  }

  /// Apply RoPE to a `[N, headDim]` query or key tensor whose rows
  /// correspond to absolute positions `[startPos, startPos + N)`.
  ///
  /// Returns a new tensor of the same shape.
  Tensor apply(Tensor qOrK, {int startPos = 0}) {
    if (qOrK.shape.length != 2 || qOrK.shape[1] != headDim) {
      throw ArgumentError(
        'RopeCache.apply: expected shape [N, $headDim], got '
        '${qOrK.shape}',
      );
    }
    final n = qOrK.shape[0];
    if (startPos + n > maxCtx) {
      throw ArgumentError(
        'RopeCache.apply: window [$startPos, ${startPos + n}) exceeds '
        'maxCtx=$maxCtx (rebuild with a larger maxCtx)',
      );
    }
    // Assemble cos/sin windows of shape [N, headDim] by stacking rows.
    final cosRows = <Tensor>[];
    final sinRows = <Tensor>[];
    for (int i = 0; i < n; i++) {
      cosRows.add(_cosPerPos[startPos + i]);
      sinRows.add(_sinPerPos[startPos + i]);
    }
    final cosBlock = cosRows.length == 1
        ? cosRows.first
        : TensorConcat.concat(cosRows, axis: 0);
    final sinBlock = sinRows.length == 1
        ? sinRows.first
        : TensorConcat.concat(sinRows, axis: 0);

    final rotated = qOrK.matmul(_rotateHalfP); // rotate_half via matmul
    return (qOrK * cosBlock) + (rotated * sinBlock);
  }
}
