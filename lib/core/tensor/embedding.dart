part of 'tensor.dart';

/// Embedding table lookup.
///
/// Contract:
///   * `table` shape `[V, D]` — the trainable weight matrix.
///   * `indices` shape `[N]` (or `[N, 1]`) with values in `[0, V)`,
///     stored as float32 and rounded to int (same convention as
///     `crossEntropy`).
///   * Output shape `[N, D]`.
///
/// Backward: only `table` receives a gradient; the `indices` tensor
/// is treated as a constant. On GPU the backward is a scatter-add
/// (atomicAdd) into the pre-allocated `table._grad`; on CPU it's an
/// inline accumulate.
extension TensorEmbedding on Tensor {
  /// Look up rows of `this` (the table) at the given `indices`.
  Tensor embedding(Tensor indices) {
    if (shape.length != 2) {
      throw ArgumentError('embedding requires 2D table; got $shape');
    }
    // Multi-axis indices are supported by flattening then re-shaping:
    // indices [B, S] with table [V, D] -> output [B, S, D].
    if (indices.shape.length > 1) {
      final outShape = [...indices.shape, shape[1]];
      return embedding(indices.reshape([indices.length])).reshape(outShape);
    }
    final v = shape[0];
    final d = shape[1];
    final n = indices.length;
    if (device != indices.device) {
      throw ArgumentError(
        'embedding: mixed devices — table=$device, '
        'indices=${indices.device}. Call .to(...) first.',
      );
    }

    final table = this;
    final outShape = [n, d];

    final out = device == Device.CPU
        ? _embeddingFwdCpu(indices, v, d, n)
        : Tensor._gpu(
            outShape,
            engine.embeddingForward(_handle!, indices._handle!, n),
          );

    if (requiresGrad) {
      out._setBackward([table], () {
        final gO = out._grad!;
        if (table.device == Device.CPU) {
          _embeddingBwdCpu(table, indices, gO, v, d, n);
        } else {
          _embeddingBwdGpu(table, indices, gO, n);
        }
      });
    }
    return out;
  }

  Tensor _embeddingFwdCpu(Tensor indices, int v, int d, int n) {
    final td = _cpuData!;
    final id = indices._cpuData!;
    final out = Float32List(n * d);
    for (int i = 0; i < n; i++) {
      final row = (id[i] + 0.5).floor();
      if (row < 0 || row >= v) {
        // Zero-fill out-of-range rows (matches the GPU kernel).
        continue;
      }
      for (int j = 0; j < d; j++) {
        out[i * d + j] = td[row * d + j];
      }
    }
    return Tensor._cpu([n, d], out);
  }

  static void _embeddingBwdCpu(
    Tensor table,
    Tensor indices,
    Tensor gOut,
    int v,
    int d,
    int n,
  ) {
    final id = indices._cpuData!;
    final gd = gOut._cpuData!;
    final gTable = Float32List(v * d);
    for (int i = 0; i < n; i++) {
      final row = (id[i] + 0.5).floor();
      if (row < 0 || row >= v) continue;
      for (int j = 0; j < d; j++) {
        gTable[row * d + j] += gd[i * d + j];
      }
    }
    table._accumulateGrad(Tensor._cpu(table.shape, gTable));
  }

  static void _embeddingBwdGpu(
    Tensor table,
    Tensor indices,
    Tensor gOut,
    int n,
  ) {
    // Prime table._grad so atomicAdds have somewhere to land.
    table._grad ??= Tensor.fill(table.shape, 0.0, device: Device.GPU);
    engine.embeddingBackward(
      indices._handle!,
      gOut._handle!,
      table._grad!._handle!,
      n,
    );
    // Mutated in place — no further accumulation.
  }
}
