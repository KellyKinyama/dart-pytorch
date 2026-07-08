part of 'tensor.dart';

/// Row-wise scatter-add: place a `[K, D]` subset of rows into an
/// otherwise-zero `[T, D]` tensor at the positions given by
/// `indices` (a `[K]` float32 tensor whose values are rounded to
/// int). If the same index appears more than once in `indices` the
/// contributions add (atomically on GPU).
///
/// This is the adjoint of [TensorEmbedding.embedding] — `embedding`
/// gathers `subset[k] = table[indices[k]]`; `scatterRowsAdd`
/// scatters `full[indices[k]] += subset[k]`. Together they let you
/// implement sparse-MoE routing: gather the tokens routed to each
/// expert, run the expert on that subset, then scatter the results
/// back into a full-sequence buffer.
///
/// Autograd:
///   * `subset` gets its gradient by gathering back from
///     `output.grad` at the same indices (i.e. `embedding`).
///   * `indices` is treated as constant.
extension TensorScatterRows on Tensor {
  /// Scatter the rows of `this` (`[K, D]`) into a fresh `[T, D]`
  /// tensor of zeros, at row positions given by [indices] (`[K]`).
  Tensor scatterRowsAdd(Tensor indices, int fullRows) {
    if (shape.length != 2) {
      throw ArgumentError(
        'scatterRowsAdd: expected 2D subset [K, D]; got $shape',
      );
    }
    if (indices.shape.length != 1) {
      throw ArgumentError(
        'scatterRowsAdd: indices must be 1D [K]; got ${indices.shape}',
      );
    }
    if (indices.shape[0] != shape[0]) {
      throw ArgumentError(
        'scatterRowsAdd: indices length (${indices.shape[0]}) must '
        'equal subset row count (${shape[0]})',
      );
    }
    if (device != indices.device) {
      throw ArgumentError(
        'scatterRowsAdd: mixed devices — subset=$device, '
        'indices=${indices.device}.',
      );
    }
    final k = shape[0];
    final d = shape[1];
    final subset = this;

    Tensor out;
    if (device == Device.CPU) {
      final sd = _cpuData!;
      final id = indices._cpuData!;
      final outData = Float32List(fullRows * d);
      for (int i = 0; i < k; i++) {
        final row = (id[i] + 0.5).floor();
        if (row < 0 || row >= fullRows) continue;
        for (int j = 0; j < d; j++) {
          outData[row * d + j] += sd[i * d + j];
        }
      }
      out = Tensor._cpu([fullRows, d], outData);
    } else {
      // Allocate zero-init full-size tensor; embedding_backward
      // atomicAdds subset rows into the requested positions.
      out = Tensor.fill([fullRows, d], 0.0, device: Device.GPU);
      engine.embeddingBackward(
        indices._handle!,
        subset._handle!,
        out._handle!,
        k,
      );
    }

    if (requiresGrad) {
      out._setBackward([subset], () {
        // Gather back into subset.grad from out.grad at the same
        // indices. `out._grad` doesn't have requiresGrad, so
        // `embedding()` runs without recording new autograd.
        subset._accumulateGrad(out._grad!.embedding(indices));
      });
    }
    return out;
  }
}
