part of 'tensor.dart';

/// Concatenate a list of 2D tensors along `axis`. Supports `axis = 0`
/// (rows) and `axis = 1` (last-axis, columns).
///
/// All inputs must be 2D with matching size on the non-concat axis and
/// the same device. Backward slices the upstream gradient back into
/// each input's range along the concat axis.
///
/// The concat itself is done in host memory (Float32List) — for GPU
/// inputs the data is downloaded, spliced, and re-uploaded. That's
/// acceptable for the sizes seen in MHA / KV-cache use; a fused GPU
/// kernel can replace this later without changing the API.
extension TensorConcat on Tensor {
  static Tensor concat(List<Tensor> tensors, {int axis = 1}) {
    if (tensors.isEmpty) {
      throw ArgumentError('concat: input list is empty');
    }
    if (axis != 0 && axis != 1) {
      throw ArgumentError('concat: only axis=0 or axis=1 supported');
    }

    final first = tensors[0];
    final device = first.device;
    // For axis=1 we validate common rows; for axis=0 we validate common cols.
    final fixedAxis = axis == 1 ? 0 : 1;
    final fixedDim = first.shape[fixedAxis];
    int concatDim = 0;
    for (final t in tensors) {
      if (t.shape.length != 2) {
        throw ArgumentError('concat: 2D inputs required; got ${t.shape}');
      }
      if (t.shape[fixedAxis] != fixedDim) {
        throw ArgumentError(
          'concat(axis=$axis): non-concat dim mismatch — expected '
          '$fixedDim, got ${t.shape[fixedAxis]}',
        );
      }
      if (t.device != device) {
        throw ArgumentError(
          'concat: device mismatch — expected $device, got ${t.device}. '
          'Call .to(...) on inputs first.',
        );
      }
      concatDim += t.shape[axis];
    }

    final rows = axis == 1 ? fixedDim : concatDim;
    final cols = axis == 1 ? concatDim : fixedDim;

    // Gather host-side data (download from GPU if needed).
    final srcData = <Float32List>[];
    for (final t in tensors) {
      if (t.device == Device.CPU) {
        srcData.add(t._cpuData!);
      } else {
        srcData.add(Float32List.fromList(t.toList()));
      }
    }

    final outData = Float32List(rows * cols);
    if (axis == 1) {
      for (int r = 0; r < rows; r++) {
        int colOff = 0;
        for (int i = 0; i < tensors.length; i++) {
          final c = tensors[i].shape[1];
          for (int j = 0; j < c; j++) {
            outData[r * cols + colOff + j] = srcData[i][r * c + j];
          }
          colOff += c;
        }
      }
    } else {
      // axis == 0 — flat row copies.
      int rowOff = 0;
      for (int i = 0; i < tensors.length; i++) {
        final r = tensors[i].shape[0];
        outData.setRange(rowOff * cols, (rowOff + r) * cols, srcData[i]);
        rowOff += r;
      }
    }

    final anyReq = tensors.any((t) => t.requiresGrad);
    final Tensor out = device == Device.CPU
        ? Tensor._cpu([rows, cols], outData, requiresGrad: anyReq)
        : Tensor._gpu(
            [rows, cols],
            Tensor._uploadToGpu([rows, cols], outData),
            requiresGrad: anyReq,
          );

    if (anyReq) {
      out._setBackward(tensors, () {
        final gO = out._grad!;
        // Download upstream grad to a host buffer, slice per input.
        final gOData = gO.device == Device.CPU
            ? gO._cpuData!
            : Float32List.fromList(gO.toList());

        if (axis == 1) {
          int colOff = 0;
          for (int i = 0; i < tensors.length; i++) {
            final t = tensors[i];
            final c = t.shape[1];
            if (t.requiresGrad) {
              final slice = Float32List(rows * c);
              for (int r = 0; r < rows; r++) {
                for (int j = 0; j < c; j++) {
                  slice[r * c + j] = gOData[r * cols + colOff + j];
                }
              }
              final gSlice = t.device == Device.CPU
                  ? Tensor._cpu(t.shape, slice)
                  : Tensor._gpu(t.shape, Tensor._uploadToGpu(t.shape, slice));
              t._accumulateGrad(gSlice);
            }
            colOff += c;
          }
        } else {
          int rowOff = 0;
          for (int i = 0; i < tensors.length; i++) {
            final t = tensors[i];
            final r = t.shape[0];
            if (t.requiresGrad) {
              final slice = Float32List(r * cols);
              slice.setRange(0, r * cols,
                  gOData.sublist(rowOff * cols, (rowOff + r) * cols));
              final gSlice = t.device == Device.CPU
                  ? Tensor._cpu(t.shape, slice)
                  : Tensor._gpu(t.shape, Tensor._uploadToGpu(t.shape, slice));
              t._accumulateGrad(gSlice);
            }
            rowOff += r;
          }
        }
      });
    }
    return out;
  }
}
