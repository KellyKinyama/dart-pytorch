part of 'tensor.dart';

/// Concatenate a list of 2D tensors along the last axis.
///
/// All inputs must be 2D with the same number of rows and the same
/// device. Result has shape `[R, sum(Ci)]`. Backward is a simple slice
/// of the upstream gradient back into each input's column range.
///
/// The concat itself is done in host memory (Float32List) — for GPU
/// inputs the data is downloaded, spliced, and re-uploaded. That's
/// acceptable for small numbers of heads (typical MHA use); a fused
/// GPU kernel can replace this later without changing the API.
extension TensorConcat on Tensor {
  static Tensor concat(List<Tensor> tensors, {int axis = 1}) {
    if (tensors.isEmpty) {
      throw ArgumentError('concat: input list is empty');
    }
    if (axis != 1) {
      throw ArgumentError('concat: only axis=1 (last axis of 2D) supported');
    }

    final first = tensors[0];
    final rows = first.shape[0];
    final device = first.device;
    int totalCols = 0;
    for (final t in tensors) {
      if (t.shape.length != 2) {
        throw ArgumentError('concat: 2D inputs required; got ${t.shape}');
      }
      if (t.shape[0] != rows) {
        throw ArgumentError(
          'concat: row mismatch — expected $rows, got ${t.shape[0]}',
        );
      }
      if (t.device != device) {
        throw ArgumentError(
          'concat: device mismatch — expected $device, got ${t.device}. '
          'Call .to(...) on inputs first.',
        );
      }
      totalCols += t.shape[1];
    }

    // Gather host-side data (download from GPU if needed).
    final srcData = <Float32List>[];
    for (final t in tensors) {
      if (t.device == Device.CPU) {
        srcData.add(t._cpuData!);
      } else {
        srcData.add(Float32List.fromList(t.toList()));
      }
    }

    final outData = Float32List(rows * totalCols);
    for (int r = 0; r < rows; r++) {
      int colOff = 0;
      for (int i = 0; i < tensors.length; i++) {
        final c = tensors[i].shape[1];
        for (int j = 0; j < c; j++) {
          outData[r * totalCols + colOff + j] = srcData[i][r * c + j];
        }
        colOff += c;
      }
    }

    final anyReq = tensors.any((t) => t.requiresGrad);
    final Tensor out = device == Device.CPU
        ? Tensor._cpu([rows, totalCols], outData, requiresGrad: anyReq)
        : Tensor._gpu(
            [rows, totalCols],
            Tensor._uploadToGpu([rows, totalCols], outData),
            requiresGrad: anyReq,
          );

    if (anyReq) {
      out._setBackward(tensors, () {
        final gO = out._grad!;
        // Download upstream grad to a host buffer, slice per input.
        final gOData = gO.device == Device.CPU
            ? gO._cpuData!
            : Float32List.fromList(gO.toList());

        int colOff = 0;
        for (int i = 0; i < tensors.length; i++) {
          final t = tensors[i];
          final c = t.shape[1];
          if (t.requiresGrad) {
            final slice = Float32List(rows * c);
            for (int r = 0; r < rows; r++) {
              for (int j = 0; j < c; j++) {
                slice[r * c + j] = gOData[r * totalCols + colOff + j];
              }
            }
            final gSlice = t.device == Device.CPU
                ? Tensor._cpu(t.shape, slice)
                : Tensor._gpu(t.shape, Tensor._uploadToGpu(t.shape, slice));
            t._accumulateGrad(gSlice);
          }
          colOff += c;
        }
      });
    }
    return out;
  }
}
