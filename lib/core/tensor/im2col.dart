part of 'tensor.dart';

/// im2col unfold for 8x8 NHWC-flat activations used by the LC0 conv
/// tower.
///
/// `this` is `[B*64, Cin]` (NHWC-flat). Result is `[B*64, Cin*k*k]`
/// with the same `(ci, ky, kx)` inner ordering the LC0 weight layout
/// `[Cout, Cin, Kh, Kw]` expects, so a subsequent
/// `.matmul(wT [Cin*k*k, Cout])` produces `[B*64, Cout]` directly.
///
/// GPU path: single fused CUDA kernel (`im2col_nhwc_op`). CPU path:
/// Dart nested loops — kept mostly as a reference and for the
/// `Device.CPU` engine variant.
extension TensorIm2ColNHWC on Tensor {
  Tensor im2colNhwc({
    required int batch,
    required int cin,
    required int k,
    required int pad,
  }) {
    const w = 8;
    const h = 8;
    final rowsPerBatch = h * w;
    final rows = batch * rowsPerBatch;
    final colsWidth = cin * k * k;

    if (device == Device.GPU) {
      return Tensor._gpu([
        rows,
        colsWidth,
      ], engine.im2colNhwc(_handle!, batch, cin, k, pad));
    }

    final data = toFloat32List();
    final out = Float32List(rows * colsWidth);
    var rowOff = 0;
    for (int bi = 0; bi < batch; bi++) {
      final batchBase = bi * rowsPerBatch * cin;
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          var colOff = rowOff;
          for (int c = 0; c < cin; c++) {
            for (int ky = 0; ky < k; ky++) {
              final yin = y + ky - pad;
              for (int kx = 0; kx < k; kx++) {
                final xin = x + kx - pad;
                if (yin >= 0 && yin < h && xin >= 0 && xin < w) {
                  out[colOff] = data[batchBase + (yin * w + xin) * cin + c];
                }
                colOff++;
              }
            }
          }
          rowOff += colsWidth;
        }
      }
    }
    return Tensor.fromFloat32List([rows, colsWidth], out, device: Device.CPU);
  }
}
