part of 'tensor.dart';

/// Scaled dot-product attention (single head, 2D).
///
/// Given `Q` shape `[N, Dk]`, `K` shape `[M, Dk]`, `V` shape `[M, Dv]`:
///
///     scores = Q @ K.T * (1 / sqrt(Dk))    // [N, M]
///     attn   = softmax(scores)             // [N, M], row-wise
///     out    = attn @ V                    // [N, Dv]
///
/// Optionally accepts an additive `mask` broadcast to `[N, M]` — typical
/// values are `0` for allowed positions and a large negative number
/// (e.g. `-1e9`) for masked positions. `mask` must be same-shape as
/// `scores`; broadcasting is not implied.
///
/// Autograd is fully wired via the underlying ops — no custom backward.
extension TensorAttention on Tensor {
  Tensor scaledDotProductAttention(Tensor k, Tensor v, {Tensor? mask}) {
    if (shape.length != 2 || k.shape.length != 2 || v.shape.length != 2) {
      throw ArgumentError(
        'scaledDotProductAttention: expected 2D q/k/v; '
        'got q=$shape k=${k.shape} v=${v.shape}',
      );
    }
    final dk = shape[1];
    if (k.shape[1] != dk) {
      throw ArgumentError(
        'scaledDotProductAttention: q and k must share last dim; '
        'got q=$shape k=${k.shape}',
      );
    }
    if (v.shape[0] != k.shape[0]) {
      throw ArgumentError(
        'scaledDotProductAttention: k and v must share first dim; '
        'got k=${k.shape} v=${v.shape}',
      );
    }

    final scale = 1.0 / math.sqrt(dk);
    var scores = matmul(k.transpose()) * scale;
    if (mask != null) {
      if (mask.shape.length != 2 ||
          mask.shape[0] != scores.shape[0] ||
          mask.shape[1] != scores.shape[1]) {
        throw ArgumentError(
          'scaledDotProductAttention: mask must be ${scores.shape}; '
          'got ${mask.shape}',
        );
      }
      scores = scores + mask;
    }
    final attn = scores.softmax();
    return attn.matmul(v);
  }
}
