/// Gradient regularization utilities.
library;

import 'dart:math' as math;

import '../tensor/tensor.dart';

/// Clip the global L2 norm of the gradients across `parameters` to at
/// most `maxNorm`, in place. Returns the pre-clip total norm — useful
/// for logging.
///
/// Parameters whose `.grad` is `null` are skipped. If the total norm
/// is already at or below `maxNorm`, this is a no-op.
double clipGradNorm(List<Tensor> parameters, double maxNorm) {
  if (maxNorm <= 0) {
    throw ArgumentError('clipGradNorm: maxNorm must be > 0; got $maxNorm');
  }

  double sumSq = 0.0;
  for (final p in parameters) {
    final g = p.grad;
    if (g == null) continue;
    for (final v in g.toList()) {
      sumSq += v * v;
    }
  }
  final total = math.sqrt(sumSq);
  if (total <= maxNorm || total == 0.0) return total;
  // Guard against non-finite totals: previously an `inf` from an
  // exploding gradient would silently yield `scale = 0`, zeroing
  // every parameter's gradient without warning; a `NaN` would yield
  // `scale = NaN`, poisoning every parameter with `NaN`. Both cases
  // corrupted training with no indication of what went wrong. Now
  // we short-circuit: caller sees the non-finite norm in the return
  // value and can decide whether to skip the step.
  if (!total.isFinite) return total;

  final scale = maxNorm / total;
  for (final p in parameters) {
    final g = p.grad;
    if (g == null) continue;
    final scaled = g * scale;
    // g.assign() destroys the old handle; but `g * scale` also
    // allocated a fresh handle that assign steals. No leak here.
    g.assign(scaled);
  }
  return total;
}
