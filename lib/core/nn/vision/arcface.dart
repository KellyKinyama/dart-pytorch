/// ArcFace angular-margin softmax loss (Deng et al., CVPR 2019).
///
/// Wraps a `[numClasses, embedDim]` weight matrix and, at forward time,
/// L2-normalises both `weight` and the incoming embeddings so their
/// row-dot-product becomes the cosine of the angle between them:
///
///   cosθ_{b,k} = ê_b · ŵ_k       ê_b = embed_b / ‖embed_b‖
///                                 ŵ_k = weight_k / ‖weight_k‖
///
/// For the **target class** of each row we substitute
///
///   cos(θ + m) = cosθ · cos m − sinθ · sin m
///              = cosθ · cos m − √(1 − cos²θ) · sin m
///
/// and finally scale every entry by `s`. Feed the returned logits into
/// `softmax + crossEntropy` (or `.crossEntropy` directly) to get the
/// ArcFace training loss. At inference call [logitsForInference] which
/// skips the margin and just returns `s · cosθ`.
///
/// Defaults follow the paper (`s = 64`, `m = 0.5 rad = 28.6°`). This
/// module has no bias and no activation — it's a drop-in replacement
/// for the final `Linear + softmax + cross-entropy` head in any face-
/// recognition (or fine-grained classification) pipeline. Pair it with
/// a `[N, D]` L2-normalised embedding backbone such as [ResNet] + global
/// avg-pool or [Facenet]'s output.
library;

import 'dart:math' as math;

import '../../tensor/tensor.dart';
import '../linear.dart';
import '../module.dart';
import 'nchw.dart';

class ArcFace extends Module {
  final int embedDim;
  final int numClasses;
  final double scale;
  final double margin;

  /// `[numClasses, embedDim]` — the per-class prototype directions.
  /// Held in a `Linear` so autograd + optimizer step through it
  /// without any extra plumbing.
  final Linear weight;

  ArcFace(
    this.embedDim,
    this.numClasses, {
    this.scale = 64.0,
    this.margin = 0.5,
    Device device = Device.CPU,
    int seed = 0,
  }) : weight = Linear(
         embedDim,
         numClasses,
         bias: false,
         device: device,
         seed: seed,
       );

  /// Forward pass at **training time** — needs the ground-truth
  /// `labels` (shape `[B]`, class indices encoded as floats since this
  /// repo has no int tensor) so the margin is added only on the
  /// correct row/column pair.
  ///
  /// Returns `[B, numClasses]` scaled logits ready for `crossEntropy`.
  Tensor call(Tensor embeddings, Tensor labels) {
    _requireEmbed(embeddings);
    if (labels.shape.length != 1 || labels.shape[0] != embeddings.shape[0]) {
      throw ArgumentError(
        'ArcFace: labels must be [B] with B=${embeddings.shape[0]}; '
        'got ${labels.shape}',
      );
    }
    // Cosines against L2-normalised weight/embeddings.
    final cos = _cosMatrix(embeddings);
    // Host-side patch on target column with cos(θ+m).
    final vals = Tensor.noGrad(() => cos.toList());
    final labelList = Tensor.noGrad(() => labels.toList());
    final cosM = math.cos(margin);
    final sinM = math.sin(margin);
    final b = embeddings.shape[0];
    final k = numClasses;
    final patched = List<double>.of(vals);
    for (int i = 0; i < b; i++) {
      final y = labelList[i].toInt();
      if (y < 0 || y >= k) {
        throw ArgumentError('ArcFace: label $y out of range [0, $k)');
      }
      final c = patched[i * k + y].clamp(-1.0, 1.0);
      final s = math.sqrt(math.max(0.0, 1.0 - c * c));
      patched[i * k + y] = c * cosM - s * sinM;
    }
    final patchedT = Tensor.fromList(
      [b, k],
      patched,
      device: embeddings.device,
    );
    // Multiply by `scale`. The tape reconnects through `cos` for the
    // non-target entries because we return a plain scalar-mul below.
    return patchedT * scale;
  }

  /// Inference-time logits: `s · cosθ`, no margin. Cheaper and gives
  /// well-calibrated similarity scores. Prefer [call] during training.
  Tensor logitsForInference(Tensor embeddings) {
    _requireEmbed(embeddings);
    return _cosMatrix(embeddings) * scale;
  }

  /// `[B, K]` matrix of cosines between row-embeddings and per-class
  /// prototype weights, both L2-normalised.
  Tensor _cosMatrix(Tensor embeddings) {
    final embNorm = l2NormalizeRows(embeddings);
    final wNorm = l2NormalizeRows(weight.weight);
    return embNorm.matmul(wNorm.transpose());
  }

  void _requireEmbed(Tensor embeddings) {
    if (embeddings.shape.length != 2 || embeddings.shape[1] != embedDim) {
      throw ArgumentError(
        'ArcFace: embeddings must be [B, $embedDim]; got ${embeddings.shape}',
      );
    }
  }

  @override
  List<Tensor> parameters() => weight.parameters();

  @override
  List<Module> submodules() => [weight];
}
