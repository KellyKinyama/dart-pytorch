/// DETR-style multi-object detector on top of a Vision Transformer.
///
/// Reads the CLS token of a [ViTBackbone] and splits it into two
/// heads:
///
///   * `classHead : Linear(embedDim, numQueries * (numClasses + 1))`
///     — per-query classification logits. The extra `+1` class slot
///     is a "no object / background" label (index `numClasses`).
///   * `boxHead   : Linear(embedDim, numQueries * 4)` — per-query
///     bounding box `(x, y, w, h)` in normalized coordinates. A
///     sigmoid is applied so predictions stay in `[0, 1]`.
///
/// Train it in one of two flavors:
///
///   1. **Fixed-order targets** — the caller pre-orders GT boxes to
///      match query slots and uses plain MSE + `crossEntropy`. See
///      `bin/vit_object_detection_demo.dart`.
///   2. **Hungarian matching** — for each step, compute a cost matrix
///      between the `numQueries` predictions and the (variable-count)
///      GT objects, run [HungarianAlgorithm], build a *masked* target
///      tensor (unmatched queries → `numClasses` background,
///      `boxMask=0`), and apply a weighted loss. See
///      `bin/vit_hungarian_matching_demo.dart`.
///
/// This is a **single-image** module: `[numPatches, patchPixels]` in,
/// `[numQueries, numClasses+1]` + `[numQueries, 4]` out.
library;

import '../../tensor/tensor.dart';
import '../linear.dart';
import '../module.dart';
import 'vit_backbone.dart';

class ViTObjectDetector extends Module {
  final ViTBackbone backbone;
  final Linear classHead;
  final Linear boxHead;
  final int numQueries;
  final int numClasses;
  final int embedDim;

  ViTObjectDetector({
    required int imageSize,
    required int patchSize,
    int numChannels = 3,
    required this.embedDim,
    required this.numClasses,
    required this.numQueries,
    int numLayers = 4,
    int numHeads = 4,
    double dropoutP = 0.0,
    Device device = Device.CPU,
    int seed = 0,
  }) : backbone = ViTBackbone(
         imageSize: imageSize,
         patchSize: patchSize,
         numChannels: numChannels,
         embedDim: embedDim,
         numLayers: numLayers,
         numHeads: numHeads,
         dropoutP: dropoutP,
         device: device,
         seed: seed,
       ),
       classHead = Linear(
         embedDim,
         numQueries * (numClasses + 1),
         bias: true,
         device: device,
         seed: seed + 111111,
       ),
       boxHead = Linear(
         embedDim,
         numQueries * 4,
         bias: true,
         device: device,
         seed: seed + 222222,
       );

  /// Forward pass. Returns:
  ///
  ///   * `'logits'` — `[numQueries, numClasses + 1]` unnormalized
  ///     class scores (the extra class is background).
  ///   * `'boxes'`  — `[numQueries, 4]` post-sigmoid box parameters
  ///     in `[0, 1]`.
  Map<String, Tensor> call(Tensor patchifiedImage) {
    final encoded = backbone(patchifiedImage);
    final cls = vitClsFeature(encoded); // [1, embedDim]

    final rawLogits = classHead(cls); // [1, numQueries*(numClasses+1)]
    final logits = rawLogits.reshape([numQueries, numClasses + 1]);

    final rawBoxes = boxHead(cls); // [1, numQueries*4]
    final boxes = rawBoxes.reshape([numQueries, 4]).sigmoid();

    return {'logits': logits, 'boxes': boxes};
  }

  @override
  List<Tensor> parameters() => [
    ...backbone.parameters(),
    ...classHead.parameters(),
    ...boxHead.parameters(),
  ];

  @override
  List<Module> submodules() => [backbone, classHead, boxHead];
}
