/// Image classifier on top of a Vision Transformer backbone.
///
/// Wraps a [ViTBackbone] and a `Linear(embedDim, numClasses)` head
/// that reads the CLS token of the encoded sequence and produces class
/// logits `[1, numClasses]`. Combine with [`crossEntropy`] on a target
/// class index tensor `[1]` for training.
///
/// This is the simplest ViT head — good for CIFAR-style classification
/// and as a sanity test of the backbone. For dense-prediction tasks
/// (object detection, segmentation), replace this head with something
/// per-patch or add learnable query tokens.
library;

import '../../tensor/tensor.dart';
import '../linear.dart';
import '../module.dart';
import 'vit_backbone.dart';

class ViTClassifier extends Module {
  final ViTBackbone backbone;
  final Linear head;
  final int numClasses;

  ViTClassifier({
    required int imageSize,
    required int patchSize,
    int numChannels = 3,
    required int embedDim,
    required this.numClasses,
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
       head = Linear(
         embedDim,
         numClasses,
         bias: true,
         device: device,
         seed: seed + 999999,
       );

  /// `patchifiedImage` — `[numPatches, patchSize * patchSize *
  /// numChannels]`. Returns class logits `[1, numClasses]`.
  Tensor call(Tensor patchifiedImage) {
    final encoded = backbone(patchifiedImage);
    final cls = vitClsFeature(encoded); // [1, embedDim]
    return head(cls); // [1, numClasses]
  }

  @override
  List<Tensor> parameters() => [...backbone.parameters(), ...head.parameters()];

  @override
  List<Module> submodules() => [backbone, head];
}
