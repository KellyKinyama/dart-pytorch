/// ResNet-50 (torchvision "V1.5") for ImageNet classification.
///
/// Input `[N, 3, 224, 224]` after ImageNet normalization
/// (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) →
/// `[N, 1000]` logits.
///
/// Architecture (torchvision `resnet50` default):
///
///   * Stem: Conv 7×7 s=2 pad=3 3→64 + BN + ReLU + MaxPool 3×3 s=2 pad=1
///   * layer1: 3× [Bottleneck 64→64→256]  (first block downsamples 64→256)
///   * layer2: 4× [Bottleneck 256→128→512]  (first block s=2)
///   * layer3: 6× [Bottleneck 512→256→1024] (first block s=2)
///   * layer4: 3× [Bottleneck 1024→512→2048](first block s=2)
///   * head: GlobalAvgPool → Linear 2048→1000
///
/// **Stride-2 goes on the 3×3 conv** (torchvision "V1.5" / He et al.
/// "Bag of Tricks" convention). Classical He-et-al V1 puts it on the
/// downsample 1×1 — we don't support that variant; both share the same
/// weight shapes so accidentally loading a V1 checkpoint won't error,
/// it'll just misalign.
///
/// **BN is folded into the preceding Conv2d at load time** via
/// [ConvBnFold]. At runtime every step is Conv2d, ReLU, MaxPool,
/// AdaptiveAvgPool, elementwise add, or Linear.
///
/// Load with [ResNetLoader.loadFile] from a
/// `scripts/convert_resnet50_pt_to_safetensors.py`-produced safetensors
/// (torchvision `state_dict` keys — see the loader for the mapping).
library;

import '../../tensor/tensor.dart';
import '../conv2d.dart';
import '../linear.dart';
import '../module.dart';
import 'pool2d.dart';

/// `Conv2d(bias=false) + BN + optional ReLU` — after ConvBnFold this
/// is a single `Conv2d(bias=true)` followed by optional ReLU.
class ConvBn extends Module {
  final Conv2d conv;
  final bool applyRelu;

  ConvBn(
    int inC,
    int outC, {
    int kernel = 3,
    int stride = 1,
    int padding = 0,
    this.applyRelu = true,
    Device device = Device.CPU,
    int seed = 0,
  }) : conv = Conv2d(
         inC,
         outC,
         kernel: kernel,
         stride: stride,
         padding: padding,
         bias: true,
         device: device,
         seed: seed,
       );

  Tensor call(Tensor x) {
    var y = conv(x);
    if (applyRelu) y = y.relu();
    return y;
  }

  @override
  List<Tensor> parameters() => conv.parameters();
}

/// Bottleneck block: 1×1 → 3×3 → 1×1 with a residual (identity or
/// projection) skip and a final ReLU. Expansion factor is fixed at 4
/// (per He et al.); i.e. `outChannels = midChannels * 4`.
class Bottleneck extends Module {
  final ConvBn conv1;
  final ConvBn conv2;
  final ConvBn conv3; // last conv: no ReLU (comes after the skip add)
  final ConvBn? downsample;

  Bottleneck(
    int inChannels,
    int midChannels, {
    int stride = 1,
    Device device = Device.CPU,
    int seed = 0,
  }) : conv1 = ConvBn(
         inChannels,
         midChannels,
         kernel: 1,
         stride: 1,
         padding: 0,
         applyRelu: true,
         device: device,
         seed: seed,
       ),
       conv2 = ConvBn(
         midChannels,
         midChannels,
         kernel: 3,
         stride: stride,
         padding: 1,
         applyRelu: true,
         device: device,
         seed: seed + 1,
       ),
       conv3 = ConvBn(
         midChannels,
         midChannels * 4,
         kernel: 1,
         stride: 1,
         padding: 0,
         applyRelu: false,
         device: device,
         seed: seed + 2,
       ),
       downsample = (stride != 1 || inChannels != midChannels * 4)
           ? ConvBn(
               inChannels,
               midChannels * 4,
               kernel: 1,
               stride: stride,
               padding: 0,
               applyRelu: false,
               device: device,
               seed: seed + 3,
             )
           : null;

  Tensor call(Tensor x) {
    final identity = downsample == null ? x : downsample!(x);
    final y = conv3(conv2(conv1(x)));
    return (y + identity).relu();
  }

  @override
  List<Tensor> parameters() => [
    ...conv1.parameters(),
    ...conv2.parameters(),
    ...conv3.parameters(),
    if (downsample != null) ...downsample!.parameters(),
  ];

  @override
  List<Module> submodules() => [
    conv1,
    conv2,
    conv3,
    if (downsample != null) downsample!,
  ];

  // Exposed for the loader.
  ConvBn get conv1Ref => conv1;
  ConvBn get conv2Ref => conv2;
  ConvBn get conv3Ref => conv3;
  ConvBn? get downsampleRef => downsample;
}

class ResNetConfig {
  final List<int> blocksPerStage; // e.g. [3, 4, 6, 3] for ResNet-50
  final int numClasses;
  final Device device;
  final int seed;

  const ResNetConfig({
    required this.blocksPerStage,
    this.numClasses = 1000,
    this.device = Device.CPU,
    this.seed = 0,
  });

  /// Standard ImageNet ResNet-50 config.
  const ResNetConfig.resnet50({
    this.numClasses = 1000,
    this.device = Device.CPU,
    this.seed = 0,
  }) : blocksPerStage = const [3, 4, 6, 3];
}

class ResNet extends Module {
  final ResNetConfig config;

  final ConvBn stem; // 7x7 s=2 pad=3, 3->64 + BN + ReLU
  final List<Bottleneck> layer1;
  final List<Bottleneck> layer2;
  final List<Bottleneck> layer3;
  final List<Bottleneck> layer4;
  final Linear fc;

  ResNet(this.config)
    : stem = ConvBn(
        3,
        64,
        kernel: 7,
        stride: 2,
        padding: 3,
        applyRelu: true,
        device: config.device,
        seed: config.seed,
      ),
      layer1 = _makeStage(
        inChannels: 64,
        midChannels: 64,
        blocks: config.blocksPerStage[0],
        stride: 1,
        device: config.device,
        seed: config.seed + 1_000,
      ),
      layer2 = _makeStage(
        inChannels: 256,
        midChannels: 128,
        blocks: config.blocksPerStage[1],
        stride: 2,
        device: config.device,
        seed: config.seed + 2_000,
      ),
      layer3 = _makeStage(
        inChannels: 512,
        midChannels: 256,
        blocks: config.blocksPerStage[2],
        stride: 2,
        device: config.device,
        seed: config.seed + 3_000,
      ),
      layer4 = _makeStage(
        inChannels: 1024,
        midChannels: 512,
        blocks: config.blocksPerStage[3],
        stride: 2,
        device: config.device,
        seed: config.seed + 4_000,
      ),
      fc = Linear(
        2048,
        config.numClasses,
        bias: true,
        device: config.device,
        seed: config.seed + 900_000,
      );

  static List<Bottleneck> _makeStage({
    required int inChannels,
    required int midChannels,
    required int blocks,
    required int stride,
    required Device device,
    required int seed,
  }) {
    final out = <Bottleneck>[];
    for (int i = 0; i < blocks; i++) {
      out.add(
        Bottleneck(
          i == 0 ? inChannels : midChannels * 4,
          midChannels,
          stride: i == 0 ? stride : 1,
          device: device,
          seed: seed + i * 10,
        ),
      );
    }
    return out;
  }

  /// Forward pass. `x` is `[N, 3, 224, 224]`. Returns `[N, numClasses]`.
  Tensor call(Tensor x) {
    if (x.shape.length != 4 || x.shape[1] != 3) {
      throw ArgumentError('ResNet: expected [N, 3, H, W]; got ${x.shape}');
    }
    var h = stem(x);
    h = maxPool2d(h, kernel: 3, stride: 2, padding: 1);
    for (final b in layer1) {
      h = b(h);
    }
    for (final b in layer2) {
      h = b(h);
    }
    for (final b in layer3) {
      h = b(h);
    }
    for (final b in layer4) {
      h = b(h);
    }
    final pooled = globalAvgPool2d(h); // [N, 2048]
    return fc(pooled);
  }

  @override
  List<Tensor> parameters() => [
    ...stem.parameters(),
    for (final b in layer1) ...b.parameters(),
    for (final b in layer2) ...b.parameters(),
    for (final b in layer3) ...b.parameters(),
    for (final b in layer4) ...b.parameters(),
    ...fc.parameters(),
  ];

  @override
  List<Module> submodules() => [
    stem,
    ...layer1,
    ...layer2,
    ...layer3,
    ...layer4,
    fc,
  ];
}
