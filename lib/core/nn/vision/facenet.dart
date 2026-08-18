/// FaceNet Inception-ResNet-V1 backbone — inference + head-only fine-tune.
///
/// Ports `facenet_pytorch.models.inception_resnet_v1.InceptionResnetV1`
/// (`pretrained='vggface2'`). Input `[N, 3, 160, 160]` (already
/// cropped + aligned) → `[N, 512]` L2-normalized embedding.
///
/// **BN is folded into the preceding Conv2d at load time**
/// ([`conv_bn_fold.dart`](conv_bn_fold.dart)), so at runtime every
/// step is either a `Conv2d`, a `Linear`, a ReLU, a channel-concat,
/// a pool, or a residual add. The folded conv weights carry
/// `requiresGrad: true` — the *last* Linear (and any user-added head
/// on top) can be fine-tuned with autograd today. Gradients into the
/// frozen conv stack require a Conv2d autograd rewrite (see
/// `doc/facenet.md`).
library;

import '../../tensor/tensor.dart';
import '../conv2d.dart';
import '../dropout.dart';
import '../linear.dart';
import '../module.dart';
import 'nchw.dart';
import 'pool2d.dart';

/// A `Conv2d(bias=False) + BatchNorm2d(eps=1e-3) + ReLU` triple.
/// Folded into a single Conv2d(bias=True) at load time; ReLU stays.
class BasicConv2d extends Module {
  final Conv2d conv;
  final bool applyRelu;

  BasicConv2d(
    int inChannels,
    int outChannels, {
    int kernel = 3,
    int? kernelH,
    int? kernelW,
    int stride = 1,
    int padding = 0,
    int? paddingH,
    int? paddingW,
    this.applyRelu = true,
    Device device = Device.CPU,
  }) : conv = Conv2d(
         inChannels,
         outChannels,
         kernel: kernel,
         kernelH: kernelH,
         kernelW: kernelW,
         stride: stride,
         padding: padding,
         paddingH: paddingH,
         paddingW: paddingW,
         bias: true,
         device: device,
       );

  Tensor call(Tensor x) {
    var y = conv(x);
    if (applyRelu) y = y.relu();
    return y;
  }

  @override
  List<Tensor> parameters() => conv.parameters();
}

/// Inception-ResNet-A block. Input/output `[N, 256, H, W]`.
class Block35 extends Module {
  final double scale;
  final BasicConv2d branch0;
  final BasicConv2d branch1a;
  final BasicConv2d branch1b;
  final BasicConv2d branch2a;
  final BasicConv2d branch2b;
  final BasicConv2d branch2c;
  final Conv2d conv;

  Block35({this.scale = 1.0, Device device = Device.CPU})
    : branch0 = BasicConv2d(256, 32, kernel: 1, device: device),
      branch1a = BasicConv2d(256, 32, kernel: 1, device: device),
      branch1b = BasicConv2d(32, 32, kernel: 3, padding: 1, device: device),
      branch2a = BasicConv2d(256, 32, kernel: 1, device: device),
      branch2b = BasicConv2d(32, 32, kernel: 3, padding: 1, device: device),
      branch2c = BasicConv2d(32, 32, kernel: 3, padding: 1, device: device),
      conv = Conv2d(96, 256, kernel: 1, bias: true, device: device);

  Tensor call(Tensor x) {
    final b0 = branch0(x);
    final b1 = branch1b(branch1a(x));
    final b2 = branch2c(branch2b(branch2a(x)));
    final mixed = catChannels([b0, b1, b2]);
    final up = conv(mixed);
    final out = x + up * scale;
    return out.relu();
  }

  @override
  List<Tensor> parameters() => [
    ...branch0.parameters(),
    ...branch1a.parameters(),
    ...branch1b.parameters(),
    ...branch2a.parameters(),
    ...branch2b.parameters(),
    ...branch2c.parameters(),
    ...conv.parameters(),
  ];
}

/// Inception-ResNet-B block. Input/output `[N, 896, H, W]`.
class Block17 extends Module {
  final double scale;
  final BasicConv2d branch0;
  final BasicConv2d branch1a;
  final BasicConv2d branch1b;
  final BasicConv2d branch1c;
  final Conv2d conv;

  Block17({this.scale = 1.0, Device device = Device.CPU})
    : branch0 = BasicConv2d(896, 128, kernel: 1, device: device),
      branch1a = BasicConv2d(896, 128, kernel: 1, device: device),
      branch1b = BasicConv2d(
        128,
        128,
        kernelH: 1,
        kernelW: 7,
        paddingH: 0,
        paddingW: 3,
        device: device,
      ),
      branch1c = BasicConv2d(
        128,
        128,
        kernelH: 7,
        kernelW: 1,
        paddingH: 3,
        paddingW: 0,
        device: device,
      ),
      conv = Conv2d(256, 896, kernel: 1, bias: true, device: device);

  Tensor call(Tensor x) {
    final b0 = branch0(x);
    final b1 = branch1c(branch1b(branch1a(x)));
    final mixed = catChannels([b0, b1]);
    final up = conv(mixed);
    final out = x + up * scale;
    return out.relu();
  }

  @override
  List<Tensor> parameters() => [
    ...branch0.parameters(),
    ...branch1a.parameters(),
    ...branch1b.parameters(),
    ...branch1c.parameters(),
    ...conv.parameters(),
  ];
}

/// Inception-ResNet-C block. Input/output `[N, 1792, H, W]`.
class Block8 extends Module {
  final double scale;
  final bool applyRelu;
  final BasicConv2d branch0;
  final BasicConv2d branch1a;
  final BasicConv2d branch1b;
  final BasicConv2d branch1c;
  final Conv2d conv;

  Block8({this.scale = 1.0, this.applyRelu = true, Device device = Device.CPU})
    : branch0 = BasicConv2d(1792, 192, kernel: 1, device: device),
      branch1a = BasicConv2d(1792, 192, kernel: 1, device: device),
      branch1b = BasicConv2d(
        192,
        192,
        kernelH: 1,
        kernelW: 3,
        paddingH: 0,
        paddingW: 1,
        device: device,
      ),
      branch1c = BasicConv2d(
        192,
        192,
        kernelH: 3,
        kernelW: 1,
        paddingH: 1,
        paddingW: 0,
        device: device,
      ),
      conv = Conv2d(384, 1792, kernel: 1, bias: true, device: device);

  Tensor call(Tensor x) {
    final b0 = branch0(x);
    final b1 = branch1c(branch1b(branch1a(x)));
    final mixed = catChannels([b0, b1]);
    final up = conv(mixed);
    var out = x + up * scale;
    if (applyRelu) out = out.relu();
    return out;
  }

  @override
  List<Tensor> parameters() => [
    ...branch0.parameters(),
    ...branch1a.parameters(),
    ...branch1b.parameters(),
    ...branch1c.parameters(),
    ...conv.parameters(),
  ];
}

/// Reduction-A: `[N, 256, 17, 17]` → `[N, 896, 8, 8]`.
class Mixed6a extends Module {
  final BasicConv2d branch0;
  final BasicConv2d branch1a;
  final BasicConv2d branch1b;
  final BasicConv2d branch1c;

  Mixed6a({Device device = Device.CPU})
    : branch0 = BasicConv2d(256, 384, kernel: 3, stride: 2, device: device),
      branch1a = BasicConv2d(256, 192, kernel: 1, device: device),
      branch1b = BasicConv2d(192, 192, kernel: 3, padding: 1, device: device),
      branch1c = BasicConv2d(192, 256, kernel: 3, stride: 2, device: device);

  Tensor call(Tensor x) {
    final b0 = branch0(x);
    final b1 = branch1c(branch1b(branch1a(x)));
    final b2 = maxPool2d(x, kernel: 3, stride: 2);
    return catChannels([b0, b1, b2]);
  }

  @override
  List<Tensor> parameters() => [
    ...branch0.parameters(),
    ...branch1a.parameters(),
    ...branch1b.parameters(),
    ...branch1c.parameters(),
  ];
}

/// Reduction-B: `[N, 896, 8, 8]` → `[N, 1792, 3, 3]`.
class Mixed7a extends Module {
  final BasicConv2d branch0a;
  final BasicConv2d branch0b;
  final BasicConv2d branch1a;
  final BasicConv2d branch1b;
  final BasicConv2d branch2a;
  final BasicConv2d branch2b;
  final BasicConv2d branch2c;

  Mixed7a({Device device = Device.CPU})
    : branch0a = BasicConv2d(896, 256, kernel: 1, device: device),
      branch0b = BasicConv2d(256, 384, kernel: 3, stride: 2, device: device),
      branch1a = BasicConv2d(896, 256, kernel: 1, device: device),
      branch1b = BasicConv2d(256, 256, kernel: 3, stride: 2, device: device),
      branch2a = BasicConv2d(896, 256, kernel: 1, device: device),
      branch2b = BasicConv2d(256, 256, kernel: 3, padding: 1, device: device),
      branch2c = BasicConv2d(256, 256, kernel: 3, stride: 2, device: device);

  Tensor call(Tensor x) {
    final b0 = branch0b(branch0a(x));
    final b1 = branch1b(branch1a(x));
    final b2 = branch2c(branch2b(branch2a(x)));
    final b3 = maxPool2d(x, kernel: 3, stride: 2);
    return catChannels([b0, b1, b2, b3]);
  }

  @override
  List<Tensor> parameters() => [
    ...branch0a.parameters(),
    ...branch0b.parameters(),
    ...branch1a.parameters(),
    ...branch1b.parameters(),
    ...branch2a.parameters(),
    ...branch2b.parameters(),
    ...branch2c.parameters(),
  ];
}

/// InceptionResnetV1 — VGGFace2 flavour, 512-d L2-normalized embeddings.
class InceptionResnetV1 extends Module {
  final Device device;

  // Stem.
  final BasicConv2d conv2d1a; // 3 → 32, k=3, s=2
  final BasicConv2d conv2d2a; // 32 → 32, k=3
  final BasicConv2d conv2d2b; // 32 → 64, k=3, p=1
  final BasicConv2d conv2d3b; // 64 → 80, k=1
  final BasicConv2d conv2d4a; // 80 → 192, k=3
  final BasicConv2d conv2d4b; // 192 → 256, k=3, s=2

  // Repeats.
  final List<Block35> repeat1; // × 5
  final Mixed6a mixed6a;
  final List<Block17> repeat2; // × 10
  final Mixed7a mixed7a;
  final List<Block8> repeat3; // × 5
  final Block8 block8Final; // scale=1.0, noReLU

  // Head.
  final Dropout dropout;
  final Linear lastLinear; // 1792 → 512, bias=false

  /// Folded [`last_bn`] parameters (`BatchNorm1d(512, eps=1e-3,
  /// affine=False)`). Folded into an affine on the 512-d embedding:
  ///   y = (x - mean) / sqrt(var + eps)
  /// which we store as `scale` (`1/sqrt(var+eps)`) and `offset`
  /// (`-mean * scale`), applied elementwise.
  late Tensor lastBnScale; // [1, 512]
  late Tensor lastBnOffset; // [1, 512]

  InceptionResnetV1({this.device = Device.CPU, double dropoutP = 0.6})
    : conv2d1a = BasicConv2d(3, 32, kernel: 3, stride: 2, device: device),
      conv2d2a = BasicConv2d(32, 32, kernel: 3, device: device),
      conv2d2b = BasicConv2d(32, 64, kernel: 3, padding: 1, device: device),
      conv2d3b = BasicConv2d(64, 80, kernel: 1, device: device),
      conv2d4a = BasicConv2d(80, 192, kernel: 3, device: device),
      conv2d4b = BasicConv2d(192, 256, kernel: 3, stride: 2, device: device),
      repeat1 = List.generate(5, (_) => Block35(scale: 0.17, device: device)),
      mixed6a = Mixed6a(device: device),
      repeat2 = List.generate(10, (_) => Block17(scale: 0.10, device: device)),
      mixed7a = Mixed7a(device: device),
      repeat3 = List.generate(5, (_) => Block8(scale: 0.20, device: device)),
      block8Final = Block8(scale: 1.0, applyRelu: false, device: device),
      dropout = Dropout(dropoutP),
      lastLinear = Linear(1792, 512, bias: false, device: device) {
    lastBnScale = Tensor.fill([1, 512], 1.0, device: device);
    lastBnOffset = Tensor.fill([1, 512], 0.0, device: device);
  }

  /// Forward pass on `[N, 3, 160, 160]` → L2-normalized `[N, 512]`.
  Tensor call(Tensor x) {
    if (x.shape.length != 4 ||
        x.shape[1] != 3 ||
        x.shape[2] != 160 ||
        x.shape[3] != 160) {
      throw ArgumentError(
        'InceptionResnetV1: expected [N, 3, 160, 160]; got ${x.shape}',
      );
    }
    var h = conv2d1a(x); //   79x79
    h = conv2d2a(h); //         77x77
    h = conv2d2b(h); //         77x77
    h = maxPool2d(h, kernel: 3, stride: 2); // 38x38
    h = conv2d3b(h); //         38x38
    h = conv2d4a(h); //         36x36
    h = conv2d4b(h); //         17x17

    for (final b in repeat1) {
      h = b(h);
    }
    h = mixed6a(h); //          8x8
    for (final b in repeat2) {
      h = b(h);
    }
    h = mixed7a(h); //          3x3
    for (final b in repeat3) {
      h = b(h);
    }
    h = block8Final(h);

    final pooled = globalAvgPool2d(h); // [N, 1792]
    final dropped = dropout(pooled);
    var emb = lastLinear(dropped); // [N, 512]
    emb = emb * lastBnScale + lastBnOffset;
    return l2NormalizeRows(emb);
  }

  @override
  List<Tensor> parameters() => [
    ...conv2d1a.parameters(),
    ...conv2d2a.parameters(),
    ...conv2d2b.parameters(),
    ...conv2d3b.parameters(),
    ...conv2d4a.parameters(),
    ...conv2d4b.parameters(),
    for (final b in repeat1) ...b.parameters(),
    ...mixed6a.parameters(),
    for (final b in repeat2) ...b.parameters(),
    ...mixed7a.parameters(),
    for (final b in repeat3) ...b.parameters(),
    ...block8Final.parameters(),
    ...lastLinear.parameters(),
  ];

  @override
  List<Module> submodules() => [dropout];
}
