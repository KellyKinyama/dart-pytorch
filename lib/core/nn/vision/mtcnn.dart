/// MTCNN face detector — Zhang et al. 2016, weights from
/// facenet-pytorch. Three cascaded networks:
///
///   * **PNet** — fully-convolutional face proposal network. Slides
///     a 12×12 receptive field over the input at multiple scales,
///     produces per-position face probability + bbox regression.
///   * **RNet** — 24×24 CNN, filters PNet proposals.
///   * **ONet** — 48×48 CNN, produces final bbox + 5 facial landmarks.
///
/// Between stages: NMS + bbox regression + crop.
///
/// Weights convention (facenet-pytorch, dumped via
/// `scripts/convert_mtcnn_pt_to_safetensors.py`):
///
///   pnet: conv{1,2,3}.{weight,bias}, prelu{1,2,3}.weight,
///         conv4_1 (face prob), conv4_2 (bbox reg)
///   rnet: conv{1,2,3}.{weight,bias}, prelu{1,2,3}.weight,
///         dense4.{weight,bias}, prelu4.weight,
///         dense5_1 (face prob), dense5_2 (bbox reg)
///   onet: conv{1,2,3,4}.{weight,bias}, prelu{1,2,3,4}.weight,
///         dense5.{weight,bias}, prelu5.weight,
///         dense6_1 (face prob), dense6_2 (bbox reg),
///         dense6_3 (5 landmarks)
///
/// The `dense{4,5}` weights expect the flattened conv output in
/// **NHWC-with-W-outer** layout (`x.permute(0, 3, 2, 1).flatten(1)`
/// in facenet-pytorch). We replicate that ordering exactly via a
/// host-side scatter — otherwise the FC output is silently garbage.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../../tensor/tensor.dart';
import '../conv2d.dart';
import '../linear.dart';
import '../module.dart';
import '../prelu.dart';
import 'pool2d.dart';

/// Proposal Network. Fully convolutional: input `[N, 3, H, W]` →
/// `probs [N, 2, H', W']` and `regs [N, 4, H', W']`.
class PNet extends Module {
  final Conv2d conv1;
  final PReLU prelu1;
  final Conv2d conv2;
  final PReLU prelu2;
  final Conv2d conv3;
  final PReLU prelu3;
  final Conv2d conv4_1; // face probs (before softmax)
  final Conv2d conv4_2; // bbox regression

  PNet({Device device = Device.CPU})
    : conv1 = Conv2d(3, 10, kernel: 3, bias: true, device: device),
      prelu1 = PReLU(10, device: device),
      conv2 = Conv2d(10, 16, kernel: 3, bias: true, device: device),
      prelu2 = PReLU(16, device: device),
      conv3 = Conv2d(16, 32, kernel: 3, bias: true, device: device),
      prelu3 = PReLU(32, device: device),
      conv4_1 = Conv2d(32, 2, kernel: 1, bias: true, device: device),
      conv4_2 = Conv2d(32, 4, kernel: 1, bias: true, device: device);

  /// Returns `(probs, regs)` — both `[N, C, H', W']` with C=2 and 4.
  (Tensor, Tensor) call(Tensor x) {
    var h = prelu1(conv1(x));
    h = maxPool2d(h, kernel: 2, stride: 2, ceilMode: true);
    h = prelu2(conv2(h));
    h = prelu3(conv3(h));
    final probsRaw = conv4_1(h);
    final regs = conv4_2(h);
    final probs = _softmaxOverChannels(probsRaw);
    return (probs, regs);
  }

  @override
  List<Tensor> parameters() => [
    ...conv1.parameters(),
    ...prelu1.parameters(),
    ...conv2.parameters(),
    ...prelu2.parameters(),
    ...conv3.parameters(),
    ...prelu3.parameters(),
    ...conv4_1.parameters(),
    ...conv4_2.parameters(),
  ];
}

/// Refine Network. Fixed 24×24 input, outputs `(probs [N,2], regs [N,4])`.
class RNet extends Module {
  final Conv2d conv1;
  final PReLU prelu1;
  final Conv2d conv2;
  final PReLU prelu2;
  final Conv2d conv3;
  final PReLU prelu3;
  final Linear dense4;
  final PReLU prelu4;
  final Linear dense5_1;
  final Linear dense5_2;

  RNet({Device device = Device.CPU})
    : conv1 = Conv2d(3, 28, kernel: 3, bias: true, device: device),
      prelu1 = PReLU(28, device: device),
      conv2 = Conv2d(28, 48, kernel: 3, bias: true, device: device),
      prelu2 = PReLU(48, device: device),
      conv3 = Conv2d(48, 64, kernel: 2, bias: true, device: device),
      prelu3 = PReLU(64, device: device),
      dense4 = Linear(576, 128, bias: true, device: device),
      prelu4 = PReLU(128, device: device),
      dense5_1 = Linear(128, 2, bias: true, device: device),
      dense5_2 = Linear(128, 4, bias: true, device: device);

  (Tensor, Tensor) call(Tensor x) {
    var h = prelu1(conv1(x));
    h = maxPool2d(h, kernel: 3, stride: 2, ceilMode: true);
    h = prelu2(conv2(h));
    h = maxPool2d(h, kernel: 3, stride: 2, ceilMode: true);
    h = prelu3(conv3(h)); // [N, 64, 3, 3]
    final flat = _permuteWHCFlatten(h); // [N, 576]
    final d4 = prelu4(dense4(flat));
    final probsRaw = dense5_1(d4);
    final regs = dense5_2(d4);
    final probs = _softmax2d(probsRaw);
    return (probs, regs);
  }

  @override
  List<Tensor> parameters() => [
    ...conv1.parameters(),
    ...prelu1.parameters(),
    ...conv2.parameters(),
    ...prelu2.parameters(),
    ...conv3.parameters(),
    ...prelu3.parameters(),
    ...dense4.parameters(),
    ...prelu4.parameters(),
    ...dense5_1.parameters(),
    ...dense5_2.parameters(),
  ];
}

/// Output Network. Fixed 48×48 input, outputs `(probs [N,2], regs
/// [N,4], landmarks [N,10])`. Landmarks are `(x1,x2,x3,x4,x5,
/// y1,y2,y3,y4,y5)` in **normalized box coords** — the caller maps
/// them back to image space.
class ONet extends Module {
  final Conv2d conv1;
  final PReLU prelu1;
  final Conv2d conv2;
  final PReLU prelu2;
  final Conv2d conv3;
  final PReLU prelu3;
  final Conv2d conv4;
  final PReLU prelu4;
  final Linear dense5;
  final PReLU prelu5;
  final Linear dense6_1;
  final Linear dense6_2;
  final Linear dense6_3;

  ONet({Device device = Device.CPU})
    : conv1 = Conv2d(3, 32, kernel: 3, bias: true, device: device),
      prelu1 = PReLU(32, device: device),
      conv2 = Conv2d(32, 64, kernel: 3, bias: true, device: device),
      prelu2 = PReLU(64, device: device),
      conv3 = Conv2d(64, 64, kernel: 3, bias: true, device: device),
      prelu3 = PReLU(64, device: device),
      conv4 = Conv2d(64, 128, kernel: 2, bias: true, device: device),
      prelu4 = PReLU(128, device: device),
      dense5 = Linear(1152, 256, bias: true, device: device),
      prelu5 = PReLU(256, device: device),
      dense6_1 = Linear(256, 2, bias: true, device: device),
      dense6_2 = Linear(256, 4, bias: true, device: device),
      dense6_3 = Linear(256, 10, bias: true, device: device);

  (Tensor, Tensor, Tensor) call(Tensor x) {
    var h = prelu1(conv1(x));
    h = maxPool2d(h, kernel: 3, stride: 2, ceilMode: true);
    h = prelu2(conv2(h));
    h = maxPool2d(h, kernel: 3, stride: 2, ceilMode: true);
    h = prelu3(conv3(h));
    h = maxPool2d(h, kernel: 2, stride: 2, ceilMode: true);
    h = prelu4(conv4(h)); // [N, 128, 3, 3]
    final flat = _permuteWHCFlatten(h); // [N, 1152]
    final d5 = prelu5(dense5(flat));
    final probsRaw = dense6_1(d5);
    final regs = dense6_2(d5);
    final landmarks = dense6_3(d5);
    final probs = _softmax2d(probsRaw);
    return (probs, regs, landmarks);
  }

  @override
  List<Tensor> parameters() => [
    ...conv1.parameters(),
    ...prelu1.parameters(),
    ...conv2.parameters(),
    ...prelu2.parameters(),
    ...conv3.parameters(),
    ...prelu3.parameters(),
    ...conv4.parameters(),
    ...prelu4.parameters(),
    ...dense5.parameters(),
    ...prelu5.parameters(),
    ...dense6_1.parameters(),
    ...dense6_2.parameters(),
    ...dense6_3.parameters(),
  ];
}

// ---------------- helpers ----------------

/// Softmax over the channel axis of a `[N, C, H, W]` tensor.
Tensor _softmaxOverChannels(Tensor x) {
  final n = x.shape[0];
  final c = x.shape[1];
  final h = x.shape[2];
  final w = x.shape[3];
  final data = x.toFloat32List();
  final out = Float32List(data.length);
  final spatial = h * w;
  for (int ni = 0; ni < n; ni++) {
    for (int hi = 0; hi < h; hi++) {
      for (int wi = 0; wi < w; wi++) {
        double maxV = -double.infinity;
        for (int ci = 0; ci < c; ci++) {
          final v = data[((ni * c + ci) * h + hi) * w + wi];
          if (v > maxV) maxV = v;
        }
        double sum = 0.0;
        for (int ci = 0; ci < c; ci++) {
          final e = _fexp(data[((ni * c + ci) * h + hi) * w + wi] - maxV);
          out[((ni * c + ci) * h + hi) * w + wi] = e;
          sum += e;
        }
        final inv = 1.0 / sum;
        for (int ci = 0; ci < c; ci++) {
          out[((ni * c + ci) * h + hi) * w + wi] *= inv;
        }
      }
    }
    if (spatial == 0) break;
  }
  return Tensor.fromFloat32List(x.shape, out, device: x.device);
}

/// 2-D softmax along axis 1 for `[N, C]`.
Tensor _softmax2d(Tensor x) {
  final n = x.shape[0];
  final c = x.shape[1];
  final data = x.toFloat32List();
  final out = Float32List(data.length);
  for (int ni = 0; ni < n; ni++) {
    double maxV = -double.infinity;
    for (int ci = 0; ci < c; ci++) {
      final v = data[ni * c + ci];
      if (v > maxV) maxV = v;
    }
    double sum = 0.0;
    for (int ci = 0; ci < c; ci++) {
      final e = _fexp(data[ni * c + ci] - maxV);
      out[ni * c + ci] = e;
      sum += e;
    }
    final inv = 1.0 / sum;
    for (int ci = 0; ci < c; ci++) {
      out[ni * c + ci] *= inv;
    }
  }
  return Tensor.fromFloat32List(x.shape, out, device: x.device);
}

double _fexp(double x) => math.exp(x);

/// facenet-pytorch's `x.permute(0, 3, 2, 1).flatten(1)`:
/// `[N, C, H, W]` -> `[N, W * H * C]` with the outer axis being W,
/// then H, then C. This matches the pretrained FC weights.
Tensor _permuteWHCFlatten(Tensor x) {
  final n = x.shape[0];
  final c = x.shape[1];
  final h = x.shape[2];
  final w = x.shape[3];
  final src = x.toFloat32List();
  final out = Float32List(n * c * h * w);
  final stride = c * h * w;
  for (int ni = 0; ni < n; ni++) {
    for (int wi = 0; wi < w; wi++) {
      for (int hi = 0; hi < h; hi++) {
        for (int ci = 0; ci < c; ci++) {
          final srcIdx = ((ni * c + ci) * h + hi) * w + wi;
          final dstIdx = ni * stride +
              ((wi * h + hi) * c + ci);
          out[dstIdx] = src[srcIdx];
        }
      }
    }
  }
  return Tensor.fromFloat32List([n, stride], out, device: x.device);
}
