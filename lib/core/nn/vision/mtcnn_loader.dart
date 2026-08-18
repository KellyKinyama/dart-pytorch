/// Safetensors loader for MTCNN P/R/O nets — pairs each Conv2d with
/// its bias and each PReLU with its per-channel slope.
library;

import 'dart:typed_data';

import '../../tensor/tensor.dart';
import '../conv2d.dart';
import '../linear.dart';
import '../prelu.dart';
import '../safetensors.dart';
import 'mtcnn.dart';

class MTCNNLoader {
  static int loadPNet(PNet net, String path) {
    final state = SafeTensors.loadFile(path);
    _loadConv2d(net.conv1, state, 'conv1', [10, 3, 3, 3]);
    _loadPrelu(net.prelu1, state, 'prelu1', 10);
    _loadConv2d(net.conv2, state, 'conv2', [16, 10, 3, 3]);
    _loadPrelu(net.prelu2, state, 'prelu2', 16);
    _loadConv2d(net.conv3, state, 'conv3', [32, 16, 3, 3]);
    _loadPrelu(net.prelu3, state, 'prelu3', 32);
    _loadConv2d(net.conv4_1, state, 'conv4_1', [2, 32, 1, 1]);
    _loadConv2d(net.conv4_2, state, 'conv4_2', [4, 32, 1, 1]);
    return state.length;
  }

  static int loadRNet(RNet net, String path) {
    final state = SafeTensors.loadFile(path);
    _loadConv2d(net.conv1, state, 'conv1', [28, 3, 3, 3]);
    _loadPrelu(net.prelu1, state, 'prelu1', 28);
    _loadConv2d(net.conv2, state, 'conv2', [48, 28, 3, 3]);
    _loadPrelu(net.prelu2, state, 'prelu2', 48);
    _loadConv2d(net.conv3, state, 'conv3', [64, 48, 2, 2]);
    _loadPrelu(net.prelu3, state, 'prelu3', 64);
    _loadLinear(net.dense4, state, 'dense4', [128, 576]);
    _loadPrelu(net.prelu4, state, 'prelu4', 128);
    _loadLinear(net.dense5_1, state, 'dense5_1', [2, 128]);
    _loadLinear(net.dense5_2, state, 'dense5_2', [4, 128]);
    return state.length;
  }

  static int loadONet(ONet net, String path) {
    final state = SafeTensors.loadFile(path);
    _loadConv2d(net.conv1, state, 'conv1', [32, 3, 3, 3]);
    _loadPrelu(net.prelu1, state, 'prelu1', 32);
    _loadConv2d(net.conv2, state, 'conv2', [64, 32, 3, 3]);
    _loadPrelu(net.prelu2, state, 'prelu2', 64);
    _loadConv2d(net.conv3, state, 'conv3', [64, 64, 3, 3]);
    _loadPrelu(net.prelu3, state, 'prelu3', 64);
    _loadConv2d(net.conv4, state, 'conv4', [128, 64, 2, 2]);
    _loadPrelu(net.prelu4, state, 'prelu4', 128);
    _loadLinear(net.dense5, state, 'dense5', [256, 1152]);
    _loadPrelu(net.prelu5, state, 'prelu5', 256);
    _loadLinear(net.dense6_1, state, 'dense6_1', [2, 256]);
    _loadLinear(net.dense6_2, state, 'dense6_2', [4, 256]);
    _loadLinear(net.dense6_3, state, 'dense6_3', [10, 256]);
    return state.length;
  }

  // ---------------- helpers ----------------

  static void _loadConv2d(
    Conv2d conv,
    Map<String, Tensor> state,
    String prefix,
    List<int> expectedShape,
  ) {
    final w = _need(state, '$prefix.weight');
    final b = _need(state, '$prefix.bias');
    if (!_shapeEq(w.shape, expectedShape)) {
      throw ArgumentError(
        'MTCNN loader: $prefix.weight expected $expectedShape got ${w.shape}',
      );
    }
    if (b.shape.length != 1 || b.shape[0] != expectedShape[0]) {
      throw ArgumentError(
        'MTCNN loader: $prefix.bias expected [${expectedShape[0]}] got ${b.shape}',
      );
    }
    final wT = Tensor.fromList(
      conv.weight.shape,
      w.toList(),
      device: conv.weight.device,
      requiresGrad: conv.weight.requiresGrad,
    );
    conv.weight.assign(wT);
    final bT = Tensor.fromList(
      conv.bias!.shape,
      b.toList(),
      device: conv.bias!.device,
      requiresGrad: conv.bias!.requiresGrad,
    );
    conv.bias!.assign(bT);
  }

  static void _loadLinear(
    Linear lin,
    Map<String, Tensor> state,
    String prefix,
    List<int> expectedShape,
  ) {
    final w = _need(state, '$prefix.weight');
    final b = _need(state, '$prefix.bias');
    if (!_shapeEq(w.shape, expectedShape)) {
      throw ArgumentError(
        'MTCNN loader: $prefix.weight expected $expectedShape got ${w.shape}',
      );
    }
    final wT = Tensor.fromList(
      lin.weight.shape,
      w.toList(),
      device: lin.weight.device,
      requiresGrad: lin.weight.requiresGrad,
    );
    lin.weight.assign(wT);
    // Linear bias in this repo is [1, outF]; source is [outF].
    final vals = b.toList();
    final wideB = Float32List(vals.length);
    for (int i = 0; i < vals.length; i++) {
      wideB[i] = vals[i];
    }
    final bT = Tensor.fromFloat32List(
      lin.bias!.shape,
      wideB,
      device: lin.bias!.device,
      requiresGrad: lin.bias!.requiresGrad,
    );
    lin.bias!.assign(bT);
  }

  static void _loadPrelu(
    PReLU prelu,
    Map<String, Tensor> state,
    String prefix,
    int expectedChannels,
  ) {
    final w = _need(state, '$prefix.weight');
    if (w.shape.length != 1 || w.shape[0] != expectedChannels) {
      throw ArgumentError(
        'MTCNN loader: $prefix.weight expected [$expectedChannels] got ${w.shape}',
      );
    }
    final wT = Tensor.fromList(
      prelu.weight.shape,
      w.toList(),
      device: prelu.weight.device,
      requiresGrad: prelu.weight.requiresGrad,
    );
    prelu.weight.assign(wT);
  }

  static Tensor _need(Map<String, Tensor> state, String name) {
    final t = state[name];
    if (t == null) {
      throw ArgumentError('MTCNN loader: missing tensor "$name"');
    }
    return t;
  }

  static bool _shapeEq(List<int> a, List<int> b) {
    if (a.length != b.length) return false;
    for (int i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }
}
