/// Loader for torchvision ResNet-50 `state_dict()` converted to
/// safetensors via `scripts/convert_resnet50_pt_to_safetensors.py`
/// (which is `torch.save(model.state_dict(), ...)` → safetensors).
///
/// Every `conv{i} + bn{i}` (and `downsample.0 + downsample.1`) pair
/// is folded at load time via [`conv_bn_fold.dart`] (eps=1e-5,
/// torchvision default), so the runtime model is just
/// Conv2d + ReLU + MaxPool + AdaptiveAvgPool + Linear.
///
/// Key mapping (torchvision `resnet50.state_dict()`):
///
///   conv1.weight                                          [64, 3, 7, 7]
///   bn1.{weight, bias, running_mean, running_var}         [64]
///   layer{s}.{i}.conv{k}.weight                           [Cout, Cin, K, K]
///   layer{s}.{i}.bn{k}.{weight, bias, running_mean, running_var}
///   layer{s}.{i}.downsample.0.weight                      [Cout, Cin, 1, 1]
///   layer{s}.{i}.downsample.1.{weight, bias, running_mean, running_var}
///   fc.{weight, bias}                                      [1000, 2048] / [1000]
///
/// Ignored: `bn*.num_batches_tracked` (int64 counter, not used by
/// the fold).
library;

import 'dart:typed_data';

import '../../tensor/tensor.dart';
import '../safetensors.dart';
import 'conv_bn_fold.dart';
import 'resnet.dart';

class ResNetLoadReport {
  final int consumedCount;
  final List<String> unusedKeys;
  const ResNetLoadReport({
    required this.consumedCount,
    required this.unusedKeys,
  });

  @override
  String toString() =>
      'ResNetLoadReport(consumed=$consumedCount, unused=${unusedKeys.length})';
}

class ResNetLoader {
  /// torchvision BatchNorm2d default eps.
  static const double bnEps = 1e-5;

  static ResNetLoadReport loadFile(ResNet model, String path) {
    final state = SafeTensors.loadFile(path);
    return loadMap(model, state);
  }

  static ResNetLoadReport loadMap(ResNet model, Map<String, Tensor> state) {
    final consumed = <String>{};

    // Stem: conv1 + bn1.
    _foldInto(
      model.stem.conv,
      state,
      consumed,
      convPrefix: 'conv1',
      bnPrefix: 'bn1',
    );

    final stages = [
      ('layer1', model.layer1),
      ('layer2', model.layer2),
      ('layer3', model.layer3),
      ('layer4', model.layer4),
    ];
    for (final (name, blocks) in stages) {
      for (int i = 0; i < blocks.length; i++) {
        final b = blocks[i];
        final p = '$name.$i';
        _foldInto(
          b.conv1Ref.conv,
          state,
          consumed,
          convPrefix: '$p.conv1',
          bnPrefix: '$p.bn1',
        );
        _foldInto(
          b.conv2Ref.conv,
          state,
          consumed,
          convPrefix: '$p.conv2',
          bnPrefix: '$p.bn2',
        );
        _foldInto(
          b.conv3Ref.conv,
          state,
          consumed,
          convPrefix: '$p.conv3',
          bnPrefix: '$p.bn3',
        );
        if (b.downsampleRef != null) {
          _foldInto(
            b.downsampleRef!.conv,
            state,
            consumed,
            convPrefix: '$p.downsample.0',
            bnPrefix: '$p.downsample.1',
          );
        }
      }
    }

    // Head — plain fc.weight + fc.bias.
    final fcW = _take(state, consumed, 'fc.weight');
    final fcB = _take(state, consumed, 'fc.bias');
    _assign(model.fc.weight, fcW);
    _assign(model.fc.bias!, _reshapeVectorTo1xN(fcB));

    // Silently absorb any per-bn `num_batches_tracked` counters.
    for (final k in state.keys.toList()) {
      if (k.endsWith('.num_batches_tracked')) consumed.add(k);
    }

    final unused = state.keys.where((k) => !consumed.contains(k)).toList()
      ..sort();
    return ResNetLoadReport(consumedCount: consumed.length, unusedKeys: unused);
  }

  static void _foldInto(
    dynamic conv, // Conv2d (avoid top-level import of ../conv2d.dart for type)
    Map<String, Tensor> state,
    Set<String> consumed, {
    required String convPrefix,
    required String bnPrefix,
  }) {
    final w = _take(state, consumed, '$convPrefix.weight');
    final gamma = _take(state, consumed, '$bnPrefix.weight');
    final beta = _take(state, consumed, '$bnPrefix.bias');
    final mean = _take(state, consumed, '$bnPrefix.running_mean');
    final varT = _take(state, consumed, '$bnPrefix.running_var');
    loadConvBnFolded(
      conv,
      convW: _toF32(w),
      bnGamma: _toF32(gamma),
      bnBeta: _toF32(beta),
      bnMean: _toF32(mean),
      bnVar: _toF32(varT),
      bnEps: bnEps,
    );
  }

  static Tensor _take(
    Map<String, Tensor> state,
    Set<String> consumed,
    String name,
  ) {
    final t = state[name];
    if (t == null) {
      throw ArgumentError('resnet loader: missing tensor "$name"');
    }
    consumed.add(name);
    return t;
  }

  static void _assign(Tensor dst, Tensor src) {
    if (dst.length != src.length) {
      throw ArgumentError(
        'resnet loader: assign length mismatch — dst=${dst.shape}, '
        'src=${src.shape}',
      );
    }
    final vals = src.toList();
    final matched = Tensor.fromList(dst.shape, vals, device: dst.device);
    dst.assign(matched);
  }

  static Float32List _toF32(Tensor t) {
    final data = t.toList();
    final out = Float32List(data.length);
    for (int i = 0; i < data.length; i++) {
      out[i] = data[i];
    }
    return out;
  }

  static Tensor _reshapeVectorTo1xN(Tensor v) {
    if (v.shape.length != 1) {
      throw ArgumentError(
        'resnet loader: expected rank 1 for bias, got ${v.shape}',
      );
    }
    return Tensor.fromList([1, v.shape[0]], v.toList(), device: Device.CPU);
  }
}
