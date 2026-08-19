/// Loader for HiFi-GAN V1 generator checkpoints (LJSpeech / VCTK /
/// UNIVERSAL_V1) converted to safetensors via
/// `scripts/convert_hifigan_pt_to_safetensors.py` (which folds
/// `torch.nn.utils.remove_weight_norm(g)` before dumping).
///
/// Expected key layout (jik876/hifi-gan `generator.state_dict()`):
///
///   conv_pre.{weight, bias}                             [512, 80, 7] / [512]
///   ups.{i}.{weight, bias}                              (ConvTranspose1d)
///   resblocks.{i * 3 + j}.convs1.{k}.{weight, bias}     (Conv1d)
///   resblocks.{i * 3 + j}.convs2.{k}.{weight, bias}     (Conv1d)
///   conv_post.{weight, bias}                            [1, 32, 7] / [1]
///
/// `resblocks` are laid out as one flat list of `numStages * numKernels`
/// blocks — stage `i`, kernel index `j` maps to
/// `resblocks[i * numKernels + j]`.
///
/// PyTorch tensor conventions:
///   * `Conv1d.weight`          shape `[Cout, Cin, K]`
///   * `ConvTranspose1d.weight` shape `[Cin, Cout, K]`
///   * bias vectors             shape `[Cout]`
///
/// Our internal `Conv1d.weight` is stored transposed as `[Cin*K, Cout]`
/// with `[1, Cout]` bias — the loader uses `Conv1d.loadFromPytorch` to
/// perform the transpose. `ConvTranspose1d.weight` matches the HF
/// layout directly.
library;

import 'dart:typed_data';

import '../../tensor/tensor.dart';
import '../conv1d.dart';
import '../conv_transpose_1d.dart';
import '../safetensors.dart';
import 'hifigan.dart';

class HiFiGanLoadReport {
  final int consumedCount;
  final List<String> unusedKeys;
  const HiFiGanLoadReport({
    required this.consumedCount,
    required this.unusedKeys,
  });

  @override
  String toString() =>
      'HiFiGanLoadReport(consumed=$consumedCount, '
      'unused=${unusedKeys.length})';
}

class HiFiGanLoader {
  static HiFiGanLoadReport loadFile(HiFiGanGenerator model, String path) {
    final state = SafeTensors.loadFile(path);
    return loadMap(model, state);
  }

  static HiFiGanLoadReport loadMap(
    HiFiGanGenerator model,
    Map<String, Tensor> state,
  ) {
    final consumed = <String>{};

    Tensor take(String name) {
      final t = state[name];
      if (t == null) {
        throw ArgumentError('hifigan loader: missing tensor "$name"');
      }
      consumed.add(name);
      return t;
    }

    // ---------- pre-conv ----------
    _loadConv1d(model.preConv, take('conv_pre.weight'), take('conv_pre.bias'));

    // ---------- upsample + resblocks per stage ----------
    final numStages = model.stages.length;
    final numKernels = model.config.resblockKernelSizes.length;
    final numDilations = model.config.resblockDilations[0].length;
    for (int i = 0; i < numStages; i++) {
      _loadConvTranspose1d(
        model.stages[i].upsample,
        take('ups.$i.weight'),
        take('ups.$i.bias'),
      );
      for (int j = 0; j < numKernels; j++) {
        final block = model.stages[i].resblocks[j];
        final flat = i * numKernels + j;
        for (int k = 0; k < numDilations; k++) {
          _loadConv1d(
            block.convs1[k],
            take('resblocks.$flat.convs1.$k.weight'),
            take('resblocks.$flat.convs1.$k.bias'),
          );
          _loadConv1d(
            block.convs2[k],
            take('resblocks.$flat.convs2.$k.weight'),
            take('resblocks.$flat.convs2.$k.bias'),
          );
        }
      }
    }

    // ---------- post-conv ----------
    _loadConv1d(
      model.postConv,
      take('conv_post.weight'),
      take('conv_post.bias'),
    );

    final unused = state.keys.where((k) => !consumed.contains(k)).toList()
      ..sort();
    return HiFiGanLoadReport(
      consumedCount: consumed.length,
      unusedKeys: unused,
    );
  }

  static void _loadConv1d(Conv1d conv, Tensor weight, Tensor bias) {
    final expectedW = conv.outChannels * conv.inChannels * conv.kernelSize;
    if (weight.length != expectedW) {
      throw ArgumentError(
        'hifigan loader: conv1d weight length ${weight.length} != '
        '$expectedW (Cout=${conv.outChannels}, Cin=${conv.inChannels}, '
        'K=${conv.kernelSize})',
      );
    }
    if (bias.length != conv.outChannels) {
      throw ArgumentError(
        'hifigan loader: conv1d bias length ${bias.length} != '
        '${conv.outChannels}',
      );
    }
    conv.loadFromPytorch(_toF32(weight), _toF32(bias));
  }

  static void _loadConvTranspose1d(
    ConvTranspose1d ct,
    Tensor weight,
    Tensor bias,
  ) {
    // HF/PyTorch ConvTranspose1d.weight is [Cin, Cout, K] — matches our
    // layout exactly, so just assign.
    if (weight.length != ct.weight.length) {
      throw ArgumentError(
        'hifigan loader: convtranspose1d weight length ${weight.length} '
        '!= ${ct.weight.length}',
      );
    }
    if (bias.length != ct.bias!.length) {
      throw ArgumentError(
        'hifigan loader: convtranspose1d bias length ${bias.length} != '
        '${ct.bias!.length}',
      );
    }
    _assign(ct.weight, weight);
    _assign(ct.bias!, bias);
  }

  static Float32List _toF32(Tensor t) {
    final data = t.toList();
    final out = Float32List(data.length);
    for (int i = 0; i < data.length; i++) {
      out[i] = data[i];
    }
    return out;
  }

  static void _assign(Tensor dst, Tensor src) {
    final vals = src.toList();
    final matched = Tensor.fromList(dst.shape, vals, device: dst.device);
    dst.assign(matched);
  }
}
