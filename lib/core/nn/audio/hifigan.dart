/// HiFi-GAN Generator (V1) — mel-spectrogram → waveform vocoder.
///
/// Kong et al. 2020, `jik876/hifi-gan`. Turns an `[N, 80, T]` log-mel
/// spectrogram into an `[N, 1, 256·T]` waveform via 4 upsample stages
/// interleaved with Multi-Receptive-Field (MRF) residual blocks:
///
///   pre       Conv1d(80 → 512, k=7, p=3, bias)
///   stage 0   LeakyReLU → ConvTranspose1d(512→256, k=16, s=8, p=4)
///             → MRF(256, {k=3,7,11}, dilations={{1,3,5},{1,3,5},{1,3,5}})
///   stage 1   LeakyReLU → ConvTranspose1d(256→128, k=16, s=8, p=4)
///             → MRF(128, …)
///   stage 2   LeakyReLU → ConvTranspose1d(128→64, k=4, s=2, p=1)
///             → MRF(64, …)
///   stage 3   LeakyReLU → ConvTranspose1d(64→32, k=4, s=2, p=1)
///             → MRF(32, …)
///   post      LeakyReLU → Conv1d(32 → 1, k=7, p=3, bias) → tanh
///
/// Upsampling ratios `[8, 8, 2, 2]` give a total 256× upsample factor,
/// matching HiFi-GAN V1 config on LJSpeech / VCTK. The generator is
/// forward-only here (no discriminator, no training loop) — it takes a
/// mel-spectrogram and produces a waveform ready for playback.
///
/// **Weight-norm is expected to be pre-folded** in the checkpoint —
/// standard `torch.nn.utils.remove_weight_norm(g)` before dumping. The
/// loader takes the plain `Conv1d.weight` and `Conv1d.bias` layouts.
library;

import 'dart:math' as math;

import '../../tensor/tensor.dart';
import '../conv1d.dart';
import '../conv_transpose_1d.dart';
import '../module.dart';

/// LeakyReLU slope used by every activation in HiFi-GAN.
const double _leakyReluSlope = 0.1;

/// Standard HiFi-GAN V1 upsample schedule: `[8, 8, 2, 2]` × kernels
/// `[16, 16, 4, 4]` and initial channels 512 → 256 → 128 → 64 → 32.
class HiFiGanV1Config {
  final int melChannels;
  final int upsampleInitialChannels;
  final List<int> upsampleRates;
  final List<int> upsampleKernelSizes;
  final List<int> resblockKernelSizes;
  final List<List<int>> resblockDilations;
  final int preKernelSize;
  final int postKernelSize;
  final Device device;
  final int seed;

  const HiFiGanV1Config({
    this.melChannels = 80,
    this.upsampleInitialChannels = 512,
    this.upsampleRates = const [8, 8, 2, 2],
    this.upsampleKernelSizes = const [16, 16, 4, 4],
    this.resblockKernelSizes = const [3, 7, 11],
    this.resblockDilations = const [
      [1, 3, 5],
      [1, 3, 5],
      [1, 3, 5],
    ],
    this.preKernelSize = 7,
    this.postKernelSize = 7,
    this.device = Device.CPU,
    this.seed = 0,
  });

  /// Total upsample factor across all stages (e.g. 8·8·2·2 = 256 for V1).
  int get totalUpsampleFactor {
    var f = 1;
    for (final r in upsampleRates) {
      f *= r;
    }
    return f;
  }
}

/// One MRF residual sub-block: three dilated convs each with a
/// LeakyReLU pre-activation, residual added at the end. For dilation
/// list `[d0, d1, d2]`:
///
///   for d in dilations:
///     x = x + Conv1d(k, dilation=d, pad=(k-1)*d/2)( LeakyReLU(x) )
///           then Conv1d(k, dilation=1, pad=(k-1)/2)( LeakyReLU(...) )
class HiFiGanResBlock1 extends Module {
  final int channels;
  final int kernelSize;
  final List<int> dilations;
  final List<Conv1d> convs1;
  final List<Conv1d> convs2;

  HiFiGanResBlock1({
    required this.channels,
    required this.kernelSize,
    required this.dilations,
    Device device = Device.CPU,
    int seed = 0,
  }) : convs1 = <Conv1d>[],
       convs2 = <Conv1d>[] {
    for (int i = 0; i < dilations.length; i++) {
      final d = dilations[i];
      final padDil = (kernelSize - 1) * d ~/ 2;
      final pad1 = (kernelSize - 1) ~/ 2;
      convs1.add(
        Conv1d(
          inChannels: channels,
          outChannels: channels,
          kernelSize: kernelSize,
          stride: 1,
          padding: padDil,
          dilation: d,
          bias: true,
          device: device,
        ),
      );
      convs2.add(
        Conv1d(
          inChannels: channels,
          outChannels: channels,
          kernelSize: kernelSize,
          stride: 1,
          padding: pad1,
          dilation: 1,
          bias: true,
          device: device,
        ),
      );
    }
  }

  Tensor call(Tensor x) {
    var h = x;
    for (int i = 0; i < convs1.length; i++) {
      final a = convs1[i](_leakyRelu(h));
      final b = convs2[i](_leakyRelu(a));
      h = h + b;
    }
    return h;
  }

  @override
  List<Tensor> parameters() => [
    for (final c in convs1) ...c.parameters(),
    for (final c in convs2) ...c.parameters(),
  ];

  @override
  List<Module> submodules() => [...convs1, ...convs2];
}

/// One upsample stage: LeakyReLU → ConvTranspose1d → MRF (three parallel
/// residual sub-blocks whose outputs are averaged).
class HiFiGanUpsampleBlock extends Module {
  final ConvTranspose1d upsample;
  final List<HiFiGanResBlock1> resblocks;

  HiFiGanUpsampleBlock({
    required int inChannels,
    required int outChannels,
    required int kernel,
    required int stride,
    required int padding,
    required List<int> resblockKernelSizes,
    required List<List<int>> resblockDilations,
    Device device = Device.CPU,
    int seed = 0,
  }) : upsample = ConvTranspose1d(
         inChannels,
         outChannels,
         kernelSize: kernel,
         stride: stride,
         padding: padding,
         bias: true,
         device: device,
         seed: seed,
       ),
       resblocks = <HiFiGanResBlock1>[] {
    for (int i = 0; i < resblockKernelSizes.length; i++) {
      resblocks.add(
        HiFiGanResBlock1(
          channels: outChannels,
          kernelSize: resblockKernelSizes[i],
          dilations: resblockDilations[i],
          device: device,
          seed: seed + 1000 * (i + 1),
        ),
      );
    }
  }

  Tensor call(Tensor x) {
    final up = upsample(_leakyRelu(x));
    // MRF: average the sub-block outputs (matches jik876/hifi-gan).
    var acc = resblocks.first(up);
    for (int i = 1; i < resblocks.length; i++) {
      acc = acc + resblocks[i](up);
    }
    return acc * (1.0 / resblocks.length);
  }

  @override
  List<Tensor> parameters() => [
    ...upsample.parameters(),
    for (final r in resblocks) ...r.parameters(),
  ];

  @override
  List<Module> submodules() => [upsample, ...resblocks];
}

class HiFiGanGenerator extends Module {
  final HiFiGanV1Config config;

  final Conv1d preConv;
  final List<HiFiGanUpsampleBlock> stages;
  final Conv1d postConv;

  HiFiGanGenerator(this.config)
    : preConv = Conv1d(
        inChannels: config.melChannels,
        outChannels: config.upsampleInitialChannels,
        kernelSize: config.preKernelSize,
        stride: 1,
        padding: (config.preKernelSize - 1) ~/ 2,
        bias: true,
        device: config.device,
      ),
      stages = <HiFiGanUpsampleBlock>[],
      postConv = Conv1d(
        inChannels:
            config.upsampleInitialChannels ~/
            math.pow(2, config.upsampleRates.length).toInt(),
        outChannels: 1,
        kernelSize: config.postKernelSize,
        stride: 1,
        padding: (config.postKernelSize - 1) ~/ 2,
        bias: true,
        device: config.device,
      ) {
    var channels = config.upsampleInitialChannels;
    for (int i = 0; i < config.upsampleRates.length; i++) {
      final s = config.upsampleRates[i];
      final k = config.upsampleKernelSizes[i];
      if ((k - s) % 2 != 0) {
        throw ArgumentError(
          'HiFiGanGenerator: kernel-stride mismatch on stage $i '
          '(k=$k, s=$s) — HiFi-GAN convention requires (k-s) even.',
        );
      }
      final pad = (k - s) ~/ 2;
      stages.add(
        HiFiGanUpsampleBlock(
          inChannels: channels,
          outChannels: channels ~/ 2,
          kernel: k,
          stride: s,
          padding: pad,
          resblockKernelSizes: config.resblockKernelSizes,
          resblockDilations: config.resblockDilations,
          device: config.device,
          seed: config.seed + 100_000 * (i + 1),
        ),
      );
      channels = channels ~/ 2;
    }
  }

  /// Forward pass. `mel` is `[N, melChannels, T]`. Returns
  /// `[N, 1, totalUpsampleFactor · T]` waveform samples in `[-1, 1]`.
  Tensor call(Tensor mel) {
    if (mel.shape.length != 3 || mel.shape[1] != config.melChannels) {
      throw ArgumentError(
        'HiFiGanGenerator: expected [N, ${config.melChannels}, T]; '
        'got ${mel.shape}',
      );
    }
    var h = preConv(mel);
    for (final s in stages) {
      h = s(h);
    }
    final logits = postConv(_leakyRelu(h));
    return _tanh(logits);
  }

  @override
  List<Tensor> parameters() => [
    ...preConv.parameters(),
    for (final s in stages) ...s.parameters(),
    ...postConv.parameters(),
  ];

  @override
  List<Module> submodules() => [preConv, ...stages, postConv];
}

Tensor _leakyRelu(Tensor x) {
  final data = x.toFloat32List();
  final out = List<double>.filled(data.length, 0);
  for (int i = 0; i < data.length; i++) {
    final v = data[i];
    out[i] = v >= 0 ? v : v * _leakyReluSlope;
  }
  return Tensor.fromList(x.shape, out, device: x.device);
}

Tensor _tanh(Tensor x) {
  final data = x.toFloat32List();
  final out = List<double>.filled(data.length, 0);
  for (int i = 0; i < data.length; i++) {
    final e2 = math.exp(2 * data[i]);
    out[i] = (e2 - 1) / (e2 + 1);
  }
  return Tensor.fromList(x.shape, out, device: x.device);
}
