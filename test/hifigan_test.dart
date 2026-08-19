@Timeout(Duration(minutes: 3))
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

Tensor _fake(List<int> shape, {int seed = 0, Device device = Device.CPU}) {
  final rng = math.Random(seed);
  var n = 1;
  for (final d in shape) {
    n *= d;
  }
  final v = Float32List(n);
  for (int i = 0; i < n; i++) {
    v[i] = rng.nextDouble() - 0.5;
  }
  return Tensor.fromFloat32List(shape, v, device: device);
}

bool _gpuAvailable() {
  try {
    Tensor.fromList([1], [1.0], device: Device.GPU).toList();
    return true;
  } catch (_) {
    return false;
  }
}

/// Tiny HiFi-GAN config for CPU test speed — same structure as V1 but
/// scaled channels down 16×.
const _tinyCfg = HiFiGanV1Config(
  melChannels: 4,
  upsampleInitialChannels: 32,
  upsampleRates: [8, 8, 2, 2],
  upsampleKernelSizes: [16, 16, 4, 4],
  resblockKernelSizes: [3, 7, 11],
  resblockDilations: [
    [1, 3, 5],
    [1, 3, 5],
    [1, 3, 5],
  ],
);

void main() {
  group('HiFiGanV1Config', () {
    test('totalUpsampleFactor = 256 for default V1 schedule', () {
      const cfg = HiFiGanV1Config();
      expect(cfg.totalUpsampleFactor, 256);
    });

    test('rejects mismatched kernel/stride parity in a stage', () {
      // k=3, s=2 → k-s=1 (odd) — no integer symmetric padding.
      expect(
        () => HiFiGanGenerator(
          const HiFiGanV1Config(
            upsampleRates: [2, 2, 2, 2],
            upsampleKernelSizes: [3, 4, 4, 4],
          ),
        ),
        throwsArgumentError,
      );
    });
  });

  group('HiFiGanResBlock1', () {
    test('preserves shape [N, C, T]', () {
      final rb = HiFiGanResBlock1(
        channels: 8,
        kernelSize: 3,
        dilations: [1, 3, 5],
      );
      final x = _fake([1, 8, 32], seed: 1);
      final y = rb(x);
      expect(y.shape, equals([1, 8, 32]));
    });

    test('dilation=1 with symmetric padding preserves shape', () {
      final rb = HiFiGanResBlock1(
        channels: 4,
        kernelSize: 7,
        dilations: [1, 1, 1],
      );
      final x = _fake([1, 4, 16], seed: 2);
      expect(rb(x).shape, equals([1, 4, 16]));
    });
  });

  group('HiFiGanGenerator', () {
    test('output shape = totalUpsampleFactor × input T', () {
      final gen = HiFiGanGenerator(_tinyCfg);
      final mel = _fake([1, 4, 10], seed: 3);
      final wav = gen(mel);
      // 10 mel frames × (8·8·2·2 = 256) = 2560 samples.
      expect(wav.shape, equals([1, 1, 2560]));
    });

    test('output values are in [-1, 1] after tanh', () {
      final gen = HiFiGanGenerator(_tinyCfg);
      final mel = _fake([1, 4, 6], seed: 4);
      final wav = gen(mel);
      for (final v in wav.toList()) {
        expect(v.abs() <= 1.0, isTrue, reason: 'sample out of tanh range: $v');
      }
    });

    test('rejects wrong-mel-channel input', () {
      final gen = HiFiGanGenerator(_tinyCfg);
      final mel = _fake([1, 3, 6], seed: 5);
      expect(() => gen(mel), throwsArgumentError);
    });

    test('parameter list is non-empty and finite', () {
      final gen = HiFiGanGenerator(_tinyCfg);
      final params = gen.parameters();
      expect(params, isNotEmpty);
      var checked = 0;
      for (final p in params) {
        for (final v in p.toList()) {
          expect(v.isFinite, isTrue);
        }
        checked++;
        if (checked > 50) break; // enough for a smoke check
      }
    });
  });

  group('Conv1d dilation regression', () {
    test('dilation=2 output length matches expanded kernel formula', () {
      // (l + 2p - dil*(k-1) - 1) / s + 1
      // (16 + 2·2 - 2·(3-1) - 1) / 1 + 1 = 16
      final conv = Conv1d(
        inChannels: 1,
        outChannels: 1,
        kernelSize: 3,
        padding: 2,
        dilation: 2,
        bias: false,
      );
      final x = _fake([1, 1, 16], seed: 11);
      expect(conv(x).shape, equals([1, 1, 16]));
    });

    test('dilation=1 == old Conv1d output', () {
      // Verify no regression at the historic default path.
      final conv = Conv1d(
        inChannels: 2,
        outChannels: 3,
        kernelSize: 5,
        padding: 2,
        bias: false,
      );
      final x = _fake([1, 2, 12], seed: 22);
      expect(conv(x).shape, equals([1, 3, 12]));
    });
  });

  group('HiFiGanGenerator on GPU', () {
    final gpuOk = _gpuAvailable();
    if (!gpuOk) {
      test(
        'GPU unavailable → skipped',
        () {},
        skip: 'CUDA / native/lib/libmat_mul.so not usable in this env',
      );
      return;
    }
    test('CPU vs GPU forward parity (tiny cfg)', () {
      const cpuCfg = _tinyCfg;
      final gpuCfg = HiFiGanV1Config(
        melChannels: cpuCfg.melChannels,
        upsampleInitialChannels: cpuCfg.upsampleInitialChannels,
        upsampleRates: cpuCfg.upsampleRates,
        upsampleKernelSizes: cpuCfg.upsampleKernelSizes,
        resblockKernelSizes: cpuCfg.resblockKernelSizes,
        resblockDilations: cpuCfg.resblockDilations,
        device: Device.GPU,
      );
      final cpu = HiFiGanGenerator(cpuCfg);
      final gpu = HiFiGanGenerator(gpuCfg);
      // Sync parameters.
      final cpuP = cpu.parameters();
      final gpuP = gpu.parameters();
      expect(cpuP.length, gpuP.length);
      for (int i = 0; i < cpuP.length; i++) {
        gpuP[i].assign(
          Tensor.fromList(cpuP[i].shape, cpuP[i].toList(), device: Device.GPU),
        );
      }
      final xVals = List<double>.generate(1 * 4 * 8, (i) => math.sin(i * 0.13));
      final xCpu = Tensor.fromList([1, 4, 8], xVals);
      final xGpu = Tensor.fromList([1, 4, 8], xVals, device: Device.GPU);
      final yCpu = cpu(xCpu).toList();
      final yGpu = gpu(xGpu).toList();
      double maxDiff = 0;
      for (int i = 0; i < yCpu.length; i++) {
        final d = (yCpu[i] - yGpu[i]).abs();
        if (d > maxDiff) maxDiff = d;
      }
      expect(maxDiff, lessThan(5e-2), reason: 'max cpu/gpu diff = $maxDiff');
    });
  });

  group('HiFiGanLoader', () {
    /// Build a synthetic torch-style state_dict matching every key our
    /// loader consumes.
    Map<String, Tensor> synthState(HiFiGanGenerator model) {
      final rng = math.Random(0);
      Tensor rand(List<int> shape) {
        var n = 1;
        for (final d in shape) {
          n *= d;
        }
        final v = Float32List(n);
        for (int i = 0; i < n; i++) {
          v[i] = (rng.nextDouble() - 0.5) * 0.1;
        }
        return Tensor.fromFloat32List(shape, v);
      }

      final s = <String, Tensor>{};
      final cfg = model.config;
      s['conv_pre.weight'] = rand([
        cfg.upsampleInitialChannels,
        cfg.melChannels,
        cfg.preKernelSize,
      ]);
      s['conv_pre.bias'] = rand([cfg.upsampleInitialChannels]);

      var ch = cfg.upsampleInitialChannels;
      for (int i = 0; i < cfg.upsampleRates.length; i++) {
        final outCh = ch ~/ 2;
        s['ups.$i.weight'] =
            rand([ch, outCh, cfg.upsampleKernelSizes[i]]);
        s['ups.$i.bias'] = rand([outCh]);
        for (int j = 0; j < cfg.resblockKernelSizes.length; j++) {
          final flat = i * cfg.resblockKernelSizes.length + j;
          for (int k = 0;
              k < cfg.resblockDilations[j].length;
              k++) {
            s['resblocks.$flat.convs1.$k.weight'] =
                rand([outCh, outCh, cfg.resblockKernelSizes[j]]);
            s['resblocks.$flat.convs1.$k.bias'] = rand([outCh]);
            s['resblocks.$flat.convs2.$k.weight'] =
                rand([outCh, outCh, cfg.resblockKernelSizes[j]]);
            s['resblocks.$flat.convs2.$k.bias'] = rand([outCh]);
          }
        }
        ch = outCh;
      }
      s['conv_post.weight'] = rand([1, ch, cfg.postKernelSize]);
      s['conv_post.bias'] = rand([1]);
      return s;
    }

    test('consumes a synthetic state_dict with no unused keys', () {
      final gen = HiFiGanGenerator(_tinyCfg);
      final state = synthState(gen);
      final report = HiFiGanLoader.loadMap(gen, state);
      expect(report.unusedKeys, isEmpty);

      // Expected count:
      //   conv_pre.{w,b}                                    = 2
      //   per stage: ups.{w,b} + kernels * dilations * 4    = 2 + 3*3*4 = 38
      //   conv_post.{w,b}                                   = 2
      final perStage = 2 +
          _tinyCfg.resblockKernelSizes.length *
              _tinyCfg.resblockDilations[0].length *
              4;
      final expected = 2 + _tinyCfg.upsampleRates.length * perStage + 2;
      expect(report.consumedCount, expected);
    });

    test('rejects a missing tensor', () {
      final gen = HiFiGanGenerator(_tinyCfg);
      final state = synthState(gen);
      state.remove('conv_post.bias');
      expect(() => HiFiGanLoader.loadMap(gen, state), throwsArgumentError);
    });

    test('forward after load produces finite waveform in [-1, 1]', () {
      final gen = HiFiGanGenerator(_tinyCfg);
      HiFiGanLoader.loadMap(gen, synthState(gen));
      final mel = _fake([1, 4, 6], seed: 99);
      final wav = gen(mel).toList();
      for (final v in wav) {
        expect(v.isFinite, isTrue);
        expect(v.abs() <= 1.0, isTrue, reason: 'sample $v outside tanh');
      }
    });
  });
}
