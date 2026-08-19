@Timeout(Duration(minutes: 3))
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

/// Tiny DeepSeek-V2 config for CPU-viable tests: 2 layers (1 dense +
/// 1 MoE), embed=32, 4 heads, 4 experts, top-2. Shares the same
/// architecture as V2-Lite / V2-full but scaled down 100×.
const _tinyCfg = DeepSeekV2Config(
  vocabSize: 128,
  maxCtx: 32,
  embedDim: 32,
  numLayers: 2,
  firstKDenseReplace: 1,
  denseFfnDim: 64,
  moeExpertHiddenDim: 32,
  numRoutedExperts: 4,
  numSharedExperts: 1,
  numExpertsPerTok: 2,
  numExpertGroups: 1,
  topKGroups: 1,
  mlaConfig: MLAConfig(
    embedDim: 32,
    numHeads: 4,
    qLoraRank: null, // Lite-style — no Q compression
    kvLoraRank: 12,
    qkNopeHeadDim: 6,
    qkRopeHeadDim: 4,
    vHeadDim: 6,
  ),
  rmsNormEps: 1e-6,
  ropeBase: 10000.0,
);

bool _gpuAvailable() {
  try {
    Tensor.fromList([1], [1.0], device: Device.GPU).toList();
    return true;
  } catch (_) {
    return false;
  }
}

void main() {
  group('DeepSeekV2Config presets', () {
    test('lite matches HF DeepSeek-V2-Lite config.json', () {
      final cfg = DeepSeekV2Config.lite();
      expect(cfg.vocabSize, 102400);
      expect(cfg.embedDim, 2048);
      expect(cfg.numLayers, 27);
      expect(cfg.firstKDenseReplace, 1);
      expect(cfg.denseFfnDim, 10944);
      expect(cfg.moeExpertHiddenDim, 1408);
      expect(cfg.numRoutedExperts, 64);
      expect(cfg.numSharedExperts, 2);
      expect(cfg.numExpertsPerTok, 6);
      expect(cfg.numExpertGroups, 1);
      expect(cfg.topKGroups, 1);
      expect(cfg.mlaConfig.qLoraRank, isNull);
      expect(cfg.mlaConfig.kvLoraRank, 512);
    });

    test('full matches HF DeepSeek-V2 config.json', () {
      final cfg = DeepSeekV2Config.full();
      expect(cfg.embedDim, 5120);
      expect(cfg.numLayers, 60);
      expect(cfg.numRoutedExperts, 160);
      expect(cfg.numExpertGroups, 8);
      expect(cfg.topKGroups, 3);
      expect(cfg.mlaConfig.qLoraRank, 1536);
    });
  });

  group('DeepSeekV2Block', () {
    test('layer 0 is dense (isMoE=false), layer 1 is MoE', () {
      final rope = RopeCache(
        maxCtx: _tinyCfg.maxCtx,
        headDim: _tinyCfg.mlaConfig.qkRopeHeadDim,
      );
      final b0 = DeepSeekV2Block(layerIndex: 0, config: _tinyCfg, rope: rope);
      final b1 = DeepSeekV2Block(layerIndex: 1, config: _tinyCfg, rope: rope);
      expect(b0.isMoE, isFalse);
      expect(b0.denseFfn, isNotNull);
      expect(b0.moeFfn, isNull);
      expect(b1.isMoE, isTrue);
      expect(b1.denseFfn, isNull);
      expect(b1.moeFfn, isNotNull);
    });

    test('dense block forward preserves [N, D]', () {
      final rope = RopeCache(
        maxCtx: _tinyCfg.maxCtx,
        headDim: _tinyCfg.mlaConfig.qkRopeHeadDim,
      );
      final b = DeepSeekV2Block(layerIndex: 0, config: _tinyCfg, rope: rope);
      final rng = math.Random(1);
      final vals =
          Float32List.fromList(List.generate(5 * 32, (_) => rng.nextDouble()));
      final x = Tensor.fromFloat32List([5, 32], vals);
      expect(b(x).shape, equals([5, 32]));
    });

    test('MoE block forward preserves [N, D]', () {
      final rope = RopeCache(
        maxCtx: _tinyCfg.maxCtx,
        headDim: _tinyCfg.mlaConfig.qkRopeHeadDim,
      );
      final b = DeepSeekV2Block(layerIndex: 1, config: _tinyCfg, rope: rope);
      final rng = math.Random(2);
      final vals =
          Float32List.fromList(List.generate(4 * 32, (_) => rng.nextDouble()));
      final x = Tensor.fromFloat32List([4, 32], vals);
      expect(b(x).shape, equals([4, 32]));
    });
  });

  group('DeepSeekV2Model', () {
    test('output shape [seqLen, vocabSize]', () {
      final m = DeepSeekV2Model(_tinyCfg);
      final tokens = Tensor.fromList([5], [1.0, 3.0, 7.0, 11.0, 42.0]);
      final logits = m(tokens);
      expect(logits.shape, equals([5, _tinyCfg.vocabSize]));
    });

    test('block count matches numLayers', () {
      final m = DeepSeekV2Model(_tinyCfg);
      expect(m.blocks.length, 2);
    });

    test('untiedHead present when tieWordEmbeddings=false', () {
      final m = DeepSeekV2Model(_tinyCfg);
      expect(m.untiedHead, isNotNull);
    });

    test('untiedHead null when tieWordEmbeddings=true', () {
      final tied = DeepSeekV2Config(
        vocabSize: _tinyCfg.vocabSize,
        maxCtx: _tinyCfg.maxCtx,
        embedDim: _tinyCfg.embedDim,
        numLayers: _tinyCfg.numLayers,
        firstKDenseReplace: _tinyCfg.firstKDenseReplace,
        denseFfnDim: _tinyCfg.denseFfnDim,
        moeExpertHiddenDim: _tinyCfg.moeExpertHiddenDim,
        numRoutedExperts: _tinyCfg.numRoutedExperts,
        numSharedExperts: _tinyCfg.numSharedExperts,
        numExpertsPerTok: _tinyCfg.numExpertsPerTok,
        mlaConfig: _tinyCfg.mlaConfig,
        tieWordEmbeddings: true,
      );
      final m = DeepSeekV2Model(tied);
      expect(m.untiedHead, isNull);
      // Tied forward still returns correct shape.
      final tokens = Tensor.fromList([3], [1.0, 2.0, 3.0]);
      expect(m(tokens).shape, equals([3, tied.vocabSize]));
    });

    test('rejects empty and too-long sequences', () {
      final m = DeepSeekV2Model(_tinyCfg);
      expect(() => m(Tensor.fromList([0], [])), throwsArgumentError);
      final tooLong = Tensor.fromList(
        [_tinyCfg.maxCtx + 1],
        List<double>.filled(_tinyCfg.maxCtx + 1, 1.0),
      );
      expect(() => m(tooLong), throwsArgumentError);
    });

    test('all logits are finite', () {
      final m = DeepSeekV2Model(_tinyCfg);
      final tokens = Tensor.fromList([4], [1.0, 5.0, 9.0, 13.0]);
      final logits = m(tokens).toList();
      for (int i = 0; i < logits.length; i++) {
        expect(logits[i].isFinite, isTrue,
            reason: 'logit $i is ${logits[i]}');
      }
    });
  });

  group('DeepSeekV2 on GPU', () {
    final gpuOk = _gpuAvailable();
    if (!gpuOk) {
      test('GPU unavailable → skipped', () {},
          skip: 'CUDA / native/lib/libmat_mul.so not usable in this env');
      return;
    }
    test('single dense block CPU vs GPU parity', () {
      final ropeCpu = RopeCache(
        maxCtx: _tinyCfg.maxCtx,
        headDim: _tinyCfg.mlaConfig.qkRopeHeadDim,
      );
      final ropeGpu = RopeCache(
        maxCtx: _tinyCfg.maxCtx,
        headDim: _tinyCfg.mlaConfig.qkRopeHeadDim,
        device: Device.GPU,
      );
      final cpu = DeepSeekV2Block(
        layerIndex: 0,
        config: _tinyCfg,
        rope: ropeCpu,
      );
      final gpuCfg = DeepSeekV2Config(
        vocabSize: _tinyCfg.vocabSize,
        maxCtx: _tinyCfg.maxCtx,
        embedDim: _tinyCfg.embedDim,
        numLayers: _tinyCfg.numLayers,
        firstKDenseReplace: _tinyCfg.firstKDenseReplace,
        denseFfnDim: _tinyCfg.denseFfnDim,
        moeExpertHiddenDim: _tinyCfg.moeExpertHiddenDim,
        numRoutedExperts: _tinyCfg.numRoutedExperts,
        numSharedExperts: _tinyCfg.numSharedExperts,
        numExpertsPerTok: _tinyCfg.numExpertsPerTok,
        mlaConfig: _tinyCfg.mlaConfig,
        device: Device.GPU,
      );
      final gpu = DeepSeekV2Block(
        layerIndex: 0,
        config: gpuCfg,
        rope: ropeGpu,
      );
      final cpuP = cpu.parameters();
      final gpuP = gpu.parameters();
      expect(cpuP.length, gpuP.length);
      for (int i = 0; i < cpuP.length; i++) {
        gpuP[i].assign(
          Tensor.fromList(cpuP[i].shape, cpuP[i].toList(), device: Device.GPU),
        );
      }
      final xVals =
          List<double>.generate(5 * 32, (i) => math.sin(i * 0.11));
      final xCpu = Tensor.fromList([5, 32], xVals);
      final xGpu = Tensor.fromList([5, 32], xVals, device: Device.GPU);
      final maskCpu = causalMask(5);
      final maskGpu = causalMask(5, device: Device.GPU);
      final yCpu = cpu(xCpu, mask: maskCpu).toList();
      final yGpu = gpu(xGpu, mask: maskGpu).toList();
      double maxDiff = 0;
      for (int i = 0; i < yCpu.length; i++) {
        final d = (yCpu[i] - yGpu[i]).abs();
        if (d > maxDiff) maxDiff = d;
      }
      expect(maxDiff, lessThan(5e-3), reason: 'max cpu/gpu diff = $maxDiff');
    });
  });
}
