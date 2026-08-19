@Timeout(Duration(minutes: 3))
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

Tensor _rand(List<int> shape, {int seed = 0, Device device = Device.CPU}) {
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

/// Tiny MLA config that exercises all the low-rank + rope-decoupled
/// machinery without eating minutes of CPU time.
const _tinyCfg = MLAConfig(
  embedDim: 32,
  numHeads: 4,
  qLoraRank: 16,
  kvLoraRank: 12,
  qkNopeHeadDim: 6,
  qkRopeHeadDim: 4,
  vHeadDim: 6,
  rmsNormEps: 1e-6,
);

void main() {
  group('MLAConfig', () {
    test('qkHeadDim = nope + rope', () {
      expect(_tinyCfg.qkHeadDim, 10);
    });

    test('deepseekV2LiteConfig matches HF config.json exactly', () {
      final cfg = MLAConfig.deepseekV2LiteConfig();
      expect(cfg.embedDim, 2048);
      expect(cfg.numHeads, 16);
      expect(cfg.qLoraRank, isNull); // Lite has no Q compression
      expect(cfg.qInDim, 2048); // falls back to embedDim
      expect(cfg.kvLoraRank, 512);
      expect(cfg.qkNopeHeadDim, 128);
      expect(cfg.qkRopeHeadDim, 64);
      expect(cfg.vHeadDim, 128);
      expect(cfg.qkHeadDim, 192);
    });

    test('deepseekV2Config (full 236B) has Q compression at 1536', () {
      final cfg = MLAConfig.deepseekV2Config();
      expect(cfg.embedDim, 5120);
      expect(cfg.numHeads, 128);
      expect(cfg.qLoraRank, 1536);
      expect(cfg.qInDim, 1536);
    });
  });

  group('MultiHeadLatentAttention forward', () {
    test('output shape [N, embedDim] (no RoPE)', () {
      final mla = MultiHeadLatentAttention(_tinyCfg);
      final x = _rand([5, 32], seed: 1);
      final y = mla(x);
      expect(y.shape, equals([5, 32]));
    });

    test('output shape unchanged when RoPE is attached', () {
      final mla = MultiHeadLatentAttention(_tinyCfg);
      mla.rope = RopeCache(maxCtx: 32, headDim: _tinyCfg.qkRopeHeadDim);
      final x = _rand([5, 32], seed: 2);
      expect(mla(x).shape, equals([5, 32]));
    });

    test('causal mask is respected — perturbing last token cannot '
        'affect earlier hidden states', () {
      final mla = MultiHeadLatentAttention(_tinyCfg);
      mla.rope = RopeCache(maxCtx: 8, headDim: _tinyCfg.qkRopeHeadDim);
      final xBase = _rand([4, 32], seed: 3);
      // Build a modified version where only row 3 differs.
      final vals = xBase.toList();
      for (int j = 0; j < 32; j++) {
        vals[3 * 32 + j] += 5.0;
      }
      final xMod = Tensor.fromList([4, 32], vals);
      final mask = causalMask(4);
      final base = mla(xBase, mask: mask).toList();
      final mod = mla(xMod, mask: mask).toList();
      // Rows 0..2 (positions 0, 1, 2) must be identical.
      for (int i = 0; i < 3 * 32; i++) {
        expect(
          (base[i] - mod[i]).abs() < 1e-5,
          isTrue,
          reason: 'causal violation at i=$i: base=${base[i]} mod=${mod[i]}',
        );
      }
      // Row 3 must actually differ.
      double totalDelta = 0;
      for (int i = 3 * 32; i < 4 * 32; i++) {
        totalDelta += (base[i] - mod[i]).abs();
      }
      expect(
        totalDelta > 0.01,
        isTrue,
        reason: 'last-token perturbation had no effect',
      );
    });

    test('rejects wrong input dim', () {
      final mla = MultiHeadLatentAttention(_tinyCfg);
      expect(() => mla(_rand([5, 33], seed: 4)), throwsArgumentError);
    });

    test('rejects mismatched RopeCache headDim', () {
      final mla = MultiHeadLatentAttention(_tinyCfg);
      mla.rope = RopeCache(maxCtx: 32, headDim: 8); // != qkRopeHeadDim
      expect(() => mla(_rand([3, 32], seed: 5)), throwsStateError);
    });

    test('parameter list surfaces every learnable tensor (with Q compression)',
        () {
      final mla = MultiHeadLatentAttention(_tinyCfg);
      final params = mla.parameters();
      // Counts:
      //   qDown: 1
      //   qLn (RMSNorm): 1 (gamma only)
      //   qUpNope: numHeads = 4
      //   qUpRope: numHeads = 4
      //   kvDown: 1
      //   kvLn: 1
      //   kRope: 1
      //   kUpNope: numHeads = 4
      //   vUp: numHeads = 4
      //   oProj: 1
      final expected = 1 + 1 + 4 + 4 + 1 + 1 + 1 + 4 + 4 + 1;
      expect(params.length, expected);
    });

    test('no-Q-compression (Lite-style) — forward works + param count drops '
        'by 2', () {
      const liteTiny = MLAConfig(
        embedDim: 32,
        numHeads: 4,
        qLoraRank: null, // no q compression
        kvLoraRank: 12,
        qkNopeHeadDim: 6,
        qkRopeHeadDim: 4,
        vHeadDim: 6,
      );
      final mla = MultiHeadLatentAttention(liteTiny);
      final x = _rand([5, 32], seed: 8);
      expect(mla(x).shape, equals([5, 32]));
      final params = mla.parameters();
      // qDown + qLn are gone (drop 2 params).
      final expected = 4 + 4 + 1 + 1 + 1 + 4 + 4 + 1;
      expect(params.length, expected);
    });
  });

  group('MLA on GPU', () {
    final gpuOk = _gpuAvailable();
    if (!gpuOk) {
      test(
        'GPU unavailable → skipped',
        () {},
        skip: 'CUDA / native/lib/libmat_mul.so not usable in this env',
      );
      return;
    }
    test('CPU vs GPU forward parity (no RoPE)', () {
      final cpu = MultiHeadLatentAttention(_tinyCfg);
      final gpu = MultiHeadLatentAttention(_tinyCfg, device: Device.GPU);
      // Sync parameters.
      final cpuP = cpu.parameters();
      final gpuP = gpu.parameters();
      expect(cpuP.length, gpuP.length);
      for (int i = 0; i < cpuP.length; i++) {
        gpuP[i].assign(
          Tensor.fromList(cpuP[i].shape, cpuP[i].toList(), device: Device.GPU),
        );
      }
      final xVals = List<double>.generate(5 * 32, (i) => math.sin(i * 0.17));
      final xCpu = Tensor.fromList([5, 32], xVals);
      final xGpu = Tensor.fromList([5, 32], xVals, device: Device.GPU);
      final cpuOut = cpu(xCpu).toList();
      final gpuOut = gpu(xGpu).toList();
      double maxDiff = 0;
      for (int i = 0; i < cpuOut.length; i++) {
        final d = (cpuOut[i] - gpuOut[i]).abs();
        if (d > maxDiff) maxDiff = d;
      }
      expect(maxDiff, lessThan(5e-4), reason: 'max cpu/gpu diff = $maxDiff');
    });

    test('CPU vs GPU parity with RoPE + causal mask', () {
      final cpu = MultiHeadLatentAttention(_tinyCfg);
      final gpu = MultiHeadLatentAttention(_tinyCfg, device: Device.GPU);
      cpu.rope = RopeCache(maxCtx: 16, headDim: _tinyCfg.qkRopeHeadDim);
      gpu.rope = RopeCache(
        maxCtx: 16,
        headDim: _tinyCfg.qkRopeHeadDim,
        device: Device.GPU,
      );
      final cpuP = cpu.parameters();
      final gpuP = gpu.parameters();
      for (int i = 0; i < cpuP.length; i++) {
        gpuP[i].assign(
          Tensor.fromList(cpuP[i].shape, cpuP[i].toList(), device: Device.GPU),
        );
      }
      final xVals = List<double>.generate(6 * 32, (i) => math.cos(i * 0.09));
      final xCpu = Tensor.fromList([6, 32], xVals);
      final xGpu = Tensor.fromList([6, 32], xVals, device: Device.GPU);
      final maskCpu = causalMask(6);
      final maskGpu = causalMask(6, device: Device.GPU);
      final cpuOut = cpu(xCpu, mask: maskCpu).toList();
      final gpuOut = gpu(xGpu, mask: maskGpu).toList();
      double maxDiff = 0;
      for (int i = 0; i < cpuOut.length; i++) {
        final d = (cpuOut[i] - gpuOut[i]).abs();
        if (d > maxDiff) maxDiff = d;
      }
      expect(maxDiff, lessThan(5e-4), reason: 'max cpu/gpu diff = $maxDiff');
    });
  });
}
