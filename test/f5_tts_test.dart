@Timeout(Duration(minutes: 2))
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

Tensor _fake(List<int> shape, {int seed = 0}) {
  final rng = math.Random(seed);
  var n = 1;
  for (final d in shape) {
    n *= d;
  }
  final v = Float32List(n);
  for (int i = 0; i < n; i++) {
    v[i] = rng.nextDouble() - 0.5;
  }
  return Tensor.fromFloat32List(shape, v);
}

void main() {
  group('SinusoidalTimestepEmbedding', () {
    test('output shape [embedDim]', () {
      final emb = SinusoidalTimestepEmbedding(freqDim: 32, embedDim: 64);
      final t = Tensor.fromList([1], [0.5]);
      final out = emb(t);
      expect(out.shape, equals([64]));
    });

    test('different timesteps produce different embeddings', () {
      final emb = SinusoidalTimestepEmbedding(freqDim: 32, embedDim: 64);
      final e0 = emb(Tensor.fromList([1], [0.0])).toList();
      final e1 = emb(Tensor.fromList([1], [1.0])).toList();
      double maxDiff = 0;
      for (int i = 0; i < e0.length; i++) {
        final d = (e0[i] - e1[i]).abs();
        if (d > maxDiff) maxDiff = d;
      }
      expect(maxDiff > 0.0, isTrue);
    });

    test('rejects wrong-shape input', () {
      final emb = SinusoidalTimestepEmbedding(freqDim: 32, embedDim: 64);
      expect(() => emb(Tensor.fromList([2], [0.5, 0.7])), throwsArgumentError);
    });

    test('rejects odd freqDim', () {
      expect(
        () => SinusoidalTimestepEmbedding(freqDim: 33, embedDim: 64),
        throwsArgumentError,
      );
    });
  });

  group('AdaLNZero', () {
    test('zero-init produces zero modulation vectors', () {
      final adaLn = AdaLNZero(embedDim: 16);
      final c = _fake([16], seed: 1).reshape([16]);
      final mod = adaLn(c).toList();
      for (int i = 0; i < mod.length; i++) {
        expect(
          mod[i].abs() < 1e-6,
          isTrue,
          reason: 'mod[$i] = ${mod[i]} — should be zero at init',
        );
      }
    });

    test('output shape [6, embedDim]', () {
      final adaLn = AdaLNZero(embedDim: 32);
      final c = _fake([32], seed: 2);
      expect(adaLn(c).shape, equals([6, 32]));
    });

    test('non-zero modulation after we perturb the projection', () {
      final adaLn = AdaLNZero(embedDim: 16);
      // Perturb the modulation weight so the block leaves the identity
      // regime. Any non-zero value across the 96-cell block is enough.
      final w = List<double>.filled(adaLn.modulation.weight.length, 0.0);
      w[0] = 0.5;
      adaLn.modulation.weight.assign(
        Tensor.fromList(
          adaLn.modulation.weight.shape,
          w,
          device: adaLn.modulation.weight.device,
        ),
      );
      final c = Tensor.fromList([16], List<double>.filled(16, 1.0));
      final mod = adaLn(c).toList();
      double maxAbs = 0;
      for (final v in mod) {
        if (v.abs() > maxAbs) maxAbs = v.abs();
      }
      expect(maxAbs > 0, isTrue);
    });
  });

  group('adaLNModulate + adaLNGate', () {
    test('modulate is xn * (1 + scale) + shift row-wise', () {
      final xn = Tensor.fromList([2, 3], [1, 2, 3, 4, 5, 6]);
      final scale = Tensor.fromList([3], [0.1, 0.0, -0.5]);
      final shift = Tensor.fromList([3], [10, 20, 30]);
      final out = adaLNModulate(xn, scale, shift).toList();
      // row 0: 1*(1.1)+10 = 11.1, 2*(1.0)+20 = 22.0, 3*(0.5)+30 = 31.5
      // row 1: 4*1.1+10 = 14.4, 5*1.0+20 = 25.0, 6*0.5+30 = 33.0
      expect((out[0] - 11.1).abs() < 1e-5, isTrue);
      expect((out[1] - 22.0).abs() < 1e-5, isTrue);
      expect((out[2] - 31.5).abs() < 1e-5, isTrue);
      expect((out[3] - 14.4).abs() < 1e-5, isTrue);
      expect((out[4] - 25.0).abs() < 1e-5, isTrue);
      expect((out[5] - 33.0).abs() < 1e-5, isTrue);
    });

    test('gate is x * gate row-wise', () {
      final x = Tensor.fromList([2, 3], [1, 2, 3, 4, 5, 6]);
      final gate = Tensor.fromList([3], [0.0, 2.0, -1.0]);
      final out = adaLNGate(x, gate).toList();
      expect(out, equals([0.0, 4.0, -3.0, 0.0, 10.0, -6.0]));
    });
  });

  group('F5DiTBlock', () {
    test('at init the block is an identity residual (zero modulation)', () {
      final block = F5DiTBlock(embedDim: 32, numHeads: 4, mlpDim: 64);
      final x = _fake([5, 32], seed: 7);
      final c = _fake([32], seed: 8);
      final y = block(x, c);
      // With AdaLNZero at init, gate_msa = gate_mlp = 0, so both
      // residual updates are zero. Output must equal input.
      final xVals = x.toList();
      final yVals = y.toList();
      double maxDiff = 0;
      for (int i = 0; i < xVals.length; i++) {
        final d = (xVals[i] - yVals[i]).abs();
        if (d > maxDiff) maxDiff = d;
      }
      expect(
        maxDiff < 1e-5,
        isTrue,
        reason: 'DiT block should be identity at init; max diff = $maxDiff',
      );
    });

    test('after perturbing modulation, block leaves identity regime', () {
      final block = F5DiTBlock(embedDim: 32, numHeads: 4, mlpDim: 64);
      // Perturb the AdaLN modulation weight matrix so gate_msa is
      // non-zero.
      final w = List<double>.filled(block.adaLn.modulation.weight.length, 0.1);
      block.adaLn.modulation.weight.assign(
        Tensor.fromList(
          block.adaLn.modulation.weight.shape,
          w,
          device: block.adaLn.modulation.weight.device,
        ),
      );
      final x = _fake([5, 32], seed: 11);
      final c = _fake([32], seed: 12);
      final y = block(x, c);
      final xVals = x.toList();
      final yVals = y.toList();
      double maxDiff = 0;
      for (int i = 0; i < xVals.length; i++) {
        final d = (xVals[i] - yVals[i]).abs();
        if (d > maxDiff) maxDiff = d;
      }
      expect(
        maxDiff > 1e-3,
        isTrue,
        reason: 'perturbed block should differ from identity',
      );
    });

    test('rejects wrong-shape input', () {
      final block = F5DiTBlock(embedDim: 32, numHeads: 4, mlpDim: 64);
      final c = _fake([32], seed: 13);
      expect(() => block(_fake([5, 33], seed: 14), c), throwsArgumentError);
    });
  });

  group('gaussianNoise', () {
    test('produces the requested shape', () {
      final n = gaussianNoise([4, 5], seed: 1);
      expect(n.shape, equals([4, 5]));
    });

    test('empirical mean/std are close to N(0, 1) for large sample', () {
      final n = gaussianNoise([1000], seed: 2).toList();
      double sum = 0;
      for (final v in n) {
        sum += v;
      }
      final mean = sum / n.length;
      double sq = 0;
      for (final v in n) {
        sq += (v - mean) * (v - mean);
      }
      final std = math.sqrt(sq / n.length);
      expect(mean.abs() < 0.1, isTrue, reason: 'mean=$mean');
      expect((std - 1.0).abs() < 0.1, isTrue, reason: 'std=$std');
    });

    test('deterministic given the same seed', () {
      final a = gaussianNoise([16], seed: 42).toList();
      final b = gaussianNoise([16], seed: 42).toList();
      expect(a, equals(b));
    });
  });

  group('FlowMatchingSampler', () {
    test('linear velocity v(x, t) = 1 integrates from 0 to 1 exactly', () {
      // dx/dt = 1  =>  x(1) = x(0) + 1.
      const sampler = FlowMatchingSampler(numSteps: 8);
      final x0 = Tensor.fromList([3], [0.0, 5.0, -2.0]);
      final x1 = sampler.sample(
        initialNoise: x0,
        velocityField: (x, t) =>
            Tensor.fromList(x.shape, List<double>.filled(x.length, 1.0)),
      );
      final vals = x1.toList();
      expect((vals[0] - 1.0).abs() < 1e-5, isTrue);
      expect((vals[1] - 6.0).abs() < 1e-5, isTrue);
      expect((vals[2] - -1.0).abs() < 1e-5, isTrue);
    });

    test('midpoint solver is exact for constant velocity too', () {
      const sampler = FlowMatchingSampler(
        numSteps: 4,
        solver: FlowSolver.midpoint,
      );
      final x0 = Tensor.fromList([2], [0.0, 10.0]);
      final x1 = sampler.sample(
        initialNoise: x0,
        velocityField: (x, t) => Tensor.fromList(x.shape, [0.5, 2.0]),
      );
      final vals = x1.toList();
      // x0 + 1 * dv where dv = velocity * 1 for whole interval.
      expect((vals[0] - 0.5).abs() < 1e-5, isTrue);
      expect((vals[1] - 12.0).abs() < 1e-5, isTrue);
    });

    test('midpoint solver has smaller error than Euler on '
        'time-varying velocity', () {
      // dx/dt = 2·t → x(1) = x(0) + t² |₀¹ = x(0) + 1.
      Tensor v(Tensor x, Tensor t) {
        final tt = t.toList()[0];
        return Tensor.fromList(x.shape, List<double>.filled(x.length, 2 * tt));
      }

      const nSteps = 4;
      const euler = FlowMatchingSampler(numSteps: nSteps);
      const midpoint = FlowMatchingSampler(
        numSteps: nSteps,
        solver: FlowSolver.midpoint,
      );
      final x0 = Tensor.fromList([1], [0.0]);
      final xE = euler.sample(initialNoise: x0, velocityField: v);
      final xM = midpoint.sample(initialNoise: x0, velocityField: v);
      final eErr = (xE.toList()[0] - 1.0).abs();
      final mErr = (xM.toList()[0] - 1.0).abs();
      expect(
        mErr < eErr,
        isTrue,
        reason: 'midpoint err $mErr should beat Euler err $eErr',
      );
    });

    test('velocity shape mismatch throws', () {
      const sampler = FlowMatchingSampler(numSteps: 4);
      expect(
        () => sampler.sample(
          initialNoise: Tensor.fromList([3], [0, 1, 2]),
          velocityField: (x, t) => Tensor.fromList([2], [0, 0]),
        ),
        throwsArgumentError,
      );
    });

    test('numSteps=32 shape roundtrip on a 2-D noise shape', () {
      const sampler = FlowMatchingSampler(numSteps: 32);
      final z = gaussianNoise([4, 8], seed: 99);
      final out = sampler.sample(
        initialNoise: z,
        velocityField: (x, t) =>
            Tensor.fromList(x.shape, List<double>.filled(x.length, 0.0)),
      );
      // Zero velocity → sample equals input noise.
      expect(out.shape, equals([4, 8]));
      final zVals = z.toList();
      final outVals = out.toList();
      for (int i = 0; i < zVals.length; i++) {
        expect((zVals[i] - outVals[i]).abs() < 1e-5, isTrue);
      }
    });
  });

  group('DepthwiseConv1d', () {
    test('preserves shape when padding = (k-1)/2', () {
      final dw = DepthwiseConv1d(channels: 4, kernelSize: 7, padding: 3);
      final x = _fake([2, 4, 16], seed: 100);
      expect(dw(x).shape, equals([2, 4, 16]));
    });

    test(
      'channel independence — perturbing one channel does not affect others',
      () {
        final dw = DepthwiseConv1d(channels: 3, kernelSize: 3, padding: 1);
        final base = _fake([1, 3, 8], seed: 200);
        final vals = base.toList();
        // Perturb only channel 0.
        for (int i = 0; i < 8; i++) {
          vals[i] += 5.0;
        }
        final perturbed = Tensor.fromList([1, 3, 8], vals);
        final outBase = dw(base).toList();
        final outPert = dw(perturbed).toList();
        // Channel 0 (indices 0..7) differs; channels 1 and 2 (indices
        // 8..15 and 16..23) are unchanged.
        double maxCh0 = 0;
        for (int i = 0; i < 8; i++) {
          final d = (outBase[i] - outPert[i]).abs();
          if (d > maxCh0) maxCh0 = d;
        }
        double maxOthers = 0;
        for (int i = 8; i < 24; i++) {
          final d = (outBase[i] - outPert[i]).abs();
          if (d > maxOthers) maxOthers = d;
        }
        expect(maxCh0 > 0.1, isTrue);
        expect(maxOthers < 1e-5, isTrue);
      },
    );

    test('rejects wrong-shape input', () {
      final dw = DepthwiseConv1d(channels: 4, kernelSize: 3);
      expect(() => dw(_fake([1, 5, 8], seed: 300)), throwsArgumentError);
    });
  });

  group('GlobalResponseNormalization', () {
    test('preserves shape', () {
      final grn = GlobalResponseNormalization(channels: 4);
      final x = _fake([2, 8, 4], seed: 400);
      expect(grn(x).shape, equals([2, 8, 4]));
    });

    test('at init (γ=0, β=0) is identity residual', () {
      final grn = GlobalResponseNormalization(channels: 6);
      final x = _fake([1, 5, 6], seed: 500);
      final out = grn(x);
      // With γ=β=0 the block collapses to x + 0 = x.
      final xVals = x.toList();
      final outVals = out.toList();
      double maxDiff = 0;
      for (int i = 0; i < xVals.length; i++) {
        final d = (xVals[i] - outVals[i]).abs();
        if (d > maxDiff) maxDiff = d;
      }
      expect(
        maxDiff < 1e-5,
        isTrue,
        reason: 'GRN identity at init; max diff = $maxDiff',
      );
    });
  });

  group('ConvNeXtV2Block', () {
    test('preserves [N, T, dim] shape', () {
      final b = ConvNeXtV2Block(dim: 16, intermediateDim: 32);
      final x = _fake([2, 10, 16], seed: 600);
      expect(b(x).shape, equals([2, 10, 16]));
    });

    test('rejects wrong-shape input', () {
      final b = ConvNeXtV2Block(dim: 16, intermediateDim: 32);
      expect(() => b(_fake([2, 10, 17], seed: 700)), throwsArgumentError);
    });
  });

  group('F5TextEncoder', () {
    test('forward output shape [T, dim]', () {
      final enc = F5TextEncoder(
        vocabSize: 100,
        dim: 32,
        intermediateDim: 64,
        numLayers: 2,
      );
      final tokens = Tensor.fromList([
        12,
      ], List<double>.generate(12, (i) => (i * 7 % 100).toDouble()));
      expect(enc(tokens).shape, equals([12, 32]));
    });

    test('rejects non-1D input', () {
      final enc = F5TextEncoder(
        vocabSize: 50,
        dim: 32,
        intermediateDim: 64,
        numLayers: 1,
      );
      expect(
        () => enc(Tensor.fromList([2, 3], [1, 2, 3, 4, 5, 6])),
        throwsArgumentError,
      );
    });
  });

  group('F5DiT', () {
    test('forward output shape [T, melDim]', () {
      final dit = F5DiT(
        melDim: 8,
        textDim: 16,
        embedDim: 32,
        numLayers: 2,
        numHeads: 4,
        mlpDim: 64,
        freqDim: 16,
      );
      final mel = _fake([6, 8], seed: 800).reshape([6, 8]);
      final text = _fake([6, 16], seed: 801).reshape([6, 16]);
      final t = Tensor.fromList([1], [0.5]);
      expect(dit(mel, text, t).shape, equals([6, 8]));
    });

    test('at init the DiT preserves the mel signal shape but does not '
        'necessarily match input (only DiT blocks are identity at init; '
        'inputProj + outputProj still transform the signal)', () {
      final dit = F5DiT(
        melDim: 8,
        textDim: 16,
        embedDim: 32,
        numLayers: 2,
        numHeads: 4,
        mlpDim: 64,
        freqDim: 16,
      );
      final mel = _fake([4, 8], seed: 900);
      final text = _fake([4, 16], seed: 901);
      final t = Tensor.fromList([1], [0.2]);
      final v = dit(mel, text, t);
      expect(v.shape, equals([4, 8]));
      for (final x in v.toList()) {
        expect(x.isFinite, isTrue);
      }
    });

    test('rejects mismatched T', () {
      final dit = F5DiT(
        melDim: 8,
        textDim: 16,
        embedDim: 32,
        numLayers: 1,
        numHeads: 4,
        mlpDim: 32,
        freqDim: 16,
      );
      expect(
        () => dit(
          _fake([5, 8], seed: 1000),
          _fake([6, 16], seed: 1001), // wrong T
          Tensor.fromList([1], [0.5]),
        ),
        throwsArgumentError,
      );
    });
  });

  group('F5DurationPredictor', () {
    test('output shape [T]', () {
      final dp = F5DurationPredictor(
        textDim: 32,
        intermediateDim: 64,
        numLayers: 2,
      );
      final text = _fake([12, 32], seed: 1100);
      expect(dp(text).shape, equals([12]));
    });

    test('all durations are non-negative (softplus output)', () {
      final dp = F5DurationPredictor(
        textDim: 32,
        intermediateDim: 64,
        numLayers: 1,
      );
      final text = _fake([8, 32], seed: 1200);
      for (final v in dp(text).toList()) {
        expect(v >= 0, isTrue);
      }
    });

    test('expandTextToFrames broadcasts by rounded durations', () {
      final text = Tensor.fromList(
        [3, 2],
        [
          1.0, 1.1, //
          2.0, 2.2, //
          3.0, 3.3, //
        ],
      );
      final dur = Tensor.fromList([3], [1.0, 2.0, 0.0]);
      final expanded = F5DurationPredictor.expandTextToFrames(text, dur);
      // 1 frame of row 0, 2 frames of row 1, 0 frames of row 2.
      expect(expanded.shape, equals([3, 2]));
      final vals = expanded.toList();
      final want = [1.0, 1.1, 2.0, 2.2, 2.0, 2.2];
      for (int i = 0; i < want.length; i++) {
        expect(
          (vals[i] - want[i]).abs() < 1e-4,
          isTrue,
          reason: 'i=$i vals=${vals[i]} want=${want[i]}',
        );
      }
    });

    test('expandTextToFrames zero-duration char dropped', () {
      final text = Tensor.fromList([2, 3], [1, 2, 3, 4, 5, 6]);
      final dur = Tensor.fromList([2], [0.0, 3.0]);
      final expanded = F5DurationPredictor.expandTextToFrames(text, dur);
      expect(expanded.shape, equals([3, 3]));
      final vals = expanded.toList();
      expect(vals, equals([4, 5, 6, 4, 5, 6, 4, 5, 6]));
    });
  });

  group('F5TtsHFLoader roundtrip', () {
    Map<String, Tensor> _dumpBundle(
      F5TextEncoder te,
      F5DiT dit,
      F5DurationPredictor? dp,
    ) {
      final s = <String, Tensor>{};
      // Text encoder.
      s['text_embed.embedding.weight'] = te.tokenEmbedding.weight;
      for (int i = 0; i < te.blocks.length; i++) {
        _dumpConvBlock(te.blocks[i], s, 'text_embed.blocks.$i');
      }
      s['text_embed.norm.weight'] = te.finalNorm.gamma;
      s['text_embed.norm.bias'] = te.finalNorm.beta;
      // DiT.
      s['input_proj.weight'] = dit.inputProj.weight;
      s['input_proj.bias'] = Tensor.fromList([
        dit.embedDim,
      ], dit.inputProj.bias!.toList());
      s['time_embed.freq_proj.weight'] = dit.timeEmbed.proj1.weight;
      s['time_embed.freq_proj.bias'] = Tensor.fromList([
        dit.embedDim,
      ], dit.timeEmbed.proj1.bias!.toList());
      s['time_embed.out_proj.weight'] = dit.timeEmbed.proj2.weight;
      s['time_embed.out_proj.bias'] = Tensor.fromList([
        dit.embedDim,
      ], dit.timeEmbed.proj2.bias!.toList());
      final nh = dit.numHeads;
      final hd = dit.embedDim ~/ nh;
      for (int i = 0; i < dit.blocks.length; i++) {
        final b = dit.blocks[i];
        final p = 'blocks.$i';
        s['$p.norm1.weight'] = b.norm1.gamma;
        s['$p.norm1.bias'] = b.norm1.beta;
        // Rebuild fused QKV.
        for (final (proj, list) in [
          ('q_proj', b.attn.wq),
          ('k_proj', b.attn.wk),
          ('v_proj', b.attn.wv),
        ]) {
          final wRows = <double>[];
          final bRows = <double>[];
          for (int h = 0; h < nh; h++) {
            wRows.addAll(list[h].weight.toList());
            bRows.addAll(list[h].bias!.toList());
          }
          s['$p.attn.$proj.weight'] = Tensor.fromList([
            dit.embedDim,
            dit.embedDim,
          ], wRows);
          s['$p.attn.$proj.bias'] = Tensor.fromList([dit.embedDim], bRows);
          // silence hd unused
          (hd);
        }
        s['$p.attn.o_proj.weight'] = b.attn.wo.weight;
        s['$p.attn.o_proj.bias'] = Tensor.fromList([
          dit.embedDim,
        ], b.attn.wo.bias!.toList());
        s['$p.norm2.weight'] = b.norm2.gamma;
        s['$p.norm2.bias'] = b.norm2.beta;
        s['$p.fc1.weight'] = b.fc1.weight;
        s['$p.fc1.bias'] = Tensor.fromList([dit.mlpDim], b.fc1.bias!.toList());
        s['$p.fc2.weight'] = b.fc2.weight;
        s['$p.fc2.bias'] = Tensor.fromList([
          dit.embedDim,
        ], b.fc2.bias!.toList());
        s['$p.adaLN.modulation.weight'] = b.adaLn.modulation.weight;
        s['$p.adaLN.modulation.bias'] = Tensor.fromList([
          6 * dit.embedDim,
        ], b.adaLn.modulation.bias!.toList());
      }
      s['norm_out.weight'] = dit.finalNorm.gamma;
      s['norm_out.bias'] = dit.finalNorm.beta;
      s['output_proj.weight'] = dit.outputProj.weight;
      s['output_proj.bias'] = Tensor.fromList([
        dit.melDim,
      ], dit.outputProj.bias!.toList());

      if (dp != null) {
        for (int i = 0; i < dp.blocks.length; i++) {
          _dumpConvBlock(dp.blocks[i], s, 'duration.blocks.$i');
        }
        s['duration.norm.weight'] = dp.finalNorm.gamma;
        s['duration.norm.bias'] = dp.finalNorm.beta;
        s['duration.head.weight'] = dp.head.weight;
        s['duration.head.bias'] = Tensor.fromList([1], dp.head.bias!.toList());
      }
      return s;
    }

    test('text-encoder + DiT + duration roundtrip consumes all keys', () {
      final te = F5TextEncoder(
        vocabSize: 100,
        dim: 32,
        intermediateDim: 64,
        numLayers: 2,
      );
      final dit = F5DiT(
        melDim: 8,
        textDim: 16,
        embedDim: 32,
        numLayers: 2,
        numHeads: 4,
        mlpDim: 64,
        freqDim: 16,
      );
      final dp = F5DurationPredictor(
        textDim: 32,
        intermediateDim: 64,
        numLayers: 1,
      );
      final state = _dumpBundle(te, dit, dp);

      final teDst = F5TextEncoder(
        vocabSize: 100,
        dim: 32,
        intermediateDim: 64,
        numLayers: 2,
        seed: 999,
      );
      final ditDst = F5DiT(
        melDim: 8,
        textDim: 16,
        embedDim: 32,
        numLayers: 2,
        numHeads: 4,
        mlpDim: 64,
        freqDim: 16,
        seed: 999,
      );
      final dpDst = F5DurationPredictor(
        textDim: 32,
        intermediateDim: 64,
        numLayers: 1,
        seed: 999,
      );
      final report = F5TtsHFLoader.loadMap(
        textEncoder: teDst,
        dit: ditDst,
        durationPredictor: dpDst,
        state: state,
      );
      expect(
        report.unusedKeys,
        isEmpty,
        reason: 'unused: ${report.unusedKeys.take(5).toList()}',
      );
    });

    test('without durationPredictor, duration keys go to unusedKeys', () {
      final te = F5TextEncoder(
        vocabSize: 100,
        dim: 32,
        intermediateDim: 64,
        numLayers: 1,
      );
      final dit = F5DiT(
        melDim: 8,
        textDim: 16,
        embedDim: 32,
        numLayers: 1,
        numHeads: 4,
        mlpDim: 32,
        freqDim: 16,
      );
      final dp = F5DurationPredictor(
        textDim: 32,
        intermediateDim: 64,
        numLayers: 1,
      );
      final state = _dumpBundle(te, dit, dp);
      final report = F5TtsHFLoader.loadMap(
        textEncoder: te,
        dit: dit,
        durationPredictor: null,
        state: state,
      );
      expect(report.unusedKeys, isNotEmpty);
      for (final k in report.unusedKeys) {
        expect(
          k.startsWith('duration.'),
          isTrue,
          reason: 'unexpected unused key: $k',
        );
      }
    });

    test('rejects missing key', () {
      final te = F5TextEncoder(
        vocabSize: 50,
        dim: 16,
        intermediateDim: 32,
        numLayers: 1,
      );
      final dit = F5DiT(
        melDim: 4,
        textDim: 8,
        embedDim: 16,
        numLayers: 1,
        numHeads: 2,
        mlpDim: 16,
        freqDim: 8,
      );
      final state = _dumpBundle(te, dit, null);
      state.remove('norm_out.weight');
      expect(
        () => F5TtsHFLoader.loadMap(textEncoder: te, dit: dit, state: state),
        throwsArgumentError,
      );
    });
  });
}

void _dumpConvBlock(ConvNeXtV2Block b, Map<String, Tensor> s, String prefix) {
  s['$prefix.dwconv.weight'] = b.dwconv.weight;
  s['$prefix.dwconv.bias'] = Tensor.fromList([b.dim], b.dwconv.bias!.toList());
  s['$prefix.norm.weight'] = b.norm.gamma;
  s['$prefix.norm.bias'] = b.norm.beta;
  s['$prefix.pwconv1.weight'] = b.pwconv1.weight;
  s['$prefix.pwconv1.bias'] = Tensor.fromList([
    b.intermediateDim,
  ], b.pwconv1.bias!.toList());
  s['$prefix.grn.gamma'] = b.grn.gamma;
  s['$prefix.grn.beta'] = b.grn.beta;
  s['$prefix.pwconv2.weight'] = b.pwconv2.weight;
  s['$prefix.pwconv2.bias'] = Tensor.fromList([
    b.dim,
  ], b.pwconv2.bias!.toList());
}
