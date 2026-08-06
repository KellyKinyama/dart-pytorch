// Tests for CLIPVisionModel + ClipHFLoader.
//
// The tests avoid needing a real 350 MB CLIP checkpoint by
// constructing a *tiny* CLIPVisionConfig (matching the HF layout
// exactly, just with smaller dims) and feeding a synthetic state
// dict that mimics HuggingFace safetensors keys. That exercises
// every code path in the loader — prefix detection, per-head
// slicing, Conv→Linear permutation, LayerNorm mapping — without
// downloading anything.

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

Tensor _randomImage(CLIPVisionConfig cfg, {int seed = 0}) {
  final rng = math.Random(seed);
  final n = cfg.numPatches * cfg.patchPixels;
  final vals = List<double>.generate(n, (_) => rng.nextDouble() * 2 - 1);
  return Tensor.fromList(
    [cfg.numPatches, cfg.patchPixels],
    vals,
    device: cfg.device,
  );
}

/// Build a synthetic `Map<String, Tensor>` that looks like an HF
/// safetensors state dict for a `CLIPVisionModel` of the given
/// [cfg]. All values are drawn from a small Gaussian so they don't
/// blow up on forward.
Map<String, Tensor> _syntheticState(
  CLIPVisionConfig cfg, {
  String prefix = 'vision_model.',
  int seed = 42,
}) {
  final rng = math.Random(seed);
  Tensor rand(List<int> shape) {
    final n = shape.fold<int>(1, (a, b) => a * b);
    final vals = List<double>.generate(n, (_) {
      final u1 = rng.nextDouble().clamp(1e-9, 1.0);
      final u2 = rng.nextDouble();
      final z = math.sqrt(-2.0 * math.log(u1)) *
          math.cos(2.0 * math.pi * u2);
      return z * 0.02;
    });
    return Tensor.fromList(shape, vals);
  }

  Tensor ones(List<int> shape) {
    final n = shape.fold<int>(1, (a, b) => a * b);
    return Tensor.fromList(shape, List<double>.filled(n, 1.0));
  }

  Tensor zeros(List<int> shape) {
    final n = shape.fold<int>(1, (a, b) => a * b);
    return Tensor.fromList(shape, List<double>.filled(n, 0.0));
  }

  final d = cfg.embedDim;
  final f = cfg.ffnDim;
  final out = <String, Tensor>{};

  out['${prefix}embeddings.patch_embedding.weight'] =
      rand([d, cfg.numChannels, cfg.patchSize, cfg.patchSize]);
  out['${prefix}embeddings.class_embedding'] = rand([d]);
  out['${prefix}embeddings.position_embedding.weight'] =
      rand([cfg.numPatches + 1, d]);

  out['${prefix}pre_layrnorm.weight'] = ones([d]);
  out['${prefix}pre_layrnorm.bias'] = zeros([d]);

  for (int i = 0; i < cfg.numLayers; i++) {
    final p = '${prefix}encoder.layers.$i';
    out['$p.layer_norm1.weight'] = ones([d]);
    out['$p.layer_norm1.bias'] = zeros([d]);
    out['$p.layer_norm2.weight'] = ones([d]);
    out['$p.layer_norm2.bias'] = zeros([d]);
    out['$p.self_attn.q_proj.weight'] = rand([d, d]);
    out['$p.self_attn.q_proj.bias'] = zeros([d]);
    out['$p.self_attn.k_proj.weight'] = rand([d, d]);
    out['$p.self_attn.k_proj.bias'] = zeros([d]);
    out['$p.self_attn.v_proj.weight'] = rand([d, d]);
    out['$p.self_attn.v_proj.bias'] = zeros([d]);
    out['$p.self_attn.out_proj.weight'] = rand([d, d]);
    out['$p.self_attn.out_proj.bias'] = zeros([d]);
    out['$p.mlp.fc1.weight'] = rand([f, d]);
    out['$p.mlp.fc1.bias'] = zeros([f]);
    out['$p.mlp.fc2.weight'] = rand([d, f]);
    out['$p.mlp.fc2.bias'] = zeros([d]);
  }

  out['${prefix}post_layernorm.weight'] = ones([d]);
  out['${prefix}post_layernorm.bias'] = zeros([d]);

  return out;
}

void main() {
  group('CLIPVisionModel', () {
    test('forward returns [numPatches + 1, embedDim]', () {
      const cfg = CLIPVisionConfig(
        imageSize: 8,
        patchSize: 4,
        embedDim: 16,
        numLayers: 2,
        numHeads: 2,
        ffnDim: 32,
      );
      final model = CLIPVisionModel(cfg);
      final x = _randomImage(cfg);
      final y = model(x);
      expect(y.shape, [cfg.numPatches + 1, cfg.embedDim]);
    });

    test('rejects mismatched input shape', () {
      const cfg = CLIPVisionConfig(
        imageSize: 8,
        patchSize: 4,
        embedDim: 16,
        numLayers: 2,
        numHeads: 2,
        ffnDim: 32,
      );
      final model = CLIPVisionModel(cfg);
      expect(
        () => model(Tensor.fromList([3, cfg.patchPixels],
            List<double>.filled(3 * cfg.patchPixels, 0.0))),
        throwsArgumentError,
      );
    });

    test('implements VisionEncoder', () {
      const cfg = CLIPVisionConfig(
        imageSize: 8,
        patchSize: 4,
        embedDim: 16,
        numLayers: 1,
        numHeads: 2,
        ffnDim: 32,
      );
      final VisionEncoder model = CLIPVisionModel(cfg);
      expect(model.embedDim, 16);
      expect(model.numPatches, 4);
    });
  });

  group('ClipHFLoader', () {
    const cfg = CLIPVisionConfig(
      imageSize: 8,
      patchSize: 4,
      embedDim: 16,
      numLayers: 2,
      numHeads: 4, // → headDim = 4
      ffnDim: 32,
    );

    test('loadMap consumes every key with vision_model prefix', () {
      final model = CLIPVisionModel(cfg);
      final state = _syntheticState(cfg);
      final report = ClipHFLoader.loadMap(model, state);
      expect(report.prefix, 'vision_model.');
      expect(report.unusedKeys, isEmpty);
      expect(report.consumedCount, state.length);
    });

    test('loadMap consumes every key with empty prefix', () {
      final model = CLIPVisionModel(cfg);
      final state = _syntheticState(cfg, prefix: '');
      final report = ClipHFLoader.loadMap(model, state);
      expect(report.prefix, '');
      expect(report.unusedKeys, isEmpty);
    });

    test('loadMap leaves unrelated keys as unused', () {
      final model = CLIPVisionModel(cfg);
      final state = _syntheticState(cfg);
      state['logit_scale'] = Tensor.fromList([1], [1.0]);
      state['text_model.embeddings.token_embedding.weight'] =
          Tensor.fromList([2, 3], [0, 0, 0, 0, 0, 0]);
      final report = ClipHFLoader.loadMap(model, state);
      expect(report.unusedKeys, contains('logit_scale'));
      expect(
        report.unusedKeys,
        contains('text_model.embeddings.token_embedding.weight'),
      );
    });

    test('missing key throws with helpful message', () {
      final model = CLIPVisionModel(cfg);
      final state = _syntheticState(cfg);
      state.remove('vision_model.pre_layrnorm.weight');
      expect(
        () => ClipHFLoader.loadMap(model, state),
        throwsA(isA<ArgumentError>().having(
          (e) => e.message,
          'message',
          contains('pre_layrnorm.weight'),
        )),
      );
    });

    test(
      'Conv→Linear permutation matches conv semantics on a single patch',
      () {
        // Sanity check: with an all-zero cls, zero posEmb, identity
        // pre-LN, identity blocks, and identity post_layernorm, the
        // patch row of the model output should equal the conv output
        // for that patch.
        //
        // We build a minimal 1-layer CLIP-like model, zero out
        // everything but the patch conv, then verify one patch
        // produces the expected dot products.

        const c = 3;
        const p = 2;
        const d = 4;
        const cfg2 = CLIPVisionConfig(
          imageSize: 4,
          patchSize: p,
          embedDim: d,
          numLayers: 1,
          numHeads: 2,
          ffnDim: 8,
        );
        final model = CLIPVisionModel(cfg2);
        final state = _syntheticState(cfg2, seed: 7);

        // Force a known patch conv weight: W[o, cc, hh, ww] = o*100 + cc*10 + hh + ww*0.1
        final convVals = <double>[];
        for (int o = 0; o < d; o++) {
          for (int cc = 0; cc < c; cc++) {
            for (int hh = 0; hh < p; hh++) {
              for (int ww = 0; ww < p; ww++) {
                convVals.add(o * 100.0 + cc * 10.0 + hh + ww * 0.1);
              }
            }
          }
        }
        state['vision_model.embeddings.patch_embedding.weight'] =
            Tensor.fromList([d, c, p, p], convVals);

        ClipHFLoader.loadMap(model, state);

        // Feed a single-patch image (well, a 4-patch one — build so
        // that patch 0 has a known pattern and the rest are zero).
        final flat = Float32List(cfg2.numPatches * cfg2.patchPixels);
        // Patch 0 pixel(h=0, w=0, c=0..2) = 1, 2, 3;
        //         pixel(h=0, w=1, c=0..2) = 4, 5, 6;
        //         pixel(h=1, w=0, c=0..2) = 7, 8, 9;
        //         pixel(h=1, w=1, c=0..2) = 10, 11, 12.
        for (int i = 0; i < 12; i++) {
          flat[i] = (i + 1).toDouble();
        }
        final img = Tensor.fromList(
          [cfg2.numPatches, cfg2.patchPixels],
          flat,
        );

        // Expected: y[o] = sum_{cc,hh,ww} conv[o,cc,hh,ww] * patch0[cc,hh,ww]
        // patch0 pixel layout in flat is (hh, ww, cc) with cc innermost.
        final expected = <double>[];
        for (int o = 0; o < d; o++) {
          var s = 0.0;
          for (int cc = 0; cc < c; cc++) {
            for (int hh = 0; hh < p; hh++) {
              for (int ww = 0; ww < p; ww++) {
                final pixel = flat[hh * p * c + ww * c + cc];
                final wConv = o * 100.0 + cc * 10.0 + hh + ww * 0.1;
                s += pixel * wConv;
              }
            }
          }
          expected.add(s);
        }

        // Run only the patch projection, not the full model, since
        // the transformer layers scramble things beyond checkability.
        final projected = model.patchProjection(img);
        final vals = projected.toList();
        // Row 0 (first patch) should equal `expected`. Tolerance
        // accommodates fp32 accumulation on values up to ~1e4.
        for (int o = 0; o < d; o++) {
          expect(
            vals[o],
            closeTo(expected[o], expected[o].abs() * 1e-4 + 1e-3),
            reason: 'patch conv output for out-neuron $o',
          );
        }
      },
    );

    test('bad state dict (no CLIP keys) fails prefix detection', () {
      final model = CLIPVisionModel(cfg);
      final state = <String, Tensor>{
        'random_key': Tensor.fromList([1], [0.0]),
      };
      expect(
        () => ClipHFLoader.loadMap(model, state),
        throwsA(isA<ArgumentError>().having(
          (e) => e.message,
          'message',
          contains('patch_embedding.weight'),
        )),
      );
    });
  });

  group('ClipHFLoader configs', () {
    test('base32 matches HF openai/clip-vit-base-patch32', () {
      final cfg = ClipHFLoader.base32Config();
      expect(cfg.imageSize, 224);
      expect(cfg.patchSize, 32);
      expect(cfg.embedDim, 768);
      expect(cfg.numLayers, 12);
      expect(cfg.numHeads, 12);
      expect(cfg.ffnDim, 3072);
      expect(cfg.numPatches, 49);
    });

    test('base16 has 196 patches', () {
      final cfg = ClipHFLoader.base16Config();
      expect(cfg.patchSize, 16);
      expect(cfg.numPatches, 196);
    });

    test('large14 matches HF openai/clip-vit-large-patch14', () {
      final cfg = ClipHFLoader.large14Config();
      expect(cfg.embedDim, 1024);
      expect(cfg.numLayers, 24);
      expect(cfg.numHeads, 16);
      expect(cfg.ffnDim, 4096);
      expect(cfg.patchSize, 14);
      expect(cfg.numPatches, 256);
    });
  });
}
