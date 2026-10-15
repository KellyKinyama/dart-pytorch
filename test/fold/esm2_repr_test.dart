@Timeout(Duration(minutes: 2))
library;

import 'package:dart_pytorch/core/nn/esm2.dart';
import 'package:dart_pytorch/core/nn/esm2_hf_loader.dart';
import 'package:dart_pytorch/core/tensor/tensor.dart';
import 'package:test/test.dart';

void main() {
  test('3B config has the expected ESMFold dimensions', () {
    final c = ESM2HFLoader.esm2_3bConfig();
    expect(c.numLayers, 36);
    expect(c.embedDim, 2560);
    expect(c.numHeads, 40);
    expect(c.embedDim ~/ c.numHeads, 64); // head dim
    expect(c.ffnDim, 10240);
    expect(c.vocabSize, 33);
  });

  test('larger configs are head-divisible', () {
    for (final c in [
      ESM2HFLoader.esm2_150mConfig(),
      ESM2HFLoader.esm2_650mConfig(),
      ESM2HFLoader.esm2_3bConfig(),
    ]) {
      expect(c.embedDim % c.numHeads, 0);
    }
  });

  group('representations()', () {
    final cfg = ESM2HFLoader.esm2_8mConfig();
    final model = ESM2Model(cfg);
    // [<cls>, M, A, G, V, <eos>]
    final tokens = Tensor.fromList(
        [6], [0, 20, 5, 6, 7, 2].map((e) => e.toDouble()).toList());

    test('returns numLayers + 1 hidden states of shape [N, D]', () {
      final reps = Tensor.noGrad(() => model.representations(tokens));
      expect(reps.length, cfg.numLayers + 1);
      for (final r in reps) {
        expect(r.shape, [6, cfg.embedDim]);
      }
    });

    test('finalLn(reps.last) equals call(tokens)', () {
      final viaCall = Tensor.noGrad(() => model.call(tokens).toList());
      final viaRepr = Tensor.noGrad(
          () => model.finalLn(model.representations(tokens).last).toList());
      expect(viaRepr.length, viaCall.length);
      for (var i = 0; i < viaCall.length; i++) {
        expect(viaRepr[i], closeTo(viaCall[i], 1e-5), reason: 'elem $i');
      }
    });

    test('index 0 is the raw token embedding (no final LayerNorm)', () {
      final emb = Tensor.noGrad(() => model.embedIn(tokens).toList());
      final rep0 = Tensor.noGrad(() => model.representations(tokens)[0].toList());
      for (var i = 0; i < emb.length; i++) {
        expect(rep0[i], closeTo(emb[i], 1e-6), reason: 'elem $i');
      }
    });
  });
}
