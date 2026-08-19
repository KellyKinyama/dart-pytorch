import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

// Tiny config for shape-only tests.
T5Config _tinyCfg({T5FfnActivation act = T5FfnActivation.relu}) => T5Config(
      vocabSize: 32,
      dModel: 16,
      dFf: 32,
      dKv: 4,
      numLayers: 2,
      numDecoderLayers: 2,
      numHeads: 4,
      feedForwardProj: act,
      maxCtx: 32,
      seed: 7,
    );

// Deterministic token id sequence.
Tensor _ids(List<int> ids) => Tensor.fromList(
      [ids.length],
      ids.map((i) => i.toDouble()).toList(),
    );

void main() {
  group('T5Config presets', () {
    test('t5-small (v1.0): relu FFN, 6 layers, dModel=512', () {
      final c = T5HFLoader.t5SmallConfig();
      expect(c.vocabSize, 32128);
      expect(c.dModel, 512);
      expect(c.dFf, 2048);
      expect(c.dKv, 64);
      expect(c.numLayers, 6);
      expect(c.numDecoderLayers, 6);
      expect(c.numHeads, 8);
      expect(c.feedForwardProj, T5FfnActivation.relu);
      expect(c.tieWordEmbeddings, isTrue);
    });

    test('t5-v1.1-small: gated-GELU FFN, 8 layers, 6 heads, dFf=1024', () {
      final c = T5HFLoader.t5V11SmallConfig();
      expect(c.dModel, 512);
      expect(c.dFf, 1024);
      expect(c.numLayers, 8);
      expect(c.numDecoderLayers, 8);
      expect(c.numHeads, 6);
      expect(c.dKv, 64);
      expect(c.feedForwardProj, T5FfnActivation.gatedGelu);
    });

    test('flan-t5-small aliases t5-v1.1-small', () {
      final a = T5HFLoader.flanT5SmallConfig();
      final b = T5HFLoader.t5V11SmallConfig();
      expect(a.dModel, b.dModel);
      expect(a.dFf, b.dFf);
      expect(a.numLayers, b.numLayers);
      expect(a.numHeads, b.numHeads);
      expect(a.feedForwardProj, b.feedForwardProj);
    });

    test('flan-t5-base: 12 layers, dModel=768, dFf=2048', () {
      final c = T5HFLoader.flanT5BaseConfig();
      expect(c.dModel, 768);
      expect(c.dFf, 2048);
      expect(c.numLayers, 12);
      expect(c.numHeads, 12);
      expect(c.dKv, 64);
      expect(c.feedForwardProj, T5FfnActivation.gatedGelu);
    });
  });

  group('T5RelativeBias bucketing', () {
    test('bidirectional buckets are symmetric under sign flip', () {
      final rb = T5RelativeBias(
        numBuckets: 32,
        numHeads: 4,
        maxDistance: 128,
        bidirectional: true,
      );
      // With bidirectional=true, bucket ids for +k and -k should
      // differ by numBuckets/2 (positive/negative half of the table).
      final biasPos = rb.maskPerHead(4, 4)[0].toList();
      // Position (0, 2): relPos=+2. Position (2, 0): relPos=-2.
      // In the bucket function, both go to small-distance exact bins,
      // but +2 gets +numBuckets/2 offset. So the two picked values
      // should generally differ (from different rows of the table).
      final v_0_2 = biasPos[0 * 4 + 2];
      final v_2_0 = biasPos[2 * 4 + 0];
      expect(v_0_2 != v_2_0, isTrue,
          reason: 'bidirectional should pick different rows for +2 and -2');
    });

    test('unidirectional: future positions map to bucket 0', () {
      final rb = T5RelativeBias(
        numBuckets: 32,
        numHeads: 4,
        maxDistance: 128,
        bidirectional: false,
      );
      final bias = rb.maskPerHead(4, 4)[0].toList();
      // All (i, j) with j > i should hit the same bucket 0.
      final v_0_1 = bias[0 * 4 + 1];
      final v_0_2 = bias[0 * 4 + 2];
      final v_0_3 = bias[0 * 4 + 3];
      expect(v_0_1, equals(v_0_2));
      expect(v_0_2, equals(v_0_3));
    });
  });

  group('T5 forward (shape only)', () {
    test('encoder output has shape [N, dModel]', () {
      final m = T5Model(_tinyCfg());
      final ids = _ids([0, 1, 2, 3]);
      final h = m.encoder(ids);
      expect(h.shape, equals([4, 16]));
    });

    test('decoder output has shape [Nq, dModel]', () {
      final m = T5Model(_tinyCfg());
      final memory = m.encoder(_ids([0, 1, 2, 3]));
      final tgt = _ids([0, 5, 6]);
      final h = m.decoder(tgt, memory: memory);
      expect(h.shape, equals([3, 16]));
    });

    test('logitsLastToken has shape [vocabSize]', () {
      final m = T5Model(_tinyCfg());
      final memory = m.encoder(_ids([0, 1, 2, 3]));
      final logits = m.logitsLastToken([0, 5, 6], memory);
      expect(logits.shape, equals([32]));
    });

    test('gated-GELU FFN also works end-to-end', () {
      final m = T5Model(_tinyCfg(act: T5FfnActivation.gatedGelu));
      final memory = m.encoder(_ids([0, 1, 2, 3]));
      final logits = m.logitsLastToken([0, 5], memory);
      expect(logits.shape, equals([32]));
    });

    test('generate returns non-empty sequence', () {
      final m = T5Model(_tinyCfg());
      final out = m.generate([1, 2, 3, 4], maxNewTokens: 5);
      // start token + up to 5 new
      expect(out.length, greaterThan(1));
      expect(out.first, equals(0));
    });
  });

  group('T5HFLoader roundtrip', () {
    // Dump a fresh T5Model's weights into a synthetic HF-style state
    // dict, then load them back into a second model and verify
    // roundtrip.
    Map<String, Tensor> dump(T5Model m) {
      final s = <String, Tensor>{};
      final cfg = m.config;
      s['shared.weight'] = m.sharedEmbedding.weight;
      // encoder blocks
      for (int i = 0; i < cfg.numLayers; i++) {
        final b = m.encoder.blocks[i];
        _dumpAttentionInto(s, b.selfAttn, cfg,
            base: 'encoder.block.$i.layer.0.SelfAttention');
        s['encoder.block.$i.layer.0.layer_norm.weight'] =
            b.selfAttnNorm.gamma;
        if (i == 0) {
          s['encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight'] =
              m.encoder.relativeBias.table.weight;
        }
        _dumpFfnInto(s, b.ffn, cfg,
            base: 'encoder.block.$i.layer.1.DenseReluDense');
        s['encoder.block.$i.layer.1.layer_norm.weight'] = b.ffnNorm.gamma;
      }
      s['encoder.final_layer_norm.weight'] = m.encoder.finalNorm.gamma;

      // decoder blocks
      for (int i = 0; i < cfg.numDecoderLayers; i++) {
        final b = m.decoder.blocks[i];
        _dumpAttentionInto(s, b.selfAttn, cfg,
            base: 'decoder.block.$i.layer.0.SelfAttention');
        s['decoder.block.$i.layer.0.layer_norm.weight'] =
            b.selfAttnNorm.gamma;
        if (i == 0) {
          s['decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight'] =
              m.decoder.relativeBias.table.weight;
        }
        _dumpAttentionInto(s, b.crossAttn, cfg,
            base: 'decoder.block.$i.layer.1.EncDecAttention');
        s['decoder.block.$i.layer.1.layer_norm.weight'] = b.crossAttnNorm.gamma;
        _dumpFfnInto(s, b.ffn, cfg,
            base: 'decoder.block.$i.layer.2.DenseReluDense');
        s['decoder.block.$i.layer.2.layer_norm.weight'] = b.ffnNorm.gamma;
      }
      s['decoder.final_layer_norm.weight'] = m.decoder.finalNorm.gamma;
      return s;
    }

    test('roundtrip (relu FFN) consumes all keys, no unused', () {
      final src = T5Model(_tinyCfg());
      final state = dump(src);
      final dst = T5Model(_tinyCfg());
      final report = T5HFLoader.loadMap(dst, state);
      expect(report.unusedKeys, isEmpty);
      expect(report.consumedCount, equals(state.length));
    });

    test('roundtrip (gated-GELU FFN) consumes all keys', () {
      final src = T5Model(_tinyCfg(act: T5FfnActivation.gatedGelu));
      final state = dump(src);
      final dst = T5Model(_tinyCfg(act: T5FfnActivation.gatedGelu));
      final report = T5HFLoader.loadMap(dst, state);
      expect(report.unusedKeys, isEmpty);
    });

    test('roundtrip rejects missing key', () {
      final src = T5Model(_tinyCfg());
      final state = dump(src);
      state.remove('encoder.final_layer_norm.weight');
      final dst = T5Model(_tinyCfg());
      expect(() => T5HFLoader.loadMap(dst, state), throwsArgumentError);
    });

    test('extra keys land in unusedKeys (not consumed)', () {
      final src = T5Model(_tinyCfg());
      final state = dump(src);
      state['bogus.extra.weight'] =
          Tensor.fromList([1], [0.0]);
      final dst = T5Model(_tinyCfg());
      final report = T5HFLoader.loadMap(dst, state);
      expect(report.unusedKeys, contains('bogus.extra.weight'));
    });
  });
}

void _dumpAttentionInto(
  Map<String, Tensor> s,
  T5Attention attn,
  T5Config cfg, {
  required String base,
}) {
  final h = cfg.numHeads;
  final d = cfg.dKv;
  final inQ = cfg.dModel;
  final inKV = attn.kvDim;
  s['$base.q.weight'] = _concatHeadRows(attn.wq, h, d, inQ);
  s['$base.k.weight'] = _concatHeadRows(attn.wk, h, d, inKV);
  s['$base.v.weight'] = _concatHeadRows(attn.wv, h, d, inKV);
  s['$base.o.weight'] = attn.wo.weight;
}

Tensor _concatHeadRows(List heads, int H, int D, int IN) {
  final out = List<double>.filled(H * D * IN, 0);
  for (int hi = 0; hi < H; hi++) {
    final head = heads[hi];
    final row = head.weight.toList();
    final base = hi * D * IN;
    for (int i = 0; i < D * IN; i++) {
      out[base + i] = row[i];
    }
  }
  return Tensor.fromList([H * D, IN], out);
}

void _dumpFfnInto(
  Map<String, Tensor> s,
  T5Ffn ffn,
  T5Config cfg, {
  required String base,
}) {
  if (ffn.wi1 == null) {
    s['$base.wi.weight'] = ffn.wi0.weight;
  } else {
    s['$base.wi_0.weight'] = ffn.wi0.weight;
    s['$base.wi_1.weight'] = ffn.wi1!.weight;
  }
  s['$base.wo.weight'] = ffn.wo.weight;
}
