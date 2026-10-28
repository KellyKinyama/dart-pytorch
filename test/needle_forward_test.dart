import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/core/nn/needle/needle_config.dart';
import 'package:dart_pytorch/core/nn/needle/needle_forward.dart';
import 'package:dart_pytorch/core/nn/needle/hadamard.dart';
import 'package:test/test.dart';

Float64List _rand(math.Random r, int n, [double scale = 0.2]) =>
    Float64List.fromList(
      List<double>.generate(n, (_) => (r.nextDouble() * 2 - 1) * scale),
    );

NeedleModelF _tinyModel(math.Random r) {
  // Small geometry with square Monarch blocks (n = 16 → 4x4).
  const cfg = NeedleConfig(
    vocab: 32,
    dModel: 16,
    numLayers: 2,
    kvBits: 8,
    ropeTheta: 100000.0,
    numQHeads: 2,
    headDimQ: 4,
    numKvHeads: 1,
    headDimV: 4,
    convTaps: 3,
    window: 1024,
    numLanes: 2,
    engramSlots: 64,
    engramSubDim: 8,
    engramOrders: [2, 3],
  );
  const d = 16;
  const n = 16; // hadamard size
  final (ba, bb) = hadaBlocks(n);
  final nLanes = cfg.numLanes;
  final heads = d ~/ (cfg.engramOrders.length * cfg.engramSubDim); // 1
  final numTables = cfg.engramOrders.length * heads; // 2
  final ed = numTables * cfg.engramSubDim; // 16

  MlpW mlp() => MlpW(
        condV: _rand(r, d * 8),
        condU: _rand(r, 8 * n),
        d1: _rand(r, n, 1.0),
        d2: _rand(r, n, 1.0),
        b2: _rand(r, n),
        d3: _rand(r, n, 1.0),
        d4: _rand(r, n),
        w1a: _rand(r, ba * ba),
        w1b: _rand(r, bb * bb),
        w2a: _rand(r, ba * ba),
        w2b: _rand(r, bb * bb),
        w3a: _rand(r, ba * ba),
        w3b: _rand(r, bb * bb),
        p1: Int32List.fromList(List<int>.generate(n, (i) => i)),
        p2: Int32List.fromList(List<int>.generate(n, (i) => i)),
      );

  AttnW attn() => AttnW(
        qProj: _rand(r, (cfg.numQHeads * cfg.headDimQ) * d),
        kProj: _rand(r, (cfg.numKvHeads * cfg.headDimQ) * d),
        vProj: _rand(r, (cfg.numKvHeads * cfg.headDimV) * d),
        qTaps: _rand(r, cfg.convTaps * (cfg.numQHeads * cfg.headDimQ)),
        kTaps: _rand(r, cfg.convTaps * (cfg.numKvHeads * cfg.headDimQ)),
        vTaps: _rand(r, cfg.convTaps * (cfg.numKvHeads * cfg.headDimV)),
        qNorm: _rand(r, cfg.headDimQ),
        kNorm: _rand(r, cfg.headDimQ),
        gateProj: _rand(r, (cfg.numQHeads * cfg.headDimV) * d),
        outProj: _rand(r, d * (cfg.numQHeads * cfg.headDimV)),
      );

  final blocks = [
    for (var l = 0; l < cfg.numLayers; l++)
      BlockW(
        normIn: _rand(r, d),
        postNorm: _rand(r, d),
        preHada: _rand(r, d),
        attnGate: r.nextDouble() - 0.5,
        attn: attn(),
        mlp: mlp(),
      ),
  ];

  final nd = nLanes * d;
  final mhc = MhcW(
    aPre: _rand(r, cfg.numLayers, 0.01),
    aPost: _rand(r, cfg.numLayers, 0.01),
    aRes: _rand(r, cfg.numLayers, 0.01),
    bPre: _rand(r, cfg.numLayers * nLanes),
    bPost: _rand(r, cfg.numLayers * nLanes),
    bRes: _rand(r, cfg.numLayers * nLanes * nLanes),
    phiPre: _rand(r, cfg.numLayers * nd * nLanes, 0.1),
    phiPost: _rand(r, cfg.numLayers * nd * nLanes, 0.1),
    phiRes: _rand(r, cfg.numLayers * nd * nLanes * nLanes, 0.1),
  );

  // engram at sites 3,7,... but tiny model has 2 layers; put a site at layer 0.
  final engram = [
    EngramW(
      site: 0,
      tables: _rand(r, numTables * cfg.engramSlots * cfg.engramSubDim),
      keyProj: _rand(r, d * ed),
      valueProj: _rand(r, d * ed),
      taps: _rand(r, 4 * d),
    ),
  ];

  return NeedleModelF(
    config: cfg,
    embedding: _rand(r, cfg.vocab * d, 1.0),
    blocks: blocks,
    mhc: mhc,
    engram: engram,
    finalNorm: _rand(r, d),
  );
}

void main() {
  group('NeedleModelF forward (synthetic weights)', () {
    test('produces finite logits of shape [T, vocab]', () {
      final m = _tinyModel(math.Random(1));
      final tokens = [2, 5, 9, 1, 7, 3];
      final logits = m.logits(tokens);
      expect(logits.length, tokens.length * m.config.vocab);
      expect(logits.every((v) => v.isFinite), isTrue);
    });

    test('is deterministic', () {
      final m = _tinyModel(math.Random(2));
      final a = m.logits([1, 2, 3, 4]);
      final b = m.logits([1, 2, 3, 4]);
      for (var i = 0; i < a.length; i++) {
        expect(a[i], a[i] == b[i] ? a[i] : closeTo(b[i], 0));
      }
    });

    test('is causal: changing the last token leaves earlier logits intact',
        () {
      final m = _tinyModel(math.Random(3));
      final v = m.config.vocab;
      final a = m.logits([3, 1, 4, 1, 5]);
      final b = m.logits([3, 1, 4, 1, 9]); // change only the last token
      // positions 0..3 must be unchanged
      for (var p = 0; p < 4; p++) {
        for (var c = 0; c < v; c++) {
          expect(a[p * v + c], closeTo(b[p * v + c], 1e-9),
              reason: 'pos $p col $c changed');
        }
      }
    });
  });
}
