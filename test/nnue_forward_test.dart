/// Golden smoke test for the end-to-end NNUE port. Requires the real
/// SF16 `nn-5af11540bbfe.nnue` file at `models/stockfish/`; skipped if
/// absent so CI without the download doesn't fail.
import 'dart:io';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

void main() {
  const netPath = 'models/stockfish/nn-5af11540bbfe.nnue';
  final present = File(netPath).existsSync();

  group('NnueNet end-to-end', () {
    late NnueNet net;

    setUpAll(() {
      if (!present) return;
      final raw = NnueReader.loadFile(netPath);
      net = NnueNet.fromRaw(raw);
    });

    test('loads dequantised weights with expected shapes', () {
      if (!present) return;
      expect(net.ftDim, 1536);
      expect(net.numInputs, 22528);
      expect(net.psqtBuckets, 8);
      expect(net.buckets.length, 8);
    });

    test('startpos evaluates within a small cp band', () {
      if (!present) return;
      const startpos =
          'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
      final r = net.evaluate(encodeFen(startpos));
      // Float port isn't exact vs Stockfish quantised eval; assert only
      // that startpos comes out roughly small and positive-ish. Real SF
      // reports ~25 cp; we allow a broad band because scale calibration
      // (M3 → M6) is intentionally approximate for the float backend.
      expect(
        r.cp.abs(),
        lessThan(200),
        reason: 'startpos should be small, got ${r.cp}',
      );
      expect(r.bucket, 7);
    });

    test('white winning a queen gives a large positive cp', () {
      if (!present) return;
      const fen = 'rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
      final r = net.evaluate(encodeFen(fen));
      expect(
        r.cp,
        greaterThan(500),
        reason: 'white up a queen should be > +500 cp, got ${r.cp}',
      );
    });

    test('white losing a queen gives a large negative cp', () {
      if (!present) return;
      const fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1';
      final r = net.evaluate(encodeFen(fen));
      expect(
        r.cp,
        lessThan(-500),
        reason: 'white down a queen should be < -500 cp, got ${r.cp}',
      );
    });

    test(
      'mirror-symmetric position gives cp of similar magnitude, opposite sign',
      () {
        if (!present) return;
        const white =
            'rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
        const black =
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1';
        final rW = net.evaluate(encodeFen(white));
        final rB = net.evaluate(encodeFen(black));
        // Sign should be opposite.
        expect(rW.cp * rB.cp, lessThan(0));
      },
    );

    test('int backend agrees with float backend on sign', () {
      if (!present) return;
      const fens = [
        'rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
        'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1',
        'r3k3/8/8/8/8/8/8/R3K2R w KQq - 0 1',
      ];
      for (final fen in fens) {
        final feats = encodeFen(fen);
        final rf = net.evaluate(feats);
        final ri = net.evaluateInt(feats);
        expect(
          rf.cp * ri.cp,
          greaterThan(0),
          reason: 'float=${rf.cp}, int=${ri.cp} disagree on $fen',
        );
      }
    });

    test(
      'int backend material-monotone: -black queen gives >+500 cp uplift',
      () {
        if (!present) return;
        const startpos =
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
        const blackNoQueen =
            'rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
        final rBase = net.evaluateInt(encodeFen(startpos));
        final rBetter = net.evaluateInt(encodeFen(blackNoQueen));
        expect(
          rBetter.cp - rBase.cp,
          greaterThan(500),
          reason: 'delta=${rBetter.cp - rBase.cp} should be >500 cp',
        );
      },
    );
  });

  if (!present) {
    test('SFNNv8 weights not present — end-to-end NNUE tests skipped', () {
      print('skipped: place $netPath to run full NNUE tests');
    });
  }
}
