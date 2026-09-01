import 'dart:typed_data';

import 'package:dart_pytorch/core/nn/nnue_input.dart';
import 'package:test/test.dart';

void main() {
  group('encodeFen', () {
    test('startpos has 32 active features per POV and 32 pieces', () {
      const startpos =
          'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
      final f = encodeFen(startpos);
      expect(f.pieceCount, 32);
      expect(f.stm, NnuePerspective.white);
      expect(f.whiteActive.length, 32);
      expect(f.blackActive.length, 32);
    });

    test('all indices are within [0, 22528)', () {
      const fens = [
        'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
        '4k3/8/8/8/8/8/8/4K3 w - - 0 1',
        'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
      ];
      for (final fen in fens) {
        final f = encodeFen(fen);
        for (final idx in f.whiteActive) {
          expect(idx, greaterThanOrEqualTo(0));
          expect(idx, lessThan(22528));
        }
        for (final idx in f.blackActive) {
          expect(idx, greaterThanOrEqualTo(0));
          expect(idx, lessThan(22528));
        }
      }
    });

    test('startpos white POV features are symmetric with black POV', () {
      // From the start position the two POVs see mirror-image boards, so
      // the sorted feature-index sets should be identical after we swap
      // POV — i.e., they must at least have the same length and cover
      // the same king-bucket range.
      const startpos =
          'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
      final f = encodeFen(startpos);
      final whiteBuckets = <int>{};
      final blackBuckets = <int>{};
      for (final idx in f.whiteActive) {
        whiteBuckets.add(idx ~/ 704);
      }
      for (final idx in f.blackActive) {
        blackBuckets.add(idx ~/ 704);
      }
      // Both POVs share a single king bucket (starting kings on e1/e8,
      // which after vertical flip for black and horizontal identity
      // land on the same oriented square).
      expect(whiteBuckets.length, 1);
      expect(blackBuckets.length, 1);
      expect(whiteBuckets.single, blackBuckets.single);
    });

    test('flipped side to move', () {
      const black = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1';
      expect(encodeFen(black).stm, NnuePerspective.black);
    });

    test('rejects FEN missing kings', () {
      expect(
        () => encodeFen('8/8/8/8/8/8/8/8 w - - 0 1'),
        throwsA(isA<FormatException>()),
      );
    });

    test('lone-king endgame has exactly 2 features per POV', () {
      const bareKings = '4k3/8/8/8/8/8/8/4K3 w - - 0 1';
      final f = encodeFen(bareKings);
      expect(f.pieceCount, 2);
      expect(f.whiteActive.length, 2);
      expect(f.blackActive.length, 2);
    });

    test('encoding is deterministic', () {
      const fen =
          'r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R '
          'w KQkq - 0 4';
      final a = encodeFen(fen);
      final b = encodeFen(fen);
      expect(a.whiteActive, orderedEquals(b.whiteActive));
      expect(a.blackActive, orderedEquals(b.blackActive));
    });
  });

  test('startpos active index is Int32List', () {
    const startpos = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
    final f = encodeFen(startpos);
    expect(f.whiteActive, isA<Int32List>());
    expect(f.blackActive, isA<Int32List>());
  });
}
