import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

double _plane(List<double> data, int p, int rank, int file) =>
    data[p * 64 + rank * 8 + file];

void main() {
  group('Lc0Input.fromFen', () {
    test('starting position has correct shape', () {
      final t = Lc0Input.fromFen(startFen);
      expect(t.shape, [1, 112, 8, 8]);
    });

    test('starting position places all 8 pawns on rank 1 (STM POV)', () {
      final t = Lc0Input.fromFen(startFen).toList();
      // White pawn plane = 0. In white-POV coordinates rank 1 = index 1.
      for (int f = 0; f < 8; f++) {
        expect(_plane(t, 0, 1, f), 1.0);
      }
      // Rank 6 = black pawns (plane 6).
      for (int f = 0; f < 8; f++) {
        expect(_plane(t, 6, 6, f), 1.0);
      }
    });

    test('starting position king on e1 white / e8 black', () {
      final t = Lc0Input.fromFen(startFen).toList();
      // White king (plane 5) on rank 0 (a1..h1), file 4 (e1).
      expect(_plane(t, 5, 0, 4), 1.0);
      // Black king (plane 11) on rank 7, file 4.
      expect(_plane(t, 11, 7, 4), 1.0);
    });

    test('castling rights KQkq set all four planes', () {
      final t = Lc0Input.fromFen(startFen).toList();
      for (int p = 104; p <= 107; p++) {
        for (int r = 0; r < 8; r++) {
          for (int f = 0; f < 8; f++) {
            expect(_plane(t, p, r, f), 1.0);
          }
        }
      }
    });

    test('side-to-move plane is zero for white, one for black', () {
      final tW = Lc0Input.fromFen(startFen).toList();
      for (int i = 0; i < 64; i++) {
        expect(tW[108 * 64 + i], 0.0);
      }
      final tB = Lc0Input.fromFen(
        'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1',
      ).toList();
      for (int i = 0; i < 64; i++) {
        expect(tB[108 * 64 + i], 1.0);
      }
    });

    test('plane 111 is constant ones', () {
      final t = Lc0Input.fromFen(startFen).toList();
      for (int i = 0; i < 64; i++) {
        expect(t[111 * 64 + i], 1.0);
      }
    });

    test('black to move mirrors pieces so STM is at bottom', () {
      final tB = Lc0Input.fromFen(
        'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1',
      ).toList();
      // After mirror, "white pawn" plane holds the STM's pawns (black
      // pawns) at rank 1 in STM-POV.
      for (int f = 0; f < 8; f++) {
        expect(_plane(tB, 0, 1, f), 1.0);
      }
      // Opponent (white in absolute terms) pawns are at rank 6.
      for (int f = 0; f < 8; f++) {
        expect(_plane(tB, 6, 6, f), 1.0);
      }
    });

    test('rejects an 8-line FEN with a bad character', () {
      expect(
        () => Lc0Input.fromFen('rnbqkbnZ/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1'),
        throwsA(isA<ArgumentError>()),
      );
    });
  });
}
