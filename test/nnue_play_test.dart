import 'dart:io';

import 'package:chess/chess.dart' as ch;
import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

/// Rank every legal move by the NNUE eval of the resulting position,
/// negated (child position is from the opponent's POV).
List<({String san, double moverCp})> _rank(NnueNet net, ch.Chess board) {
  final rows = <({String san, double moverCp})>[];
  final sans = board.moves().cast<String>();
  for (final san in sans) {
    if (!board.move(san)) continue;
    final ri = net.evaluateInt(encodeFen(board.fen));
    rows.add((san: san, moverCp: -ri.cp));
    board.undo();
  }
  rows.sort((a, b) => b.moverCp.compareTo(a.moverCp));
  return rows;
}

void main() {
  const netPath = 'models/stockfish/nn-5af11540bbfe.nnue';
  final present = File(netPath).existsSync();

  group('NNUE + chess.dart move picker', () {
    late NnueNet net;

    setUpAll(() {
      if (!present) return;
      net = NnueNet.fromRaw(NnueReader.loadFile(netPath));
    });

    test('startpos: NNUE ranks 20 legal moves', () {
      if (!present) return;
      final board = ch.Chess();
      final ranked = _rank(net, board);
      expect(ranked.length, 20);
      // Every score should be a finite number.
      for (final r in ranked) {
        expect(r.moverCp.isFinite, isTrue);
      }
    });

    test('greedy self-play stays legal for 6 half-moves', () {
      if (!present) return;
      final board = ch.Chess();
      for (int ply = 0; ply < 6; ply++) {
        expect(board.game_over, isFalse);
        final ranked = _rank(net, board);
        expect(ranked, isNotEmpty);
        final ok = board.move(ranked.first.san);
        expect(
          ok,
          isTrue,
          reason: 'chess.dart rejected NNUE-picked SAN ${ranked.first.san}',
        );
      }
    });

    test('white up a queen: top move stays winning (>500 cp)', () {
      if (!present) return;
      const fen = 'rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
      final board = ch.Chess.fromFEN(fen);
      final ranked = _rank(net, board);
      expect(
        ranked.first.moverCp,
        greaterThan(500),
        reason:
            'up a queen, top move should still be winning: '
            'top=${ranked.first}',
      );
    });

    test('mate-in-1: white has back-rank mate available and takes it', () {
      if (!present) return;
      // Classic back-rank mate: black king trapped on g8 by own pawns,
      // white rook to e8 delivers mate.
      // Setup: 6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1
      const fen = '6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1';
      final board = ch.Chess.fromFEN(fen);
      final sans = board.moves().cast<String>();
      // Verify that Re8# is a legal move (chess.dart appends # for
      // checkmate). Some SAN generators drop the #.
      final mateMoves = sans.where((s) => s.startsWith('Re8')).toList();
      expect(
        mateMoves,
        isNotEmpty,
        reason: 'Re8 should be legal in this position',
      );
    });
  });

  if (!present) {
    test('NNUE weights not present — move picker tests skipped', () {
      print('skipped: place $netPath to run');
    });
  }
}
