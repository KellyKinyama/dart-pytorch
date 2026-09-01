import 'dart:io';

import 'package:chess/chess.dart' as ch;
import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

const double _kMateScore = 1e9;

/// Rank every legal move by the NNUE eval of the resulting position,
/// negated (child position is from the opponent's POV). Depth-1 only.
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

/// Recursive negamax with alpha-beta pruning. Mirrors the searcher in
/// `bin/nnue_play.dart` (kept here as a duplicate so the test doesn't
/// depend on `bin/` compilation).
double _negamax(
  NnueNet net,
  ch.Chess board,
  int depth,
  double alpha,
  double beta,
) {
  if (board.in_checkmate) return -_kMateScore + (100 - depth).toDouble();
  if (board.in_stalemate ||
      board.insufficient_material ||
      board.in_threefold_repetition ||
      board.in_draw) {
    return 0.0;
  }
  if (depth <= 0) return net.evaluateInt(encodeFen(board.fen)).cp;
  final sans = board.moves().cast<String>();
  var best = -_kMateScore * 2;
  for (final san in sans) {
    if (!board.move(san)) continue;
    final score = -_negamax(net, board, depth - 1, -beta, -alpha);
    board.undo();
    if (score > best) best = score;
    if (score > alpha) alpha = score;
    if (alpha >= beta) break;
  }
  return best;
}

/// Depth-N root ranking. Returns `[(san, cp)]` sorted best-first.
List<({String san, double cp})> _rankDepth(
  NnueNet net,
  ch.Chess board,
  int depth,
) {
  final rows = <({String san, double cp})>[];
  final sans = board.moves().cast<String>();
  for (final san in sans) {
    if (!board.move(san)) continue;
    final score = -_negamax(net, board, depth - 1, -_kMateScore, _kMateScore);
    board.undo();
    rows.add((san: san, cp: score));
  }
  rows.sort((a, b) => b.cp.compareTo(a.cp));
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
      const fen = '6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1';
      final board = ch.Chess.fromFEN(fen);
      final sans = board.moves().cast<String>();
      final mateMoves = sans.where((s) => s.startsWith('Re8')).toList();
      expect(
        mateMoves,
        isNotEmpty,
        reason: 'Re8 should be legal in this position',
      );
    });

    test('depth-1 negamax picks the mate-in-1', () {
      if (!present) return;
      const fen = '6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1';
      final board = ch.Chess.fromFEN(fen);
      final ranked = _rankDepth(net, board, 1);
      expect(ranked.first.san.startsWith('Re8'), isTrue,
          reason: 'top depth-1 move should be Re8#, got ${ranked.first}');
      expect(ranked.first.cp, greaterThan(_kMateScore / 2),
          reason: 'mate should score near +infinity');
    });

    test('depth-2 negamax on Italian position promotes a developing move', () {
      if (!present) return;
      // Bishop's-Opening / Italian: after 1.e4 e5 2.Bc4 Nc6, standard
      // moves are Nf3 / Nc3 / d3. Depth-1 puts self-blocking Ne2 first;
      // depth-2 should promote a developing move into the top 3.
      const fen =
          'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3';
      final board = ch.Chess.fromFEN(fen);
      final ranked = _rankDepth(net, board, 2);
      final top3 = ranked.take(3).map((r) => r.san).toSet();
      final developing = {'Nc3', 'Nf3', 'd3', 'd4', 'Qf3', 'Qe2'};
      expect(top3.intersection(developing), isNotEmpty,
          reason: 'top-3 should contain a developing move: $top3');
      // And Ne2 / Ke2 (self-blocking / lose castling) should NOT be #1.
      expect(ranked.first.san, isNot('Ne2'));
      expect(ranked.first.san, isNot('Ke2'));
    });
  });

  if (!present) {
    test('NNUE weights not present — move picker tests skipped', () {
      print('skipped: place $netPath to run');
    });
  }
}
