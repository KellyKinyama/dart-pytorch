/// NNUE-guided move picker and self-play demo.
///
/// Uses the `chess` package for legal-move generation and board state
/// and the Stockfish NNUE port from `lib/core/nn/nnue.dart` to score
/// each resulting position. Every move is scored by pushing it,
/// running one full NNUE forward pass on the child position, and
/// negating the cp (child position is evaluated from the opponent's
/// perspective, so a lower opponent cp = better for us).
///
/// Modes:
///   * `--top N` — list the top N moves for the given position.
///   * `--play N` — greedily play N half-moves from the given start
///                  position, printing the game as it goes.
///
/// Usage:
///
///   dart run bin/nnue_play.dart               # top 5 moves from startpos
///   dart run bin/nnue_play.dart --top 10 '<fen>'
///   dart run bin/nnue_play.dart --play 12     # 6-move greedy self-play
///   dart run bin/nnue_play.dart --play 20 '<fen>'
library;

import 'dart:io';

import 'package:chess/chess.dart' as ch;
import 'package:dart_pytorch/dart_pytorch.dart';

const String _defaultNetPath = 'models/stockfish/nn-5af11540bbfe.nnue';
const String _defaultFen =
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

Future<void> main(List<String> args) async {
  var mode = _Mode.top;
  var n = 5;
  var fen = _defaultFen;
  var netPath = _defaultNetPath;

  for (int i = 0; i < args.length; i++) {
    final a = args[i];
    if (a == '--top' && i + 1 < args.length) {
      mode = _Mode.top;
      n = int.parse(args[++i]);
    } else if (a == '--play' && i + 1 < args.length) {
      mode = _Mode.play;
      n = int.parse(args[++i]);
    } else if (a == '--net' && i + 1 < args.length) {
      netPath = args[++i];
    } else if (!a.startsWith('--')) {
      fen = a;
    }
  }

  if (!File(netPath).existsSync()) {
    stderr.writeln('missing $netPath — download with:');
    stderr.writeln('  mkdir -p models/stockfish');
    stderr.writeln('  curl -sL -o $netPath \\');
    stderr.writeln(
      '    '
      'https://tests.stockfishchess.org/api/nn/nn-5af11540bbfe.nnue',
    );
    exit(64);
  }

  stdout.writeln('Loading $netPath …');
  final raw = NnueReader.loadFile(netPath);
  final net = NnueNet.fromRaw(raw);
  stdout.writeln('  ready.\n');

  switch (mode) {
    case _Mode.top:
      _showTop(net, fen, n);
      break;
    case _Mode.play:
      _play(net, fen, n);
      break;
  }
}

enum _Mode { top, play }

/// Evaluate every legal move from [fen] and print the top [n] sorted by
/// resulting cp (from the side-to-move's POV).
void _showTop(NnueNet net, String fen, int n) {
  final board = ch.Chess.fromFEN(fen);
  stdout.writeln(board.ascii);
  stdout.writeln(
    'side to move: ${board.turn == ch.Chess.WHITE ? "white" : "black"}',
  );
  if (board.game_over) {
    stdout.writeln('game over: ${_status(board)}');
    return;
  }

  final rows = _rankMoves(net, board);
  final show = rows.length < n ? rows.length : n;
  stdout.writeln('');
  stdout.writeln('Top $show / ${rows.length} legal moves (cp from mover POV):');
  stdout.writeln('  rank   move         resulting-cp');
  for (int k = 0; k < show; k++) {
    final r = rows[k];
    stdout.writeln(
      '  ${(k + 1).toString().padLeft(4)}   '
      '${r.san.padRight(10)}   ${r.moverCp.toStringAsFixed(1).padLeft(10)} cp',
    );
  }
}

/// Greedy self-play: repeatedly pick the top-ranked NNUE move for the
/// current side until game over or half-move budget is exhausted.
void _play(NnueNet net, String fen, int halfMoves) {
  final board = ch.Chess.fromFEN(fen);
  stdout.writeln(board.ascii);
  stdout.writeln('starting fen: $fen');
  stdout.writeln('');

  for (int ply = 0; ply < halfMoves; ply++) {
    if (board.game_over) {
      stdout.writeln('game over: ${_status(board)}');
      return;
    }
    final rows = _rankMoves(net, board);
    if (rows.isEmpty) {
      stdout.writeln('no legal moves — game over');
      return;
    }
    final pick = rows.first;
    final moveNo = (ply ~/ 2) + 1;
    final side = board.turn == ch.Chess.WHITE ? '.' : '…';
    stdout.writeln(
      '  $moveNo$side  ${pick.san.padRight(8)}  '
      '→ ${pick.moverCp.toStringAsFixed(1)} cp',
    );
    board.move(pick.san);
  }
  stdout.writeln('');
  stdout.writeln('final fen: ${board.fen}');
  if (board.game_over) {
    stdout.writeln('game over: ${_status(board)}');
  }
}

class _Ranked {
  const _Ranked(this.san, this.moverCp);
  final String san;
  final double moverCp;
}

/// Rank every legal move by the NNUE eval of the resulting position,
/// negated (since after our move the opponent is to move and cp is
/// reported from their POV).
List<_Ranked> _rankMoves(NnueNet net, ch.Chess board) {
  final sans = board.moves().cast<String>();
  final rows = <_Ranked>[];
  for (final san in sans) {
    final ok = board.move(san);
    if (!ok) continue;
    final childFen = board.fen;
    final feats = encodeFen(childFen);
    final ri = net.evaluateInt(feats);
    // ri.cp is from the child's STM POV = our opponent. Flip sign.
    rows.add(_Ranked(san, -ri.cp));
    board.undo();
  }
  rows.sort((a, b) => b.moverCp.compareTo(a.moverCp));
  return rows;
}

String _status(ch.Chess board) {
  if (board.in_checkmate) {
    final loser = board.turn == ch.Chess.WHITE ? 'white' : 'black';
    return 'checkmate ($loser to move)';
  }
  if (board.in_stalemate) return 'stalemate';
  if (board.insufficient_material) return 'insufficient material';
  if (board.in_threefold_repetition) return 'threefold repetition';
  if (board.in_draw) return 'draw (50-move / other)';
  return 'unknown';
}
