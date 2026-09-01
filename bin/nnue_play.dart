/// NNUE-guided move picker and self-play demo.
///
/// Uses the `chess` package for legal-move generation and the Stockfish
/// NNUE port for position evaluation. Two search modes:
///
///   * **depth 1** (default): rank each legal move by the NNUE eval
///     of the resulting position. Fast but ignores tactics beyond one
///     half-move.
///   * **depth ≥ 2**: negamax with alpha-beta pruning, moves ordered
///     by their depth-1 NNUE score. Depth 2 typically finds standard
///     opening moves and one-move tactics; depth 3 sees two-move
///     combinations. Cost scales roughly `branchingFactor^depth /
///     alpha_beta_reduction`.
///
/// Modes:
///   * `--top N` — list the top N moves.
///   * `--play N` — greedily play N half-moves.
///   * `--depth D` — search depth (default 1, applies to both modes).
///
/// Usage:
///
///   dart run bin/nnue_play.dart                       # top 5, depth 1
///   dart run bin/nnue_play.dart --top 10 '<fen>'
///   dart run bin/nnue_play.dart --depth 2 --top 5     # 1-ply lookahead
///   dart run bin/nnue_play.dart --depth 3 --play 12   # deeper self-play
library;

import 'dart:io';

import 'package:chess/chess.dart' as ch;
import 'package:dart_pytorch/dart_pytorch.dart';

const String _defaultNetPath = 'models/stockfish/nn-5af11540bbfe.nnue';
const String _defaultFen =
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

const double _kMateScore = 1e9;

Future<void> main(List<String> args) async {
  var mode = _Mode.top;
  var n = 5;
  var depth = 1;
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
    } else if (a == '--depth' && i + 1 < args.length) {
      depth = int.parse(args[++i]);
    } else if (a == '--net' && i + 1 < args.length) {
      netPath = args[++i];
    } else if (!a.startsWith('--')) {
      fen = a;
    }
  }
  if (depth < 1) {
    stderr.writeln('--depth must be >= 1');
    exit(64);
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
  stdout.writeln('  ready (depth=$depth).\n');

  switch (mode) {
    case _Mode.top:
      _showTop(net, fen, n, depth);
      break;
    case _Mode.play:
      _play(net, fen, n, depth);
      break;
  }
}

enum _Mode { top, play }

/// Evaluate every legal move from [fen] and print the top [n] sorted by
/// resulting cp (from the side-to-move's POV) using search depth [depth].
void _showTop(NnueNet net, String fen, int n, int depth) {
  final board = ch.Chess.fromFEN(fen);
  stdout.writeln(board.ascii);
  stdout.writeln(
    'side to move: ${board.turn == ch.Chess.WHITE ? "white" : "black"}',
  );
  if (board.game_over) {
    stdout.writeln('game over: ${_status(board)}');
    return;
  }

  final searcher = _NnueSearcher(net);
  final start = DateTime.now();
  final rows = searcher.rankRoot(board, depth);
  final ms = DateTime.now().difference(start).inMilliseconds;
  final show = rows.length < n ? rows.length : n;
  stdout.writeln('');
  stdout.writeln(
    'Top $show / ${rows.length} legal moves at depth $depth '
    '(${searcher.nodesVisited} nodes, ${ms}ms):',
  );
  stdout.writeln('  rank   move         resulting-cp');
  for (int k = 0; k < show; k++) {
    final r = rows[k];
    stdout.writeln(
      '  ${(k + 1).toString().padLeft(4)}   '
      '${r.san.padRight(10)}   ${r.moverCp.toStringAsFixed(1).padLeft(10)} cp',
    );
  }
}

/// Greedy self-play: pick the top-ranked search move each ply.
void _play(NnueNet net, String fen, int halfMoves, int depth) {
  final board = ch.Chess.fromFEN(fen);
  stdout.writeln(board.ascii);
  stdout.writeln('starting fen: $fen');
  stdout.writeln('search depth: $depth');
  stdout.writeln('');

  final searcher = _NnueSearcher(net);
  for (int ply = 0; ply < halfMoves; ply++) {
    if (board.game_over) {
      stdout.writeln('game over: ${_status(board)}');
      return;
    }
    final start = DateTime.now();
    searcher.nodesVisited = 0;
    final rows = searcher.rankRoot(board, depth);
    final ms = DateTime.now().difference(start).inMilliseconds;
    if (rows.isEmpty) {
      stdout.writeln('no legal moves — game over');
      return;
    }
    final pick = rows.first;
    final moveNo = (ply ~/ 2) + 1;
    final side = board.turn == ch.Chess.WHITE ? '.' : '…';
    stdout.writeln(
      '  $moveNo$side  ${pick.san.padRight(8)}  '
      '→ ${pick.moverCp.toStringAsFixed(1)} cp '
      '(${searcher.nodesVisited} nodes, ${ms}ms)',
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

/// Negamax + alpha-beta searcher backed by [NnueNet.evaluateInt].
///
/// Move ordering at each node: sort children by their depth-0 NNUE
/// score (best-looking first) to maximise alpha-beta cutoffs.
class _NnueSearcher {
  _NnueSearcher(this.net);

  final NnueNet net;
  int nodesVisited = 0;

  /// Root: enumerate every legal move, run a depth-`depth` search on the
  /// child, and return the ranked list.
  List<_Ranked> rankRoot(ch.Chess board, int depth) {
    final ordered = _orderMoves(board);
    final rows = <_Ranked>[];
    for (final entry in ordered) {
      if (!board.move(entry.san)) continue;
      final score = -_negamax(board, depth - 1, -_kMateScore, _kMateScore);
      board.undo();
      rows.add(_Ranked(entry.san, score));
    }
    rows.sort((a, b) => b.moverCp.compareTo(a.moverCp));
    return rows;
  }

  /// Standard negamax with alpha-beta. Returns the score from the side-
  /// to-move's POV at the current [board] state.
  double _negamax(ch.Chess board, int depth, double alpha, double beta) {
    nodesVisited++;
    if (board.in_checkmate) return -_kMateScore + (100 - depth).toDouble();
    if (board.in_stalemate ||
        board.insufficient_material ||
        board.in_threefold_repetition ||
        board.in_draw) {
      return 0.0;
    }
    if (depth <= 0) {
      return net.evaluateInt(encodeFen(board.fen)).cp;
    }

    final ordered = _orderMoves(board);
    var best = -_kMateScore * 2;
    for (final entry in ordered) {
      if (!board.move(entry.san)) continue;
      final score = -_negamax(board, depth - 1, -beta, -alpha);
      board.undo();
      if (score > best) best = score;
      if (score > alpha) alpha = score;
      if (alpha >= beta) break; // beta cutoff
    }
    return best;
  }

  /// Move-ordering helper: score each legal move by the NNUE cp of the
  /// resulting position (from the current mover's POV) and return the
  /// list sorted best-first. This is a cheap "depth-0" heuristic that
  /// dramatically improves alpha-beta pruning on deeper searches.
  List<_Ranked> _orderMoves(ch.Chess board) {
    final sans = board.moves().cast<String>();
    final rows = <_Ranked>[];
    for (final san in sans) {
      if (!board.move(san)) continue;
      final ri = net.evaluateInt(encodeFen(board.fen));
      board.undo();
      rows.add(_Ranked(san, -ri.cp));
    }
    rows.sort((a, b) => b.moverCp.compareTo(a.moverCp));
    return rows;
  }
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
