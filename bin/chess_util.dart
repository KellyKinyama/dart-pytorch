/// Build a UCI chess-move corpus for the character LM, preferring **real
/// games downloaded from Lichess** and falling back to synthetic random
/// legal games if the download is unavailable.
///
/// Pipeline:
///   1. download a user's games as PGN  → data/lichess_games.pgn
///   2. parse each game with the `chess` package, replay it, and emit the
///      moves as space-separated **UCI** (e.g. `e2e4 e7e5 g1f3 ...`),
///      one game per line → data/chess_uci.txt
///
/// UCI is chosen over SAN because it is uniform (from-square, to-square,
/// optional promotion) — easier for a small char model to produce
/// well-formed moves, and trivial to validate/apply (`board.move({...})`).
library;

import 'dart:io';
import 'dart:math' as math;

import 'package:chess/chess.dart' as ch;

const String kLichessPgnPath = 'data/lichess_games.pgn';
const String kChessUciPath = 'data/chess_uci.txt';
const String _startFen =
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

/// Sanitised cache key for a given set of users + per-user game count, so a
/// different request (more games, more users) rebuilds instead of returning
/// a stale corpus.
String _cacheKey(List<String> users, int maxGames) {
  final who = users.map((u) => u.toLowerCase()).join('-');
  final safe = who.replaceAll(RegExp(r'[^a-z0-9\-]'), '');
  return '${safe}_g$maxGames';
}

/// Load (and cache) the UCI corpus. On first use it downloads [maxGames]
/// games **per user** in [user] (a single name or a comma-separated list)
/// from Lichess and converts them to UCI; if the download fails it
/// synthesises [synthGames] random legal games instead.
Future<String> loadChessUci({
  String user = 'DrNykterstein',
  int maxGames = 400,
  int synthGames = 2000,
  int maxPlies = 40,
  int seed = 0,
  bool regenerate = false,
}) async {
  final users = user
      .split(',')
      .map((s) => s.trim())
      .where((s) => s.isNotEmpty)
      .toList();
  final key = _cacheKey(users, maxGames);
  final out = File('data/chess_uci_$key.txt');
  if (out.existsSync() && !regenerate) return out.readAsStringSync();

  String uci;
  try {
    final pgnBuf = StringBuffer();
    for (final u in users) {
      pgnBuf.writeln(await _ensureLichessPgn(u, maxGames, regenerate));
    }
    uci = _pgnToUci(pgnBuf.toString());
    if (uci.trim().isEmpty) throw StateError('no games parsed');
  } catch (e) {
    stdout.writeln('lichess download/parse failed ($e); '
        'falling back to $synthGames synthetic random games');
    uci = _synthUci(synthGames, maxPlies, seed);
  }

  out.parent.createSync(recursive: true);
  out.writeAsStringSync(uci);
  stdout.writeln('corpus: ${uci.length} chars -> ${out.path}');
  return uci;
}

/// Download `maxGames` of a single user's games as PGN (cached per user).
Future<String> _ensureLichessPgn(
    String user, int maxGames, bool regenerate) async {
  final safeUser = user.toLowerCase().replaceAll(RegExp(r'[^a-z0-9\-]'), '');
  final f = File('data/lichess_${safeUser}_g$maxGames.pgn');
  if (f.existsSync() && !regenerate) return f.readAsStringSync();

  final uri = Uri.parse(
    'https://lichess.org/api/games/user/$user'
    '?max=$maxGames&moves=true&tags=true&clocks=false&evals=false&opening=false',
  );
  stdout.writeln('downloading up to $maxGames games of "$user" from Lichess ...');
  final client = HttpClient()..connectionTimeout = const Duration(seconds: 30);
  try {
    final req = await client.getUrl(uri);
    req.headers.set(HttpHeaders.acceptHeader, 'application/x-chess-pgn');
    req.headers.set(HttpHeaders.userAgentHeader, 'dart-pytorch-char-lm');
    final resp = await req.close();
    if (resp.statusCode != 200) {
      throw HttpException('Lichess HTTP ${resp.statusCode}');
    }
    final buf = StringBuffer();
    await for (final chunk in resp.transform(const SystemEncoding().decoder)) {
      buf.write(chunk);
    }
    final pgn = buf.toString();
    f.parent.createSync(recursive: true);
    f.writeAsStringSync(pgn);
    stdout.writeln('saved PGN (${pgn.length} chars) -> ${f.path}');
    return pgn;
  } finally {
    client.close(force: true);
  }
}

/// Split a multi-game PGN, replay each game, and emit UCI move lines.
String _pgnToUci(String pgnText) {
  // Each game starts with an [Event "..."] header line.
  final games = pgnText.split(RegExp(r'(?=\[Event )'));
  final sb = StringBuffer();
  var ok = 0, bad = 0;
  for (final g in games) {
    if (g.trim().isEmpty) continue;
    try {
      final board = ch.Chess.fromFEN(_startFen);
      board.reset();
      if (!board.load_pgn(g)) {
        bad++;
        continue;
      }
      final hist = board.getHistory({'verbose': true});
      if (hist.isEmpty) continue;
      final line = hist
          .map((m) => '${(m as Map)['from']}${m['to']}${m['promotion'] ?? ''}')
          .join(' ');
      sb.writeln(line);
      ok++;
    } catch (_) {
      bad++;
    }
  }
  stdout.writeln('parsed $ok games to UCI ($bad skipped)');
  return sb.toString();
}

/// Random legal games as UCI (offline fallback).
String _synthUci(int games, int maxPlies, int seed) {
  final rng = math.Random(seed);
  final sb = StringBuffer();
  for (var g = 0; g < games; g++) {
    final board = ch.Chess.fromFEN(_startFen);
    final moves = <String>[];
    for (var ply = 0; ply < maxPlies; ply++) {
      if (board.game_over) break;
      final legal = board.generate_moves();
      if (legal.isEmpty) break;
      final mv = legal[rng.nextInt(legal.length)];
      moves.add('${mv.fromAlgebraic}${mv.toAlgebraic}'
          '${mv.promotion?.name ?? ''}');
      board.move(mv);
    }
    if (moves.isNotEmpty) sb.writeln(moves.join(' '));
  }
  return sb.toString();
}

// ---------------------------------------------------------------------------
// Legality tester — shared by every chess demo so the replay-and-count logic
// lives in one place.
// ---------------------------------------------------------------------------

/// Parse a UCI move string ("e2e4", "e7e8q") into a `chess` move map, or
/// `null` if it is too short to be a move.
Map<String, String>? uciToMove(String u) {
  if (u.length < 4) return null;
  final m = {'from': u.substring(0, 2), 'to': u.substring(2, 4)};
  if (u.length > 4) m['promotion'] = u.substring(4, 5);
  return m;
}

/// UCI string for a `chess` [ch.Move] ("e2e4", "e7e8q").
String moveToUci(ch.Move m) =>
    m.fromAlgebraic + m.toAlgebraic + (m.promotion?.name ?? '');

/// The legal moves in the current position as UCI strings.
List<String> legalUciMoves(ch.Chess board) =>
    [for (final m in board.generate_moves()) moveToUci(m)];

/// Result of replaying a UCI move sequence from the starting position.
class LegalityReport {
  /// Number of leading moves that were legal before the first illegal one.
  final int legal;

  /// Total whitespace-separated tokens examined.
  final int total;

  /// The first token that could not be legally played, or `null` if all were.
  final String? firstIllegal;

  /// ASCII board after the legal prefix was applied.
  final String boardAscii;

  const LegalityReport({
    required this.legal,
    required this.total,
    required this.firstIllegal,
    required this.boardAscii,
  });

  bool get allLegal => firstIllegal == null && total > 0;

  @override
  String toString() => 'legal move prefix: $legal / $total'
      '${firstIllegal != null ? '  (stopped at "$firstIllegal")' : ''}';
}

/// Replay space-separated UCI [movesText] from the initial position and report
/// how many leading moves are legal. This is the legality tester the demos
/// use to score a generated continuation.
LegalityReport checkUciLegality(String movesText) {
  final board = ch.Chess.fromFEN(_startFen);
  board.reset();
  final tokens =
      movesText.split(RegExp(r'\s+')).where((t) => t.isNotEmpty).toList();
  var legal = 0;
  String? firstIllegal;
  for (final uci in tokens) {
    final mv = uciToMove(uci);
    if (board.game_over || mv == null || !board.move(mv)) {
      firstIllegal = uci;
      break;
    }
    legal++;
  }
  return LegalityReport(
    legal: legal,
    total: tokens.length,
    firstIllegal: firstIllegal,
    boardAscii: board.ascii,
  );
}

