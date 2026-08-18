/// Chess FEN -> 112-plane LC0 input tensor.
///
/// Minimal encoder covering the current-position slice of LC0's
/// classical input format (Chess1 / T1 layout, no history):
///
///   planes  0..5    white P, N, B, R, Q, K on the given side-to-move POV
///   planes  6..11   black p, n, b, r, q, k
///   plane   12      repetitions counter (zeroed here, no history)
///   planes 13..103  history slots (7 more half-moves; zeroed)
///   planes 104..107 castling rights (STM K, STM Q, opponent k, opponent q)
///   plane   108     side to move (0 for white, 1 for black — flat plane)
///   plane   109     rule-50 half-move counter / 99
///   plane   110     ply (0-based total half-move count, unused for demo)
///   plane   111     all ones (constant)
///
/// LC0 mirrors the board so the side-to-move sits at the bottom rank.
/// When it's black to move we flip vertically and swap colours.
///
/// Not implemented (returns a plausible but non-authoritative input
/// for arbitrary mid-game positions): repetition detection, en-passant
/// target square, move-history planes. For a real chess UI you would
/// track those alongside the FEN.
library;

import 'dart:typed_data';

import '../tensor/tensor.dart';

const _pieceIndex = <String, int>{
  'P': 0,
  'N': 1,
  'B': 2,
  'R': 3,
  'Q': 4,
  'K': 5,
  'p': 6,
  'n': 7,
  'b': 8,
  'r': 9,
  'q': 10,
  'k': 11,
};

class Lc0Input {
  /// Parse a FEN into a `[1, 112, 8, 8]` CPU tensor ready to feed
  /// into [Lc0Net].
  ///
  /// This is a convenience wrapper around [fromFens] with a
  /// single-position history (all older history planes zeroed —
  /// LC0's `FillEmptyHistory::NO` behavior).
  static Tensor fromFen(String fen) => fromFens(<String>[fen]);

  /// Encode up to 8 game positions with proper LC0 `kMoveHistory`
  /// layout: `fens[0]` = current position, `fens[k]` = k plies ago.
  /// Missing history plies (fewer than 8 FENs) leave those planes
  /// zeroed, matching LC0's `FillEmptyHistory::NO` policy for
  /// engine play.
  ///
  /// Per-slot layout (13 planes each, base = slot * 13):
  ///   base+0..5    STM's pieces (P N B R Q K), current-STM POV
  ///   base+6..11   opponent's pieces
  ///   base+12      set to 1.0 if this position is a repetition of a
  ///                later ancestor position (2-fold detection)
  ///
  /// Auxiliary planes 104..111 are computed from the current FEN
  /// (castling, side-to-move, rule50, constant-1).
  static Tensor fromFens(List<String> fens) {
    if (fens.isEmpty) {
      throw ArgumentError('Lc0Input.fromFens: at least one FEN required');
    }
    final data = Float32List(112 * 8 * 8);

    // Parse the current position's meta (STM, castling, rule50) once.
    final current = _parseFen(fens[0]);
    final blackToMove = current.blackToMove;

    // Compute board hashes for repetition detection across history.
    final hashes = <int>[];
    for (final f in fens) {
      hashes.add(_parseFen(f).boardHash);
    }

    // Fill up to 8 history slots. i=0 is current, i=7 is 7 plies back.
    for (int i = 0; i < 8 && i < fens.length; i++) {
      final base = i * 13;
      final parsed = i == 0 ? current : _parseFen(fens[i]);
      _fillPiecePlanes(data, parsed, base, blackToMove);
      // Repetition flag: set if this slot's board hash matches any
      // strictly-later slot (i.e. an earlier ply in time). Mirrors
      // LC0's `GetRepetitions() >= 1` check at encoder.cc:290.
      final h = hashes[i];
      for (int j = i + 1; j < fens.length; j++) {
        if (hashes[j] == h) {
          _fillPlane(data, base + 12, 1.0);
          break;
        }
      }
    }

    // Aux planes computed from the CURRENT position only.
    if (current.stmQ) _fillPlane(data, 104, 1.0);
    if (current.stmK) _fillPlane(data, 105, 1.0);
    if (current.oppQ) _fillPlane(data, 106, 1.0);
    if (current.oppK) _fillPlane(data, 107, 1.0);
    _fillPlane(data, 108, blackToMove ? 1.0 : 0.0);
    _fillPlane(data, 109, current.rule50.toDouble());
    _fillPlane(data, 110, 0.0);
    _fillPlane(data, 111, 1.0);

    return Tensor.fromFloat32List([1, 112, 8, 8], data, device: Device.CPU);
  }

  /// Parses a FEN's fields we care about and computes a lightweight
  /// hash of just the board placement (used for repetition detection
  /// across the history).
  static _ParsedFen _parseFen(String fen) {
    final parts = fen.trim().split(RegExp(r'\s+'));
    if (parts.isEmpty) throw ArgumentError('Lc0Input: empty FEN');
    final placement = parts[0];
    final stm = parts.length > 1 ? parts[1] : 'w';
    final castling = parts.length > 2 ? parts[2] : '-';
    final rule50 = parts.length > 4 ? int.tryParse(parts[4]) ?? 0 : 0;
    if (stm != 'w' && stm != 'b') {
      throw ArgumentError('Lc0Input: side-to-move must be w or b, got "$stm"');
    }
    final blackToMove = stm == 'b';
    final wK = castling.contains('K');
    final wQ = castling.contains('Q');
    final bK = castling.contains('k');
    final bQ = castling.contains('q');
    return _ParsedFen(
      placement: placement,
      blackToMove: blackToMove,
      stmK: blackToMove ? bK : wK,
      stmQ: blackToMove ? bQ : wQ,
      oppK: blackToMove ? wK : bK,
      oppQ: blackToMove ? wQ : bQ,
      rule50: rule50,
      boardHash: Object.hashAll([placement, blackToMove]),
    );
  }

  /// Fill piece planes `[base..base+11]` for the given position,
  /// mirrored vertically + colours swapped iff the CURRENT position
  /// (root of the history) has black to move.
  static void _fillPiecePlanes(
    Float32List data,
    _ParsedFen p,
    int base,
    bool currentIsBlackToMove,
  ) {
    final ranks = p.placement.split('/');
    if (ranks.length != 8) {
      throw ArgumentError(
        'Lc0Input: expected 8 ranks in placement, got ${ranks.length}',
      );
    }
    // Encode white-POV into a temp block first, then mirror if needed.
    final block = Float32List(12 * 64);
    for (int fenRank = 0; fenRank < 8; fenRank++) {
      final rank = 7 - fenRank;
      final row = ranks[fenRank];
      int file = 0;
      for (int k = 0; k < row.length; k++) {
        final ch = row[k];
        final digit = int.tryParse(ch);
        if (digit != null) {
          file += digit;
          continue;
        }
        final idx = _pieceIndex[ch];
        if (idx == null) {
          throw ArgumentError('Lc0Input: bad FEN char "$ch"');
        }
        block[idx * 64 + rank * 8 + file] = 1.0;
        file++;
      }
    }
    if (currentIsBlackToMove) _mirrorForBlackBlock(block);
    for (int i = 0; i < 12; i++) {
      final srcOff = i * 64;
      final dstOff = (base + i) * 64;
      for (int j = 0; j < 64; j++) {
        data[dstOff + j] = block[srcOff + j];
      }
    }
  }

  /// [_mirrorForBlack] on a standalone 12*64 block.
  static void _mirrorForBlackBlock(Float32List block) {
    final tmp = Float32List(64);
    for (int i = 0; i < 6; i++) {
      final aBase = i * 64;
      final bBase = (i + 6) * 64;
      for (int r = 0; r < 8; r++) {
        for (int f = 0; f < 8; f++) {
          tmp[(7 - r) * 8 + f] = block[aBase + r * 8 + f];
        }
      }
      for (int r = 0; r < 8; r++) {
        for (int f = 0; f < 8; f++) {
          block[aBase + r * 8 + f] = block[bBase + (7 - r) * 8 + f];
        }
      }
      for (int j = 0; j < 64; j++) {
        block[bBase + j] = tmp[j];
      }
    }
  }

  static void _fillPlane(Float32List d, int plane, double v) {
    final base = plane * 64;
    for (int i = 0; i < 64; i++) {
      d[base + i] = v;
    }
  }
}

/// Standard chess starting position in FEN.
const startFen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

class _ParsedFen {
  final String placement;
  final bool blackToMove;
  final bool stmK;
  final bool stmQ;
  final bool oppK;
  final bool oppQ;
  final int rule50;
  final int boardHash;
  const _ParsedFen({
    required this.placement,
    required this.blackToMove,
    required this.stmK,
    required this.stmQ,
    required this.oppK,
    required this.oppQ,
    required this.rule50,
    required this.boardHash,
  });
}
