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
  static Tensor fromFen(String fen) {
    final parts = fen.trim().split(RegExp(r'\s+'));
    if (parts.isEmpty) {
      throw ArgumentError('Lc0Input: empty FEN');
    }
    final placement = parts[0];
    final stm = parts.length > 1 ? parts[1] : 'w';
    final castling = parts.length > 2 ? parts[2] : '-';
    final rule50 = parts.length > 4 ? int.tryParse(parts[4]) ?? 0 : 0;
    final fullMove = parts.length > 5 ? int.tryParse(parts[5]) ?? 1 : 1;

    if (stm != 'w' && stm != 'b') {
      throw ArgumentError('Lc0Input: side-to-move must be w or b, got "$stm"');
    }
    final blackToMove = stm == 'b';

    final data = Float32List(112 * 8 * 8);

    // Piece placement is given top-down (rank 8 first). Store into
    // whitePOV[rank][file] with rank 0 = white's back rank (a1),
    // rank 7 = black's back rank (a8).
    final ranks = placement.split('/');
    if (ranks.length != 8) {
      throw ArgumentError(
        'Lc0Input: expected 8 ranks in placement, got ${ranks.length}',
      );
    }
    // Fill piece planes 0..11 in white POV first, then mirror below
    // if it's black to move.
    for (int fenRank = 0; fenRank < 8; fenRank++) {
      final rank = 7 - fenRank; // rank index in white-POV coords
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
        _set(data, idx, rank, file, 1.0);
        file++;
      }
    }

    // If black to move, mirror board vertically and swap piece colours
    // so STM's back rank sits at rank 0.
    if (blackToMove) {
      _mirrorForBlack(data);
    }

    // Castling rights, INPUT_CLASSICAL_112_PLANE layout (see
    // lc0/src/neural/encoder.cc EncodePositionForNN):
    //   plane 104 = our queenside (000)
    //   plane 105 = our kingside  (00)
    //   plane 106 = their queenside
    //   plane 107 = their kingside
    final wK = castling.contains('K');
    final wQ = castling.contains('Q');
    final bK = castling.contains('k');
    final bQ = castling.contains('q');
    final stmK = blackToMove ? bK : wK;
    final stmQ = blackToMove ? bQ : wQ;
    final oppK = blackToMove ? wK : bK;
    final oppQ = blackToMove ? wQ : bQ;
    if (stmQ) _fillPlane(data, 104, 1.0);
    if (stmK) _fillPlane(data, 105, 1.0);
    if (oppQ) _fillPlane(data, 106, 1.0);
    if (oppK) _fillPlane(data, 107, 1.0);

    // Plane 108 = "we_are_black" (1.0 iff STM is black).
    _fillPlane(data, 108, blackToMove ? 1.0 : 0.0);

    // Plane 109 = rule-50 half-move counter, RAW (not normalized) for
    // the classical INPUT_CLASSICAL_112_PLANE format. LC0 only
    // divides by 100 in the "hectoplies" input formats.
    _fillPlane(data, 109, rule50.toDouble());

    // Plane 110 used to be a movecount plane; the modern encoder
    // leaves it zero for the classical format.
    _fillPlane(data, 110, 0.0);

    // Plane 111 = constant ones (helps the network find board edges).
    _fillPlane(data, 111, 1.0);

    // fullMove parameter is intentionally unused for the classical
    // format — LC0's own encoder does not put it anywhere.
    // ignore: unused_local_variable
    final _ = fullMove;

    return Tensor.fromList([1, 112, 8, 8], data.toList(), device: Device.CPU);
  }

  static void _set(Float32List d, int plane, int rank, int file, double v) {
    d[plane * 64 + rank * 8 + file] = v;
  }

  static void _fillPlane(Float32List d, int plane, double v) {
    final base = plane * 64;
    for (int i = 0; i < 64; i++) {
      d[base + i] = v;
    }
  }

  static void _mirrorForBlack(Float32List d) {
    // Swap white / black piece planes (0..5 <-> 6..11) AND flip each
    // board vertically (rank -> 7-rank).
    final tmp = Float32List(64);
    for (int i = 0; i < 6; i++) {
      final aBase = i * 64;
      final bBase = (i + 6) * 64;
      for (int r = 0; r < 8; r++) {
        for (int f = 0; f < 8; f++) {
          tmp[(7 - r) * 8 + f] = d[aBase + r * 8 + f];
        }
      }
      for (int r = 0; r < 8; r++) {
        for (int f = 0; f < 8; f++) {
          d[aBase + r * 8 + f] = d[bBase + (7 - r) * 8 + f];
        }
      }
      for (int j = 0; j < 64; j++) {
        d[bBase + j] = tmp[j];
      }
    }
  }
}

/// Standard chess starting position in FEN.
const startFen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
