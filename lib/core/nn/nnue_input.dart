/// FEN → HalfKAv2_hm sparse feature index encoder for Stockfish NNUE.
///
/// The HalfKAv2_hm feature space, per side-to-move perspective, has
/// 22,528 possible active indices. A position typically activates
/// ~30 of them (roughly one per piece × two POVs). Every index is
/// computed via
///
///     idx(perspective, sq, piece, ksq) =
///         orient(perspective, sq, ksq)          // 0..63
///       + PieceSquareIndex[perspective][piece]  // 0..640 in steps of 64
///       + PS_NB * KingBuckets[orient(perspective, ksq, ksq)]
///
/// where `PS_NB = 11 * 64 = 704` and `KingBuckets` maps the (possibly
/// mirrored) king square to a bucket in `[0, 32)`. Horizontal mirroring
/// (XOR 7) is applied iff the king sits on files a–d, so the actual
/// king square is always on files e–h after orientation. Vertical flip
/// (XOR 56) is applied iff the perspective is black, so the perspective
/// side always sits on the bottom four ranks after orientation.
///
/// Reference: `Stockfish/src/nnue/features/half_ka_v2_hm.h` and
/// `.../nnue_common.h`.
library;

import 'dart:typed_data';

/// Perspective / side-to-move flag.
enum NnuePerspective { white, black }

/// The bare sparse-feature output that [encodeFen] produces.
class NnueFeatures {
  const NnueFeatures({
    required this.whiteActive,
    required this.blackActive,
    required this.stm,
    required this.pieceCount,
  });

  /// Active feature indices from white's perspective. Sorted ascending.
  final Int32List whiteActive;

  /// Active feature indices from black's perspective. Sorted ascending.
  final Int32List blackActive;

  /// Side to move in the input FEN.
  final NnuePerspective stm;

  /// Total number of pieces (both colors) on the board. Used by the
  /// runtime to pick the material bucket via `(pieceCount - 1) / 4`.
  final int pieceCount;
}

// Piece encoding: index into `_pieceChars`. 0 = W_PAWN … 5 = W_KING,
// 6 = B_PAWN … 11 = B_KING. Standard FEN characters.
const List<String> _pieceChars = [
  'P', 'N', 'B', 'R', 'Q', 'K',
  'p', 'n', 'b', 'r', 'q', 'k',
];

int _pieceIndex(String ch) {
  final i = _pieceChars.indexOf(ch);
  if (i < 0) throw FormatException('invalid FEN piece char "$ch"');
  return i;
}

// Bit 0 of `_pieceIndex` is 0 for white, 1 for black? No — layout above
// stacks W_PAWN..W_KING then B_PAWN..B_KING. Use color():
bool _isBlackPiece(int p) => p >= 6;
int _pieceTypeOf(int p) => p % 6; // 0=P, 1=N, 2=B, 3=R, 4=Q, 5=K

// PieceSquareIndex[perspective][piece] — base offset in `[0, PS_NB)`
// per (POV, piece) pair. Follows the SF table layout where each POV
// re-labels the piece color so that "our" pieces get the low offsets.
//
// Ordering within a POV:
//   friendly P, enemy P, friendly N, enemy N, ..., friendly K == enemy K.
const int _psNb = 11 * 64;
const int _psFriendPawn = 0 * 64;
const int _psEnemyPawn = 1 * 64;
const int _psFriendKnight = 2 * 64;
const int _psEnemyKnight = 3 * 64;
const int _psFriendBishop = 4 * 64;
const int _psEnemyBishop = 5 * 64;
const int _psFriendRook = 6 * 64;
const int _psEnemyRook = 7 * 64;
const int _psFriendQueen = 8 * 64;
const int _psEnemyQueen = 9 * 64;
const int _psKing = 10 * 64;

int _pieceSquareIndex(NnuePerspective persp, int piece) {
  final black = _isBlackPiece(piece);
  final type = _pieceTypeOf(piece);
  // "friend" means the POV's own color; from POV=white, white pieces
  // are friendly; from POV=black, black pieces are friendly.
  final friendly = (persp == NnuePerspective.white) ? !black : black;
  switch (type) {
    case 0: // pawn
      return friendly ? _psFriendPawn : _psEnemyPawn;
    case 1: // knight
      return friendly ? _psFriendKnight : _psEnemyKnight;
    case 2: // bishop
      return friendly ? _psFriendBishop : _psEnemyBishop;
    case 3: // rook
      return friendly ? _psFriendRook : _psEnemyRook;
    case 4: // queen
      return friendly ? _psFriendQueen : _psEnemyQueen;
    case 5: // king — both colors share the same "king" slot in
      // HalfKAv2_hm (the piece-square-index table has PS_KING for both
      // W_KING and B_KING). This is what makes it "v2" over HalfKA.
      return _psKing;
    default:
      throw StateError('unreachable piece type $type');
  }
}

// KingBuckets[64]: map an oriented king square (guaranteed on files
// e–h after horizontal mirror) to a bucket in `[0, 32)`. Squares on
// files a–d get -1 because they should never be seen after mirror.
//
// Values from `Stockfish/src/nnue/features/half_ka_v2_hm.h`.
const List<int> _kingBuckets = [
  -1, -1, -1, -1, 31, 30, 29, 28,
  -1, -1, -1, -1, 27, 26, 25, 24,
  -1, -1, -1, -1, 23, 22, 21, 20,
  -1, -1, -1, -1, 19, 18, 17, 16,
  -1, -1, -1, -1, 15, 14, 13, 12,
  -1, -1, -1, -1, 11, 10,  9,  8,
  -1, -1, -1, -1,  7,  6,  5,  4,
  -1, -1, -1, -1,  3,  2,  1,  0,
];

/// SF's HalfKAv2_hm `orient`: XOR square with `56` for black
/// perspective (vertical flip so friend sits on ranks 1–4) and with
/// `7` if the king is on files a–d (horizontal mirror so king lands
/// on files e–h).
int _orient(NnuePerspective persp, int sq, int ksq) {
  final flipV = persp == NnuePerspective.black ? 56 : 0;
  final kingFile = ksq & 7;
  final mirrorH = kingFile < 4 ? 7 : 0;
  return sq ^ flipV ^ mirrorH;
}

int _makeIndex(NnuePerspective persp, int sq, int piece, int ksq) {
  final orientedSq = _orient(persp, sq, ksq);
  final orientedKsq = _orient(persp, ksq, ksq);
  final bucket = _kingBuckets[orientedKsq];
  if (bucket < 0) {
    throw StateError(
      'KingBuckets returned -1 for oriented ksq=$orientedKsq '
      '(king should have been mirrored to files e-h)',
    );
  }
  return orientedSq + _pieceSquareIndex(persp, piece) + _psNb * bucket;
}

/// Parse [fen] and return the two sparse HalfKAv2_hm feature index
/// sets (one per POV). Only the piece-placement + side-to-move fields
/// of the FEN are consulted; castling / ep / clocks are ignored
/// because they don't feed into the NNUE input.
NnueFeatures encodeFen(String fen) {
  final parts = fen.trim().split(RegExp(r'\s+'));
  if (parts.isEmpty) throw const FormatException('empty FEN');
  final placement = parts[0];
  final stmChar = parts.length > 1 ? parts[1] : 'w';
  final stm = stmChar == 'b' ? NnuePerspective.black : NnuePerspective.white;

  // Parse board. rank 8 comes first in FEN; convert to SF square
  // convention where rank 1 = squares 0..7 and rank 8 = squares 56..63.
  final board = List<int>.filled(64, -1); // -1 = empty, else piece 0..11
  int fenRank = 7;
  int fenFile = 0;
  for (final ch in placement.split('')) {
    if (ch == '/') {
      fenRank -= 1;
      fenFile = 0;
      continue;
    }
    final digit = int.tryParse(ch);
    if (digit != null) {
      fenFile += digit;
      continue;
    }
    if (fenRank < 0 || fenRank > 7 || fenFile < 0 || fenFile > 7) {
      throw FormatException('FEN overran board at "$ch"');
    }
    final sq = fenRank * 8 + fenFile;
    board[sq] = _pieceIndex(ch);
    fenFile += 1;
  }

  // Locate both kings.
  int wKing = -1;
  int bKing = -1;
  int pieces = 0;
  for (int sq = 0; sq < 64; sq++) {
    final p = board[sq];
    if (p < 0) continue;
    pieces++;
    if (p == 5) wKing = sq; // W_KING
    if (p == 11) bKing = sq; // B_KING
  }
  if (wKing < 0 || bKing < 0) {
    throw const FormatException('FEN missing one or both kings');
  }

  final whiteActive = <int>[];
  final blackActive = <int>[];
  for (int sq = 0; sq < 64; sq++) {
    final p = board[sq];
    if (p < 0) continue;
    whiteActive.add(_makeIndex(NnuePerspective.white, sq, p, wKing));
    blackActive.add(_makeIndex(NnuePerspective.black, sq, p, bKing));
  }
  whiteActive.sort();
  blackActive.sort();

  return NnueFeatures(
    whiteActive: Int32List.fromList(whiteActive),
    blackActive: Int32List.fromList(blackActive),
    stm: stm,
    pieceCount: pieces,
  );
}
