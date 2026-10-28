/// Engram n-gram hashing for Needle 3, matching `engram_indices` in the
/// reference `architecture.py`. For each sequence position the last `order`
/// tokens are folded into an FNV-style 32-bit hash, once per table, giving a
/// row index into that table's `slots` rows.
///
/// Geometry for the shipped model: orders (2, 3), `heads` = 3, so
/// `numTables = len(orders) * heads = 6`, `slots = 18432`.
library;

import 'dart:typed_data';

const int _engramSeed = 0x9E3779B9;
const int _engramPrime = 0x01000193;
const int _mask32 = 0xFFFFFFFF;

/// Number of engram tables for the given geometry.
int engramTableCount(List<int> orders, int heads) => orders.length * heads;

/// Compute engram row indices for [tokens]. Returns `[T]` rows, each with
/// `orders.length * heads` table indices in `[0, slots)`.
///
/// Mirrors the reference exactly: `shift_right(u, j)` means token `t-j`
/// (zero before the start), `acc = (acc ^ u) * PRIME` per n-gram step, a
/// final `acc ^= acc >> 15`, then `acc % slots`.
List<Int32List> engramIndices(
  List<int> tokens, {
  List<int> orders = const [2, 3],
  int heads = 3,
  int slots = 18432,
  int seedHeads = 0,
}) {
  final t = tokens.length;
  final numTables = orders.length * heads;
  final stride = seedHeads != 0 ? seedHeads : heads;
  final out = List<Int32List>.generate(t, (_) => Int32List(numTables));

  var col = 0;
  for (var oi = 0; oi < orders.length; oi++) {
    final order = orders[oi];
    for (var h = 0; h < heads; h++) {
      final seed = (_engramSeed * (oi * stride + h + 1)) & _mask32;
      for (var p = 0; p < t; p++) {
        var acc = seed;
        for (var j = 0; j < order; j++) {
          final uj = (p - j >= 0) ? (tokens[p - j] & _mask32) : 0;
          acc = ((acc ^ uj) * _engramPrime) & _mask32;
        }
        acc ^= (acc >> 15);
        out[p][col] = acc % slots;
      }
      col++;
    }
  }
  return out;
}
