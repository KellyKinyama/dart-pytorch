/// Binary-vector companion to [Index].
///
/// FAISS' `IndexBinary` family stores fixed-length bit strings and
/// scores them by Hamming distance (popcount of the XOR). Because the
/// input/output types differ from the float base class (`Uint8List`
/// vs `Float32List`), we keep the hierarchy separate rather than
/// forcing an awkward generic parameter through everything.
library;

import 'dart:typed_data';

import 'index.dart';

/// Precomputed byte popcount table (index → number of set bits).
final Uint8List _popcount8 = Uint8List.fromList(
  List<int>.generate(256, (i) {
    var c = 0;
    var b = i;
    while (b != 0) {
      if ((b & 1) != 0) c++;
      b >>= 1;
    }
    return c;
  }),
);

/// Hamming distance between `a` and `b[bStart..bStart + a.length)`.
///
/// Not exposed publicly; used by binary indexes in their hot loops.
int _hamming(Uint8List a, Uint8List b, int bStart, int codeSize) {
  var s = 0;
  for (var i = 0; i < codeSize; i++) {
    s += _popcount8[a[i] ^ b[bStart + i]];
  }
  return s;
}

/// Base class for binary (bit-string) indexes.
///
/// * [codeSize] — bytes per vector (so vector dim `d = codeSize * 8`).
/// * [ntotal] — vectors currently indexed.
/// * [search] returns Hamming distances as float32 (integer values
///   fitting easily) so callers can share [SearchResult] with the
///   float family.
abstract class IndexBinary {
  IndexBinary(this.codeSize);

  /// Bytes per vector.
  final int codeSize;

  /// Bit dimensionality.
  int get d => codeSize * 8;

  int ntotal = 0;
  bool isTrained = true;

  /// Train (no-op for indexes that don't need it).
  void train(List<Uint8List> xs) {
    isTrained = true;
  }

  /// Add vectors. Ids are sequential from [ntotal].
  void add(List<Uint8List> xs);

  /// k-NN by Hamming distance. Distances are integers cast to float32.
  SearchResult search(List<Uint8List> queries, int k);
}

/// Public alias so binary-index implementations can reuse the shared
/// popcount + Hamming primitives without exposing library-private
/// names.
int hammingDistance(Uint8List a, Uint8List b, int bStart, int codeSize) =>
    _hamming(a, b, bStart, codeSize);

/// Public accessor for the popcount lookup table.
Uint8List get popcount8Table => _popcount8;
