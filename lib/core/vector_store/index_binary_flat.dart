/// Brute-force Hamming-distance index over fixed-length bit strings.
///
/// Mirrors FAISS' `IndexBinaryFlat`. Stores `ntotal * codeSize` bytes
/// contiguously and scans linearly at search time. This is the exact
/// ground truth for the other binary indexes.
library;

import 'dart:typed_data';

import 'index.dart';
import 'index_binary.dart';
import 'index_io.dart';

class IndexBinaryFlat extends IndexBinary {
  IndexBinaryFlat(super.codeSize);

  Uint8List _storage = Uint8List(0);
  int _capacity = 0; // in vectors

  @override
  void add(List<Uint8List> xs) {
    if (xs.isEmpty) return;
    final n = xs.length;
    if (ntotal + n > _capacity) {
      var newCap = _capacity == 0 ? 1024 : _capacity;
      while (newCap < ntotal + n) {
        newCap *= 2;
      }
      final grown = Uint8List(newCap * codeSize);
      for (var i = 0; i < ntotal * codeSize; i++) {
        grown[i] = _storage[i];
      }
      _storage = grown;
      _capacity = newCap;
    }
    for (var i = 0; i < n; i++) {
      if (xs[i].length != codeSize) {
        throw ArgumentError(
          'vector $i has length ${xs[i].length}, expected $codeSize',
        );
      }
      final base = (ntotal + i) * codeSize;
      for (var j = 0; j < codeSize; j++) {
        _storage[base + j] = xs[i][j];
      }
    }
    ntotal += n;
  }

  /// Direct read-only view of the code at position `id`.
  Uint8List reconstruct(int id) {
    if (id < 0 || id >= ntotal) {
      throw RangeError('id $id out of range [0, $ntotal)');
    }
    return Uint8List.sublistView(_storage, id * codeSize, (id + 1) * codeSize);
  }

  /// Read-only view of the packed code region actually in use
  /// (`ntotal * codeSize` bytes). Used by FAISS-format interop to emit
  /// the `xb` payload without copying.
  Uint8List get codes =>
      Uint8List.sublistView(_storage, 0, ntotal * codeSize);

  /// I/O hook: replace the packed code buffer with [newCodes] and set
  /// `ntotal = newNtotal`. Used by FAISS-format readers to rehydrate
  /// an `IndexBinaryFlat` in one shot instead of round-tripping through
  /// [add].
  ///
  /// Throws [ArgumentError] if `newCodes.length != newNtotal * codeSize`.
  void ioSetCodes(Uint8List newCodes, int newNtotal) {
    if (newCodes.length != newNtotal * codeSize) {
      throw ArgumentError(
        'ioSetCodes: got ${newCodes.length} bytes, expected '
        '${newNtotal * codeSize} (= newNtotal * codeSize)',
      );
    }
    _storage = Uint8List(newCodes.length);
    for (var i = 0; i < newCodes.length; i++) {
      _storage[i] = newCodes[i];
    }
    _capacity = newNtotal;
    ntotal = newNtotal;
  }

  @override
  SearchResult search(List<Uint8List> queries, int k) {
    final nq = queries.length;
    final distances = List<Float32List>.generate(nq, (_) => Float32List(k));
    final ids = List<Int32List>.generate(nq, (_) => Int32List(k));

    for (var qi = 0; qi < nq; qi++) {
      final q = queries[qi];
      if (q.length != codeSize) {
        throw ArgumentError('query $qi length ${q.length} != $codeSize');
      }
      final heap = TopK(k, maxIsWorst: true); // smaller Hamming = closer
      for (var vi = 0; vi < ntotal; vi++) {
        final h = hammingDistance(q, _storage, vi * codeSize, codeSize);
        heap.push(h.toDouble(), vi);
      }
      final sorted = heap.sorted();
      distances[qi] = sorted.scores;
      ids[qi] = sorted.ids;
    }
    return SearchResult(distances, ids);
  }

  // --- persistence --------------------------------------------------------

  void writeTo(IoWriter w) {
    w.writeU32(codeSize);
    w.writeU32(ntotal);
    w.writeU8(isTrained ? 1 : 0);
    if (ntotal > 0) {
      w.writeU8List(Uint8List.sublistView(_storage, 0, ntotal * codeSize));
    }
  }

  static IndexBinaryFlat readFrom(IoReader r) {
    final codeSize = r.readU32();
    final ntotal = r.readU32();
    final isTrained = r.readU8() != 0;
    final idx = IndexBinaryFlat(codeSize);
    if (ntotal > 0) {
      idx._storage = r.readU8List(ntotal * codeSize);
      idx._capacity = ntotal;
    }
    idx.ntotal = ntotal;
    idx.isTrained = isTrained;
    return idx;
  }
}
