/// `IndexLSH` — random-projection locality-sensitive hashing.
///
/// Learns a projection matrix `W` of shape `[nbits, d]` with i.i.d.
/// standard-normal entries and encodes each database vector `x` into a
/// [nbits]-bit code via `sign(W · x)`. At search time queries are
/// encoded the same way and compared to stored codes via Hamming
/// distance — extremely fast and cache-friendly, with recall that
/// grows monotonically with `nbits`.
///
/// This mirrors FAISS' `IndexLSH`. It extends the float [Index] base
/// class (float in, binary storage) rather than [IndexBinary].
///
/// Memory: `ceil(nbits / 8)` bytes per stored vector, plus the
/// `nbits * d * 4` byte projection matrix.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'index.dart';
import 'index_binary.dart';
import 'index_io.dart';

class IndexLSH extends Index {
  IndexLSH({required int d, required this.nbits, this.seed = 1234})
    : codeSize = (nbits + 7) ~/ 8,
      _proj = Float32List(nbits * d),
      super(d, Metric.l2) {
    isTrained = false;
  }

  /// Number of hash bits per vector.
  final int nbits;

  /// Bytes used to store each code (`ceil(nbits / 8)`).
  final int codeSize;

  final int seed;

  /// Row-major projection matrix `W` of shape `[nbits, d]`.
  final Float32List _proj;

  Uint8List _codes = Uint8List(0);
  int _capacityCodes = 0;

  Float32List get projection => _proj;

  /// Live view of the packed code buffer, sliced to the currently
  /// occupied region (`ntotal * codeSize` bytes). Backed by the same
  /// underlying storage as internal writes, so callers must not mutate
  /// it. Provided as an I/O hook for FAISS-format serialization.
  Uint8List get codes => Uint8List.sublistView(_codes, 0, ntotal * codeSize);

  @override
  void train(List<Float32List> xs) {
    // The random projection is data-independent, so training data is
    // ignored. We simply sample W from N(0, 1). Idempotent.
    trainProjection();
  }

  /// Sample the projection matrix from `N(0, 1)`. Idempotent.
  void trainProjection() {
    final rng = math.Random(seed);
    for (var i = 0; i < _proj.length; i++) {
      // Box-Muller (approximation via sum of two uniforms is enough for
      // a random-projection basis — anything mean-zero symmetric works).
      final u1 = math.max(rng.nextDouble(), 1e-12);
      final u2 = rng.nextDouble();
      _proj[i] = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2);
    }
    isTrained = true;
  }

  /// Encode a batch of vectors to `codeSize`-byte binary codes.
  Uint8List _encode(List<Float32List> xs) {
    final n = xs.length;
    final out = Uint8List(n * codeSize);
    for (var i = 0; i < n; i++) {
      final row = xs[i];
      if (row.length != d) {
        throw ArgumentError('vector $i length ${row.length} != $d');
      }
      final base = i * codeSize;
      for (var b = 0; b < nbits; b++) {
        var s = 0.0;
        final projRow = b * d;
        for (var j = 0; j < d; j++) {
          s += _proj[projRow + j] * row[j];
        }
        if (s > 0) {
          out[base + (b >> 3)] |= 1 << (b & 7);
        }
      }
    }
    return out;
  }

  @override
  void add(List<Float32List> xs) {
    if (!isTrained) {
      // Auto-train on the first `add` so the caller can skip the
      // separate `trainProjection` step (FAISS behaves similarly).
      trainProjection();
    }
    if (xs.isEmpty) return;
    final newCodes = _encode(xs);
    final need = (ntotal + xs.length) * codeSize;
    if (need > _capacityCodes) {
      var newCap = _capacityCodes == 0 ? 1024 * codeSize : _capacityCodes;
      while (newCap < need) {
        newCap *= 2;
      }
      final grown = Uint8List(newCap);
      for (var i = 0; i < ntotal * codeSize; i++) {
        grown[i] = _codes[i];
      }
      _codes = grown;
      _capacityCodes = newCap;
    }
    final off = ntotal * codeSize;
    for (var i = 0; i < newCodes.length; i++) {
      _codes[off + i] = newCodes[i];
    }
    ntotal += xs.length;
  }

  @override
  SearchResult search(List<Float32List> queries, int k) {
    if (!isTrained) throw StateError('IndexLSH.search before add/train');
    final nq = queries.length;
    final qCodes = _encode(queries);
    final distances = List<Float32List>.generate(nq, (_) => Float32List(k));
    final ids = List<Int32List>.generate(nq, (_) => Int32List(k));

    for (var qi = 0; qi < nq; qi++) {
      final qBase = qi * codeSize;
      final qView = Uint8List.sublistView(qCodes, qBase, qBase + codeSize);
      final heap = TopK(k, maxIsWorst: true); // smaller Hamming = closer
      for (var vi = 0; vi < ntotal; vi++) {
        final h = hammingDistance(qView, _codes, vi * codeSize, codeSize);
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
    w.writeU32(d);
    w.writeU32(metricToU32(metric));
    w.writeU32(ntotal);
    w.writeU8(isTrained ? 1 : 0);
    w.writeU32(nbits);
    w.writeU32(seed);
    w.writeF32List(_proj);
    if (ntotal > 0) {
      w.writeU8List(Uint8List.sublistView(_codes, 0, ntotal * codeSize));
    }
  }

  static IndexLSH readFrom(IoReader r) {
    final d = r.readU32();
    // Metric byte is present for symmetry but LSH is always L2-based.
    metricFromU32(r.readU32());
    final ntotal = r.readU32();
    final isTrained = r.readU8() != 0;
    final nbits = r.readU32();
    final seed = r.readU32();
    final idx = IndexLSH(d: d, nbits: nbits, seed: seed);
    final proj = r.readF32List(nbits * d);
    for (var i = 0; i < proj.length; i++) {
      idx._proj[i] = proj[i];
    }
    if (ntotal > 0) {
      idx._codes = r.readU8List(ntotal * idx.codeSize);
      idx._capacityCodes = ntotal * idx.codeSize;
    }
    idx.ntotal = ntotal;
    idx.isTrained = isTrained;
    return idx;
  }

  /// I/O hook used by FAISS-interop readers to install an externally
  /// supplied projection matrix and code payload without going through
  /// [add]. [proj] must be `nbits * d` floats (row-major `[nbits, d]`);
  /// [codes] must be exactly `newNtotal * codeSize` bytes.
  void ioSetProjectionAndCodes(
    Float32List proj,
    Uint8List codes,
    int newNtotal,
  ) {
    if (proj.length != _proj.length) {
      throw ArgumentError(
        'ioSetProjectionAndCodes: proj length ${proj.length} != '
        'nbits * d = ${_proj.length}',
      );
    }
    if (codes.length != newNtotal * codeSize) {
      throw ArgumentError(
        'ioSetProjectionAndCodes: codes length ${codes.length} != '
        'newNtotal * codeSize = ${newNtotal * codeSize}',
      );
    }
    for (var i = 0; i < proj.length; i++) {
      _proj[i] = proj[i];
    }
    _codes = Uint8List(codes.length);
    for (var i = 0; i < codes.length; i++) {
      _codes[i] = codes[i];
    }
    _capacityCodes = codes.length;
    ntotal = newNtotal;
  }
}
