/// Product Quantizer + `IndexPQ`.
///
/// A [ProductQuantizer] splits every `d`-dim vector into [m] equal-size
/// subvectors, each independently quantized to one of `2^nbits`
/// codewords learned by k-means in the subspace. Vectors are therefore
/// encoded as `m` byte-sized (assuming `nbits == 8`) codes.
///
/// [IndexPQ] holds one `Uint8List` of length `ntotal * m` and performs
/// Asymmetric Distance Computation (ADC) at search time via a
/// per-query lookup table.
library;

import 'dart:typed_data';

import 'index.dart';
import 'index_io.dart';
import 'kmeans.dart';

/// Product quantizer.
class ProductQuantizer {
  ProductQuantizer({
    required this.d,
    required this.m,
    this.nbits = 8,
    this.kmeansIters = 25,
    this.seed = 1234,
  }) : ksub = 1 << nbits {
    if (d % m != 0) {
      throw ArgumentError(
        'ProductQuantizer: d ($d) must be divisible by m ($m)',
      );
    }
    if (nbits != 8) {
      // Only 8 bits fully supported for simplicity (byte-packed codes).
      throw ArgumentError('ProductQuantizer: only nbits=8 supported');
    }
    dsub = d ~/ m;
    _codebooks = List<List<Float32List>>.generate(
      m,
      (_) => List<Float32List>.generate(ksub, (_) => Float32List(dsub)),
    );
  }

  final int d;
  final int m;
  final int nbits;
  final int ksub; // 2^nbits centroids per sub-quantizer.
  final int kmeansIters;
  final int seed;

  late final int dsub;
  bool isTrained = false;

  /// `_codebooks[sub][k][j]` = j'th component of k'th centroid of
  /// subquantizer `sub`.
  late List<List<Float32List>> _codebooks;

  List<List<Float32List>> get codebooks => _codebooks;

  void train(List<Float32List> xs) {
    if (xs.length < ksub) {
      throw ArgumentError(
        'ProductQuantizer.train: need >= ksub=$ksub vectors, got ${xs.length}',
      );
    }
    for (var sub = 0; sub < m; sub++) {
      // Slice subvectors.
      final sliced = List<Float32List>.generate(
        xs.length,
        (_) => Float32List(dsub),
      );
      final off = sub * dsub;
      for (var i = 0; i < xs.length; i++) {
        final row = xs[i];
        final s = sliced[i];
        for (var j = 0; j < dsub; j++) {
          s[j] = row[off + j];
        }
      }
      final km = Kmeans(
        d: dsub,
        k: ksub,
        niter: kmeansIters,
        seed: seed + sub, // different seed per subquantizer
      );
      final res = km.train(sliced);
      for (var c = 0; c < ksub; c++) {
        final src = res.centroids[c];
        final dst = _codebooks[sub][c];
        for (var j = 0; j < dsub; j++) {
          dst[j] = src[j];
        }
      }
    }
    isTrained = true;
  }

  /// Encode a batch of vectors to their `m`-byte PQ codes.
  Uint8List encode(List<Float32List> xs) {
    if (!isTrained) throw StateError('ProductQuantizer.encode before train()');
    final n = xs.length;
    final codes = Uint8List(n * m);
    for (var i = 0; i < n; i++) {
      final row = xs[i];
      for (var sub = 0; sub < m; sub++) {
        final off = sub * dsub;
        // Closest centroid in this subspace.
        var bestC = 0;
        var bestD = double.infinity;
        final book = _codebooks[sub];
        for (var c = 0; c < ksub; c++) {
          final cen = book[c];
          var s = 0.0;
          for (var j = 0; j < dsub; j++) {
            final diff = row[off + j] - cen[j];
            s += diff * diff;
          }
          if (s < bestD) {
            bestD = s;
            bestC = c;
          }
        }
        codes[i * m + sub] = bestC;
      }
    }
    return codes;
  }

  /// Reconstruct a single code back to a `d`-length vector.
  Float32List decodeOne(Uint8List codes, int i) {
    final out = Float32List(d);
    for (var sub = 0; sub < m; sub++) {
      final off = sub * dsub;
      final cen = _codebooks[sub][codes[i * m + sub]];
      for (var j = 0; j < dsub; j++) {
        out[off + j] = cen[j];
      }
    }
    return out;
  }

  /// Build the `m × ksub` **squared-L2** distance lookup table for one
  /// query vector. `lut[sub * ksub + c] = ||q_sub - codebook[sub][c]||²`.
  Float32List buildL2LUT(Float32List q) {
    final lut = Float32List(m * ksub);
    for (var sub = 0; sub < m; sub++) {
      final off = sub * dsub;
      final book = _codebooks[sub];
      final base = sub * ksub;
      for (var c = 0; c < ksub; c++) {
        final cen = book[c];
        var s = 0.0;
        for (var j = 0; j < dsub; j++) {
          final diff = q[off + j] - cen[j];
          s += diff * diff;
        }
        lut[base + c] = s;
      }
    }
    return lut;
  }

  /// Build the `m × ksub` **inner-product** lookup table.
  Float32List buildIPLUT(Float32List q) {
    final lut = Float32List(m * ksub);
    for (var sub = 0; sub < m; sub++) {
      final off = sub * dsub;
      final book = _codebooks[sub];
      final base = sub * ksub;
      for (var c = 0; c < ksub; c++) {
        final cen = book[c];
        var s = 0.0;
        for (var j = 0; j < dsub; j++) {
          s += q[off + j] * cen[j];
        }
        lut[base + c] = s;
      }
    }
    return lut;
  }

  // --- persistence --------------------------------------------------------

  /// Write the PQ's isTrained flag + codebooks. Hyperparameters
  /// (`d`, `m`, `nbits`, seed, iters) are the caller's responsibility.
  void writeTo(IoWriter w) {
    w.writeU8(isTrained ? 1 : 0);
    if (isTrained) {
      for (var sub = 0; sub < m; sub++) {
        for (var c = 0; c < ksub; c++) {
          w.writeF32List(_codebooks[sub][c]);
        }
      }
    }
  }

  void readFrom(IoReader r) {
    isTrained = r.readU8() != 0;
    if (isTrained) {
      for (var sub = 0; sub < m; sub++) {
        for (var c = 0; c < ksub; c++) {
          _codebooks[sub][c] = r.readF32List(dsub);
        }
      }
    }
  }
}

/// Flat PQ index — every vector stored as an `m`-byte code, searched
/// via Asymmetric Distance Computation.
class IndexPQ extends Index {
  IndexPQ({
    required int d,
    required this.m,
    int nbits = 8,
    Metric metric = Metric.l2,
    int kmeansIters = 25,
    int seed = 1234,
  }) : pq = ProductQuantizer(
         d: d,
         m: m,
         nbits: nbits,
         kmeansIters: kmeansIters,
         seed: seed,
       ),
       super(d, metric) {
    isTrained = false;
  }

  final ProductQuantizer pq;
  final int m;

  Uint8List _codes = Uint8List(0);
  int _capacityCodes = 0;

  /// Read-only view of the currently used byte codes buffer (length
  /// `ntotal * m`). Zero-copy; do not mutate.
  Uint8List get codes => Uint8List.sublistView(_codes, 0, ntotal * m);

  /// I/O hook: replace the codes buffer wholesale and set `ntotal`.
  /// Used by `faiss_io.dart` when reading a persisted index.
  void ioSetCodes(Uint8List codes, int newNtotal) {
    if (codes.length != newNtotal * m) {
      throw ArgumentError(
        'IndexPQ.ioSetCodes: got ${codes.length} bytes, expected '
        '${newNtotal * m} for ntotal=$newNtotal, m=$m',
      );
    }
    _codes = Uint8List.fromList(codes);
    _capacityCodes = _codes.length;
    ntotal = newNtotal;
  }

  @override
  void train(List<Float32List> xs) {
    pq.train(xs);
    isTrained = true;
  }

  @override
  void add(List<Float32List> xs) {
    if (!isTrained) throw StateError('IndexPQ.add before train()');
    if (xs.isEmpty) return;
    final newCodes = pq.encode(xs);
    // Grow storage.
    final need = (ntotal + xs.length) * m;
    if (need > _capacityCodes) {
      var newCap = _capacityCodes == 0 ? 1024 * m : _capacityCodes;
      while (newCap < need) {
        newCap *= 2;
      }
      final grown = Uint8List(newCap);
      for (var i = 0; i < ntotal * m; i++) {
        grown[i] = _codes[i];
      }
      _codes = grown;
      _capacityCodes = newCap;
    }
    final off = ntotal * m;
    for (var i = 0; i < newCodes.length; i++) {
      _codes[off + i] = newCodes[i];
    }
    ntotal += xs.length;
  }

  @override
  SearchResult search(List<Float32List> queries, int k) {
    if (!isTrained) throw StateError('IndexPQ.search before train()');
    final nq = queries.length;
    final distances = List<Float32List>.generate(nq, (_) => Float32List(k));
    final ids = List<Int32List>.generate(nq, (_) => Int32List(k));

    for (var qi = 0; qi < nq; qi++) {
      final q = queries[qi];
      final lut = metric == Metric.l2 ? pq.buildL2LUT(q) : pq.buildIPLUT(q);
      final heap = TopK(k, maxIsWorst: metric == Metric.l2);
      final ksub = pq.ksub;
      for (var vi = 0; vi < ntotal; vi++) {
        final base = vi * m;
        var s = 0.0;
        for (var sub = 0; sub < m; sub++) {
          s += lut[sub * ksub + _codes[base + sub]];
        }
        heap.push(s, vi);
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
    w.writeU32(m);
    w.writeU32(pq.nbits);
    w.writeU32(pq.kmeansIters);
    w.writeU32(pq.seed);
    pq.writeTo(w);
    if (ntotal > 0) {
      w.writeU8List(Uint8List.sublistView(_codes, 0, ntotal * m));
    }
  }

  static IndexPQ readFrom(IoReader r) {
    final d = r.readU32();
    final metric = metricFromU32(r.readU32());
    final ntotal = r.readU32();
    final isTrained = r.readU8() != 0;
    final m = r.readU32();
    final nbits = r.readU32();
    final kmeansIters = r.readU32();
    final seed = r.readU32();
    final idx = IndexPQ(
      d: d,
      m: m,
      nbits: nbits,
      metric: metric,
      kmeansIters: kmeansIters,
      seed: seed,
    );
    idx.pq.readFrom(r);
    if (ntotal > 0) {
      idx._codes = r.readU8List(ntotal * m);
      idx._capacityCodes = ntotal * m;
    }
    idx.ntotal = ntotal;
    idx.isTrained = isTrained;
    return idx;
  }
}
