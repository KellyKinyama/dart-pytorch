/// `IndexScalarQuantizer` (SQ8) — per-dimension 8-bit scalar
/// quantization.
///
/// For each dimension `j`, learns a min/max range from the training
/// data and linearly quantizes to `[0, 255]`:
///
/// ```
///   scale[j]  = (vmax[j] - vmin[j]) / 255
///   code      = round((x[j] - vmin[j]) / scale[j])
///   decoded   = vmin[j] + code * scale[j]
/// ```
///
/// Storage cost drops from `4*d` bytes/vector (fp32) to `d` bytes/vector
/// — 4× compression with typically <5 % recall loss on well-distributed
/// data. This is FAISS' `IndexScalarQuantizer` with `qtype = QT_8bit`.
///
/// Search decodes each vector on the fly during the distance loop.
/// This is `SDC`-style (Symmetric Distance Computation): the query is
/// itself never quantized; distances are computed against the
/// reconstructed database vector.
library;

import 'dart:typed_data';

import 'index.dart';
import 'index_io.dart';

class IndexScalarQuantizer extends Index {
  IndexScalarQuantizer(int d, {Metric metric = Metric.l2})
    : _vmin = Float32List(d),
      _scale = Float32List(d),
      super(d, metric) {
    isTrained = false;
  }

  /// Per-dimension range minima (`d` floats).
  final Float32List _vmin;

  /// Per-dimension scale = `(vmax - vmin) / 255` (`d` floats).
  final Float32List _scale;

  Uint8List _codes = Uint8List(0);
  int _capacityCodes = 0;

  Float32List get vmin => _vmin;
  Float32List get scale => _scale;

  @override
  void train(List<Float32List> xs) {
    if (xs.isEmpty) {
      throw ArgumentError('IndexScalarQuantizer.train: empty training set');
    }
    final vmax = Float32List(d);
    for (var j = 0; j < d; j++) {
      _vmin[j] = double.infinity;
      vmax[j] = double.negativeInfinity;
    }
    for (var i = 0; i < xs.length; i++) {
      final row = xs[i];
      for (var j = 0; j < d; j++) {
        final v = row[j];
        if (v < _vmin[j]) _vmin[j] = v;
        if (v > vmax[j]) vmax[j] = v;
      }
    }
    for (var j = 0; j < d; j++) {
      final range = vmax[j] - _vmin[j];
      _scale[j] = range == 0 ? 1.0 : range / 255.0;
    }
    isTrained = true;
  }

  Uint8List _encode(List<Float32List> xs) {
    final n = xs.length;
    final codes = Uint8List(n * d);
    for (var i = 0; i < n; i++) {
      final row = xs[i];
      final base = i * d;
      for (var j = 0; j < d; j++) {
        final q = ((row[j] - _vmin[j]) / _scale[j]).round();
        codes[base + j] = q < 0
            ? 0
            : q > 255
            ? 255
            : q;
      }
    }
    return codes;
  }

  /// Reconstruct vector at position `i` from its 8-bit codes.
  Float32List reconstruct(int i) {
    if (i < 0 || i >= ntotal) {
      throw RangeError('id $i out of range [0, $ntotal)');
    }
    final out = Float32List(d);
    final base = i * d;
    for (var j = 0; j < d; j++) {
      out[j] = _vmin[j] + _codes[base + j] * _scale[j];
    }
    return out;
  }

  @override
  void add(List<Float32List> xs) {
    if (!isTrained) throw StateError('IndexScalarQuantizer.add before train()');
    if (xs.isEmpty) return;
    final newCodes = _encode(xs);
    final need = (ntotal + xs.length) * d;
    if (need > _capacityCodes) {
      var newCap = _capacityCodes == 0 ? 1024 * d : _capacityCodes;
      while (newCap < need) {
        newCap *= 2;
      }
      final grown = Uint8List(newCap);
      for (var i = 0; i < ntotal * d; i++) {
        grown[i] = _codes[i];
      }
      _codes = grown;
      _capacityCodes = newCap;
    }
    final off = ntotal * d;
    for (var i = 0; i < newCodes.length; i++) {
      _codes[off + i] = newCodes[i];
    }
    ntotal += xs.length;
  }

  @override
  SearchResult search(List<Float32List> queries, int k) {
    if (!isTrained)
      throw StateError('IndexScalarQuantizer.search before train()');
    final nq = queries.length;
    final distances = List<Float32List>.generate(nq, (_) => Float32List(k));
    final ids = List<Int32List>.generate(nq, (_) => Int32List(k));

    for (var qi = 0; qi < nq; qi++) {
      final q = queries[qi];
      if (q.length != d) {
        throw ArgumentError('query $qi length ${q.length} != $d');
      }
      final heap = TopK(k, maxIsWorst: metric == Metric.l2);
      for (var vi = 0; vi < ntotal; vi++) {
        final base = vi * d;
        var s = 0.0;
        if (metric == Metric.l2) {
          for (var j = 0; j < d; j++) {
            final decoded = _vmin[j] + _codes[base + j] * _scale[j];
            final diff = q[j] - decoded;
            s += diff * diff;
          }
        } else {
          for (var j = 0; j < d; j++) {
            final decoded = _vmin[j] + _codes[base + j] * _scale[j];
            s += q[j] * decoded;
          }
        }
        heap.push(s, vi);
      }
      final sorted = heap.sorted();
      distances[qi] = sorted.scores;
      ids[qi] = sorted.ids;
    }
    return SearchResult(distances, ids);
  }

  @override
  RangeSearchResult rangeSearch(List<Float32List> queries, double radius) {
    if (!isTrained) {
      throw StateError('IndexScalarQuantizer.rangeSearch before train()');
    }
    final nq = queries.length;
    final perQueryDist = List<List<double>>.generate(nq, (_) => <double>[]);
    final perQueryIds = List<List<int>>.generate(nq, (_) => <int>[]);

    for (var qi = 0; qi < nq; qi++) {
      final q = queries[qi];
      if (q.length != d) {
        throw ArgumentError('query $qi length ${q.length} != $d');
      }
      final dists = perQueryDist[qi];
      final idsRow = perQueryIds[qi];
      for (var vi = 0; vi < ntotal; vi++) {
        final base = vi * d;
        var s = 0.0;
        if (metric == Metric.l2) {
          for (var j = 0; j < d; j++) {
            final decoded = _vmin[j] + _codes[base + j] * _scale[j];
            final diff = q[j] - decoded;
            s += diff * diff;
          }
          if (s <= radius) {
            dists.add(s);
            idsRow.add(vi);
          }
        } else {
          for (var j = 0; j < d; j++) {
            final decoded = _vmin[j] + _codes[base + j] * _scale[j];
            s += q[j] * decoded;
          }
          if (s >= radius) {
            dists.add(s);
            idsRow.add(vi);
          }
        }
      }
    }
    return RangeSearchResult.fromPerQuery(perQueryDist, perQueryIds);
  }

  // --- persistence --------------------------------------------------------

  void writeTo(IoWriter w) {
    w.writeU32(d);
    w.writeU32(metricToU32(metric));
    w.writeU32(ntotal);
    w.writeU8(isTrained ? 1 : 0);
    w.writeF32List(_vmin);
    w.writeF32List(_scale);
    if (ntotal > 0) {
      w.writeU8List(Uint8List.sublistView(_codes, 0, ntotal * d));
    }
  }

  static IndexScalarQuantizer readFrom(IoReader r) {
    final d = r.readU32();
    final metric = metricFromU32(r.readU32());
    final ntotal = r.readU32();
    final isTrained = r.readU8() != 0;
    final idx = IndexScalarQuantizer(d, metric: metric);
    final vmin = r.readF32List(d);
    final scale = r.readF32List(d);
    for (var j = 0; j < d; j++) {
      idx._vmin[j] = vmin[j];
      idx._scale[j] = scale[j];
    }
    if (ntotal > 0) {
      idx._codes = r.readU8List(ntotal * d);
      idx._capacityCodes = ntotal * d;
    }
    idx.ntotal = ntotal;
    idx.isTrained = isTrained;
    return idx;
  }
}
