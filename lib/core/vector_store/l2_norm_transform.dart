/// `L2NormTransform` — divides each vector by its L2 norm so that all
/// outputs lie on the unit hypersphere. Data-independent; no training.
///
/// This transform is a cheap and effective preprocessor for indexes
/// whose relevance signal is directional (cosine similarity, LSH,
/// inner-product search on unit vectors). On unit vectors,
/// `‖x − y‖² = 2 − 2·<x, y>`, so L2 nearest neighbour == largest
/// inner product == largest cosine similarity.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'index_io.dart';
import 'vector_transform.dart';

class L2NormTransform extends VectorTransform {
  L2NormTransform(int d) : super(d, d) {
    isTrained = true;
  }

  @override
  List<Float32List> apply(List<Float32List> xs) {
    final out = List<Float32List>.generate(xs.length, (i) {
      final row = xs[i];
      if (row.length != dIn) {
        throw ArgumentError('vector $i length ${row.length} != $dIn');
      }
      var s = 0.0;
      for (var j = 0; j < dIn; j++) {
        final v = row[j];
        s += v * v;
      }
      final norm = math.sqrt(s);
      final o = Float32List(dIn);
      if (norm > 0) {
        final inv = 1.0 / norm;
        for (var j = 0; j < dIn; j++) {
          o[j] = row[j] * inv;
        }
      }
      // if norm == 0 leave o as zeros
      return o;
    });
    return out;
  }

  @override
  void writeTo(IoWriter w) {
    w.writeU32(TransformKind.l2Norm);
    w.writeU32(dIn);
  }

  static L2NormTransform readFrom(IoReader r) {
    // Subkind byte already consumed by IndexPreTransform.readFrom.
    final d = r.readU32();
    return L2NormTransform(d);
  }
}
