/// `IndexPreTransform` — wraps an inner [Index] with an ordered chain
/// of [VectorTransform]s. All inputs to train / add / search /
/// rangeSearch flow through the chain first; the inner index sees
/// only transformed vectors.
///
/// Mirrors FAISS' `IndexPreTransform`. Composition rules:
///  * `d` (this wrapper's input dimension) == `chain.first.dIn`
///    (or `inner.d` if the chain is empty).
///  * `chain[i].dOut == chain[i+1].dIn`.
///  * `chain.last.dOut == inner.d`.
library;

import 'dart:typed_data';

import 'index.dart';
import 'index_io.dart';
import 'vector_transform.dart';
import 'l2_norm_transform.dart';
import 'random_rotation_transform.dart';
import 'pca_transform.dart';

class IndexPreTransform extends Index {
  IndexPreTransform({required List<VectorTransform> chain, required this.inner})
    : chain = List.unmodifiable(chain),
      super(chain.isEmpty ? inner.d : chain.first.dIn, inner.metric) {
    _validateChain();
    // Aggregate isTrained: wrapper is trained iff every transform
    // AND the inner index is trained.
    _syncTrainedFlag();
  }

  final List<VectorTransform> chain;
  final Index inner;

  void _validateChain() {
    var expected = d;
    for (var i = 0; i < chain.length; i++) {
      if (chain[i].dIn != expected) {
        throw ArgumentError(
          'IndexPreTransform: chain[$i].dIn=${chain[i].dIn} '
          'expected $expected',
        );
      }
      expected = chain[i].dOut;
    }
    if (expected != inner.d) {
      throw ArgumentError(
        'IndexPreTransform: chain outputs dim $expected but inner.d=${inner.d}',
      );
    }
  }

  void _syncTrainedFlag() {
    var t = inner.isTrained;
    for (final tr in chain) {
      if (!tr.isTrained) {
        t = false;
        break;
      }
    }
    isTrained = t;
  }

  List<Float32List> _forward(List<Float32List> xs) {
    var cur = xs;
    for (final tr in chain) {
      cur = tr.apply(cur);
    }
    return cur;
  }

  @override
  void train(List<Float32List> xs) {
    // Train each transform on the (progressively transformed) data,
    // then train the inner index on the fully-transformed data.
    var cur = xs;
    for (final tr in chain) {
      if (!tr.isTrained) tr.train(cur);
      cur = tr.apply(cur);
    }
    inner.train(cur);
    _syncTrainedFlag();
  }

  @override
  void add(List<Float32List> xs) {
    if (!isTrained) {
      // Auto-train on first add, matching FAISS behaviour for
      // transforms that need data (currently none of ours, but the
      // hook is here for future PCA/OPQ).
      train(xs);
    }
    inner.add(_forward(xs));
    ntotal = inner.ntotal;
  }

  @override
  SearchResult search(List<Float32List> queries, int k) =>
      inner.search(_forward(queries), k);

  @override
  RangeSearchResult rangeSearch(List<Float32List> queries, double radius) =>
      inner.rangeSearch(_forward(queries), radius);

  @override
  int removeIds(Set<int> ids) {
    final removed = inner.removeIds(ids);
    ntotal = inner.ntotal;
    return removed;
  }

  // --- persistence --------------------------------------------------------

  void writeTo(IoWriter w) {
    w.writeU32(d);
    w.writeU32(metricToU32(metric));
    w.writeU32(inner.ntotal);
    w.writeU8(isTrained ? 1 : 0);
    // Chain: u32 count, then each transform (subkind + payload).
    w.writeU32(chain.length);
    for (final tr in chain) {
      tr.writeTo(w);
    }
    // Inner index: standalone blob via writeChild (writes its own
    // magic + version + kind + payload).
    writeChild(w, inner);
  }

  static IndexPreTransform readFrom(IoReader r) {
    // Order below MUST mirror writeTo:
    //   d, metric, inner_ntotal, isTrained, chainCount, chain..., inner_blob
    r.readU32(); // d — recomputed from chain/inner, discarded
    r.readU32(); // metric — recomputed from inner
    r.readU32(); // inner ntotal — will match after inner is loaded
    r.readU8(); // isTrained — recomputed
    final n = r.readU32();
    final chain = <VectorTransform>[];
    for (var i = 0; i < n; i++) {
      final subkind = r.readU32();
      switch (subkind) {
        case TransformKind.l2Norm:
          chain.add(L2NormTransform.readFrom(r));
          break;
        case TransformKind.randomRotation:
          chain.add(RandomRotationTransform.readFrom(r));
          break;
        case TransformKind.pca:
          chain.add(PCATransform.readFrom(r));
          break;
        default:
          throw FormatException(
            'IndexPreTransform: unknown transform subkind '
            '0x${subkind.toRadixString(16)}',
          );
      }
    }
    final inner = readChild(r);
    final idx = IndexPreTransform(chain: chain, inner: inner);
    idx.ntotal = inner.ntotal;
    return idx;
  }
}
