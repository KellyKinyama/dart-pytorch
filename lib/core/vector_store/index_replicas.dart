/// `IndexReplicas` — hold N functionally-identical copies of an index.
///
/// Mirrors FAISS' `IndexReplicas`. Every write (train / add) is
/// broadcast to all replicas; every read (search) is answered by a
/// single replica chosen round-robin. In a multi-threaded runtime this
/// permits N queries to run concurrently on N cores against N private
/// copies of the corpus. This Dart port dispatches sequentially, so
/// replicas here are primarily a semantic wrapper preserving that
/// contract for future concurrency work and for byte-compatible
/// persistence.
library;

import 'dart:typed_data';

import 'index.dart';
import 'index_io.dart';

class IndexReplicas extends Index {
  IndexReplicas({required List<Index> replicas})
    : replicas = List.unmodifiable(replicas),
      super(replicas.first.d, replicas.first.metric) {
    if (replicas.isEmpty) {
      throw ArgumentError('IndexReplicas: need at least one replica');
    }
    for (var i = 1; i < replicas.length; i++) {
      if (replicas[i].d != d) {
        throw ArgumentError(
          'IndexReplicas: replica $i has d=${replicas[i].d}, expected $d',
        );
      }
      if (replicas[i].metric != metric) {
        throw ArgumentError(
          'IndexReplicas: replica $i has metric ${replicas[i].metric}, '
          'expected $metric',
        );
      }
    }
    _sync();
  }

  final List<Index> replicas;
  int _nextReplica = 0;

  int get nreplicas => replicas.length;

  void _sync() {
    ntotal = replicas.first.ntotal;
    var trained = true;
    for (final r in replicas) {
      if (!r.isTrained) trained = false;
    }
    isTrained = trained;
  }

  @override
  void train(List<Float32List> xs) {
    for (final r in replicas) {
      if (!r.isTrained) r.train(xs);
    }
    _sync();
  }

  @override
  void add(List<Float32List> xs) {
    for (final r in replicas) {
      r.add(xs);
    }
    ntotal = replicas.first.ntotal;
  }

  @override
  SearchResult search(List<Float32List> queries, int k) {
    final chosen = replicas[_nextReplica];
    _nextReplica = (_nextReplica + 1) % replicas.length;
    return chosen.search(queries, k);
  }

  // --- persistence --------------------------------------------------------

  void writeTo(IoWriter w) {
    w.writeU32(d);
    w.writeU32(metricToU32(metric));
    w.writeU32(ntotal);
    w.writeU8(isTrained ? 1 : 0);
    w.writeU32(nreplicas);
    for (final r in replicas) {
      writeChild(w, r);
    }
  }

  static IndexReplicas readFrom(IoReader r) {
    r.readU32(); // d — recomputed
    r.readU32(); // metric — recomputed
    final ntotal = r.readU32();
    final isTrained = r.readU8() != 0;
    final n = r.readU32();
    final reps = <Index>[];
    for (var i = 0; i < n; i++) {
      reps.add(readChild(r));
    }
    final idx = IndexReplicas(replicas: reps);
    idx.ntotal = ntotal;
    idx.isTrained = isTrained;
    return idx;
  }
}
