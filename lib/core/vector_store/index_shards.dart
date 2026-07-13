/// `IndexShards` — partition the corpus across a fixed set of inner
/// indexes and merge search results at query time.
///
/// Mirrors FAISS' `IndexShards`. Every child shard sees only a slice
/// of the added vectors; searches are dispatched to every shard in
/// sequence and their top-`k` results are folded into a single global
/// heap. This is the primary composition primitive for horizontal
/// scaling — in a multi-threaded runtime the per-shard searches run
/// concurrently, though this Dart port dispatches sequentially.
///
/// ID scheme (round-robin, mirrors FAISS' `successive_ids=false` mode
/// with a deterministic mapping):
///
/// ```
/// externalId(id) → shard = id % nshards, local = id ~/ nshards
/// externalId(local, shard) = local * nshards + shard
/// ```
///
/// This lets each shard use its own contiguous `0..listSize` id space
/// while the wrapper exposes a single monotonically-growing external
/// id space to callers.
library;

import 'dart:typed_data';

import 'index.dart';
import 'index_io.dart';

class IndexShards extends Index {
  IndexShards({required List<Index> shards})
    : shards = List.unmodifiable(shards),
      super(shards.first.d, shards.first.metric) {
    if (shards.isEmpty) {
      throw ArgumentError('IndexShards: need at least one shard');
    }
    for (var i = 1; i < shards.length; i++) {
      if (shards[i].d != d) {
        throw ArgumentError(
          'IndexShards: shard $i has d=${shards[i].d}, expected $d',
        );
      }
      if (shards[i].metric != metric) {
        throw ArgumentError(
          'IndexShards: shard $i has metric ${shards[i].metric}, expected $metric',
        );
      }
    }
    _syncTrainedFlag();
    ntotal = shards.fold<int>(0, (a, s) => a + s.ntotal);
  }

  final List<Index> shards;

  int get nshards => shards.length;

  void _syncTrainedFlag() {
    var t = true;
    for (final s in shards) {
      if (!s.isTrained) {
        t = false;
        break;
      }
    }
    isTrained = t;
  }

  @override
  void train(List<Float32List> xs) {
    for (final s in shards) {
      if (!s.isTrained) s.train(xs);
    }
    _syncTrainedFlag();
  }

  @override
  void add(List<Float32List> xs) {
    if (xs.isEmpty) return;
    // Partition xs across shards round-robin by external id.
    final buckets = List<List<Float32List>>.generate(nshards, (_) => []);
    for (var i = 0; i < xs.length; i++) {
      final extId = ntotal + i;
      buckets[extId % nshards].add(xs[i]);
    }
    for (var s = 0; s < nshards; s++) {
      if (buckets[s].isNotEmpty) shards[s].add(buckets[s]);
    }
    ntotal += xs.length;
  }

  @override
  SearchResult search(List<Float32List> queries, int k) {
    final nq = queries.length;
    final distances = List<Float32List>.generate(nq, (_) => Float32List(k));
    final ids = List<Int32List>.generate(nq, (_) => Int32List(k));
    // Metric direction: L2 → smaller is closer; IP → larger is closer.
    final maxIsWorst = metric == Metric.l2;
    for (var qi = 0; qi < nq; qi++) {
      final heap = TopK(k, maxIsWorst: maxIsWorst);
      final q = <Float32List>[queries[qi]];
      for (var s = 0; s < nshards; s++) {
        final r = shards[s].search(q, k);
        for (var j = 0; j < k; j++) {
          final localId = r.ids[0][j];
          if (localId < 0) continue; // padding
          final extId = localId * nshards + s;
          heap.push(r.distances[0][j].toDouble(), extId);
        }
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
    w.writeU32(nshards);
    for (final s in shards) {
      writeChild(w, s);
    }
  }

  static IndexShards readFrom(IoReader r) {
    r.readU32(); // d — recomputed
    r.readU32(); // metric — recomputed
    final ntotal = r.readU32();
    final isTrained = r.readU8() != 0;
    final n = r.readU32();
    final shards = <Index>[];
    for (var i = 0; i < n; i++) {
      shards.add(readChild(r));
    }
    final idx = IndexShards(shards: shards);
    idx.ntotal = ntotal;
    idx.isTrained = isTrained;
    return idx;
  }
}
