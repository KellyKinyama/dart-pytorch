/// `IndexHNSW` — Hierarchical Navigable Small World graph index.
///
/// Direct port of the Malkov-Yashunin algorithm (arXiv:1603.09320) at
/// the level of detail used by `hnswlib` and FAISS' `IndexHNSWFlat`:
///
///   • Multi-layer graph, geometrically decreasing layer probability.
///   • Greedy descent through upper layers to find an entry point.
///   • Beam search of size `ef` at each layer for k-NN.
///   • Bidirectional edges added at insertion, with pruning by the
///     "select-M-heuristic" (Malkov §4.3) when a neighbour exceeds its
///     out-degree budget.
///
/// Vectors are stored uncompressed (flat float32). This is the
/// canonical `IndexHNSWFlat`. For cosine similarity, L2-normalise
/// before adding and use [Metric.innerProduct].
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'index.dart';
import 'index_io.dart';

/// Node metadata: level + per-layer adjacency lists.
class _HnswNode {
  _HnswNode(this.level)
    : neighbors = List<List<int>>.generate(level + 1, (_) => <int>[]);
  final int level;
  final List<List<int>> neighbors;
}

class IndexHNSW extends Index {
  IndexHNSW({
    required int d,
    Metric metric = Metric.l2,
    this.M = 16,
    this.efConstruction = 100,
    this.efSearch = 32,
    this.seed = 1234,
  }) : Mmax = M,
       Mmax0 = 2 * M,
       _mL = 1.0 / math.log(M.toDouble()),
       super(d, metric);

  /// Max connections per node in layers > 0.
  final int M;

  /// Same as `M` (Malkov's default).
  final int Mmax;

  /// Larger budget for layer 0 (Malkov's default: `2 * M`).
  final int Mmax0;

  /// Beam size at insertion time (higher = better graph, slower build).
  final int efConstruction;

  /// Beam size at search time (higher = better recall, slower search).
  int efSearch;

  /// Level-sampling constant `mL = 1/ln(M)` (Malkov §4.1).
  final double _mL;

  final int seed;
  late final math.Random _rng = math.Random(seed);

  Float32List _storage = Float32List(0);
  int _capacity = 0;
  final List<_HnswNode> _nodes = [];
  int _entryPoint = -1;
  int _topLevel = -1;

  // --- vector access ------------------------------------------------------

  void _grow(int upto) {
    if (upto <= _capacity) return;
    var newCap = _capacity == 0 ? 1024 : _capacity;
    while (newCap < upto) {
      newCap *= 2;
    }
    final grown = Float32List(newCap * d);
    for (var i = 0; i < ntotal * d; i++) {
      grown[i] = _storage[i];
    }
    _storage = grown;
    _capacity = newCap;
  }

  double _dist(int a, Float32List q) {
    final base = a * d;
    if (metric == Metric.l2) {
      var s = 0.0;
      for (var j = 0; j < d; j++) {
        final diff = q[j] - _storage[base + j];
        s += diff * diff;
      }
      return s;
    }
    // Inner-product mode: we treat *smaller is worse* by returning -dot,
    // so the same graph algorithms (which think "smaller is closer")
    // work unchanged.
    var s = 0.0;
    for (var j = 0; j < d; j++) {
      s += q[j] * _storage[base + j];
    }
    return -s;
  }

  double _distIds(int a, int b) {
    final base = b * d;
    final q = Float32List.sublistView(_storage, base, base + d);
    return _dist(a, q);
  }

  int _sampleLevel() {
    // Malkov §4.1: level = floor(-ln(U(0,1)) * mL)
    var u = _rng.nextDouble();
    if (u <= 0) u = 1e-12;
    return (-math.log(u) * _mL).floor();
  }

  // --- core routines ------------------------------------------------------

  /// Greedy descent through a single layer starting at [entry], returning
  /// the closest node found (i.e. the local minimum). Used for the
  /// zoom-in phase from `topLevel` down to `targetLevel + 1`.
  int _greedyDescent(Float32List q, int entry, int layer) {
    var curr = entry;
    var currDist = _dist(curr, q);
    var improved = true;
    while (improved) {
      improved = false;
      final nbrs = _nodes[curr].neighbors[layer];
      for (var i = 0; i < nbrs.length; i++) {
        final nb = nbrs[i];
        final nd = _dist(nb, q);
        if (nd < currDist) {
          currDist = nd;
          curr = nb;
          improved = true;
        }
      }
    }
    return curr;
  }

  /// Beam search at a given [layer]. Returns up to [ef] candidates
  /// (both ids and distances) sorted best-first.
  ///
  /// Standard Malkov: maintain a min-heap of unexplored candidates and
  /// a bounded max-heap of accepted results. Prune whenever the min of
  /// candidates exceeds the worst of results.
  _BeamResult _searchLayer(
    Float32List q,
    List<int> entries,
    int ef,
    int layer,
  ) {
    final visited = <int>{};
    // Min-heap of candidates by distance (nearest first).
    final cand = _MinHeap();
    // Max-heap of results by distance (farthest at root).
    final res = _MaxHeap();
    for (final e in entries) {
      if (visited.contains(e)) continue;
      visited.add(e);
      final de = _dist(e, q);
      cand.push(de, e);
      res.push(de, e);
    }

    while (cand.isNotEmpty) {
      final (dc, c) = cand.peek();
      if (res.size >= ef && dc > res.worst) break;
      cand.pop();
      final nbrs = _nodes[c].neighbors[layer];
      for (var i = 0; i < nbrs.length; i++) {
        final nb = nbrs[i];
        if (visited.contains(nb)) continue;
        visited.add(nb);
        final dnb = _dist(nb, q);
        if (res.size < ef || dnb < res.worst) {
          cand.push(dnb, nb);
          res.push(dnb, nb);
          if (res.size > ef) res.popWorst();
        }
      }
    }
    return res.sortedBestFirst();
  }

  /// The "select M heuristic" (Malkov §4.3): keep a candidate `e` iff
  /// it is closer to the target `q` than to every already-selected
  /// neighbour. Improves graph diversity over naive top-M.
  List<int> _selectNeighbors(
    Float32List q,
    List<int> candIds,
    List<double> candDists,
    int m,
  ) {
    // Sort candidates by ascending distance (nearest first).
    final order = List<int>.generate(candIds.length, (i) => i)
      ..sort((a, b) => candDists[a].compareTo(candDists[b]));
    final selected = <int>[];
    for (final idx in order) {
      if (selected.length >= m) break;
      final e = candIds[idx];
      final eDist = candDists[idx];
      var good = true;
      for (final s in selected) {
        // If e is closer to some already-selected s than to q, reject.
        final esDist = _distIds(e, s);
        if (esDist < eDist) {
          good = false;
          break;
        }
      }
      if (good) selected.add(e);
    }
    return selected;
  }

  // --- add / search -------------------------------------------------------

  void _addOne(Float32List x) {
    final id = ntotal;
    _grow(id + 1);
    final base = id * d;
    for (var j = 0; j < d; j++) {
      _storage[base + j] = x[j];
    }
    final level = _sampleLevel();
    _nodes.add(_HnswNode(level));
    ntotal++;

    if (_entryPoint < 0) {
      _entryPoint = id;
      _topLevel = level;
      return;
    }

    // 1. Zoom in from topLevel down to level+1.
    var curr = _entryPoint;
    for (var l = _topLevel; l > level; l--) {
      curr = _greedyDescent(x, curr, l);
    }

    // 2. From min(topLevel, level) down to 0: insert with beam search.
    var entries = <int>[curr];
    for (var l = math.min(_topLevel, level); l >= 0; l--) {
      final beam = _searchLayer(x, entries, efConstruction, l);
      final mLayer = l == 0 ? Mmax0 : Mmax;
      final chosen = _selectNeighbors(x, beam.ids, beam.dists, mLayer);
      // Add bidirectional edges.
      _nodes[id].neighbors[l].addAll(chosen);
      for (final nb in chosen) {
        final nbList = _nodes[nb].neighbors[l];
        nbList.add(id);
        final budget = l == 0 ? Mmax0 : Mmax;
        if (nbList.length > budget) {
          // Prune with select-M heuristic centred on nb.
          final nbBase = nb * d;
          final nbVec = Float32List.sublistView(_storage, nbBase, nbBase + d);
          final dists = List<double>.generate(
            nbList.length,
            (i) => _dist(nbList[i], nbVec),
          );
          final kept = _selectNeighbors(
            nbVec,
            List<int>.of(nbList),
            dists,
            budget,
          );
          nbList
            ..clear()
            ..addAll(kept);
        }
      }
      // Next layer's entries are this layer's results.
      entries = beam.ids;
    }

    if (level > _topLevel) {
      _topLevel = level;
      _entryPoint = id;
    }
  }

  @override
  void add(List<Float32List> xs) {
    for (var i = 0; i < xs.length; i++) {
      if (xs[i].length != d) {
        throw ArgumentError('vector $i length ${xs[i].length} != $d');
      }
      _addOne(xs[i]);
    }
  }

  @override
  SearchResult search(List<Float32List> queries, int k) {
    final nq = queries.length;
    final distances = List<Float32List>.generate(nq, (_) => Float32List(k));
    final ids = List<Int32List>.generate(nq, (_) => Int32List(k));

    if (ntotal == 0 || _entryPoint < 0) {
      for (var qi = 0; qi < nq; qi++) {
        for (var j = 0; j < k; j++) {
          distances[qi][j] = metric == Metric.l2
              ? double.infinity
              : double.negativeInfinity;
          ids[qi][j] = -1;
        }
      }
      return SearchResult(distances, ids);
    }

    for (var qi = 0; qi < nq; qi++) {
      final q = queries[qi];
      // Zoom in from topLevel down to layer 1.
      var curr = _entryPoint;
      for (var l = _topLevel; l > 0; l--) {
        curr = _greedyDescent(q, curr, l);
      }
      // Beam search at layer 0.
      final ef = math.max(efSearch, k);
      final beam = _searchLayer(q, [curr], ef, 0);
      // Emit top-k.
      final n = math.min(k, beam.ids.length);
      for (var j = 0; j < n; j++) {
        // Convert back to metric-native distance for IP (we stored -dot).
        final dj = metric == Metric.l2 ? beam.dists[j] : -beam.dists[j];
        distances[qi][j] = dj;
        ids[qi][j] = beam.ids[j];
      }
      for (var j = n; j < k; j++) {
        distances[qi][j] = metric == Metric.l2
            ? double.infinity
            : double.negativeInfinity;
        ids[qi][j] = -1;
      }
    }
    return SearchResult(distances, ids);
  }

  // --- persistence --------------------------------------------------------

  void writeTo(IoWriter w) {
    w.writeU32(d);
    w.writeU32(metricToU32(metric));
    w.writeU32(ntotal);
    w.writeU8(isTrained ? 1 : 0);
    w.writeU32(M);
    w.writeU32(efConstruction);
    w.writeU32(efSearch);
    w.writeU32(seed);
    w.writeI32(_entryPoint);
    w.writeI32(_topLevel);
    if (ntotal > 0) {
      w.writeF32List(Float32List.sublistView(_storage, 0, ntotal * d));
    }
    for (var i = 0; i < ntotal; i++) {
      final node = _nodes[i];
      w.writeU32(node.level);
      for (var l = 0; l <= node.level; l++) {
        final layer = node.neighbors[l];
        w.writeU32(layer.length);
        for (final nb in layer) {
          w.writeI32(nb);
        }
      }
    }
  }

  static IndexHNSW readFrom(IoReader r) {
    final d = r.readU32();
    final metric = metricFromU32(r.readU32());
    final ntotal = r.readU32();
    final isTrained = r.readU8() != 0;
    final M = r.readU32();
    final efConstruction = r.readU32();
    final efSearch = r.readU32();
    final seed = r.readU32();
    final entryPoint = r.readI32();
    final topLevel = r.readI32();
    final idx = IndexHNSW(
      d: d,
      metric: metric,
      M: M,
      efConstruction: efConstruction,
      efSearch: efSearch,
      seed: seed,
    );
    if (ntotal > 0) {
      idx._storage = r.readF32List(ntotal * d);
      idx._capacity = ntotal;
    }
    for (var i = 0; i < ntotal; i++) {
      final level = r.readU32();
      final node = _HnswNode(level);
      for (var l = 0; l <= level; l++) {
        final len = r.readU32();
        final layer = node.neighbors[l];
        for (var j = 0; j < len; j++) {
          layer.add(r.readI32());
        }
      }
      idx._nodes.add(node);
    }
    idx.ntotal = ntotal;
    idx.isTrained = isTrained;
    idx._entryPoint = entryPoint;
    idx._topLevel = topLevel;
    return idx;
  }

  // --- FAISS interop hooks -----------------------------------------------
  //
  // These accessors expose the raw graph state so `faiss_io.dart` can
  // translate between the port's per-node adjacency lists and FAISS's
  // flat `offsets` + `neighbors` CSR layout.

  /// Read-only view into the packed float32 storage (`ntotal * d` values).
  Float32List get storage => Float32List.sublistView(_storage, 0, ntotal * d);

  /// Current graph entry point (storage id or `-1` when empty).
  int get entryPoint => _entryPoint;

  /// Top populated layer index (or `-1` when empty).
  int get topLevel => _topLevel;

  /// Top layer of node `i` (equal to FAISS's `levels[i] - 1`).
  int nodeLevel(int i) => _nodes[i].level;

  /// Neighbor list of node `i` at layer `layer` (compact, no `-1` padding).
  List<int> nodeNeighbors(int i, int layer) => _nodes[i].neighbors[layer];

  /// Bulk hydration hook used by `readFaissIndex`. Replaces the graph +
  /// storage in one shot after a fresh construction. Callers must pass:
  ///
  ///  * `newStorage`  — length `newNtotal * d`.
  ///  * `nodeLevels`  — per-node top-layer indices (length `newNtotal`).
  ///  * `perNodePerLayerNeighbors[i][l]` — compact neighbor id list for
  ///    node `i` at layer `l`, with `l` ranging over `0..nodeLevels[i]`.
  ///  * `newEntryPoint` / `newTopLevel` / `newNtotal` — top-level graph
  ///    stats matching FAISS's `entry_point` / `max_level` / `ntotal`.
  void ioSetGraph({
    required Float32List newStorage,
    required List<int> nodeLevels,
    required List<List<List<int>>> perNodePerLayerNeighbors,
    required int newEntryPoint,
    required int newTopLevel,
    required int newNtotal,
  }) {
    if (newStorage.length != newNtotal * d) {
      throw ArgumentError(
        'newStorage length ${newStorage.length} != newNtotal * d '
        '(${newNtotal * d})',
      );
    }
    if (nodeLevels.length != newNtotal) {
      throw ArgumentError(
        'nodeLevels length ${nodeLevels.length} != newNtotal $newNtotal',
      );
    }
    if (perNodePerLayerNeighbors.length != newNtotal) {
      throw ArgumentError(
        'perNodePerLayerNeighbors length ${perNodePerLayerNeighbors.length}'
        ' != newNtotal $newNtotal',
      );
    }
    final fresh = Float32List(newNtotal * d);
    for (var i = 0; i < newStorage.length; i++) {
      fresh[i] = newStorage[i];
    }
    _storage = fresh;
    _capacity = newNtotal;
    _nodes
      ..clear()
      ..addAll(
        List<_HnswNode>.generate(newNtotal, (i) {
          final level = nodeLevels[i];
          final layers = perNodePerLayerNeighbors[i];
          if (layers.length != level + 1) {
            throw ArgumentError(
              'node $i: expected ${level + 1} layers, got ${layers.length}',
            );
          }
          final node = _HnswNode(level);
          for (var l = 0; l <= level; l++) {
            node.neighbors[l].addAll(layers[l]);
          }
          return node;
        }),
      );
    ntotal = newNtotal;
    _entryPoint = newEntryPoint;
    _topLevel = newTopLevel;
  }
}

// -----------------------------------------------------------------------------
// Small binary heaps used only inside HNSW.
class _BeamResult {
  _BeamResult(this.ids, this.dists);
  final List<int> ids;
  final List<double> dists;
}

class _MinHeap {
  final _dists = <double>[];
  final _ids = <int>[];

  bool get isNotEmpty => _ids.isNotEmpty;
  int get size => _ids.length;

  void push(double d, int id) {
    _dists.add(d);
    _ids.add(id);
    var i = _ids.length - 1;
    while (i > 0) {
      final p = (i - 1) >> 1;
      if (_dists[i] < _dists[p]) {
        _swap(i, p);
        i = p;
      } else {
        break;
      }
    }
  }

  (double, int) peek() => (_dists[0], _ids[0]);

  (double, int) pop() {
    final r = (_dists[0], _ids[0]);
    final last = _ids.length - 1;
    _dists[0] = _dists[last];
    _ids[0] = _ids[last];
    _dists.removeLast();
    _ids.removeLast();
    if (_ids.isEmpty) return r;
    var i = 0;
    while (true) {
      final l = 2 * i + 1;
      final rc = 2 * i + 2;
      var best = i;
      if (l < _ids.length && _dists[l] < _dists[best]) best = l;
      if (rc < _ids.length && _dists[rc] < _dists[best]) best = rc;
      if (best == i) break;
      _swap(i, best);
      i = best;
    }
    return r;
  }

  void _swap(int a, int b) {
    final td = _dists[a];
    _dists[a] = _dists[b];
    _dists[b] = td;
    final ti = _ids[a];
    _ids[a] = _ids[b];
    _ids[b] = ti;
  }
}

class _MaxHeap {
  final _dists = <double>[];
  final _ids = <int>[];

  int get size => _ids.length;
  double get worst => _dists[0];

  void push(double d, int id) {
    _dists.add(d);
    _ids.add(id);
    var i = _ids.length - 1;
    while (i > 0) {
      final p = (i - 1) >> 1;
      if (_dists[i] > _dists[p]) {
        _swap(i, p);
        i = p;
      } else {
        break;
      }
    }
  }

  void popWorst() {
    final last = _ids.length - 1;
    _dists[0] = _dists[last];
    _ids[0] = _ids[last];
    _dists.removeLast();
    _ids.removeLast();
    if (_ids.isEmpty) return;
    var i = 0;
    while (true) {
      final l = 2 * i + 1;
      final r = 2 * i + 2;
      var best = i;
      if (l < _ids.length && _dists[l] > _dists[best]) best = l;
      if (r < _ids.length && _dists[r] > _dists[best]) best = r;
      if (best == i) break;
      _swap(i, best);
      i = best;
    }
  }

  _BeamResult sortedBestFirst() {
    final n = _ids.length;
    final order = List<int>.generate(n, (i) => i)
      ..sort((a, b) => _dists[a].compareTo(_dists[b]));
    final ids = <int>[];
    final ds = <double>[];
    for (final o in order) {
      ids.add(_ids[o]);
      ds.add(_dists[o]);
    }
    return _BeamResult(ids, ds);
  }

  void _swap(int a, int b) {
    final td = _dists[a];
    _dists[a] = _dists[b];
    _dists[b] = td;
    final ti = _ids[a];
    _ids[a] = _ids[b];
    _ids[b] = ti;
  }
}
