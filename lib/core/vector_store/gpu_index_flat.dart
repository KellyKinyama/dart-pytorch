/// GPU-backed brute-force flat index.
///
/// Wraps a plain [IndexFlat] and swaps its CPU-native triple-loop
/// search for a batched `Q @ Dᵀ` matmul run through the tensor engine
/// (which dispatches to the tiled CUDA kernel when the operand sizes
/// clear [Tensor.autoDeviceThreshold]).
///
/// Distance conversion:
///
///   * `metric = Metric.innerProduct` — pick top-k largest dot
///     products directly out of `Q @ Dᵀ`.
///   * `metric = Metric.l2` — recover squared L2 distance via the
///     identity `||q - d||² = ||q||² + ||d||² - 2·q·d`. `||d||²` is
///     precomputed once and cached alongside the database tensor.
///
/// The database tensor is materialised lazily on the first `search`
/// after any `add`, and reused across queries. Callers that stream
/// insertions between searches should batch their adds to avoid
/// re-uploads.
///
/// Fall-through: if the effective problem size is below
/// [gpuIndexFlatMinDot] (product of `nq * ntotal * d` fp32 mults) the
/// wrapper transparently defers to [IndexFlat.search] — the FFI +
/// upload overhead eats any speedup at tiny scales.
library;

import 'dart:typed_data';

import '../tensor/tensor.dart';
import 'index.dart';
import 'index_flat.dart';

/// Below this many fp32 multiplications (roughly `nq · ntotal · d`),
/// [GpuIndexFlat.search] transparently falls back to CPU. Chosen so
/// the smoke tests still exercise both branches at the sizes we ship
/// unit tests at.
const int gpuIndexFlatMinDot = 1 << 18; // 262 144

/// Brute-force flat index using [Tensor] matmul for the score step.
class GpuIndexFlat extends Index {
  GpuIndexFlat._(this._backing) : super(_backing.d, _backing.metric) {
    ntotal = _backing.ntotal;
    isTrained = _backing.isTrained;
  }

  /// Construct an L2 GPU flat.
  factory GpuIndexFlat.l2(int d) => GpuIndexFlat._(IndexFlat(d, Metric.l2));

  /// Construct an inner-product GPU flat.
  factory GpuIndexFlat.ip(int d) =>
      GpuIndexFlat._(IndexFlat(d, Metric.innerProduct));

  /// Wrap an existing [IndexFlat] without copying its storage — the
  /// wrapper takes ownership; do not keep using the underlying flat
  /// after this call unless you understand that adds will bypass the
  /// GPU cache invalidation.
  factory GpuIndexFlat.wrap(IndexFlat src) => GpuIndexFlat._(src);

  final IndexFlat _backing;

  /// Cached `[ntotal, d]` database tensor. `null` when stale.
  Tensor? _dbTensor;

  /// Cached `||d[i]||²` for each database vector (L2 only).
  Float32List? _dbNormsSq;

  @override
  void train(List<Float32List> xs) => _backing.train(xs);

  @override
  void add(List<Float32List> xs) {
    _backing.add(xs);
    _invalidate();
    ntotal = _backing.ntotal;
    isTrained = _backing.isTrained;
  }

  void _invalidate() {
    _dbTensor?.dispose();
    _dbTensor = null;
    _dbNormsSq = null;
  }

  /// Access to the wrapped [IndexFlat] (e.g. for `reconstruct`).
  IndexFlat get backing => _backing;

  /// Force a rebuild of the cached GPU database tensor. Cheap when
  /// the cache is already fresh (no-op).
  void warmup() => _ensureDbCache();

  void _ensureDbCache() {
    if (_dbTensor != null) return;
    if (_backing.ntotal == 0) return;
    final n = _backing.ntotal;
    final flat = List<double>.filled(n * d, 0.0);
    // Copy row-major.
    for (var i = 0; i < n; i++) {
      final view = _backing.reconstruct(i);
      final base = i * d;
      for (var j = 0; j < d; j++) {
        flat[base + j] = view[j];
      }
    }
    _dbTensor = Tensor.fromList([n, d], flat);
    if (metric == Metric.l2) {
      final norms = Float32List(n);
      for (var i = 0; i < n; i++) {
        final view = _backing.reconstruct(i);
        var s = 0.0;
        for (var j = 0; j < d; j++) {
          s += view[j] * view[j];
        }
        norms[i] = s;
      }
      _dbNormsSq = norms;
    }
  }

  @override
  SearchResult search(List<Float32List> queries, int k) {
    final nq = queries.length;
    if (nq == 0) {
      return SearchResult(List<Float32List>.empty(), List<Int32List>.empty());
    }
    if (_backing.ntotal == 0 || nq * _backing.ntotal * d < gpuIndexFlatMinDot) {
      return _backing.search(queries, k);
    }

    _ensureDbCache();
    final db = _dbTensor!;
    final n = _backing.ntotal;

    // Upload queries as `[nq, d]`, matching the DB tensor's device to
    // avoid the "mixed devices" throw in matmul.
    final qFlat = List<double>.filled(nq * d, 0.0);
    for (var qi = 0; qi < nq; qi++) {
      final q = queries[qi];
      if (q.length != d) {
        throw ArgumentError('query $qi length ${q.length} != $d');
      }
      final base = qi * d;
      for (var j = 0; j < d; j++) {
        qFlat[base + j] = q[j];
      }
    }
    final qT = Tensor.fromList([nq, d], qFlat, device: db.device);

    // dot = Q @ Dᵀ  →  `[nq, ntotal]`.
    final dbT = db.transpose();
    final dotT = qT.matmul(dbT);
    final dot = dotT.toList();
    qT.dispose();
    dbT.dispose();
    dotT.dispose();

    // Query norms (L2 only).
    Float32List? qNorm;
    if (metric == Metric.l2) {
      qNorm = Float32List(nq);
      for (var qi = 0; qi < nq; qi++) {
        final q = queries[qi];
        var s = 0.0;
        for (var j = 0; j < d; j++) {
          s += q[j] * q[j];
        }
        qNorm[qi] = s;
      }
    }
    final dNorm = _dbNormsSq;

    // Per-query top-k extraction.
    final distances = List<Float32List>.generate(nq, (_) => Float32List(k));
    final ids = List<Int32List>.generate(nq, (_) => Int32List(k));
    for (var qi = 0; qi < nq; qi++) {
      final heap = TopK(k, maxIsWorst: metric == Metric.l2);
      final rowBase = qi * n;
      if (metric == Metric.l2) {
        final qn = qNorm![qi];
        for (var vi = 0; vi < n; vi++) {
          final s = qn + dNorm![vi] - 2.0 * dot[rowBase + vi];
          heap.push(s, vi);
        }
      } else {
        for (var vi = 0; vi < n; vi++) {
          heap.push(dot[rowBase + vi], vi);
        }
      }
      final sorted = heap.sorted();
      distances[qi] = sorted.scores;
      ids[qi] = sorted.ids;
    }
    return SearchResult(distances, ids);
  }
}
