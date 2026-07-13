/// `IndexIDMap` — wraps any [Index] to support custom `int64` ids.
///
/// The FAISS base `Index` API only assigns sequential ids starting at
/// zero. In practice you often want to attach an external key (row id,
/// primary key, etc.) to each vector. `IndexIDMap` stores a
/// `List<int>` mapping "internal id" → "external id" alongside the
/// inner index and rewrites [search] results to return external ids.
library;

import 'dart:typed_data';

import 'index.dart';
import 'index_io.dart';

class IndexIDMap extends Index {
  IndexIDMap(this.inner) : super(inner.d, inner.metric) {
    isTrained = inner.isTrained;
  }

  final Index inner;

  /// `_extIds[internalId]` = external id assigned by the user.
  final List<int> _extIds = [];

  /// Reverse lookup — rebuilt lazily. Not exposed; used by [reconstruct].
  Map<int, int>? _reverse;

  int get size => _extIds.length;

  @override
  void train(List<Float32List> xs) {
    inner.train(xs);
    isTrained = inner.isTrained;
  }

  @override
  void add(List<Float32List> xs) {
    // Preserve FAISS semantics: `add` without ids assigns sequential
    // external ids continuing from wherever we left off.
    final start = _extIds.length;
    addWithIds(xs, List<int>.generate(xs.length, (i) => start + i));
  }

  /// Add vectors with explicit external ids. Length of [ids] must
  /// equal `xs.length` and ids should be unique (this is not enforced
  /// — duplicates are legal but make [idOf] ambiguous).
  ///
  /// Ids must fit in `int32` range (`[-2^31, 2^31 - 1]`) because
  /// [SearchResult] uses `Int32List`. This mirrors FAISS' pre-1.7
  /// behaviour; if you need 64-bit ids, wrap and translate yourself.
  void addWithIds(List<Float32List> xs, List<int> ids) {
    if (xs.length != ids.length) {
      throw ArgumentError(
        'addWithIds: xs (${xs.length}) and ids (${ids.length}) length mismatch',
      );
    }
    const int32Min = -0x80000000;
    const int32Max = 0x7fffffff;
    for (var i = 0; i < ids.length; i++) {
      if (ids[i] < int32Min || ids[i] > int32Max) {
        throw ArgumentError(
          'addWithIds: id ${ids[i]} at position $i does not fit in int32',
        );
      }
    }
    inner.add(xs);
    _extIds.addAll(ids);
    ntotal = _extIds.length;
    _reverse = null;
  }

  /// External id for internal offset `i`.
  int idOf(int i) => _extIds[i];

  /// Internal offset (into `inner`) for external id `id`, or `-1`.
  int internalOf(int id) {
    _reverse ??= {for (var i = 0; i < _extIds.length; i++) _extIds[i]: i};
    return _reverse![id] ?? -1;
  }

  @override
  SearchResult search(List<Float32List> queries, int k) {
    final r = inner.search(queries, k);
    // Rewrite internal ids → external ids in place.
    for (var qi = 0; qi < r.nq; qi++) {
      final row = r.ids[qi];
      for (var j = 0; j < k; j++) {
        final internal = row[j];
        if (internal < 0 || internal >= _extIds.length) continue;
        row[j] = _extIds[internal];
      }
    }
    return r;
  }

  @override
  RangeSearchResult rangeSearch(List<Float32List> queries, double radius) {
    final r = inner.rangeSearch(queries, radius);
    // Translate internal → external ids in the flat ids buffer.
    for (var i = 0; i < r.ids.length; i++) {
      final internal = r.ids[i];
      if (internal >= 0 && internal < _extIds.length) {
        r.ids[i] = _extIds[internal];
      }
    }
    return r;
  }

  /// Remove vectors by their **external** ids. Returns the number
  /// actually removed. Delegates to the inner index's [removeIds]
  /// after translating to internal offsets, then compacts the
  /// external-id table to stay in sync with the renumbered inner.
  @override
  int removeIds(Set<int> ids) {
    if (ids.isEmpty) return 0;
    _reverse ??= {for (var i = 0; i < _extIds.length; i++) _extIds[i]: i};
    final internal = <int>{};
    for (final ext in ids) {
      final i = _reverse![ext];
      if (i != null) internal.add(i);
    }
    if (internal.isEmpty) return 0;
    final removed = inner.removeIds(internal);
    // Rebuild _extIds compact to match the inner's new numbering.
    final kept = <int>[];
    for (var i = 0; i < _extIds.length; i++) {
      if (!internal.contains(i)) kept.add(_extIds[i]);
    }
    _extIds
      ..clear()
      ..addAll(kept);
    ntotal = _extIds.length;
    _reverse = null;
    return removed;
  }

  // --- persistence --------------------------------------------------------

  void writeTo(IoWriter w) {
    w.writeU32(d);
    w.writeU32(metricToU32(metric));
    w.writeU32(ntotal);
    w.writeU8(isTrained ? 1 : 0);
    writeChild(w, inner);
    w.writeU32(_extIds.length);
    for (final id in _extIds) {
      w.writeI32(id);
    }
  }

  static IndexIDMap readFrom(IoReader r) {
    // Common header (d, metric, ntotal, isTrained) — we mostly ignore
    // these because the wrapped inner index carries authoritative
    // values, but validate d/metric for sanity.
    final d = r.readU32();
    final metric = metricFromU32(r.readU32());
    final ntotal = r.readU32();
    final isTrained = r.readU8() != 0;
    final inner = readChild(r);
    if (inner.d != d || inner.metric != metric) {
      throw FormatException(
        'IndexIDMap: header (d=$d, metric=$metric) disagrees with '
        'inner (d=${inner.d}, metric=${inner.metric})',
      );
    }
    final idx = IndexIDMap(inner);
    final n = r.readU32();
    for (var i = 0; i < n; i++) {
      idx._extIds.add(r.readI32());
    }
    idx.ntotal = ntotal;
    idx.isTrained = isTrained;
    return idx;
  }
}
