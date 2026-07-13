/// Inverted-list search over binary vectors.
///
/// FAISS' `IndexBinaryIVF`: cluster the corpus into `nlist` cells via
/// binary k-means (bit-majority centroids), scan only the `nprobe`
/// closest cells at search time. Same wire-level layout as
/// [IndexBinaryFlat] except each vector is stored inside its coarse
/// cell instead of a flat array.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'index.dart';
import 'index_binary.dart';
import 'index_binary_flat.dart';
import 'index_io.dart';

class IndexBinaryIVF extends IndexBinary {
  IndexBinaryIVF({
    required int codeSize,
    required this.nlist,
    this.nprobe = 1,
    this.kmeansIters = 20,
    this.seed = 1234,
  }) : quantizer = IndexBinaryFlat(codeSize),
       super(codeSize) {
    isTrained = false;
    _invLists = List<List<int>>.generate(nlist, (_) => <int>[]);
    _invCodes = List<Uint8List>.generate(nlist, (_) => Uint8List(0));
    _invSize = Int32List(nlist);
  }

  /// Coarse quantizer holding the `nlist` bit-majority centroids.
  final IndexBinaryFlat quantizer;

  final int nlist;
  int nprobe;
  final int kmeansIters;
  final int seed;

  // For every list: parallel arrays of external ids and packed codes.
  late final List<List<int>> _invLists;
  late final List<Uint8List> _invCodes; // capacity buffers
  late final Int32List _invSize; // valid vectors per list

  /// Read-only view: number of vectors in cell [listNo].
  int listSize(int listNo) => _invSize[listNo];

  @override
  void train(List<Uint8List> xs) {
    if (xs.length < nlist) {
      throw ArgumentError(
        'IndexBinaryIVF.train: need >= nlist=$nlist vectors, got ${xs.length}',
      );
    }
    // ---- Binary k-means ------------------------------------------------
    final rng = math.Random(seed);
    // Init: pick `nlist` distinct random training codes.
    final chosen = <int>{};
    while (chosen.length < nlist) {
      chosen.add(rng.nextInt(xs.length));
    }
    final centroids = List<Uint8List>.generate(nlist, (_) => Uint8List(codeSize));
    final chosenList = chosen.toList(growable: false);
    for (var c = 0; c < nlist; c++) {
      final src = xs[chosenList[c]];
      for (var j = 0; j < codeSize; j++) {
        centroids[c][j] = src[j];
      }
    }
    final assign = Int32List(xs.length);
    // Bit-counters: per cluster, per bit position, count of set bits.
    final bitCounts = List<Int32List>.generate(
      nlist,
      (_) => Int32List(codeSize * 8),
    );
    final memberCount = Int32List(nlist);

    for (var iter = 0; iter < kmeansIters; iter++) {
      // Assign.
      for (var i = 0; i < xs.length; i++) {
        var best = 1 << 30;
        var bestC = 0;
        for (var c = 0; c < nlist; c++) {
          final h = hammingDistance(centroids[c], xs[i], 0, codeSize);
          if (h < best) {
            best = h;
            bestC = c;
          }
        }
        assign[i] = bestC;
      }
      // Update: bit-majority per position.
      for (var c = 0; c < nlist; c++) {
        for (var j = 0; j < bitCounts[c].length; j++) {
          bitCounts[c][j] = 0;
        }
      }
      for (var c = 0; c < nlist; c++) {
        memberCount[c] = 0;
      }
      for (var i = 0; i < xs.length; i++) {
        final c = assign[i];
        memberCount[c]++;
        final row = xs[i];
        final counts = bitCounts[c];
        for (var byte = 0; byte < codeSize; byte++) {
          final v = row[byte];
          final base = byte * 8;
          if ((v & 0x01) != 0) counts[base]++;
          if ((v & 0x02) != 0) counts[base + 1]++;
          if ((v & 0x04) != 0) counts[base + 2]++;
          if ((v & 0x08) != 0) counts[base + 3]++;
          if ((v & 0x10) != 0) counts[base + 4]++;
          if ((v & 0x20) != 0) counts[base + 5]++;
          if ((v & 0x40) != 0) counts[base + 6]++;
          if ((v & 0x80) != 0) counts[base + 7]++;
        }
      }
      for (var c = 0; c < nlist; c++) {
        if (memberCount[c] == 0) {
          // Re-seed empty cluster from a random point.
          final src = xs[rng.nextInt(xs.length)];
          for (var j = 0; j < codeSize; j++) {
            centroids[c][j] = src[j];
          }
          continue;
        }
        final half = memberCount[c] / 2.0;
        for (var byte = 0; byte < codeSize; byte++) {
          var b = 0;
          final base = byte * 8;
          if (bitCounts[c][base] > half) b |= 0x01;
          if (bitCounts[c][base + 1] > half) b |= 0x02;
          if (bitCounts[c][base + 2] > half) b |= 0x04;
          if (bitCounts[c][base + 3] > half) b |= 0x08;
          if (bitCounts[c][base + 4] > half) b |= 0x10;
          if (bitCounts[c][base + 5] > half) b |= 0x20;
          if (bitCounts[c][base + 6] > half) b |= 0x40;
          if (bitCounts[c][base + 7] > half) b |= 0x80;
          centroids[c][byte] = b;
        }
      }
    }
    // Load centroids into the coarse quantizer.
    quantizer.add(centroids);
    isTrained = true;
  }

  int _coarseAssign(Uint8List code) {
    var best = 1 << 30;
    var bestC = 0;
    for (var c = 0; c < nlist; c++) {
      final h = hammingDistance(quantizer.reconstruct(c), code, 0, codeSize);
      if (h < best) {
        best = h;
        bestC = c;
      }
    }
    return bestC;
  }

  List<int> _nprobeCells(Uint8List code) {
    // Small nlist here (typ. <= 1024) → simple partial sort.
    final dists = Int32List(nlist);
    for (var c = 0; c < nlist; c++) {
      dists[c] = hammingDistance(quantizer.reconstruct(c), code, 0, codeSize);
    }
    final idx = List<int>.generate(nlist, (i) => i);
    idx.sort((a, b) => dists[a] - dists[b]);
    return idx.sublist(0, math.min(nprobe, nlist));
  }

  @override
  void add(List<Uint8List> xs) {
    if (!isTrained) {
      throw StateError('IndexBinaryIVF.add before train');
    }
    if (xs.isEmpty) return;
    for (var i = 0; i < xs.length; i++) {
      if (xs[i].length != codeSize) {
        throw ArgumentError(
          'IndexBinaryIVF.add: vector $i has length ${xs[i].length}, '
          'expected $codeSize',
        );
      }
      final id = ntotal + i;
      final c = _coarseAssign(xs[i]);
      final ids = _invLists[c];
      ids.add(id);
      // Grow codes buffer if needed.
      final have = _invSize[c];
      final needed = (have + 1) * codeSize;
      if (needed > _invCodes[c].length) {
        var newCap = _invCodes[c].length == 0 ? 4 * codeSize : _invCodes[c].length;
        while (newCap < needed) {
          newCap *= 2;
        }
        final grown = Uint8List(newCap);
        for (var j = 0; j < have * codeSize; j++) {
          grown[j] = _invCodes[c][j];
        }
        _invCodes[c] = grown;
      }
      final off = have * codeSize;
      for (var j = 0; j < codeSize; j++) {
        _invCodes[c][off + j] = xs[i][j];
      }
      _invSize[c] = have + 1;
    }
    ntotal += xs.length;
  }

  @override
  SearchResult search(List<Uint8List> queries, int k) {
    if (!isTrained) throw StateError('IndexBinaryIVF.search before train');
    final nq = queries.length;
    final distances = List<Float32List>.generate(nq, (_) => Float32List(k));
    final ids = List<Int32List>.generate(nq, (_) => Int32List(k));
    for (var qi = 0; qi < nq; qi++) {
      final q = queries[qi];
      if (q.length != codeSize) {
        throw ArgumentError(
          'IndexBinaryIVF.search: query $qi length ${q.length} != $codeSize',
        );
      }
      final cells = _nprobeCells(q);
      final heap = TopK(k, maxIsWorst: true);
      for (final c in cells) {
        final size = _invSize[c];
        final codes = _invCodes[c];
        final idsC = _invLists[c];
        for (var i = 0; i < size; i++) {
          final h = hammingDistance(q, codes, i * codeSize, codeSize);
          heap.push(h.toDouble(), idsC[i]);
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
    w.writeU32(codeSize);
    w.writeU32(nlist);
    w.writeU32(nprobe);
    w.writeU32(ntotal);
    w.writeU8(isTrained ? 1 : 0);
    // Coarse quantizer centroids (as raw codes).
    for (var c = 0; c < nlist; c++) {
      if (isTrained) {
        w.writeU8List(quantizer.reconstruct(c));
      } else {
        w.writeU8List(Uint8List(codeSize));
      }
    }
    // Inverted lists.
    for (var c = 0; c < nlist; c++) {
      final size = _invSize[c];
      w.writeU32(size);
      for (var i = 0; i < size; i++) {
        w.writeU32(_invLists[c][i]);
      }
      if (size > 0) {
        w.writeU8List(
          Uint8List.sublistView(_invCodes[c], 0, size * codeSize),
        );
      }
    }
  }

  static IndexBinaryIVF readFrom(IoReader r) {
    final codeSize = r.readU32();
    final nlist = r.readU32();
    final nprobe = r.readU32();
    final ntotal = r.readU32();
    final isTrained = r.readU8() != 0;
    final idx = IndexBinaryIVF(
      codeSize: codeSize,
      nlist: nlist,
      nprobe: nprobe,
    );
    final centroids = <Uint8List>[];
    for (var c = 0; c < nlist; c++) {
      centroids.add(r.readU8List(codeSize));
    }
    if (isTrained) {
      idx.quantizer.add(centroids);
      idx.isTrained = true;
    }
    for (var c = 0; c < nlist; c++) {
      final size = r.readU32();
      final ids = List<int>.generate(size, (_) => r.readU32());
      final codes = size > 0 ? r.readU8List(size * codeSize) : Uint8List(0);
      idx._invLists[c] = ids;
      idx._invCodes[c] = codes;
      idx._invSize[c] = size;
    }
    idx.ntotal = ntotal;
    return idx;
  }
}
