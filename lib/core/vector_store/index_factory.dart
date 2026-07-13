/// `indexFactory` — build a composed [Index] from a compact FAISS-
/// style description string.
///
/// A description is a comma-separated chain of tokens:
///   * pre-transforms (left side, in order): `L2Norm`, `RR`,
///     `PCA<dOut>`, `PCAW<dOut>` (whitened).
///   * an index specifier (right side, exactly one): `Flat`,
///     `IVF<nlist>,Flat`, `IVF<nlist>,PQ<m>[x<nbits>]`, `PQ<m>[x<nbits>]`,
///     `HNSW<M>`, `SQ8`, `RFlat<inner>` (refine wrapper), `LSH<nbits>`.
///
/// Examples:
///   * `"Flat"`
///   * `"PCA32,IVF64,Flat"`
///   * `"L2Norm,IVF128,PQ16"`
///   * `"HNSW32"`
///   * `"L2Norm,LSH128"`
///
/// Metric is a parameter, not part of the string, matching FAISS'
/// `index_factory(d, description, metric)` API.
library;

import 'index.dart';
import 'index_flat.dart';
import 'index_hnsw.dart';
import 'index_ivf_flat.dart';
import 'index_ivf_pq.dart';
import 'index_lsh.dart';
import 'index_pq.dart';
import 'index_pre_transform.dart';
import 'index_scalar_quantizer.dart';
import 'l2_norm_transform.dart';
import 'pca_transform.dart';
import 'random_rotation_transform.dart';
import 'vector_transform.dart';

/// Build an [Index] from a description string.
///
/// Throws [FormatException] on unparseable input.
Index indexFactory(int d, String description, {Metric metric = Metric.l2}) {
  final tokens = description
      .split(',')
      .map((t) => t.trim())
      .where((t) => t.isNotEmpty)
      .toList();
  if (tokens.isEmpty) {
    throw FormatException('indexFactory: empty description');
  }

  // Split into (pre-transform tokens, index tokens). The index side
  // may be one or two tokens (IVF*,PQ* or IVF*,Flat) so we peel from
  // the left as long as we recognize a pre-transform.
  final chain = <VectorTransform>[];
  var currentDim = d;
  var i = 0;
  while (i < tokens.length) {
    final tok = tokens[i];
    final tr = _tryParseTransform(tok, currentDim);
    if (tr == null) break;
    chain.add(tr);
    currentDim = tr.dOut;
    i++;
  }

  final indexTokens = tokens.sublist(i);
  if (indexTokens.isEmpty) {
    throw FormatException('indexFactory: no index specifier in "$description"');
  }
  final inner = _parseIndex(currentDim, indexTokens, metric);

  if (chain.isEmpty) return inner;
  return IndexPreTransform(chain: chain, inner: inner);
}

VectorTransform? _tryParseTransform(String tok, int dIn) {
  if (tok == 'L2Norm') return L2NormTransform(dIn);
  if (tok == 'RR') return RandomRotationTransform(d: dIn);
  final pca = RegExp(r'^PCA(W?)(\d+)$').firstMatch(tok);
  if (pca != null) {
    final whitened = pca.group(1) == 'W';
    final dOut = int.parse(pca.group(2)!);
    return PCATransform(
      dIn: dIn,
      dOut: dOut,
      eigenPower: whitened ? -0.5 : 0.0,
    );
  }
  return null;
}

Index _parseIndex(int d, List<String> tokens, Metric metric) {
  final head = tokens.first;

  // Flat.
  if (head == 'Flat') {
    if (tokens.length != 1) {
      throw FormatException(
        'indexFactory: unexpected trailing tokens after Flat: $tokens',
      );
    }
    return IndexFlat(d, metric);
  }

  // SQ8 — 8-bit scalar quantizer.
  if (head == 'SQ8') {
    if (tokens.length != 1) {
      throw FormatException('indexFactory: SQ8 takes no arguments');
    }
    return IndexScalarQuantizer(d, metric: metric);
  }

  // HNSW<M>.
  final hnsw = RegExp(r'^HNSW(\d+)$').firstMatch(head);
  if (hnsw != null) {
    if (tokens.length != 1) {
      throw FormatException('indexFactory: HNSW takes no trailing tokens');
    }
    final m = int.parse(hnsw.group(1)!);
    return IndexHNSW(d: d, M: m, metric: metric);
  }

  // LSH<nbits>.
  final lsh = RegExp(r'^LSH(\d+)$').firstMatch(head);
  if (lsh != null) {
    if (tokens.length != 1) {
      throw FormatException('indexFactory: LSH takes no trailing tokens');
    }
    final nbits = int.parse(lsh.group(1)!);
    return IndexLSH(d: d, nbits: nbits);
  }

  // PQ<m>[x<nbits>].
  final pq = RegExp(r'^PQ(\d+)(?:x(\d+))?$').firstMatch(head);
  if (pq != null) {
    if (tokens.length != 1) {
      throw FormatException('indexFactory: PQ takes no trailing tokens');
    }
    final m = int.parse(pq.group(1)!);
    final nbits = pq.group(2) != null ? int.parse(pq.group(2)!) : 8;
    return IndexPQ(d: d, m: m, nbits: nbits, metric: metric);
  }

  // IVF<nlist>,{Flat|PQ<m>[x<nbits>]}.
  final ivf = RegExp(r'^IVF(\d+)$').firstMatch(head);
  if (ivf != null) {
    if (tokens.length != 2) {
      throw FormatException(
        'indexFactory: IVF must be followed by Flat or PQ (got $tokens)',
      );
    }
    final nlist = int.parse(ivf.group(1)!);
    final tail = tokens[1];
    if (tail == 'Flat') {
      return IndexIVFFlat(d: d, nlist: nlist, metric: metric);
    }
    final pqTail = RegExp(r'^PQ(\d+)(?:x(\d+))?$').firstMatch(tail);
    if (pqTail != null) {
      final m = int.parse(pqTail.group(1)!);
      final nbits = pqTail.group(2) != null ? int.parse(pqTail.group(2)!) : 8;
      return IndexIVFPQ(d: d, nlist: nlist, m: m, nbits: nbits, metric: metric);
    }
    throw FormatException('indexFactory: unknown IVF tail "$tail"');
  }

  throw FormatException(
    'indexFactory: unknown index specifier "$head" in $tokens',
  );
}
