/// PyTorch-style `Dataset` + `DataLoader` abstractions.
///
/// A [Dataset] is any indexable collection of examples of type `T`.
/// A [DataLoader] wraps a dataset and yields batches (`List<T>`) in
/// either sequential or shuffled order, optionally dropping the
/// last (short) batch.
///
/// Concrete datasets in this package:
///
/// * [ImageFolderDataset] — folder-per-class image classification /
///   triplet sampling.
/// * [TextTokenDataset] — sliding-window language-modeling examples
///   from a text file, using [CharTokenizer] or [BpeTokenizer].
/// * [CsvDataset] — tabular `(features, label)` regression /
///   classification.
///
/// All three build tensors on a caller-specified [Device].
library;

import 'dart:math' as math;

/// Read-only, indexable collection of examples of type [T].
abstract class Dataset<T> {
  int get length;
  T operator [](int index);
}

/// In-memory dataset built from a plain `List<T>`.
class ListDataset<T> extends Dataset<T> {
  final List<T> _items;
  ListDataset(this._items);

  @override
  int get length => _items.length;

  @override
  T operator [](int index) => _items[index];
}

/// Yields fixed-size batches of items from a [Dataset].
///
/// * `batchSize` — items per batch.
/// * `shuffle`   — reshuffle the index order at every call to
///   [batches]. Uses [seed] for determinism.
/// * `dropLast`  — drop the trailing short batch when
///   `dataset.length % batchSize != 0`. Default keeps it.
class DataLoader<T> {
  final Dataset<T> dataset;
  final int batchSize;
  final bool shuffle;
  final bool dropLast;
  final math.Random _rng;

  DataLoader(
    this.dataset, {
    this.batchSize = 1,
    this.shuffle = false,
    this.dropLast = false,
    int? seed,
  }) : _rng = math.Random(seed ?? 0) {
    if (batchSize < 1) {
      throw ArgumentError('batchSize must be >= 1, got $batchSize');
    }
  }

  /// Number of batches produced per full iteration.
  int get length {
    final n = dataset.length;
    if (dropLast) return n ~/ batchSize;
    return (n + batchSize - 1) ~/ batchSize;
  }

  /// Lazily produce one full pass over the dataset as batches.
  Iterable<List<T>> batches() sync* {
    final n = dataset.length;
    final order = List<int>.generate(n, (i) => i);
    if (shuffle) order.shuffle(_rng);
    for (int start = 0; start < n; start += batchSize) {
      final end = math.min(start + batchSize, n);
      if (dropLast && end - start < batchSize) break;
      yield [for (int k = start; k < end; k++) dataset[order[k]]];
    }
  }
}
