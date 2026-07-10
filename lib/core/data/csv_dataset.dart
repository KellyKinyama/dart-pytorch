/// Tabular CSV dataset for regression and classification.
///
/// Reads a UTF-8 CSV file with one example per row. Each row is
/// split into `features` (numeric columns) and an optional `label`
/// (either a numeric target for regression or a class index for
/// classification):
///
/// * If `labelColumn` is `null`, every column is treated as a
///   feature and `label` on each item is `-1`.
/// * If `labelColumn` is an int, that column is peeled off as the
///   label and the remaining columns become the feature vector.
/// * A `classMap` may be provided to map string class names to
///   integer indices when the label column is categorical text.
///
/// Numeric parsing uses [double.tryParse]; unparseable cells are
/// filled with `0.0` (regression) or trigger a `FormatException` on
/// the label column.
///
/// Every item is a [CsvSample] of one `[numFeatures]` feature
/// [Tensor] and a scalar label. Tensors are built on the requested
/// device.
library;

import 'dart:io';

import '../tensor/tensor.dart';
import 'dataset.dart';

class CsvSample {
  final Tensor features; // [numFeatures]
  final double label; // -1 if no label column
  const CsvSample(this.features, this.label);
}

class CsvDataset extends Dataset<CsvSample> {
  final List<List<double>> _rows;
  final List<double> _labels;
  final int numFeatures;
  final Device device;
  final List<String> headers;

  int get numExamples => _rows.length;

  CsvDataset._(
    this._rows,
    this._labels,
    this.numFeatures,
    this.headers,
    this.device,
  );

  factory CsvDataset.fromFile(
    String path, {
    int? labelColumn,
    Map<String, int>? classMap,
    bool hasHeader = true,
    String delimiter = ',',
    Device device = Device.CPU,
  }) {
    final raw = File(path).readAsLinesSync();
    return CsvDataset.fromLines(
      raw,
      labelColumn: labelColumn,
      classMap: classMap,
      hasHeader: hasHeader,
      delimiter: delimiter,
      device: device,
    );
  }

  factory CsvDataset.fromLines(
    List<String> lines, {
    int? labelColumn,
    Map<String, int>? classMap,
    bool hasHeader = true,
    String delimiter = ',',
    Device device = Device.CPU,
  }) {
    if (lines.isEmpty) {
      throw ArgumentError('CsvDataset: empty input');
    }
    final headers = <String>[];
    int start = 0;
    if (hasHeader) {
      headers.addAll(_splitLine(lines[0], delimiter));
      start = 1;
    }
    if (start >= lines.length) {
      throw ArgumentError('CsvDataset: no data rows');
    }
    final firstCells = _splitLine(lines[start], delimiter);
    final totalCols = firstCells.length;
    if (labelColumn != null && (labelColumn < 0 || labelColumn >= totalCols)) {
      throw ArgumentError(
        'labelColumn=$labelColumn out of range [0, $totalCols)',
      );
    }
    final numFeatures = labelColumn == null ? totalCols : totalCols - 1;

    final rows = <List<double>>[];
    final labels = <double>[];
    for (int i = start; i < lines.length; i++) {
      final line = lines[i].trim();
      if (line.isEmpty) continue;
      final cells = _splitLine(line, delimiter);
      if (cells.length != totalCols) {
        throw FormatException(
          'row $i: expected $totalCols columns, got ${cells.length}',
        );
      }
      final feats = <double>[];
      double label = -1;
      for (int c = 0; c < cells.length; c++) {
        if (c == labelColumn) {
          final raw = cells[c].trim();
          if (classMap != null) {
            final v = classMap[raw];
            if (v == null) {
              throw FormatException(
                'row $i col $c: label "$raw" not in classMap',
              );
            }
            label = v.toDouble();
          } else {
            final v = double.tryParse(raw);
            if (v == null) {
              throw FormatException(
                'row $i col $c: label "$raw" is not numeric '
                '(pass a classMap for categorical labels)',
              );
            }
            label = v;
          }
        } else {
          feats.add(double.tryParse(cells[c].trim()) ?? 0.0);
        }
      }
      rows.add(feats);
      labels.add(label);
    }
    if (rows.isEmpty) {
      throw ArgumentError('CsvDataset: parsed 0 data rows');
    }
    return CsvDataset._(rows, labels, numFeatures, headers, device);
  }

  /// Simple delimiter split — no quoting / escaping. Fine for the
  /// well-formed numeric CSVs we target here.
  static List<String> _splitLine(String line, String delimiter) =>
      line.split(delimiter);

  @override
  int get length => _rows.length;

  @override
  CsvSample operator [](int index) => CsvSample(
    Tensor.fromList([numFeatures], _rows[index], device: device),
    _labels[index],
  );
}
