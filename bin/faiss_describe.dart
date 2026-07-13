/// Describe FAISS-format index files without decoding them.
///
/// Reads at most the first 37 bytes of each argument, parses the
/// fourcc + common header via [probeFaissIndexFile], and prints a
/// one-liner per file plus a detailed metadata block. Handy for
/// triaging unknown blobs, sanity-checking dumps written by upstream
/// FAISS, and answering "how big is this index?" without paying the
/// full decode cost.
///
/// Run:
///
///     dart run bin/faiss_describe.dart index.faiss [more.faiss ...]
///
/// Exit codes:
///   0 — every file parsed successfully
///   1 — usage error (no arguments)
///   2 — one or more files could not be probed
library;

import 'dart:convert';
import 'dart:io';

import 'package:dart_pytorch/dart_pytorch.dart';

const String _usage =
    'Usage: dart run bin/faiss_describe.dart [--json] <file> [<file> ...]';

int main(List<String> args) {
  if (args.isEmpty || args.contains('-h') || args.contains('--help')) {
    stderr.writeln(_usage);
    return args.isEmpty ? 1 : 0;
  }

  var json = false;
  final paths = <String>[];
  for (final a in args) {
    if (a == '--json') {
      json = true;
    } else {
      paths.add(a);
    }
  }
  if (paths.isEmpty) {
    stderr.writeln('faiss_describe: no input files\n$_usage');
    return 1;
  }

  if (json) return _mainJson(paths);

  var failures = 0;
  for (var i = 0; i < paths.length; i++) {
    final path = paths[i];
    if (i > 0) stdout.writeln();
    if (!_describe(path)) failures++;
  }
  return failures == 0 ? 0 : 2;
}

/// Emit a single JSON array of per-file records — easy to pipe into
/// jq or another tool. Files that fail to probe get an `error` field.
int _mainJson(List<String> paths) {
  var failures = 0;
  final out = <Map<String, Object?>>[];
  for (final path in paths) {
    final rec = _probeToJson(path);
    out.add(rec);
    if (rec['error'] != null) failures++;
  }
  stdout.writeln(const JsonEncoder.withIndent('  ').convert(out));
  return failures == 0 ? 0 : 2;
}

Map<String, Object?> _probeToJson(String path) {
  final file = File(path);
  if (!file.existsSync()) {
    return {'path': path, 'error': 'no such file'};
  }
  final int size;
  try {
    size = file.lengthSync();
  } on FileSystemException catch (e) {
    return {'path': path, 'error': 'stat failed: ${e.message}'};
  }
  final FaissIndexInfo info;
  try {
    info = probeFaissIndexFile(path);
  } on FormatException catch (e) {
    return {'path': path, 'size_bytes': size, 'error': e.message};
  } on FileSystemException catch (e) {
    return {'path': path, 'size_bytes': size, 'error': e.message};
  }
  return {'path': path, 'size_bytes': size, ..._infoToJson(info)};
}

Map<String, Object?> _infoToJson(FaissIndexInfo info) {
  return {
    'fourcc': info.fourccStr,
    'fourcc_hex': '0x${info.fourcc.toRadixString(16).padLeft(8, '0')}',
    'kind': _kindTag(info.kind),
    if (info.d != null) 'd': info.d,
    if (info.codeSize != null) 'code_size': info.codeSize,
    if (info.ntotal != null) 'ntotal': info.ntotal,
    if (info.isTrained != null) 'is_trained': info.isTrained,
    if (info.metric != null) 'metric': _metricTag(info.metric!),
    if (info.tuning != null) 'tuning': _tuningToJson(info.tuning!),
    if (info.inner != null) 'inner': _infoToJson(info.inner!),
  };
}

Map<String, Object?> _tuningToJson(TuningMetadata meta) {
  return {
    'created_at': meta.createdAt.toIso8601String(),
    'metric_hint': _metricTag(meta.metric),
    'chosen_param_value': meta.chosenParamValue,
    'points': [
      for (final p in meta.points)
        {
          'param_value': p.paramValue,
          'param_label': p.paramLabel,
          'recall': p.recall,
          'mean_us': p.meanUs,
        },
    ],
  };
}

String _kindTag(FaissIndexKind kind) {
  switch (kind) {
    case FaissIndexKind.floatIndex:
      return 'float_index';
    case FaissIndexKind.binaryIndex:
      return 'binary_index';
    case FaissIndexKind.tunedWrapper:
      return 'tuned_wrapper';
    case FaissIndexKind.unknown:
      return 'unknown';
  }
}

String _metricTag(Metric m) {
  switch (m) {
    case Metric.l2:
      return 'l2';
    case Metric.innerProduct:
      return 'inner_product';
  }
}

/// Probes `path` and prints the result. Returns `true` on success.
bool _describe(String path) {
  final file = File(path);
  if (!file.existsSync()) {
    stderr.writeln('$path: no such file');
    return false;
  }
  final int size;
  try {
    size = file.lengthSync();
  } on FileSystemException catch (e) {
    stderr.writeln('$path: stat failed — ${e.message}');
    return false;
  }

  final FaissIndexInfo info;
  try {
    info = probeFaissIndexFile(path);
  } on FormatException catch (e) {
    stderr.writeln('$path: not a FAISS blob — ${e.message}');
    return false;
  } on FileSystemException catch (e) {
    stderr.writeln('$path: read failed — ${e.message}');
    return false;
  }

  stdout.writeln('$path  (${_humanBytes(size)})');
  stdout.writeln(
    '  fourcc      : ${info.fourccStr}  '
    '(0x${info.fourcc.toRadixString(16).padLeft(8, '0')})',
  );
  stdout.writeln('  kind        : ${_kindLabel(info.kind)}');

  if (info.kind == FaissIndexKind.unknown) {
    stdout.writeln(
      '  (fourcc not in the port\'s known-index table — '
      'header layout unknown)',
    );
    return true;
  }

  if (info.kind == FaissIndexKind.tunedWrapper) {
    _describeTuning(info.tuning!);
    stdout.writeln('  inner index :');
    _describeInner(info.inner!);
    return true;
  }

  stdout.writeln(
    '  d           : ${info.d}'
    '${info.kind == FaissIndexKind.binaryIndex ? ' bits' : ''}',
  );
  if (info.codeSize != null) {
    stdout.writeln('  code_size   : ${info.codeSize} bytes/vector');
  }
  stdout.writeln('  ntotal      : ${info.ntotal}');
  stdout.writeln('  is_trained  : ${info.isTrained}');
  stdout.writeln('  metric      : ${_metricLabel(info)}');
  return true;
}

void _describeTuning(TuningMetadata meta) {
  stdout.writeln('  tuning      :');
  stdout.writeln('    created_at   : ${meta.createdAt.toIso8601String()}');
  stdout.writeln('    metric hint  : ${_metricEnumLabel(meta.metric)}');
  stdout.writeln('    points       : ${meta.points.length}');
  final chosen = meta.chosenParamValue;
  if (chosen != null) {
    stdout.writeln('    chosen param : $chosen');
  } else {
    stdout.writeln('    chosen param : (none)');
  }
  for (final p in meta.points) {
    stdout.writeln(
      '      - ${p.paramLabel.padRight(16)}  '
      'recall=${p.recall.toStringAsFixed(3)}  '
      'mean_us=${p.meanUs.toStringAsFixed(1)}',
    );
  }
}

void _describeInner(FaissIndexInfo inner) {
  stdout.writeln(
    '    fourcc     : ${inner.fourccStr}  '
    '(0x${inner.fourcc.toRadixString(16).padLeft(8, '0')})',
  );
  stdout.writeln('    kind       : ${_kindLabel(inner.kind)}');
  if (inner.kind == FaissIndexKind.unknown) return;
  stdout.writeln(
    '    d          : ${inner.d}'
    '${inner.kind == FaissIndexKind.binaryIndex ? ' bits' : ''}',
  );
  if (inner.codeSize != null) {
    stdout.writeln('    code_size  : ${inner.codeSize} bytes/vector');
  }
  stdout.writeln('    ntotal     : ${inner.ntotal}');
  stdout.writeln('    is_trained : ${inner.isTrained}');
  stdout.writeln('    metric     : ${_metricLabel(inner)}');
}

String _metricEnumLabel(Metric m) {
  switch (m) {
    case Metric.l2:
      return 'L2';
    case Metric.innerProduct:
      return 'inner product';
  }
}

String _kindLabel(FaissIndexKind kind) {
  switch (kind) {
    case FaissIndexKind.floatIndex:
      return 'float index';
    case FaissIndexKind.binaryIndex:
      return 'binary index';
    case FaissIndexKind.tunedWrapper:
      return 'tuned wrapper (IxDT + inner FAISS blob)';
    case FaissIndexKind.unknown:
      return 'unknown';
  }
}

String _metricLabel(FaissIndexInfo info) {
  final m = info.metric;
  if (m != null) {
    switch (m) {
      case Metric.l2:
        return info.kind == FaissIndexKind.binaryIndex
            ? 'L2 (binary indexes always report L2; distance is Hamming)'
            : 'L2';
      case Metric.innerProduct:
        return 'inner product';
    }
  }
  return 'unrecognized (non-L2/IP metric code)';
}

/// Compact human-readable byte count (base-1024).
String _humanBytes(int n) {
  if (n < 1024) return '$n B';
  const units = ['KiB', 'MiB', 'GiB', 'TiB'];
  var v = n / 1024.0;
  var u = 0;
  while (v >= 1024 && u < units.length - 1) {
    v /= 1024.0;
    u++;
  }
  return '${v.toStringAsFixed(v >= 100 ? 0 : (v >= 10 ? 1 : 2))} ${units[u]}';
}
