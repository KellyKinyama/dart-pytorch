/// Strip the port-specific `IxDT` tuning-metadata wrapper off one or
/// more FAISS blobs, leaving the raw inner index bytes so the result
/// can be read by upstream `faiss::read_index` (or any other FAISS
/// consumer that doesn't know about `IxDT`).
///
/// The tuning block is discarded, not moved — if you need to preserve
/// it, run `faiss_describe` first and save the human-readable output
/// alongside the stripped file.
///
/// Run:
///
///     # Write next to the input as `foo.stripped.faiss`.
///     dart run bin/faiss_strip.dart tuned.faiss
///
///     # Single input, explicit output path.
///     dart run bin/faiss_strip.dart -o plain.faiss tuned.faiss
///
///     # Batch: strip several files in place-adjacent mode.
///     dart run bin/faiss_strip.dart a.faiss b.faiss c.faiss
///
/// Exit codes:
///   0 — every file stripped successfully
///   1 — usage error
///   2 — one or more files could not be stripped
library;

import 'dart:io';

import 'package:dart_pytorch/dart_pytorch.dart';

const String _usage =
    'Usage: dart run bin/faiss_strip.dart [-o OUTPUT] <file> [<file> ...]';

int main(List<String> args) {
  if (args.isEmpty || args.contains('-h') || args.contains('--help')) {
    stderr.writeln(_usage);
    return args.isEmpty ? 1 : 0;
  }

  String? explicitOut;
  final inputs = <String>[];
  for (var i = 0; i < args.length; i++) {
    final a = args[i];
    if (a == '-o' || a == '--output') {
      if (i + 1 >= args.length) {
        stderr.writeln('faiss_strip: $a needs a value\n$_usage');
        return 1;
      }
      explicitOut = args[++i];
    } else {
      inputs.add(a);
    }
  }
  if (inputs.isEmpty) {
    stderr.writeln('faiss_strip: no input files\n$_usage');
    return 1;
  }
  if (explicitOut != null && inputs.length != 1) {
    stderr.writeln(
      'faiss_strip: -o only works with a single input file '
      '(got ${inputs.length})\n$_usage',
    );
    return 1;
  }

  var failures = 0;
  for (final input in inputs) {
    final outPath = explicitOut ?? _adjacentPath(input);
    if (!_strip(input, outPath)) failures++;
  }
  return failures == 0 ? 0 : 2;
}

/// Build a `foo.stripped.faiss` sibling next to [input]. Preserves the
/// original extension so downstream tooling that keys off `.faiss` /
/// `.index` keeps working.
String _adjacentPath(String input) {
  final dot = input.lastIndexOf('.');
  if (dot < 0 || dot < input.lastIndexOf(Platform.pathSeparator)) {
    return '$input.stripped';
  }
  return '${input.substring(0, dot)}.stripped${input.substring(dot)}';
}

bool _strip(String inPath, String outPath) {
  final file = File(inPath);
  if (!file.existsSync()) {
    stderr.writeln('$inPath: no such file');
    return false;
  }
  try {
    stripTuningWrapperFile(inPath, outPath);
  } on FormatException catch (e) {
    stderr.writeln('$inPath: ${e.message}');
    return false;
  } on FileSystemException catch (e) {
    stderr.writeln('$inPath: ${e.message}');
    return false;
  }
  stdout.writeln('$inPath  ->  $outPath');
  return true;
}
