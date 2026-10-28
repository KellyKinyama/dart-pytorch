/// Inspect a Cactus `.cact` archive (Needle 3 / Whistle weights): print the
/// header geometry, codebooks, and the tensor inventory by dtype. This is
/// the Dart equivalent of the 20-line reference parser in
/// https://cactuscompute.com/blog/cact-format and the first step of a Dart
/// port of Needle 3 — a loader to validate against their oracle before any
/// dequant or forward pass is written.
///
/// Get a `.cact` file first (Apache-2.0 tooling, weights on Hugging Face):
///
///   pip install cactus-needle
///   needle build --platform <your-platform>     # writes needle3.cact
///   # or download from https://huggingface.co/Cactus-Compute/needle3
///
/// Then:
///
///   dart run bin/needle_describe.dart path/to/needle3.cact
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:dart_pytorch/core/nn/needle/cact_format.dart';
import 'package:dart_pytorch/core/nn/needle/needle_config.dart';
import 'package:dart_pytorch/core/nn/needle/needle_weights.dart';

String _mb(int bytes) => '${(bytes / 1e6).toStringAsFixed(2)} MB';

void main(List<String> args) {
  final path = args.isNotEmpty ? args[0] : 'models/needle3/needle3.cact';
  final file = File(path);
  if (!file.existsSync()) {
    stderr.writeln('no such file: $path');
    stderr.writeln('pass a path to a .cact archive, e.g. '
        'dart run bin/needle_describe.dart models/needle3/needle3.cact');
    exitCode = 1;
    return;
  }

  final bytes = file.readAsBytesSync();
  print('file: $path');
  print('size: ${bytes.length} bytes (${_mb(bytes.length)})');

  final CactFile cact;
  try {
    cact = CactFile.parse(Uint8List.fromList(bytes));
  } on FormatException catch (e) {
    stderr.writeln('parse error: $e');
    exitCode = 1;
    return;
  }

  final h = cact.header;
  final gen = h.isNeedle3
      ? 'Needle 3'
      : (h.isNeedle2 ? 'Needle 2' : 'unknown');
  print('');
  print('=== header ===');
  print('format tag : 0x${h.formatTag.toRadixString(16)}  ($gen)');
  print('tensors    : ${h.numTensors}');
  print('codebookLen: ${h.codebookLen}');
  print('vocab      : ${h.vocab}');
  print('d_model    : ${h.dModel}');
  print('layers     : ${h.numLayers}');
  print('kv_bits    : ${h.kvBits}');
  print('rope_theta : ${h.ropeTheta}');

  print('');
  print('raw header fields (49 x u32, last is rope_theta as float):');
  for (var i = 0; i < h.rawFields.length; i += 8) {
    final end = (i + 8).clamp(0, h.rawFields.length);
    final chunk = h.rawFields.sublist(i, end).join(' ');
    print('  [${i.toString().padLeft(2)}] $chunk');
  }

  print('');
  print('=== codebooks (${cact.codebooks.length} floats) ===');
  final cb = cact.codebooks
      .take(16)
      .map((v) => v.toStringAsFixed(4))
      .join(' ');
  print('  $cb${cact.codebooks.length > 16 ? ' ...' : ''}');

  print('');
  print('=== tensor inventory ===');
  final counts = cact.countByKind();
  final kinds = cact.bytesByKind();
  final total = kinds.values.fold<int>(0, (a, b) => a + b);
  final sortedKinds = kinds.keys.toList()
    ..sort((a, b) => kinds[b]!.compareTo(kinds[a]!));
  for (final k in sortedKinds) {
    final pct = total == 0 ? 0.0 : 100.0 * kinds[k]! / total;
    print('  ${k.padRight(5)} ${counts[k].toString().padLeft(4)} tensors  '
        '${_mb(kinds[k]!).padLeft(10)}  ${pct.toStringAsFixed(1)}%');
  }
  print('  ----');
  print('  total ${cact.directory.length.toString().padLeft(4)} tensors  '
      '${_mb(total).padLeft(10)}');

  final tok = cact.tokenizerBlob;
  if (tok != null) {
    print('');
    print('tokenizer (RAW): ${_mb(tok.nbytes)} at offset ${tok.offset}');
  }

  print('');
  print('=== first tensors ===');
  for (final t in cact.directory.take(8)) {
    print('  $t');
  }

  // Validate the positional tensor-order contract and print the structure.
  print('');
  print('=== typed structure (contract check) ===');
  try {
    final w = NeedleWeights.map(cact);
    print('config : ${w.config}');
    print('blocks : ${w.blocks.length} x 27 tensors');
    print('mHC    : 9 tensors (${cact.header.numLayers}L x '
        '${w.config.numLanes} lanes)');
    print('engram : ${w.engram.length} sites ${w.engram.map((e) => e.site)}');
    print('heads  : ${w.heads.length} (manifest ${w.manifest.shape})');
    print('contract OK: 1 + ${w.blocks.length}x27 + 9 + 2 + '
        '${w.engram.length}x4 + 1 + ${1 + w.heads.length * 6} + 1 = '
        '${cact.directory.length} records');
  } on FormatException catch (e) {
    print('contract MISMATCH: $e');
  }
}
