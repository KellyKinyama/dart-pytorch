/// Debug tool: dump the header + first bytes of a `.nnue` file so we
/// can eyeball the version tag, description string, and section
/// framing during the NNUE port.
///
/// Usage:
///   dart run bin/_nnue_probe.dart [path]
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:dart_pytorch/core/nn/nnue_proto.dart';

void main(List<String> args) {
  final path = args.isNotEmpty
      ? args[0]
      : 'models/stockfish/nn-5af11540bbfe.nnue';
  final bytes = File(path).readAsBytesSync();
  print('file: $path');
  print(
    'size: ${bytes.length} bytes '
    '(${(bytes.length / (1024 * 1024)).toStringAsFixed(1)} MB)',
  );

  final view = ByteData.view(bytes.buffer, bytes.offsetInBytes, bytes.length);
  final v = view.getUint32(0, Endian.little);
  final h = view.getUint32(4, Endian.little);
  final dLen = view.getUint32(8, Endian.little);
  print('raw header:');
  print('  version   = 0x${v.toRadixString(16)}');
  print('  hashValue = 0x${h.toRadixString(16)}');
  print('  descLen   = $dLen');
  final descEnd = 12 + dLen;
  if (descEnd < bytes.length) {
    final desc = String.fromCharCodes(bytes.sublist(12, descEnd));
    print('  description = "$desc"');
  }

  try {
    final header = NnueReader.parseHeader(bytes);
    print('parsed header: $header');
    print('decoding feature transformer …');
    final raw = NnueReader.parse(bytes);
    final ft = raw.featureTransformer;
    print('FT section hash = 0x${ft.sectionHash.toRadixString(16)}');
    print(
      'FT dims: numInputs=${ft.numInputs}, ftDim=${ft.ftDim}, '
      'psqtBuckets=${ft.psqtBuckets}',
    );
    _stats('biases (int16)', ft.biasesI16);
    _stats('weights (int16)', ft.weightsI16);
    _stats('psqt (int32)', ft.psqtI32);

    print('FT section ended at byte ${ft.byteEnd} / ${bytes.length} '
        '(${bytes.length - ft.byteEnd} bytes remain for network body)');

    print('decoding network body …');
    final net = raw.network;
    print('${net.buckets.length} buckets (no outer section hash — '
        'each bucket carries its own):');
    for (int b = 0; b < net.buckets.length; b++) {
      final bk = net.buckets[b];
      final l1min = bk.l1WeightsI8.reduce((a, b) => a < b ? a : b);
      final l1max = bk.l1WeightsI8.reduce((a, b) => a > b ? a : b);
      final l1bMin = bk.l1BiasesI32.reduce((a, b) => a < b ? a : b);
      final l1bMax = bk.l1BiasesI32.reduce((a, b) => a > b ? a : b);
      final l3b = bk.l3BiasesI32[0];
      print('  bucket $b: hash=0x${bk.bucketHash.toRadixString(16).padLeft(8, '0')} '
          'L1 w∈[$l1min,$l1max] b∈[$l1bMin,$l1bMax] L3 bias=$l3b');
    }
    print('network body ended at byte ${net.byteEnd} / ${bytes.length} '
        '(${bytes.length - net.byteEnd} unread)');
  } catch (e, st) {
    print('parse failed: $e');
    print(st);
  }
}

void _stats(String label, List<int> xs) {
  int mn = 0x7fffffff;
  int mx = -0x80000000;
  double sum = 0;
  int nonzero = 0;
  for (final v in xs) {
    if (v < mn) mn = v;
    if (v > mx) mx = v;
    sum += v;
    if (v != 0) nonzero++;
  }
  final mean = sum / xs.length;
  print(
    '  $label: n=${xs.length}, min=$mn, max=$mx, mean=${mean.toStringAsFixed(3)}, nonzero=$nonzero',
  );
}
