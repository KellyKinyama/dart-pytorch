/// End-to-end Stockfish NNUE evaluation demo.
///
/// Loads `nn-5af11540bbfe.nnue` (SF16 default net), encodes a FEN into
/// HalfKAv2_hm sparse features, runs the float forward pass, and
/// prints the centipawn eval and internals.
///
/// Prerequisites (one-time weights download, ~40 MB):
///
///   mkdir -p models/stockfish
///   curl -sL -o models/stockfish/nn-5af11540bbfe.nnue \
///     https://tests.stockfishchess.org/api/nn/nn-5af11540bbfe.nnue
///
/// Usage:
///
///   dart run bin/nnue_demo.dart
///   dart run bin/nnue_demo.dart \
///     'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
library;

import 'dart:io';

import 'package:dart_pytorch/dart_pytorch.dart';

const String _defaultNetPath = 'models/stockfish/nn-5af11540bbfe.nnue';

const String _defaultFen =
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

Future<void> main(List<String> args) async {
  final fen = args.isNotEmpty ? args[0] : _defaultFen;
  final netPath = args.length > 1 ? args[1] : _defaultNetPath;

  if (!File(netPath).existsSync()) {
    stderr.writeln('missing $netPath — download with:');
    stderr.writeln('  mkdir -p models/stockfish');
    stderr.writeln('  curl -sL -o $netPath \\');
    stderr.writeln('    '
        'https://tests.stockfishchess.org/api/nn/nn-5af11540bbfe.nnue');
    exit(64);
  }

  stdout.writeln('Loading $netPath …');
  final loadStart = DateTime.now();
  final raw = NnueReader.loadFile(netPath);
  stdout.writeln('  ${raw.header}');
  stdout.writeln('  FT: numInputs=${raw.featureTransformer.numInputs}, '
      'ftDim=${raw.featureTransformer.ftDim}, '
      'psqtBuckets=${raw.featureTransformer.psqtBuckets}');
  stdout.writeln('  ${raw.network.buckets.length} network buckets');

  final net = NnueNet.fromRaw(raw);
  final loadMs = DateTime.now().difference(loadStart).inMilliseconds;
  stdout.writeln('Dequantised → floats in ${loadMs} ms');

  stdout.writeln('');
  stdout.writeln('Position:');
  stdout.writeln('  $fen');
  final features = encodeFen(fen);
  stdout.writeln('  side to move: ${features.stm.name}');
  stdout.writeln('  active features: white=${features.whiteActive.length}, '
      'black=${features.blackActive.length}');
  stdout.writeln('  piece count: ${features.pieceCount} '
      '(bucket ${(features.pieceCount - 1) >> 2})');

  final r = net.evaluate(features);
  stdout.writeln('');
  stdout.writeln('Evaluation:');
  stdout.writeln('  ${r.cp.toStringAsFixed(1)} cp '
      '(${_cpBar(r.cp)} for ${features.stm.name})');
  stdout.writeln('  bucket ......... ${r.bucket}');
  stdout.writeln('  L3 output ...... ${r.rawL3Output.toStringAsFixed(2)} cp');
  stdout.writeln('  PSQT ........... '
      '${r.psqtContribution.toStringAsFixed(2)} cp');
}

String _cpBar(double cp) {
  if (cp.abs() < 30) return 'balanced';
  if (cp > 0) return '+${cp.toStringAsFixed(0)} STM';
  return '${cp.toStringAsFixed(0)} STM';
}
