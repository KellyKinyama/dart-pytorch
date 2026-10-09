// Batched-forward validation + throughput: compares forwardBatch against
// per-position forward (must match) and times single vs batched evaluation.
import 'dart:io';
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

void main(List<String> args) {
  final path = args.isNotEmpty
      ? args[0]
      : r'C:\projects\chess\models\lc0\t1-256x10-distilled.pb.gz';
  if (!File(path).existsSync()) {
    stderr.writeln('missing $path');
    exit(64);
  }
  final w = Lc0AttnReader.readFile(path);
  final net = Lc0AttnNet(w);

  const fens = [
    startFen,
    'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1',
    'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2',
    'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3',
  ];
  final inputs = [
    for (final f in fens) Lc0Input.fromFen(f).toFloat32List()
  ];

  // Correctness: batched vs per-position.
  final single = [for (final i in inputs) net.forward(i)];
  final batched = net.forwardBatch(inputs);
  var maxPolDiff = 0.0, maxWdlDiff = 0.0, maxMlhDiff = 0.0;
  for (var b = 0; b < inputs.length; b++) {
    for (var i = 0; i < single[b].policy.length; i++) {
      final d = (single[b].policy[i] - batched[b].policy[i]).abs();
      if (d > maxPolDiff) maxPolDiff = d;
    }
    for (var i = 0; i < 3; i++) {
      final d = (single[b].wdl[i] - batched[b].wdl[i]).abs();
      if (d > maxWdlDiff) maxWdlDiff = d;
    }
    final d = (single[b].movesLeft - batched[b].movesLeft).abs();
    if (d > maxMlhDiff) maxMlhDiff = d;
  }
  print('batch==single: dPolicy=${maxPolDiff.toStringAsExponential(2)} '
      'dWdl=${maxWdlDiff.toStringAsExponential(2)} '
      'dMlh=${maxMlhDiff.toStringAsExponential(2)}');

  // Throughput: B per-position forwards vs one forwardBatch of B.
  final big = <Float32List>[for (var i = 0; i < 16; i++) inputs[i % inputs.length]];
  final B = big.length;

  // warmup
  net.forward(big[0]);
  net.forwardBatch(big.sublist(0, 2));

  final sw1 = Stopwatch()..start();
  for (final inp in big) {
    net.forward(inp);
  }
  sw1.stop();

  final sw2 = Stopwatch()..start();
  net.forwardBatch(big);
  sw2.stop();

  final perSingle = sw1.elapsedMilliseconds / B;
  final perBatch = sw2.elapsedMilliseconds / B;
  print('single: ${sw1.elapsedMilliseconds} ms for $B '
      '(${perSingle.toStringAsFixed(1)} ms/pos)');
  print('batch : ${sw2.elapsedMilliseconds} ms for $B '
      '(${perBatch.toStringAsFixed(1)} ms/pos)  '
      'speedup ${(perSingle / perBatch).toStringAsFixed(2)}x');
}
