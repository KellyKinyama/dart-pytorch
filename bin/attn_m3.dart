// M3 validation: run the full attention encoder stack (embedding + 10 layers
// with smolgen) and check the body output is finite and well-scaled.
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
  final input = Lc0Input.fromFen(startFen).toFloat32List();

  final sw = Stopwatch()..start();
  final body = net.encode(input);
  sw.stop();

  var mn = double.infinity, mx = -double.infinity, sum = 0.0, sumSq = 0.0;
  var finite = true;
  for (final v in body) {
    if (!v.isFinite) finite = false;
    if (v < mn) mn = v;
    if (v > mx) mx = v;
    sum += v;
    sumSq += v * v;
  }
  final n = body.length;
  final mean = sum / n;
  final std = (sumSq / n - mean * mean);
  print('body: shape=[64,${w.embDim}] finite=$finite '
      'min=${mn.toStringAsFixed(3)} max=${mx.toStringAsFixed(3)} '
      'mean=${mean.toStringAsFixed(4)} var=${std.toStringAsFixed(4)} '
      '(${sw.elapsedMilliseconds} ms)');
  print(finite && n == 64 * w.embDim ? 'M3-ENCODE-OK' : 'M3-FAIL');
  final head = <String>[for (var i = 0; i < 8; i++) body[i].toStringAsFixed(3)];
  print('sq0[0..7] = ${head.join(', ')}');
}
