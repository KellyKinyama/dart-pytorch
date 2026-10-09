// M2 validation: parse an attention-body net and run the input embedding.
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
  if (!Lc0AttnWeights.isAttentionNet(path)) {
    print('not an attention-body net: $path');
    return;
  }
  final w = Lc0AttnReader.readFile(path);
  print('parsed: embDim=${w.embDim} heads=${w.heads} '
      'layers=${w.encoders.length} dff=${w.dff} wdl=${w.wdl}');
  print('  smolgen=${w.encoders.first.hasSmolgen} '
      'gating=${w.ipMultGate != null} '
      'smolgen_w=${w.smolgenW?.length ?? 0}');

  final input = Lc0Input.fromFen(startFen).toFloat32List(); // [112*64]
  final net = Lc0AttnNet(w);
  final emb = net.embed(input);

  var mn = double.infinity, mx = -double.infinity, sum = 0.0;
  var finite = true;
  for (final v in emb) {
    if (!v.isFinite) finite = false;
    if (v < mn) mn = v;
    if (v > mx) mx = v;
    sum += v;
  }
  print('embed: shape=[64,${w.embDim}] finite=$finite '
      'min=${mn.toStringAsFixed(3)} max=${mx.toStringAsFixed(3)} '
      'mean=${(sum / emb.length).toStringAsFixed(4)}');
  print(finite && emb.length == 64 * w.embDim ? 'M2-EMBED-OK' : 'M2-FAIL');

  // Show first square's first 8 embedding channels.
  final head = <String>[for (var i = 0; i < 8; i++) emb[i].toStringAsFixed(3)];
  print('sq0[0..7] = ${head.join(', ')}');
}
