/// Parameter averaging for DiLoCo-style cooperative training.
///
/// Given N checkpoints produced by [Checkpoint.saveBytes] over the
/// *same* module architecture, [averageCheckpoints] returns a fresh
/// DPTC checkpoint whose parameter tensors are the (optionally weighted)
/// arithmetic mean of the sources.
///
/// All checkpoints must share an identical header (same version, same
/// parameter shapes in the same order). This is verified by comparing
/// header bytes byte-for-byte; a mismatch throws [ArgumentError].
///
/// Optional per-checkpoint [weights] let you weight by the number of
/// local training steps each peer performed (as in FedAvg), so
/// long-training peers pull the average further toward their solution.
/// Weights are renormalised to sum to 1; passing null means uniform.
library;

import 'dart:typed_data';

Uint8List averageCheckpoints(
  List<Uint8List> checkpoints, {
  List<double>? weights,
}) {
  if (checkpoints.isEmpty) {
    throw ArgumentError('averageCheckpoints: no inputs');
  }
  if (weights != null && weights.length != checkpoints.length) {
    throw ArgumentError(
      'averageCheckpoints: weights length ${weights.length} != '
      'checkpoints length ${checkpoints.length}',
    );
  }

  // Normalise weights (uniform if none provided).
  final ws = List<double>.filled(checkpoints.length, 0.0);
  if (weights == null) {
    for (var i = 0; i < ws.length; i++) {
      ws[i] = 1.0 / ws.length;
    }
  } else {
    var sum = 0.0;
    for (final w in weights) {
      if (w < 0) {
        throw ArgumentError('averageCheckpoints: negative weight $w');
      }
      sum += w;
    }
    if (sum <= 0) {
      throw ArgumentError('averageCheckpoints: weights sum to $sum');
    }
    for (var i = 0; i < ws.length; i++) {
      ws[i] = weights[i] / sum;
    }
  }

  // Parse the first checkpoint's header, then verify every subsequent
  // checkpoint has identical preamble+header bytes.
  final first = checkpoints[0];
  if (first.length < 12) {
    throw ArgumentError('averageCheckpoints: checkpoint 0 too small');
  }
  final view0 = ByteData.sublistView(first);
  // Magic.
  if (first[0] != 0x44 ||
      first[1] != 0x50 ||
      first[2] != 0x54 ||
      first[3] != 0x43) {
    throw ArgumentError('averageCheckpoints: bad DPTC magic');
  }
  final headerLen = view0.getUint32(8, Endian.little);
  final headerEnd = 12 + headerLen;
  if (first.length < headerEnd) {
    throw ArgumentError('averageCheckpoints: checkpoint 0 truncated header');
  }
  final expectedDataLen = first.length - headerEnd;
  if (expectedDataLen % 4 != 0) {
    throw ArgumentError(
      'averageCheckpoints: checkpoint 0 data blob not float32-aligned',
    );
  }
  final numScalars = expectedDataLen ~/ 4;
  final preambleAndHeader = first.sublist(0, headerEnd);

  for (var i = 1; i < checkpoints.length; i++) {
    final ci = checkpoints[i];
    if (ci.length != first.length) {
      throw ArgumentError(
        'averageCheckpoints: checkpoint $i has length ${ci.length} '
        'but checkpoint 0 has ${first.length}',
      );
    }
    for (var b = 0; b < headerEnd; b++) {
      if (ci[b] != preambleAndHeader[b]) {
        throw ArgumentError(
          'averageCheckpoints: checkpoint $i header differs at byte $b',
        );
      }
    }
  }

  // Weighted-average the data blob.
  final acc = Float32List(numScalars);
  for (var i = 0; i < checkpoints.length; i++) {
    final w = ws[i];
    final view = ByteData.sublistView(checkpoints[i]);
    var off = headerEnd;
    for (var k = 0; k < numScalars; k++) {
      acc[k] += view.getFloat32(off, Endian.little) * w;
      off += 4;
    }
  }

  // Emit fresh DPTC bytes: header verbatim, data replaced with `acc`.
  final out = Uint8List(headerEnd + numScalars * 4);
  out.setRange(0, headerEnd, preambleAndHeader);
  final outView = ByteData.sublistView(out);
  var off = headerEnd;
  for (var k = 0; k < numScalars; k++) {
    outView.setFloat32(off, acc[k], Endian.little);
    off += 4;
  }
  return out;
}
