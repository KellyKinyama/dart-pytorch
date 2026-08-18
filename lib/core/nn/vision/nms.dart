/// Non-maximum suppression + IoU helpers used by MTCNN and other
/// object detectors. All boxes are `[x1, y1, x2, y2]` in image coords.
library;

import 'dart:math' as math;

double _iou(List<double> a, List<double> b, {bool minOverlap = false}) {
  final xx1 = math.max(a[0], b[0]);
  final yy1 = math.max(a[1], b[1]);
  final xx2 = math.min(a[2], b[2]);
  final yy2 = math.min(a[3], b[3]);
  final iw = math.max(0.0, xx2 - xx1);
  final ih = math.max(0.0, yy2 - yy1);
  final inter = iw * ih;
  if (inter == 0.0) return 0.0;
  final areaA = (a[2] - a[0]) * (a[3] - a[1]);
  final areaB = (b[2] - b[0]) * (b[3] - b[1]);
  if (minOverlap) {
    // MTCNN "Min" variant: IoU = inter / min(areaA, areaB).
    final denom = math.min(areaA, areaB);
    return denom <= 0 ? 0.0 : inter / denom;
  }
  final denom = areaA + areaB - inter;
  return denom <= 0 ? 0.0 : inter / denom;
}

/// Standard IoU-based NMS. Returns indices into [boxes] that survive.
///
/// [boxes] is `List<[x1, y1, x2, y2]>`; [scores] parallel List<double>.
/// [method] = `'Union'` (standard IoU) or `'Min'` (MTCNN post-ONet).
List<int> nms(
  List<List<double>> boxes,
  List<double> scores, {
  required double threshold,
  String method = 'Union',
}) {
  if (boxes.length != scores.length) {
    throw ArgumentError('nms: boxes and scores length mismatch');
  }
  if (boxes.isEmpty) return const [];
  final order = List<int>.generate(boxes.length, (i) => i)
    ..sort((a, b) => scores[b].compareTo(scores[a]));

  final kept = <int>[];
  final suppressed = List<bool>.filled(boxes.length, false);
  final minOverlap = method == 'Min';
  for (final i in order) {
    if (suppressed[i]) continue;
    kept.add(i);
    for (final j in order) {
      if (j == i || suppressed[j]) continue;
      final iou = _iou(boxes[i], boxes[j], minOverlap: minOverlap);
      if (iou > threshold) {
        suppressed[j] = true;
      }
    }
  }
  return kept;
}
