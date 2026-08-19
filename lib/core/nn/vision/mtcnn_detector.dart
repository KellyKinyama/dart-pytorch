/// MTCNN detection pipeline: raw RGB image bytes (via `package:image`)
/// -> list of detected faces with box + 5 landmarks + probability.
///
/// Implements the classic three-stage cascade of Zhang et al. 2016
/// (as ported by facenet-pytorch), on top of our own P/R/O nets.
///
/// See `bin/facenet/detect.dart` for a runnable example that saves
/// annotated JPEGs, and `bin/facenet/identify_photo.dart` for a
/// raw-photo → aligned crop → InceptionResnetV1 lookup demo.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart' as img;

import '../../tensor/tensor.dart';
import 'mtcnn.dart';
import 'nms.dart';

class DetectedFace {
  /// `[x1, y1, x2, y2]` in original-image pixel coords.
  final List<double> box;

  /// Face probability from ONet (0-1).
  final double prob;

  /// 5 landmarks in pixel coords, each `[x, y]`.
  /// Order: `[leftEye, rightEye, nose, leftMouth, rightMouth]`.
  final List<List<double>> landmarks;

  DetectedFace({
    required this.box,
    required this.prob,
    required this.landmarks,
  });

  @override
  String toString() =>
      'DetectedFace(box=[${box.map((v) => v.toStringAsFixed(1)).join(", ")}], '
      'prob=${prob.toStringAsFixed(3)}, landmarks=${landmarks.length})';
}

/// Full three-stage detector.
class MTCNN {
  final PNet pnet;
  final RNet rnet;
  final ONet onet;

  /// Minimum face size in pixels (min side of face box).
  final int minFaceSize;

  /// Image-pyramid scale factor (default 0.709).
  final double scaleFactor;

  /// Per-stage face-probability thresholds.
  final List<double> thresholds;

  /// Per-stage NMS thresholds.
  final List<double> nmsThresholds;

  MTCNN({
    required this.pnet,
    required this.rnet,
    required this.onet,
    this.minFaceSize = 20,
    this.scaleFactor = 0.709,
    List<double>? thresholds,
    List<double>? nmsThresholds,
  }) : thresholds = thresholds ?? const [0.6, 0.7, 0.7],
       nmsThresholds = nmsThresholds ?? const [0.7, 0.7, 0.7];

  /// Detect faces in [image]. Returns an empty list if none pass the
  /// ONet threshold.
  List<DetectedFace> detect(img.Image image) {
    final h = image.height;
    final w = image.width;
    final rgb = _imageToNCHW(image);

    // ---- image pyramid ----
    final scales = <double>[];
    var m = 12.0 / minFaceSize;
    var minl = math.min(h, w) * m;
    var scale = m;
    while (minl >= 12) {
      scales.add(scale);
      scale *= scaleFactor;
      minl *= scaleFactor;
    }

    // ---- stage 1: PNet at each scale ----
    var boxes = <_ScoredBox>[];
    for (final s in scales) {
      final scaledH = (h * s).ceil();
      final scaledW = (w * s).ceil();
      if (scaledH < 12 || scaledW < 12) continue;
      final resized = _resizeAndNormalize(rgb, h, w, scaledH, scaledW);
      final input = Tensor.fromFloat32List([1, 3, scaledH, scaledW], resized);
      final (probs, regs) = pnet(input);
      // probs shape [1, 2, Hp, Wp]; regs [1, 4, Hp, Wp].
      final scaledBoxes = _generatePNetBoxes(
        probs: probs,
        regs: regs,
        scale: s,
        threshold: thresholds[0],
      );
      final kept = nms(
        [for (final b in scaledBoxes) b.box],
        [for (final b in scaledBoxes) b.score],
        threshold: 0.5,
      );
      boxes.addAll([for (final i in kept) scaledBoxes[i]]);
    }
    if (boxes.isEmpty) return const [];

    // NMS across all scales.
    var kept = nms(
      [for (final b in boxes) b.box],
      [for (final b in boxes) b.score],
      threshold: nmsThresholds[0],
    );
    boxes = [for (final i in kept) boxes[i]];
    boxes = _applyBBoxReg(boxes);
    boxes = _rerec(boxes); // pad to square

    // ---- stage 2: RNet ----
    boxes = _runRNet(
      rgb: rgb,
      imgH: h,
      imgW: w,
      boxes: boxes,
      threshold: thresholds[1],
      nmsThreshold: nmsThresholds[1],
    );
    if (boxes.isEmpty) return const [];
    boxes = _applyBBoxReg(boxes);
    boxes = _rerec(boxes);

    // ---- stage 3: ONet ----
    final onetOut = _runONet(
      rgb: rgb,
      imgH: h,
      imgW: w,
      boxes: boxes,
      threshold: thresholds[2],
      nmsThreshold: nmsThresholds[2],
    );
    return onetOut;
  }

  // ---------------- helpers ----------------

  /// Extract [1, 3, H, W] float32 in `(x − 127.5) / 128` normalization
  /// (facenet-pytorch convention).
  Float32List _imageToNCHW(img.Image image) {
    final w = image.width;
    final h = image.height;
    final out = Float32List(3 * h * w);
    final chSize = h * w;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final px = image.getPixel(x, y);
        final off = y * w + x;
        out[off] = (px.r.toDouble() - 127.5) / 128.0;
        out[chSize + off] = (px.g.toDouble() - 127.5) / 128.0;
        out[2 * chSize + off] = (px.b.toDouble() - 127.5) / 128.0;
      }
    }
    return out;
  }

  /// Resize [src] (`[3, H, W]` normalized) to `outH x outW` via
  /// bilinear interpolation.
  Float32List _resizeAndNormalize(
    Float32List src,
    int inH,
    int inW,
    int outH,
    int outW,
  ) {
    final out = Float32List(3 * outH * outW);
    final chIn = inH * inW;
    final chOut = outH * outW;
    for (int c = 0; c < 3; c++) {
      for (int y = 0; y < outH; y++) {
        final srcY = (y + 0.5) * inH / outH - 0.5;
        final y0 = srcY.floor().clamp(0, inH - 1);
        final y1 = (y0 + 1).clamp(0, inH - 1);
        final wy = (srcY - y0).clamp(0.0, 1.0);
        for (int x = 0; x < outW; x++) {
          final srcX = (x + 0.5) * inW / outW - 0.5;
          final x0 = srcX.floor().clamp(0, inW - 1);
          final x1 = (x0 + 1).clamp(0, inW - 1);
          final wx = (srcX - x0).clamp(0.0, 1.0);
          final v00 = src[c * chIn + y0 * inW + x0];
          final v01 = src[c * chIn + y0 * inW + x1];
          final v10 = src[c * chIn + y1 * inW + x0];
          final v11 = src[c * chIn + y1 * inW + x1];
          final v0 = v00 * (1 - wx) + v01 * wx;
          final v1 = v10 * (1 - wx) + v11 * wx;
          out[c * chOut + y * outW + x] = v0 * (1 - wy) + v1 * wy;
        }
      }
    }
    return out;
  }

  List<_ScoredBox> _generatePNetBoxes({
    required Tensor probs,
    required Tensor regs,
    required double scale,
    required double threshold,
  }) {
    const stride = 2.0;
    const cellSize = 12.0;
    final ph = probs.shape[2];
    final pw = probs.shape[3];
    final probData = probs.toFloat32List();
    final regData = regs.toFloat32List();
    final out = <_ScoredBox>[];
    // face prob is channel 1 (channel 0 is background).
    final faceOff = 1 * ph * pw;
    for (int y = 0; y < ph; y++) {
      for (int x = 0; x < pw; x++) {
        final score = probData[faceOff + y * pw + x];
        if (score < threshold) continue;
        final x1 = (stride * x + 1) / scale;
        final y1 = (stride * y + 1) / scale;
        final x2 = (stride * x + cellSize) / scale;
        final y2 = (stride * y + cellSize) / scale;
        final dx1 = regData[0 * ph * pw + y * pw + x];
        final dy1 = regData[1 * ph * pw + y * pw + x];
        final dx2 = regData[2 * ph * pw + y * pw + x];
        final dy2 = regData[3 * ph * pw + y * pw + x];
        out.add(
          _ScoredBox(
            box: [x1, y1, x2, y2],
            score: score,
            reg: [dx1, dy1, dx2, dy2],
          ),
        );
      }
    }
    return out;
  }

  List<_ScoredBox> _applyBBoxReg(List<_ScoredBox> boxes) {
    return [
      for (final b in boxes)
        _ScoredBox(
          box: [
            b.box[0] + (b.box[2] - b.box[0]) * b.reg[0],
            b.box[1] + (b.box[3] - b.box[1]) * b.reg[1],
            b.box[2] + (b.box[2] - b.box[0]) * b.reg[2],
            b.box[3] + (b.box[3] - b.box[1]) * b.reg[3],
          ],
          score: b.score,
          reg: b.reg,
        ),
    ];
  }

  /// Convert to square boxes centered on the original center, with side
  /// equal to max(w, h). MTCNN's `rerec`.
  List<_ScoredBox> _rerec(List<_ScoredBox> boxes) {
    final out = <_ScoredBox>[];
    for (final b in boxes) {
      final w = b.box[2] - b.box[0];
      final h = b.box[3] - b.box[1];
      final side = math.max(w, h);
      final cx = b.box[0] + w * 0.5;
      final cy = b.box[1] + h * 0.5;
      out.add(
        _ScoredBox(
          box: [
            cx - side * 0.5,
            cy - side * 0.5,
            cx + side * 0.5,
            cy + side * 0.5,
          ],
          score: b.score,
          reg: b.reg,
        ),
      );
    }
    return out;
  }

  /// Crop each box out of the image, resize to [size]×[size], and
  /// stack into a batched tensor.
  Tensor _batchCropAndResize({
    required Float32List rgb,
    required int imgH,
    required int imgW,
    required List<_ScoredBox> boxes,
    required int size,
  }) {
    final n = boxes.length;
    final out = Float32List(n * 3 * size * size);
    final chIn = imgH * imgW;
    final chOut = size * size;
    for (int b = 0; b < n; b++) {
      final box = boxes[b].box;
      final x1 = box[0];
      final y1 = box[1];
      final x2 = box[2];
      final y2 = box[3];
      final boxW = x2 - x1;
      final boxH = y2 - y1;
      // Bilinear resize from the (possibly out-of-bounds) box area
      // into `size x size`. Pixels outside the source image become 0
      // (matches facenet-pytorch's zero padding).
      for (int c = 0; c < 3; c++) {
        for (int oy = 0; oy < size; oy++) {
          final srcY = y1 + (oy + 0.5) * boxH / size - 0.5;
          for (int ox = 0; ox < size; ox++) {
            final srcX = x1 + (ox + 0.5) * boxW / size - 0.5;
            final ix0 = srcX.floor();
            final iy0 = srcY.floor();
            final ix1 = ix0 + 1;
            final iy1 = iy0 + 1;
            final wx = (srcX - ix0).clamp(0.0, 1.0);
            final wy = (srcY - iy0).clamp(0.0, 1.0);
            double sample(int y, int x) {
              if (x < 0 || x >= imgW || y < 0 || y >= imgH) return 0.0;
              return rgb[c * chIn + y * imgW + x];
            }

            final v00 = sample(iy0, ix0);
            final v01 = sample(iy0, ix1);
            final v10 = sample(iy1, ix0);
            final v11 = sample(iy1, ix1);
            final v0 = v00 * (1 - wx) + v01 * wx;
            final v1 = v10 * (1 - wx) + v11 * wx;
            out[b * 3 * chOut + c * chOut + oy * size + ox] =
                v0 * (1 - wy) + v1 * wy;
          }
        }
      }
    }
    return Tensor.fromFloat32List([n, 3, size, size], out);
  }

  List<_ScoredBox> _runRNet({
    required Float32List rgb,
    required int imgH,
    required int imgW,
    required List<_ScoredBox> boxes,
    required double threshold,
    required double nmsThreshold,
  }) {
    final input = _batchCropAndResize(
      rgb: rgb,
      imgH: imgH,
      imgW: imgW,
      boxes: boxes,
      size: 24,
    );
    final (probs, regs) = rnet(input);
    final probData = probs.toFloat32List();
    final regData = regs.toFloat32List();
    final kept = <_ScoredBox>[];
    for (int i = 0; i < boxes.length; i++) {
      final score = probData[i * 2 + 1]; // face prob
      if (score < threshold) continue;
      kept.add(
        _ScoredBox(
          box: List<double>.from(boxes[i].box),
          score: score,
          reg: [
            regData[i * 4],
            regData[i * 4 + 1],
            regData[i * 4 + 2],
            regData[i * 4 + 3],
          ],
        ),
      );
    }
    if (kept.isEmpty) return const [];
    final surv = nms(
      [for (final b in kept) b.box],
      [for (final b in kept) b.score],
      threshold: nmsThreshold,
    );
    return [for (final i in surv) kept[i]];
  }

  List<DetectedFace> _runONet({
    required Float32List rgb,
    required int imgH,
    required int imgW,
    required List<_ScoredBox> boxes,
    required double threshold,
    required double nmsThreshold,
  }) {
    final input = _batchCropAndResize(
      rgb: rgb,
      imgH: imgH,
      imgW: imgW,
      boxes: boxes,
      size: 48,
    );
    final (probs, regs, landmarks) = onet(input);
    final probData = probs.toFloat32List();
    final regData = regs.toFloat32List();
    final landData = landmarks.toFloat32List();
    final kept = <_ScoredBox>[];
    final keptLand = <List<double>>[];
    for (int i = 0; i < boxes.length; i++) {
      final score = probData[i * 2 + 1];
      if (score < threshold) continue;
      // ONet landmarks are 5 x-coords then 5 y-coords, all in
      // [0, 1] relative to the crop box.
      final box = boxes[i].box;
      final w = box[2] - box[0];
      final h = box[3] - box[1];
      final pts = <double>[];
      for (int p = 0; p < 5; p++) {
        pts.add(box[0] + w * landData[i * 10 + p]);
        pts.add(box[1] + h * landData[i * 10 + 5 + p]);
      }
      kept.add(
        _ScoredBox(
          box: List<double>.from(box),
          score: score,
          reg: [
            regData[i * 4],
            regData[i * 4 + 1],
            regData[i * 4 + 2],
            regData[i * 4 + 3],
          ],
        ),
      );
      keptLand.add(pts);
    }
    if (kept.isEmpty) return const [];

    // Apply bbox reg then NMS-Min.
    final regd = _applyBBoxReg(kept);
    final surv = nms(
      [for (final b in regd) b.box],
      [for (final b in regd) b.score],
      threshold: nmsThreshold,
      method: 'Min',
    );

    final results = <DetectedFace>[];
    for (final i in surv) {
      final ptsFlat = keptLand[i];
      final lm = <List<double>>[];
      for (int p = 0; p < 5; p++) {
        lm.add([ptsFlat[p * 2], ptsFlat[p * 2 + 1]]);
      }
      results.add(
        DetectedFace(box: regd[i].box, prob: regd[i].score, landmarks: lm),
      );
    }
    return results;
  }
}

class _ScoredBox {
  final List<double> box; // [x1, y1, x2, y2]
  final double score;
  final List<double> reg; // [dx1, dy1, dx2, dy2]
  _ScoredBox({required this.box, required this.score, required this.reg});
}
