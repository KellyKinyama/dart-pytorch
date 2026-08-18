import 'dart:io';
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'dsp.dart'; // For hzToMel

/// Maps a normalised value [0..1] through matplotlib's 'magma' colormap.
img.ColorRgb8 _magma(double t) {
  const stops = [
    [0,   0,   4  ],
    [28,  16,  68 ],
    [79,  18,  123],
    [129, 37,  129],
    [181, 54,  122],
    [229, 89,  104],
    [251, 149, 131],
    [252, 253, 191],
  ];
  final scaled = t.clamp(0.0, 1.0) * (stops.length - 1);
  final lo     = scaled.floor().clamp(0, stops.length - 2);
  final hi     = lo + 1;
  final frac   = scaled - lo;
  int lerp(int a, int b) => (a + (b - a) * frac).round().clamp(0, 255);
  return img.ColorRgb8(
    lerp(stops[lo][0], stops[hi][0]),
    lerp(stops[lo][1], stops[hi][1]),
    lerp(stops[lo][2], stops[hi][2]),
  );
}

void _drawTick(
  img.Image canvas,
  int x, int y,
  String label, {
  bool vertical = false,
  bool right    = false,
  img.Color? color,
}) {
  final c = color ?? img.ColorRgb8(200, 200, 200);
  const tickLen = 5;

  if (vertical) {
    for (int dx = -tickLen; dx <= 0; dx++) {
      canvas.setPixel(x + dx, y, c);
    }
  } else if (right) {
    for (int dx = 0; dx <= tickLen; dx++) {
      canvas.setPixel(x + dx, y, c);
    }
  } else {
    for (int dy = 0; dy <= tickLen; dy++) {
      canvas.setPixel(x, y + dy, c);
    }
  }

  img.drawString(
    canvas,
    label,
    font: img.arial14,
    x: vertical ? x - tickLen - label.length * 8 - 2
                : right ? x + tickLen + 3
                        : x - label.length * 4,
    y: vertical ? y - 7
                : right ? y - 7
                        : y + tickLen + 3,
    color: c,
  );
}

/// Renders a Mel spectrogram in dB to a PNG image file.
///
/// This generates a visualization similar to `librosa.display.specshow`.
///
/// [spectrogramDb]: The spectrogram in dB (rows x frames).
/// [outputPath]: The path to save the .png file.
Future<void> saveSpectrogramImage(
  List<Float64List> spectrogramDb,
  String outputPath, {
  required int sampleRate,
  required int hopLength,
  double fMin    = 0.0,
  double? fMax,
  double topDb   = 80.0,
  String title   = 'Spectrogram',
  int plotWidth  = 800,
  int plotHeight = 256,
  bool useNearest = true,
  bool isMel      = true,
}) async {
  final nRows   = spectrogramDb.length;
  final nFrames = spectrogramDb.isEmpty ? 0 : spectrogramDb[0].length;
  if (nFrames == 0) throw ArgumentError('Empty spectrogram.');
  if (sampleRate <= 0) throw ArgumentError('Sample rate must be positive.');

  fMax ??= sampleRate / 2.0;

  const marginLeft   = 70;
  const marginRight  = 90;
  const marginTop    = 40;
  const marginBottom = 50;
  const colorbarW    = 18;
  const colorbarGap  = 10;

  final canvasW = marginLeft + plotWidth  + colorbarGap + colorbarW + marginRight;
  final canvasH = marginTop  + plotHeight + marginBottom;

  final canvas = img.Image(width: canvasW, height: canvasH);
  final bgColor = img.ColorRgb8(255, 255, 255);
  img.fill(canvas, color: bgColor);

  final dbFloor = -topDb.abs();

  for (int px = 0; px < plotWidth; px++) {
    final frameF = px / (plotWidth - 1) * (nFrames - 1);
    final f0     = frameF.floor().clamp(0, nFrames - 1);
    final f1     = (f0 + 1).clamp(0, nFrames - 1);
    final frac   = frameF - f0;

    for (int py = 0; py < plotHeight; py++) {
      final rowF = (1.0 - py / (plotHeight - 1)) * (nRows - 1);
      final r0   = rowF.floor().clamp(0, nRows - 1);
      final r1   = (r0 + 1).clamp(0, nRows - 1);
      final rFrac = rowF - r0;

      double v;
      if (useNearest) {
        v = spectrogramDb[r0][f0];
      } else {
        v = spectrogramDb[r0][f0] * (1 - rFrac) * (1 - frac) +
            spectrogramDb[r1][f0] * rFrac       * (1 - frac) +
            spectrogramDb[r0][f1] * (1 - rFrac) * frac        +
            spectrogramDb[r1][f1] * rFrac       * frac;
      }

      final t = ((v - dbFloor) / topDb).clamp(0.0, 1.0);
      canvas.setPixel(marginLeft + px, marginTop + py, _magma(t));
    }
  }

  final axisColor = img.ColorRgb8(0, 0, 0);
  for (int x = marginLeft; x < marginLeft + plotWidth; x++) {
    canvas.setPixel(x, marginTop + plotHeight, axisColor);
  }
  for (int y = marginTop; y <= marginTop + plotHeight; y++) {
    canvas.setPixel(marginLeft, y, axisColor);
  }

  final totalTime = nFrames * hopLength / sampleRate;
  final approxTicks = 8;
  final rawInterval = totalTime / approxTicks;
  final niceIntervals = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1.0, 2.0, 5.0];
  double tickInterval = niceIntervals.last;
  for (final iv in niceIntervals) {
    if (iv >= rawInterval) { tickInterval = iv; break; }
  }

  double t = 0.0;
  while (t <= totalTime + 1e-9) {
    final px = ((t / totalTime) * (plotWidth - 1)).round().clamp(0, plotWidth - 1);
    final label = t == t.truncateToDouble() ? t.toStringAsFixed(0) : t.toStringAsFixed(2);
    _drawTick(canvas, marginLeft + px, marginTop + plotHeight, label, color: axisColor);
    t = ((t * 1e6) + (tickInterval * 1e6)).round() / 1e6;
  }

  img.drawString(canvas, 'Time', font: img.arial14, x: marginLeft + plotWidth ~/ 2 - 16, y: canvasH - 18, color: axisColor);

  final hzLandmarks = <double>[];
  if (fMin == 0.0) hzLandmarks.add(0.0);
  double hz = 512.0;
  while (hz <= (fMax)) {
    if (hz >= fMin) hzLandmarks.add(hz);
    hz *= 2;
  }

  final yMin = isMel ? hzToMel(fMin) : fMin;
  final yMax = isMel ? hzToMel(fMax) : fMax;

  for (final hzVal in hzLandmarks) {
    final yVal   = isMel ? hzToMel(hzVal) : hzVal;
    final norm   = (yVal - yMin) / (yMax - yMin);
    final py     = marginTop + plotHeight - (norm * (plotHeight - 1)).round();
    if (py >= marginTop && py <= marginTop + plotHeight) {
      final label = hzVal >= 1000 ? '${(hzVal / 1).round()}' : '${hzVal.round()}';
      _drawTick(canvas, marginLeft, py, label, vertical: true, color: axisColor);
    }
  }

  img.drawString(canvas, 'Hz', font: img.arial14, x: 4, y: marginTop + plotHeight ~/ 2 - 7, color: axisColor);

  final cbX = marginLeft + plotWidth + colorbarGap;
  for (int py = 0; py < plotHeight; py++) {
    final t = 1.0 - py / (plotHeight - 1);
    final c = _magma(t);
    for (int dx = 0; dx < colorbarW; dx++) {
      canvas.setPixel(cbX + dx, marginTop + py, c);
    }
  }

  for (int py = marginTop; py <= marginTop + plotHeight; py++) {
    canvas.setPixel(cbX,              py, axisColor);
    canvas.setPixel(cbX + colorbarW, py, axisColor);
  }
  canvas.setPixel(cbX, marginTop,              axisColor);
  canvas.setPixel(cbX, marginTop + plotHeight, axisColor);

  final dbStep = topDb <= 40 ? 10.0 : 20.0;
  for (double db = 0; db >= -topDb; db -= dbStep) {
    final norm = (db - (-topDb)) / topDb;
    final py   = marginTop + plotHeight - (norm * (plotHeight - 1)).round();
    final label = db == 0 ? '+0 dB' : '${db.toInt()} dB';
    _drawTick(canvas, cbX + colorbarW, py, label, right: true, color: axisColor);
  }

  img.drawString(canvas, title,
    font: img.arial14,
    x: marginLeft + plotWidth ~/ 2 - title.length * 4,
    y: 12,
    color: axisColor,
  );

  final pngBytes = img.encodePng(canvas);
  await File(outputPath).writeAsBytes(pngBytes);
}