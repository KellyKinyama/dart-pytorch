/// Shared helpers for `bin/facenet/*` demos.
///
/// Kept as a private under-`_`-prefixed dart source rather than a
/// package export because Dart-scripts under `bin/` share sources
/// via relative import (see `bin/_gpt2_hf_api_common.dart` for the
/// same pattern). No public API here.
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:dart_pytorch/core/tensor/tensor.dart';
import 'package:image/image.dart' as img;

/// Decode a JPEG, resize to 160×160 (bilinear), apply
/// facenet-pytorch's `(x − 127.5) / 128` normalization, and upload
/// to [device]. Returns `[1, 3, 160, 160]` fp32.
Tensor decodeFaceJpeg(String path, {Device device = Device.CPU}) {
  final bytes = File(path).readAsBytesSync();
  final decoded = img.decodeImage(bytes);
  if (decoded == null) {
    throw StateError('could not decode $path');
  }
  final resized = img.copyResize(
    decoded,
    width: 160,
    height: 160,
    interpolation: img.Interpolation.linear,
  );
  final data = Float32List(3 * 160 * 160);
  const chSize = 160 * 160;
  for (int y = 0; y < 160; y++) {
    for (int x = 0; x < 160; x++) {
      final px = resized.getPixel(x, y);
      final off = y * 160 + x;
      data[off] = (px.r.toDouble() - 127.5) / 128.0;
      data[chSize + off] = (px.g.toDouble() - 127.5) / 128.0;
      data[2 * chSize + off] = (px.b.toDouble() - 127.5) / 128.0;
    }
  }
  return Tensor.fromFloat32List([1, 3, 160, 160], data, device: device);
}

/// Cosine similarity of two L2-normalized 512-d embeddings
/// (`sum(a[i] * b[i])`). No normalization here — assumes both are
/// already unit vectors, as our [`InceptionResnetV1`] output always is.
double cosine(List<double> a, List<double> b) {
  if (a.length != b.length) {
    throw ArgumentError('cosine: length mismatch ${a.length} vs ${b.length}');
  }
  double d = 0.0;
  for (int i = 0; i < a.length; i++) {
    d += a[i] * b[i];
  }
  return d;
}

/// Enumerate `{identityName: [file1, file2, ...]}` from a gallery
/// directory of the shape `gallery/{IdentityName}/*.jpg`. Every
/// per-identity list is sorted alphabetically, then optionally
/// truncated to [perId] samples.
Map<String, List<String>> scanGallery(String path, {int? perId}) {
  final out = <String, List<String>>{};
  for (final e in Directory(path).listSync()) {
    if (e is Directory) {
      final files = e.listSync().whereType<File>().map((f) => f.path).where((
        p,
      ) {
        final low = p.toLowerCase();
        return low.endsWith('.jpg') ||
            low.endsWith('.jpeg') ||
            low.endsWith('.png');
      }).toList()..sort();
      if (files.isNotEmpty) {
        final name = e.path.split(RegExp(r'[/\\]')).last;
        out[name] = perId == null ? files : files.take(perId).toList();
      }
    }
  }
  return out;
}

/// Short display label for a gallery path: `"IdentityName/file.jpg"`.
String prettyLabel(String path) {
  final parts = path.split(RegExp(r'[/\\]'));
  if (parts.length < 2) return path;
  return '${parts[parts.length - 2]}/${parts.last}';
}
