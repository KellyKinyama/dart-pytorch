/// Cactus-Quant (CQ) dequantisation for `.cact` weights.
///
/// A matrix row is stored in groups of 128 weights. Each group is the
/// Walsh-Hadamard rotation of the original weights, split into an fp16
/// **norm** (its length) and a unit **direction** whose 128 coordinates are
/// each snapped to a Lloyd-Max scalar codebook of `2^bits` centroids. The
/// per-weight cost is `bits + 1/8` (2.125 at CQ2, 4.125 at CQ4).
///
/// Reconstruction is `w = (codebook[idx] · norm) · H`, and since the
/// normalised Walsh-Hadamard `H` is symmetric and orthonormal (`H·H = I`),
/// that is just `norm · fwht(direction)`.
///
/// The exact byte layout of indices vs norms within a row is pinned by the
/// exporter (`needle/model/export.py`); [CqLayout] captures the plausible
/// variants so a decoder can be matched to the oracle without touching the
/// maths. See https://cactuscompute.com/blog/cact-format.
library;

import 'dart:typed_data';

import 'hadamard.dart';

const int kCqGroup = 128;

/// Where a row keeps its per-group fp16 norms relative to its packed
/// index stream.
enum CqLayout {
  /// All groups' packed indices, then all groups' fp16 norms.
  indicesThenNorms,

  /// All groups' fp16 norms, then all groups' packed indices.
  normsThenIndices,
}

/// Slice the per-width scalar codebook out of the 28-float codebook blob.
/// CQ2 uses the first 4 centroids, CQ3 the next 8, CQ4 the last 16.
Float32List codebookForBits(Float32List all, int bits) {
  switch (bits) {
    case 2:
      return Float32List.sublistView(all, 0, 4);
    case 3:
      return Float32List.sublistView(all, 4, 12);
    case 4:
      return Float32List.sublistView(all, 12, 28);
    default:
      throw ArgumentError('unsupported CQ width: $bits bits');
  }
}

/// Decode a half-precision (IEEE 754 binary16) value stored in the low 16
/// bits of [h].
double fp16ToDouble(int h) {
  final sign = (h & 0x8000) != 0 ? -1.0 : 1.0;
  final exp = (h >> 10) & 0x1f;
  final frac = h & 0x3ff;
  if (exp == 0) {
    // subnormal / zero
    return sign * frac * 5.960464477539063e-8; // 2^-24
  }
  if (exp == 0x1f) {
    return frac == 0 ? sign * double.infinity : double.nan;
  }
  // normal: (1 + frac/1024) * 2^(exp-15)
  final m = 1.0 + frac / 1024.0;
  return sign * m * _pow2(exp - 15);
}

/// Encode [v] to IEEE 754 binary16 (round-to-nearest-even, no NaN/Inf
/// handling beyond saturation) — used by the round-trip test encoder.
int doubleToFp16(double v) {
  if (v == 0) return v.isNegative ? 0x8000 : 0x0000;
  var sign = 0;
  var x = v;
  if (x < 0) {
    sign = 0x8000;
    x = -x;
  }
  if (x.isNaN) return 0x7e00;
  if (x.isInfinite) return sign | 0x7c00;
  var exp = 0;
  while (x >= 2.0) {
    x /= 2.0;
    exp++;
  }
  while (x < 1.0) {
    x *= 2.0;
    exp--;
  }
  final e = exp + 15;
  if (e >= 0x1f) return sign | 0x7c00; // overflow -> inf
  if (e <= 0) return sign; // underflow -> 0
  final frac = ((x - 1.0) * 1024.0).round() & 0x3ff;
  return sign | (e << 10) | frac;
}

/// Read [count] little-endian fp16 values from [blob] into a float64 buffer.
Float64List readFp16Array(Uint8List blob, int count) {
  final out = Float64List(count);
  for (var i = 0; i < count; i++) {
    final o = i * 2;
    out[i] = fp16ToDouble(blob[o] | (blob[o + 1] << 8));
  }
  return out;
}

/// Read [count] little-endian fp32 values from [blob] into a float64 buffer.
Float64List readFp32Array(Uint8List blob, int count) {
  final view = ByteData.view(blob.buffer, blob.offsetInBytes, blob.length);
  final out = Float64List(count);
  for (var i = 0; i < count; i++) {
    out[i] = view.getFloat32(i * 4, Endian.little);
  }
  return out;
}

double _pow2(int e) {
  var r = 1.0;
  if (e >= 0) {
    for (var i = 0; i < e; i++) {
      r *= 2.0;
    }
  } else {
    for (var i = 0; i < -e; i++) {
      r /= 2.0;
    }
  }
  return r;
}

int _bytesPerRow(int inDim, int bits) {
  final groups = inDim ~/ kCqGroup;
  final idxBytes = (inDim * bits) ~/ 8; // packed indices, bits-per-weight
  final normBytes = groups * 2; // one fp16 per group
  return idxBytes + normBytes;
}

/// Dequantise one CQ row of [inDim] weights (a multiple of 128) from
/// [rowBytes], returning `inDim` reconstructed float32 weights.
Float32List dequantizeRow(
  Uint8List rowBytes,
  int inDim,
  int bits,
  Float32List codebook, {
  CqLayout layout = CqLayout.indicesThenNorms,
}) {
  if (inDim % kCqGroup != 0) {
    throw ArgumentError('inDim $inDim is not a multiple of $kCqGroup');
  }
  final groups = inDim ~/ kCqGroup;
  final idxBytes = (inDim * bits) ~/ 8;
  final idxBase = layout == CqLayout.indicesThenNorms ? 0 : groups * 2;
  final normBase = layout == CqLayout.indicesThenNorms ? idxBytes : 0;

  final out = Float32List(inDim);
  final mask = (1 << bits) - 1;
  final dir = Float64List(kCqGroup);

  for (var g = 0; g < groups; g++) {
    // fp16 norm for this group (little-endian)
    final nOff = normBase + g * 2;
    final norm = fp16ToDouble(rowBytes[nOff] | (rowBytes[nOff + 1] << 8));

    // unpack 128 bits-wide indices, LSB-first, into direction centroids
    final groupBitStart = g * kCqGroup * bits;
    for (var j = 0; j < kCqGroup; j++) {
      final bitPos = groupBitStart + j * bits;
      final bytePos = idxBase + (bitPos >> 3);
      final shift = bitPos & 7;
      // indices never straddle a byte for bits in {2,4} (8 % bits == 0),
      // but read two bytes to stay correct for bits == 3.
      final lo = rowBytes[bytePos];
      final hi = (bytePos + 1 < rowBytes.length) ? rowBytes[bytePos + 1] : 0;
      final idx = ((lo | (hi << 8)) >> shift) & mask;
      dir[j] = codebook[idx];
    }

    // w = norm * H·direction
    fwhtInPlace(dir);
    final base = g * kCqGroup;
    for (var j = 0; j < kCqGroup; j++) {
      out[base + j] = norm * dir[j];
    }
  }
  return out;
}

/// Dequantise a full `[outDim, inDim]` CQ matrix into a flat `outDim*inDim`
/// float32 buffer, using the **real `.cact` layout**: the blob is all packed
/// LSB-first indices for every row (`outDim * inPad*bits/8` bytes), then all
/// per-group fp16 L2 norms (`outDim * inPad/group`). `inDim` is padded up to a
/// multiple of [group]; padding columns are dropped. See `export.py`.
Float32List dequantizeMatrix(
  Uint8List blob,
  int outDim,
  int inDim,
  int bits,
  Float32List codebook, {
  int group = kCqGroup,
}) {
  final inPad = ((inDim + group - 1) ~/ group) * group;
  final groupsPerRow = inPad ~/ group;
  final idxBytesPerRow = (inPad * bits) ~/ 8;
  final normsBase = outDim * idxBytesPerRow;
  final out = Float32List(outDim * inDim);
  final mask = (1 << bits) - 1;
  final dir = Float64List(group);

  for (var r = 0; r < outDim; r++) {
    final idxRowBase = r * idxBytesPerRow;
    for (var g = 0; g < groupsPerRow; g++) {
      final nOff = normsBase + (r * groupsPerRow + g) * 2;
      final norm = fp16ToDouble(blob[nOff] | (blob[nOff + 1] << 8));
      final groupBitStart = g * group * bits;
      for (var j = 0; j < group; j++) {
        final bitPos = groupBitStart + j * bits;
        final bytePos = idxRowBase + (bitPos >> 3);
        final shift = bitPos & 7;
        final lo = blob[bytePos];
        final hi = (bytePos + 1 < blob.length) ? blob[bytePos + 1] : 0;
        dir[j] = codebook[((lo | (hi << 8)) >> shift) & mask];
      }
      fwhtInPlace(dir);
      final col0 = g * group;
      for (var j = 0; j < group; j++) {
        final col = col0 + j;
        if (col < inDim) out[r * inDim + col] = norm * dir[j];
      }
    }
  }
  return out;
}

/// Quantise a `[outDim, inDim]` matrix to the real `.cact` CQ layout
/// (indices block then fp16 norms block) — inverse of [dequantizeMatrix],
/// used to round-trip the real layout without a weights file.
Uint8List quantizeMatrix(
  Float64List w,
  int outDim,
  int inDim,
  int bits,
  Float32List codebook, {
  int group = kCqGroup,
}) {
  final inPad = ((inDim + group - 1) ~/ group) * group;
  final groupsPerRow = inPad ~/ group;
  final idxBytesPerRow = (inPad * bits) ~/ 8;
  final idxBlock = Uint8List(outDim * idxBytesPerRow);
  final normBlock = Uint8List(outDim * groupsPerRow * 2);
  final r = Float64List(group);

  for (var row = 0; row < outDim; row++) {
    final idxRowBase = row * idxBytesPerRow;
    for (var g = 0; g < groupsPerRow; g++) {
      for (var j = 0; j < group; j++) {
        final col = g * group + j;
        r[j] = col < inDim ? w[row * inDim + col] : 0.0;
      }
      fwhtInPlace(r);
      var norm = 0.0;
      for (var j = 0; j < group; j++) {
        norm += r[j] * r[j];
      }
      norm = norm <= 0 ? 0.0 : _sqrt(norm);
      final nOff = (row * groupsPerRow + g) * 2;
      final hn = doubleToFp16(norm);
      normBlock[nOff] = hn & 0xff;
      normBlock[nOff + 1] = (hn >> 8) & 0xff;
      for (var j = 0; j < group; j++) {
        final d = norm == 0 ? 0.0 : r[j] / norm;
        var bestIdx = 0;
        var bestErr = double.infinity;
        for (var c = 0; c < codebook.length; c++) {
          final e = (codebook[c] - d).abs();
          if (e < bestErr) {
            bestErr = e;
            bestIdx = c;
          }
        }
        final bitPos = g * group * bits + j * bits;
        final bytePos = idxRowBase + (bitPos >> 3);
        final shift = bitPos & 7;
        idxBlock[bytePos] |= (bestIdx << shift) & 0xff;
        if (shift + bits > 8) {
          idxBlock[bytePos + 1] |= (bestIdx >> (8 - shift)) & 0xff;
        }
      }
    }
  }
  final out = Uint8List(idxBlock.length + normBlock.length);
  out.setRange(0, idxBlock.length, idxBlock);
  out.setRange(idxBlock.length, out.length, normBlock);
  return out;
}

/// Quantise one row of [inDim] float32 weights to CQ bytes — the inverse of
/// [dequantizeRow], used to round-trip-test the maths without a real file.
Uint8List quantizeRow(
  Float32List w,
  int bits,
  Float32List codebook, {
  CqLayout layout = CqLayout.indicesThenNorms,
}) {
  final inDim = w.length;
  if (inDim % kCqGroup != 0) {
    throw ArgumentError('inDim $inDim is not a multiple of $kCqGroup');
  }
  final groups = inDim ~/ kCqGroup;
  final idxBytes = (inDim * bits) ~/ 8;
  final bytes = Uint8List(_bytesPerRow(inDim, bits));
  final idxBase = layout == CqLayout.indicesThenNorms ? 0 : groups * 2;
  final normBase = layout == CqLayout.indicesThenNorms ? idxBytes : 0;

  for (var g = 0; g < groups; g++) {
    final base = g * kCqGroup;
    // rotate the group: r = H·w
    final r = Float64List(kCqGroup);
    for (var j = 0; j < kCqGroup; j++) {
      r[j] = w[base + j];
    }
    fwhtInPlace(r);
    // norm = |r|, direction = r / norm
    var norm = 0.0;
    for (var j = 0; j < kCqGroup; j++) {
      norm += r[j] * r[j];
    }
    norm = norm <= 0 ? 0.0 : _sqrt(norm);
    final nOff = normBase + g * 2;
    final hn = doubleToFp16(norm);
    bytes[nOff] = hn & 0xff;
    bytes[nOff + 1] = (hn >> 8) & 0xff;

    // snap each direction coordinate to the nearest codebook centroid
    for (var j = 0; j < kCqGroup; j++) {
      final d = norm == 0 ? 0.0 : r[j] / norm;
      var bestIdx = 0;
      var bestErr = double.infinity;
      for (var c = 0; c < codebook.length; c++) {
        final e = (codebook[c] - d).abs();
        if (e < bestErr) {
          bestErr = e;
          bestIdx = c;
        }
      }
      final bitPos = g * kCqGroup * bits + j * bits;
      final bytePos = idxBase + (bitPos >> 3);
      final shift = bitPos & 7;
      bytes[bytePos] |= (bestIdx << shift) & 0xff;
      if (shift + bits > 8) {
        bytes[bytePos + 1] |= (bestIdx >> (8 - shift)) & 0xff;
      }
    }
  }
  return bytes;
}

double _sqrt(double x) {
  // avoid importing dart:math just for sqrt in this hot file
  if (x <= 0) return 0.0;
  var g = x;
  for (var i = 0; i < 40; i++) {
    g = 0.5 * (g + x / g);
  }
  return g;
}
