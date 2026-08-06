/// Tensor storage dtypes and codec helpers.
///
/// `dart-pytorch` compute is always fp32 — every kernel (CPU and GPU)
/// reads and writes `Float32List`. This file adds a **storage-only**
/// fp16 path: a [Tensor] may hold its host bytes as `Uint16List` of
/// IEEE-754 half-precision bits (`DType.fp16`), and ops materialise a
/// fp32 copy at read time via `Tensor._readAsFp32()`.
///
/// The purpose is to halve the resident memory of large read-only
/// **weight** tensors (e.g. Llama-3.1-8B), where the load-time fp32
/// promotion is what pushes them off the machine. Activation tensors
/// stay fp32 throughout.
///
/// Semantics:
///
///   * fp16 tensors are **read-only for autograd**. They cannot be
///     leaves with `requiresGrad = true`, cannot be assigned into, and
///     cannot be optimizer-updated in place. Backward through an fp16
///     leaf therefore just skips accumulating a grad (there is no
///     `Adam.step` path that could apply it).
///   * fp16 tensors are **CPU-only** for now. Moving an fp16 tensor
///     to GPU (`.to(Device.GPU)`) materialises fp32 first.
///   * Every CPU op that participates in an fp16 forward pass must
///     read via `_readAsFp32()` rather than `_cpuData!`. Ops that
///     haven't been ported yet will hit a null-check on `_cpuData`
///     and throw with a clear message.
library;

import 'dart:typed_data';

/// Storage precision of a [Tensor]. Compute is always fp32.
enum DType {
  /// 32-bit IEEE-754. Default. Backing is `Float32List`.
  fp32,

  /// 16-bit IEEE-754 (half). Backing is `Uint16List` of raw bits.
  /// Ops decode to fp32 at read time.
  fp16;

  /// Bytes per stored element.
  int get itemBytes => switch (this) {
    DType.fp32 => 4,
    DType.fp16 => 2,
  };
}

/// IEEE-754 half-precision → single-precision decode. Handles
/// subnormals, ±0, ±Inf and NaN. Bit-identical to
/// `torch.Tensor.to(torch.float32)` on the same input.
double fp16BitsToFp32(int bits) {
  final sign = (bits >> 15) & 0x1;
  final exp = (bits >> 10) & 0x1f;
  final mant = bits & 0x3ff;
  final int sign32 = sign << 31;
  int exp32;
  int mant32;
  if (exp == 0) {
    if (mant == 0) {
      return _u32ToFp32(sign32); // ±0
    }
    // Subnormal — normalise.
    var m = mant;
    var e = 1;
    while ((m & 0x400) == 0) {
      m <<= 1;
      e -= 1;
    }
    m &= 0x3ff;
    exp32 = (127 - 15 + e) << 23;
    mant32 = m << 13;
  } else if (exp == 0x1f) {
    exp32 = 0xff << 23; // Inf / NaN
    mant32 = mant << 13;
  } else {
    exp32 = (exp - 15 + 127) << 23;
    mant32 = mant << 13;
  }
  return _u32ToFp32(sign32 | exp32 | mant32);
}

/// Single-precision → IEEE-754 half-precision encode. Uses
/// round-to-nearest-even. Values outside fp16 dynamic range are
/// saturated to ±Inf; NaN is preserved as a canonical fp16 NaN.
int fp32ToFp16Bits(double value) {
  final u = _fp32ToU32(value);
  final sign = (u >> 31) & 0x1;
  final exp = (u >> 23) & 0xff;
  final mant = u & 0x7fffff;
  final int sign16 = sign << 15;

  if (exp == 0xff) {
    // Inf or NaN.
    if (mant == 0) return sign16 | 0x7c00; // ±Inf
    return sign16 | 0x7e00; // canonical NaN
  }

  final signed = exp - 127; // unbiased fp32 exponent
  if (signed >= 16) {
    // Overflow — saturate to ±Inf.
    return sign16 | 0x7c00;
  }
  if (signed >= -14) {
    // Normal fp16 range.
    final exp16 = (signed + 15) & 0x1f;
    // Round-to-nearest-even on the 13 discarded low bits.
    final mant16 = _roundHalfToEven(mant, 13);
    if (mant16 == 0x400) {
      // Rounded mantissa carried into the exponent.
      if (exp16 + 1 >= 0x1f) return sign16 | 0x7c00;
      return sign16 | ((exp16 + 1) << 10);
    }
    return sign16 | (exp16 << 10) | mant16;
  }
  // Subnormal fp16 (or underflow to zero).
  final shift = -14 - signed; // 1..24
  if (shift > 24) return sign16; // underflow to ±0
  final mantWithImplicit = mant | 0x800000;
  final mant16 = _roundHalfToEven(mantWithImplicit, 13 + shift);
  return sign16 | mant16;
}

/// Bulk decode `Uint16List` (fp16 bits) into a freshly-allocated
/// `Float32List`. This is the hot path used by
/// `Tensor._readAsFp32()` on fp16 tensors.
Float32List decodeFp16Bulk(Uint16List bits) {
  final out = Float32List(bits.length);
  for (int i = 0; i < bits.length; i++) {
    out[i] = fp16BitsToFp32(bits[i]);
  }
  return out;
}

/// Bulk encode `Float32List` values into a freshly-allocated
/// `Uint16List` of fp16 bits. Used by `Tensor.toFp16()`.
Uint16List encodeFp16Bulk(Float32List vals) {
  final out = Uint16List(vals.length);
  for (int i = 0; i < vals.length; i++) {
    out[i] = fp32ToFp16Bits(vals[i]);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Internal bit tricks.
// ---------------------------------------------------------------------------

final ByteData _scratch = ByteData(4);

double _u32ToFp32(int bits) {
  _scratch.setUint32(0, bits & 0xffffffff, Endian.little);
  return _scratch.getFloat32(0, Endian.little);
}

int _fp32ToU32(double v) {
  _scratch.setFloat32(0, v, Endian.little);
  return _scratch.getUint32(0, Endian.little);
}

/// Round `x` right by `shift` bits, breaking ties to nearest-even.
int _roundHalfToEven(int x, int shift) {
  if (shift <= 0) return x;
  final half = 1 << (shift - 1);
  final mask = (1 << shift) - 1;
  final low = x & mask;
  final trunc = x >> shift;
  if (low < half) return trunc;
  if (low > half) return trunc + 1;
  // Exactly half — round to even.
  return (trunc & 1) == 0 ? trunc : trunc + 1;
}
