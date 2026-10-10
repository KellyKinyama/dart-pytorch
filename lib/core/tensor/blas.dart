/// OpenBLAS FFI bindings for the CPU matmul fast path.
///
/// [Blas.sgemm] wraps `cblas_sgemm` from the system-installed OpenBLAS
/// (`libopenblas.so.0`). Falls back gracefully when the library isn't
/// present or the caller sets `DART_PYTORCH_NO_BLAS=1` — in either
/// case [Blas.enabled] returns false and callers should use the
/// Dart matmul path.
///
/// On WSL 2 with `libopenblas-pthread` 0.3.26, this gives roughly a
/// **20-50× speedup** on the shapes we hit in Qwen1.5-MoE forward
/// (`[T, D=2048] @ [D, hidden=1408..151936]`) compared to the
/// hand-unrolled Dart triple loop.
library;

import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

// cblas_sgemm signature — see `<cblas.h>`. All shape args are Int32.
typedef _CblasSgemmNative =
    Void Function(
      Int32 order,
      Int32 transA,
      Int32 transB,
      Int32 m,
      Int32 n,
      Int32 k,
      Float alpha,
      Pointer<Float> a,
      Int32 lda,
      Pointer<Float> b,
      Int32 ldb,
      Float beta,
      Pointer<Float> c,
      Int32 ldc,
    );
typedef _CblasSgemmDart =
    void Function(
      int order,
      int transA,
      int transB,
      int m,
      int n,
      int k,
      double alpha,
      Pointer<Float> a,
      int lda,
      Pointer<Float> b,
      int ldb,
      double beta,
      Pointer<Float> c,
      int ldc,
    );

// ILP64 variant (OpenBLAS built with INTERFACE64, e.g. numpy/scipy's
// `scipy_openblas64_`): dimension args are 64-bit; the CBLAS enums stay int.
typedef _CblasSgemm64Native =
    Void Function(
      Int64 order,
      Int64 transA,
      Int64 transB,
      Int64 m,
      Int64 n,
      Int64 k,
      Float alpha,
      Pointer<Float> a,
      Int64 lda,
      Pointer<Float> b,
      Int64 ldb,
      Float beta,
      Pointer<Float> c,
      Int64 ldc,
    );
typedef _CblasSgemm64Dart =
    void Function(
      int order,
      int transA,
      int transB,
      int m,
      int n,
      int k,
      double alpha,
      Pointer<Float> a,
      int lda,
      Pointer<Float> b,
      int ldb,
      double beta,
      Pointer<Float> c,
      int ldc,
    );

const int _cblasRowMajor = 101;
const int _cblasNoTrans = 111;

class Blas {
  static bool _initialised = false;
  static _CblasSgemmDart? _sgemm; // LP64 (32-bit dims)
  static _CblasSgemm64Dart? _sgemm64; // ILP64 (64-bit dims)

  /// True when OpenBLAS was found and loaded. Lazy-initialised on
  /// first access; safe to poll from hot loops.
  static bool get enabled {
    _init();
    return _sgemm != null || _sgemm64 != null;
  }

  /// Human-readable status string for diagnostics — reports whether
  /// OpenBLAS is wired up, and the reason if not.
  static String status() {
    _init();
    if (_sgemm64 != null) return 'OpenBLAS: enabled (cblas_sgemm64_ ILP64)';
    if (_sgemm != null) return 'OpenBLAS: enabled (cblas_sgemm LP64)';
    if (Platform.environment.containsKey('DART_PYTORCH_NO_BLAS')) {
      return 'OpenBLAS: disabled by DART_PYTORCH_NO_BLAS env var';
    }
    return 'OpenBLAS: not loaded (set DART_PYTORCH_BLAS to an openblas dll/so)';
  }

  static void _init() {
    if (_initialised) return;
    _initialised = true;
    if (Platform.environment.containsKey('DART_PYTORCH_NO_BLAS')) return;
    final candidates = <String>[
      if (Platform.environment['DART_PYTORCH_BLAS'] != null)
        Platform.environment['DART_PYTORCH_BLAS']!,
      'libopenblas.dll',
      'openblas.dll',
      'libopenblas.so.0',
      'libopenblas.so',
      'libblas.so.3',
    ];
    for (final name in candidates) {
      try {
        final lib = DynamicLibrary.open(name);
        // Try ILP64 first (numpy/scipy ship an int64 OpenBLAS exporting the
        // plain `cblas_sgemm` name), then LP64. The self-test picks the one
        // whose ABI actually matches this build.
        try {
          _sgemm64 = lib
              .lookup<NativeFunction<_CblasSgemm64Native>>('cblas_sgemm')
              .asFunction();
          _setThreads(lib);
          if (_selfTest()) return;
        } catch (_) {}
        _sgemm64 = null;
        try {
          _sgemm = lib
              .lookup<NativeFunction<_CblasSgemmNative>>('cblas_sgemm')
              .asFunction();
          _setThreads(lib);
          if (_selfTest()) return;
        } catch (_) {}
        _sgemm = null;
      } catch (_) {
        // try next candidate
      }
    }
  }

  static void _setThreads(DynamicLibrary lib) {
    if (Platform.environment.containsKey('OPENBLAS_NUM_THREADS')) return;
    // Default to 1: dart_pytorch is typically used across worker isolates, so
    // multi-threaded BLAS per isolate would oversubscribe the CPU.
    for (final sym in const [
      'openblas_set_num_threads64_',
      'openblas_set_num_threads',
      'goto_set_num_threads',
    ]) {
      try {
        lib
            .lookup<NativeFunction<Void Function(Int32)>>(sym)
            .asFunction<void Function(int)>()(1);
        return;
      } catch (_) {}
    }
  }

  /// Verifies the FFI ABI with a tiny known product; guards against an
  /// ILP64/LP64 mismatch silently producing garbage.
  static bool _selfTest() {
    final a = Float32List.fromList([1, 2, 3, 4]); // [[1,2],[3,4]]
    final b = Float32List.fromList([5, 6, 7, 8]); // [[5,6],[7,8]]
    final c = Float32List(4);
    try {
      sgemm(a, b, c, 2, 2, 2);
    } catch (_) {
      return false;
    }
    // Expected [[19,22],[43,50]].
    return (c[0] - 19).abs() < 1e-3 &&
        (c[1] - 22).abs() < 1e-3 &&
        (c[2] - 43).abs() < 1e-3 &&
        (c[3] - 50).abs() < 1e-3;
  }

  /// `C = A * B` where A is `[m, k]`, B is `[k, n]`, C is `[m, n]`,
  /// all row-major fp32. Assumes [enabled] is true; caller is
  /// responsible for the length checks. Copies through native
  /// buffers because Dart FFI can't hand a raw pointer to a
  /// `Float32List`'s backing store portably. Buffers are drawn from
  /// a per-isolate reusable pool ([_pool]) so we don't `malloc`+`free`
  /// on every call.
  static void sgemm(
    Float32List a,
    Float32List b,
    Float32List c,
    int m,
    int k,
    int n,
  ) {
    final aPtr = _pool.grab(m * k);
    final bPtr = _pool.grab(k * n);
    final cPtr = _pool.grab(m * n);
    try {
      aPtr.asTypedList(m * k).setAll(0, a);
      bPtr.asTypedList(k * n).setAll(0, b);
      if (_sgemm64 != null) {
        _sgemm64!(
          _cblasRowMajor,
          _cblasNoTrans,
          _cblasNoTrans,
          m,
          n,
          k,
          1.0,
          aPtr,
          k,
          bPtr,
          n,
          0.0,
          cPtr,
          n,
        );
      } else {
        _sgemm!(
          _cblasRowMajor,
          _cblasNoTrans,
          _cblasNoTrans,
          m,
          n,
          k,
          1.0,
          aPtr,
          k,
          bPtr,
          n,
          0.0,
          cPtr,
          n,
        );
      }
      c.setAll(0, cPtr.asTypedList(m * n));
    } finally {
      _pool.release(aPtr, m * k);
      _pool.release(bPtr, k * n);
      _pool.release(cPtr, m * n);
    }
  }

  static final _BufPool _pool = _BufPool();
}

/// Very small "one power-of-two slot per size class" pool over
/// native fp32 buffers. sgemm calls with the same shape (which
/// dominates in transformer forward loops) keep hitting the same
/// slot; the buffer is reused without touching `malloc`/`free`.
class _BufPool {
  final Map<int, Pointer<Float>> _slots = {};

  Pointer<Float> grab(int n) {
    final key = _bucketFor(n);
    final cached = _slots.remove(key);
    if (cached != null) return cached;
    return malloc.allocate<Float>(key * sizeOf<Float>());
  }

  void release(Pointer<Float> p, int n) {
    final key = _bucketFor(n);
    final existing = _slots[key];
    if (existing == null) {
      _slots[key] = p;
    } else {
      malloc.free(p);
    }
  }

  static int _bucketFor(int n) {
    var b = 1;
    while (b < n) {
      b <<= 1;
    }
    return b;
  }
}
