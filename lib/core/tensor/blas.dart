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

const int _cblasRowMajor = 101;
const int _cblasNoTrans = 111;

class Blas {
  static bool _initialised = false;
  static _CblasSgemmDart? _sgemm;

  /// True when OpenBLAS was found and loaded. Lazy-initialised on
  /// first access; safe to poll from hot loops.
  static bool get enabled {
    _init();
    return _sgemm != null;
  }

  /// Human-readable status string for diagnostics — reports whether
  /// OpenBLAS is wired up, and the reason if not.
  static String status() {
    _init();
    if (_sgemm != null) return 'OpenBLAS: enabled (cblas_sgemm loaded)';
    if (Platform.environment.containsKey('DART_PYTORCH_NO_BLAS')) {
      return 'OpenBLAS: disabled by DART_PYTORCH_NO_BLAS env var';
    }
    return 'OpenBLAS: not loaded (libopenblas.so.0 not found)';
  }

  static void _init() {
    if (_initialised) return;
    _initialised = true;
    if (Platform.environment.containsKey('DART_PYTORCH_NO_BLAS')) return;
    for (final name in const [
      'libopenblas.so.0',
      'libopenblas.so',
      'libblas.so.3',
    ]) {
      try {
        final lib = DynamicLibrary.open(name);
        _sgemm = lib
            .lookup<NativeFunction<_CblasSgemmNative>>('cblas_sgemm')
            .asFunction();
        // OpenBLAS defaults to all CPUs, which oversubscribes WSL.
        // Cap at a reasonable default when the user hasn't asked for
        // a specific count.
        if (!Platform.environment.containsKey('OPENBLAS_NUM_THREADS')) {
          try {
            final setNumThreads = lib
                .lookup<NativeFunction<Void Function(Int32)>>(
                    'openblas_set_num_threads')
                .asFunction<void Function(int)>();
            final cpus = Platform.numberOfProcessors;
            final want = cpus >= 8 ? 4 : (cpus >= 4 ? 2 : 1);
            setNumThreads(want);
          } catch (_) {
            // openblas_set_num_threads absent (generic BLAS build) — ok
          }
        }
        return;
      } catch (_) {
        // try next
      }
    }
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
