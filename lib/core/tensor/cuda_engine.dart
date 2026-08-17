/// `dart:ffi` bindings for the CUDA shared library `libmat_mul.so`.
///
/// Exposes forward + backward symbols for matmul, elementwise ops,
/// activations, transpose, reductions, and LayerNorm.
library;

import 'dart:ffi' as ffi;
import 'dart:io';

// C-side (native) and D-side (Dart) FFI signatures.
typedef CCreate =
    ffi.Pointer<ffi.Void> Function(
      ffi.Int32,
      ffi.Int32,
      ffi.Pointer<ffi.Float>,
    );
typedef DCreate =
    ffi.Pointer<ffi.Void> Function(int, int, ffi.Pointer<ffi.Float>);

typedef CDestroy = ffi.Void Function(ffi.Pointer<ffi.Void>);
typedef DDestroy = void Function(ffi.Pointer<ffi.Void>);

typedef CCopy =
    ffi.Void Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Float>);
typedef DCopy = void Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Float>);

typedef COp1 = ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>);
typedef DOp1 = ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>);

typedef COp2 =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
    );
typedef DOp2 =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
    );

typedef CPow = ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, ffi.Float);
typedef DPow = ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, double);

// LayerNorm forward: (x, gamma, beta, eps) -> out
typedef CLnFwd =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Float,
    );
typedef DLnFwd =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      double,
    );

// LayerNorm backward: (x, gamma, gO, gGammaAccum, gBetaAccum, eps) -> gX
typedef CLnBwd =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Float,
    );
typedef DLnBwd =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      double,
    );

// RMSNorm forward: (x, gamma, eps) -> out
typedef CRmsFwd =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Float,
    );
typedef DRmsFwd =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      double,
    );

// RMSNorm backward: (x, gamma, gO, gGammaAccum, eps) -> gX
typedef CRmsBwd =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Float,
    );
typedef DRmsBwd =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      double,
    );

// Cross-entropy backward: (x, targets, gLoss) -> gX
typedef COp3 =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
    );
typedef DOp3 =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
    );

// Embedding forward: (table, indices, N) -> out
typedef CEmbFwd =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Int32,
    );
typedef DEmbFwd =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      int,
    );

// Embedding backward: (indices, gOut, gTableAccum, N) -> void
typedef CEmbBwd =
    ffi.Void Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Int32,
    );
typedef DEmbBwd =
    void Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      int,
    );

// AFT forward: (Q, K, V, WB, masked) -> out
typedef CAftFwd =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Int32,
    );
typedef DAftFwd =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      int,
    );

// AFT backward: (Q, K, V, WB, gOut, masked, gQ, gK, gV, gWB) -> void.
// gQ/gK/gV/gWB are caller-allocated zero-init tensors; the kernel
// atomicAdds analytical gradients into them.
typedef CAftBwd =
    ffi.Void Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Int32,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
    );
typedef DAftBwd =
    void Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      int,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
    );

// slice_top_left_forward: (x, rows, cols) -> out
typedef CSliceFwd =
    ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, ffi.Int32, ffi.Int32);
typedef DSliceFwd =
    ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, int, int);

// slice_top_left_backward: (gOut, R, C) -> gIn (freshly-allocated zeroed).
typedef CSliceBwd = CSliceFwd;
typedef DSliceBwd = DSliceFwd;

// im2col_nhwc_op: (input, batch, cin, k, pad) -> out
typedef CIm2colNhwc =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Int32,
      ffi.Int32,
      ffi.Int32,
      ffi.Int32,
    );
typedef DIm2colNhwc =
    ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, int, int, int, int);

// relu_backward_op: (a, gO, gA) -> void.
// gA is caller-allocated zero-init; kernel atomicAdds into it.
typedef CReluBwd =
    ffi.Void Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
    );
typedef DReluBwd =
    void Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
    );

class CudaEngine {
  late ffi.DynamicLibrary _lib;

  // Lifecycle + I/O
  late DCreate createTensor;
  late DDestroy destroyTensor;
  late DCopy getTensorData;

  /// Native finalizer that calls `destroy_tensor` on the attached
  /// pointer when the owning Dart object is garbage-collected. Used
  /// by `Tensor._gpu` to free intermediate GPU handles that the
  /// autograd graph doesn't explicitly dispose.
  late ffi.NativeFinalizer destroyTensorFinalizer;

  // Matmul
  late DOp2 matmulTensors;

  // Elementwise binary
  late DOp2 addTensors;
  late DOp2 subTensors;
  late DOp2 mulTensors;
  late DOp2 divTensors;

  // Scalar broadcast
  late DOp2 addTensorScalar;
  late DOp2 subTensorScalar;
  late DOp2 mulTensorScalar;
  late DOp2 divTensorScalar;

  // Row broadcast
  late DOp2 addTensorRowBroadcast;

  // Unary
  late DOp1 absTensor;
  late DOp1 logTensor;
  late DPow powTensor;
  late DOp1 reluTensor;
  late DOp1 sigmoidTensor;
  late DOp1 tanhTensor;

  // Rearrangement
  late DOp1 transposeTensor;

  // Reductions
  late DOp1 sumTensor;
  late DOp1 meanTensor;

  // LayerNorm
  late DLnFwd layernormForward;
  late DLnBwd layernormBackward;

  // RMSNorm
  late DRmsFwd rmsnormForward;
  late DRmsBwd rmsnormBackward;

  // Softmax
  late DOp1 softmaxForward;
  late DOp2 softmaxBackward;

  // Cross-entropy (fused with softmax)
  late DOp2 crossEntropyForward;
  late DOp3 crossEntropyBackward;

  // Embedding
  late DEmbFwd embeddingForward;
  late DEmbBwd embeddingBackward;

  // AFT-full
  late DAftFwd aftFullForward;
  late DAftBwd aftFullBackward;

  // sliceTopLeft
  late DSliceFwd sliceTopLeftForward;
  late DSliceBwd sliceTopLeftBackward;

  // im2col NHWC-flat for the LC0 conv tower.
  late DIm2colNhwc im2colNhwc;

  // ReLU backward (fwd is a plain `reluTensor` above).
  late DReluBwd reluBackwardOp;

  // abs backward: gA += sign(a) * gO. Fwd is `absTensor` above.
  late DReluBwd absBackwardOp;

  CudaEngine() {
    _lib = ffi.DynamicLibrary.open(_findNativeLib());

    createTensor = _lib.lookupFunction<CCreate, DCreate>('create_tensor');
    destroyTensor = _lib.lookupFunction<CDestroy, DDestroy>('destroy_tensor');
    getTensorData = _lib.lookupFunction<CCopy, DCopy>('get_tensor_data');

    // Same underlying symbol as `destroyTensor`, exposed as a native
    // function pointer so it can be attached as a NativeFinalizer.
    final destroyPtr = _lib.lookup<ffi.NativeFunction<CDestroy>>(
      'destroy_tensor',
    );
    destroyTensorFinalizer = ffi.NativeFinalizer(destroyPtr.cast());

    matmulTensors = _lib.lookupFunction<COp2, DOp2>('matmul_tensors');

    addTensors = _lib.lookupFunction<COp2, DOp2>('add_tensors');
    subTensors = _lib.lookupFunction<COp2, DOp2>('sub_tensors');
    mulTensors = _lib.lookupFunction<COp2, DOp2>('mul_tensors');
    divTensors = _lib.lookupFunction<COp2, DOp2>('div_tensors');

    addTensorScalar = _lib.lookupFunction<COp2, DOp2>('add_tensor_scalar');
    subTensorScalar = _lib.lookupFunction<COp2, DOp2>('sub_tensor_scalar');
    mulTensorScalar = _lib.lookupFunction<COp2, DOp2>('mul_tensor_scalar');
    divTensorScalar = _lib.lookupFunction<COp2, DOp2>('div_tensor_scalar');

    addTensorRowBroadcast = _lib.lookupFunction<COp2, DOp2>(
      'add_tensor_row_broadcast',
    );

    absTensor = _lib.lookupFunction<COp1, DOp1>('abs_tensor');
    logTensor = _lib.lookupFunction<COp1, DOp1>('log_tensor');
    powTensor = _lib.lookupFunction<CPow, DPow>('pow_tensor');
    reluTensor = _lib.lookupFunction<COp1, DOp1>('relu_tensor');
    sigmoidTensor = _lib.lookupFunction<COp1, DOp1>('sigmoid_tensor');
    tanhTensor = _lib.lookupFunction<COp1, DOp1>('tanh_tensor');

    transposeTensor = _lib.lookupFunction<COp1, DOp1>('transpose_tensor');

    sumTensor = _lib.lookupFunction<COp1, DOp1>('sum_tensor');
    meanTensor = _lib.lookupFunction<COp1, DOp1>('mean_tensor');

    layernormForward = _lib.lookupFunction<CLnFwd, DLnFwd>('layernorm_forward');
    layernormBackward = _lib.lookupFunction<CLnBwd, DLnBwd>(
      'layernorm_backward',
    );

    rmsnormForward = _lib.lookupFunction<CRmsFwd, DRmsFwd>('rmsnorm_forward');
    rmsnormBackward = _lib.lookupFunction<CRmsBwd, DRmsBwd>('rmsnorm_backward');

    softmaxForward = _lib.lookupFunction<COp1, DOp1>('softmax_forward');
    softmaxBackward = _lib.lookupFunction<COp2, DOp2>('softmax_backward');

    crossEntropyForward = _lib.lookupFunction<COp2, DOp2>(
      'cross_entropy_forward',
    );
    crossEntropyBackward = _lib.lookupFunction<COp3, DOp3>(
      'cross_entropy_backward',
    );

    embeddingForward = _lib.lookupFunction<CEmbFwd, DEmbFwd>(
      'embedding_forward',
    );
    embeddingBackward = _lib.lookupFunction<CEmbBwd, DEmbBwd>(
      'embedding_backward',
    );

    aftFullForward = _lib.lookupFunction<CAftFwd, DAftFwd>('aft_full_forward');
    aftFullBackward = _lib.lookupFunction<CAftBwd, DAftBwd>(
      'aft_full_backward',
    );

    sliceTopLeftForward = _lib.lookupFunction<CSliceFwd, DSliceFwd>(
      'slice_top_left_forward',
    );
    sliceTopLeftBackward = _lib.lookupFunction<CSliceBwd, DSliceBwd>(
      'slice_top_left_backward',
    );

    im2colNhwc = _lib.lookupFunction<CIm2colNhwc, DIm2colNhwc>(
      'im2col_nhwc_op',
    );

    reluBackwardOp = _lib.lookupFunction<CReluBwd, DReluBwd>(
      'relu_backward_op',
    );
    absBackwardOp = _lib.lookupFunction<CReluBwd, DReluBwd>('abs_backward_op');
  }
}

/// Resolves the platform-specific native library. Search order:
///   1. `DART_PYTORCH_NATIVE_LIB` env var (full path, wins outright).
///   2. `<cwd>/native/lib/<name>` — the in-repo build location or
///      where `ensureNativeLib()` writes prebuilt downloads.
///   3. `<executable dir>/native/lib/<name>` and `<executable dir>/<name>`
///      — for `dart compile exe` artifacts distributed alongside the lib.
///   4. Bare `<name>` — falls back to the OS loader search path
///      (LD_LIBRARY_PATH / PATH / DYLD_LIBRARY_PATH).
///
/// If none exist, returns the bare name so `DynamicLibrary.open` gets
/// to throw the familiar "cannot open shared object file" error. Users
/// on a pub.dev install should `await ensureNativeLib()` from `main`
/// before touching GPU tensors.
String _findNativeLib() {
  final envPath = Platform.environment['DART_PYTORCH_NATIVE_LIB'];
  if (envPath != null && envPath.isNotEmpty) return envPath;

  final String name;
  if (Platform.isWindows) {
    name = 'mat_mul.dll';
  } else if (Platform.isMacOS) {
    name = 'libmat_mul.dylib';
  } else {
    name = 'libmat_mul.so';
  }

  final candidates = <String>['${Directory.current.path}/native/lib/$name'];
  try {
    final exeDir = File(Platform.resolvedExecutable).parent.path;
    candidates
      ..add('$exeDir/native/lib/$name')
      ..add('$exeDir/$name');
  } catch (_) {
    // Platform.resolvedExecutable is unavailable in some embedders.
  }

  for (final c in candidates) {
    if (File(c).existsSync()) return c;
  }
  return name;
}

final engine = CudaEngine();
