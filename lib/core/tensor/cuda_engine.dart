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
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Int32,
      ffi.Int32,
    );
typedef DSliceFwd =
    ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, int, int);

// slice_top_left_backward: (gOut, R, C) -> gIn (freshly-allocated zeroed).
typedef CSliceBwd = CSliceFwd;
typedef DSliceBwd = DSliceFwd;

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

  // ReLU backward (fwd is a plain `reluTensor` above).
  late DReluBwd reluBackwardOp;

  CudaEngine() {
    _lib = ffi.DynamicLibrary.open(
      '${Directory.current.path}/native/lib/libmat_mul.so',
    );

    createTensor = _lib.lookupFunction<CCreate, DCreate>('create_tensor');
    destroyTensor = _lib.lookupFunction<CDestroy, DDestroy>('destroy_tensor');
    getTensorData = _lib.lookupFunction<CCopy, DCopy>('get_tensor_data');

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

    aftFullForward = _lib.lookupFunction<CAftFwd, DAftFwd>(
      'aft_full_forward',
    );
    aftFullBackward = _lib.lookupFunction<CAftBwd, DAftBwd>(
      'aft_full_backward',
    );

    sliceTopLeftForward = _lib.lookupFunction<CSliceFwd, DSliceFwd>(
      'slice_top_left_forward',
    );
    sliceTopLeftBackward = _lib.lookupFunction<CSliceBwd, DSliceBwd>(
      'slice_top_left_backward',
    );

    reluBackwardOp = _lib.lookupFunction<CReluBwd, DReluBwd>(
      'relu_backward_op',
    );
  }
}

final engine = CudaEngine();
