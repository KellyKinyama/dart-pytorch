/// Device-aware `Tensor` with a CPU (`Float32List`) or GPU (FFI handle)
/// backing, plus a Dart-side reverse-mode autograd tape.
///
/// Exactly one of `_cpuData` / `_handle` is populated at any time; the
/// [device] field records which. Operations dispatch by device — see
/// `docs/device-placement.md` for the policy and per-op decisions.
///
/// Autograd: tensors with [requiresGrad] = true participate in the
/// backward graph. Every differentiable op sets `_backward` on its
/// output; calling [backward] on a scalar output walks the graph in
/// reverse topological order and accumulates gradients into each
/// leaf's [grad]. Gradient math is expressed via the same ops as the
/// forward pass, so backward works on CPU and GPU without any extra
/// per-device code.
library;

// ignore_for_file: constant_identifier_names, unused_element, unused_field

import 'dart:math' as math;
import 'dart:typed_data';

import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';
import 'cuda_engine.dart';
import 'dtype.dart';

part 'mat_mul.dart';
part 'ops.dart';
part 'layer_norm.dart';
part 'rms_norm.dart';
part 'softmax.dart';
part 'embedding.dart';
part 'attention.dart';
part 'dropout.dart';
part 'concat.dart';
part 'aft.dart';
part 'scatter.dart';

enum Device { CPU, GPU }

/// Debug hook fired after each backward closure. If non-null, called
/// with (stepIndex, tensor) after `tensor._backward()` runs. Use to
/// bisect which closure corrupted state.
void Function(int, Tensor)? debugBackwardHook;

class Tensor implements ffi.Finalizable {
  final List<int> shape;
  final int length;
  Device device;

  /// Storage precision. Compute is always fp32; when this is
  /// [DType.fp16] the CPU backing lives in [_cpuF16Bits] as raw
  /// half-precision bits and ops materialise a fp32 copy at read
  /// time via [_readAsFp32]. GPU storage is always fp32 for now, so
  /// fp16 tensors are CPU-only.
  DType dtype;

  // Backing store — exactly one of these is populated (dictated by
  // [device] and [dtype]):
  //   * fp32 CPU  → `_cpuData` non-null.
  //   * fp16 CPU  → `_cpuF16Bits` non-null.
  //   * GPU (any) → `_handle` non-null.
  Float32List? _cpuData;
  Uint16List? _cpuF16Bits;
  ffi.Pointer<ffi.Void>? _handle;

  // Autograd state.
  bool requiresGrad;
  Tensor? _grad;
  void Function()? _backward;
  List<Tensor> _children = const [];

  /// Debug-only accessor: shapes of this tensor's autograd children.
  List<List<int>> get debugChildShapes =>
      _children.map((c) => c.shape).toList();

  /// Accumulated gradient for this tensor, or `null` before backward or
  /// after [zeroGrad].
  Tensor? get grad => _grad;

  /// Elementwise ops with fewer than this many elements default to CPU;
  /// at or above, they default to GPU. See `docs/device-placement.md`.
  static const int autoDeviceThreshold = 4096;

  static Device _pickDevice(int length, Device? explicit) {
    if (explicit != null) return explicit;
    return length >= autoDeviceThreshold ? Device.GPU : Device.CPU;
  }

  Tensor._cpu(this.shape, Float32List data, {this.requiresGrad = false})
    : length = data.length,
      device = Device.CPU,
      dtype = DType.fp32,
      _cpuData = data;

  Tensor._cpuF16(this.shape, Uint16List bits)
    : length = bits.length,
      device = Device.CPU,
      dtype = DType.fp16,
      requiresGrad = false,
      _cpuF16Bits = bits;

  Tensor._gpu(
    this.shape,
    ffi.Pointer<ffi.Void> handle, {
    this.requiresGrad = false,
  }) : length = shape.reduce((a, b) => a * b),
       device = Device.GPU,
       dtype = DType.fp32,
       _handle = handle {
    // Register a native finalizer so autograd-graph intermediates and
    // other implicitly-owned GPU tensors get their handle freed when
    // the Dart object is GC'd. Without this, a training loop leaks
    // ~O(200) handles per step and exhausts VRAM within ~30 steps at
    // mid-model sizes, silently corrupting future allocations.
    engine.destroyTensorFinalizer.attach(
      this,
      handle,
      detach: this,
      externalSize: length * 4,
    );
  }

  /// Construct from a flat host list. Device defaults to CPU below
  /// [autoDeviceThreshold] elements, GPU above. Pass `requiresGrad:
  /// true` to make this a trainable leaf.
  factory Tensor.fromList(
    List<int> shape,
    List<double> vals, {
    Device? device,
    bool requiresGrad = false,
  }) {
    final expected = shape.reduce((a, b) => a * b);
    if (vals.length != expected) {
      throw ArgumentError(
        'fromList: shape $shape expects $expected values, got ${vals.length}',
      );
    }
    final dev = _pickDevice(vals.length, device);
    if (dev == Device.CPU) {
      return Tensor._cpu(
        shape,
        Float32List.fromList(vals),
        requiresGrad: requiresGrad,
      );
    }
    return Tensor._gpu(
      shape,
      _uploadToGpu(shape, vals),
      requiresGrad: requiresGrad,
    );
  }

  /// Fill a tensor with a constant. Device defaults follow the same
  /// size-based rule as [Tensor.fromList].
  factory Tensor.fill(
    List<int> shape,
    double val, {
    Device? device,
    bool requiresGrad = false,
  }) {
    final n = shape.reduce((a, b) => a * b);
    final dev = _pickDevice(n, device);
    if (dev == Device.CPU) {
      final d = Float32List(n);
      for (int i = 0; i < n; i++) {
        d[i] = val;
      }
      return Tensor._cpu(shape, d, requiresGrad: requiresGrad);
    }
    final vals = List<double>.filled(n, val);
    return Tensor._gpu(
      shape,
      _uploadToGpu(shape, vals),
      requiresGrad: requiresGrad,
    );
  }

  static ffi.Pointer<ffi.Void> _uploadToGpu(
    List<int> shape,
    List<double> vals,
  ) {
    final ptr = calloc<ffi.Float>(vals.length);
    for (int i = 0; i < vals.length; i++) {
      ptr[i] = vals[i];
    }
    final rows = shape[0];
    final cols = shape.length > 1
        ? shape.sublist(1).reduce((a, b) => a * b)
        : 1;
    final h = engine.createTensor(rows, cols, ptr);
    calloc.free(ptr);
    return h;
  }

  /// Construct a **read-only fp16** CPU tensor directly from raw
  /// IEEE-754 half-precision bits. Ops decode to fp32 on read via
  /// [_readAsFp32]. See `dtype.dart` for the semantics — fp16 tensors
  /// cannot be trainable leaves, cannot be assigned into, and are
  /// promoted to fp32 on `.to(Device.GPU)`.
  ///
  /// The primary use is holding large read-only weights (e.g. loaded
  /// straight from a HF safetensors `F16` block) without incurring
  /// the 2× memory blow-up of the fp32 promotion. `bits.length` must
  /// equal `shape.reduce(*)`.
  factory Tensor.fromFp16Bits(List<int> shape, Uint16List bits) {
    final expected = shape.reduce((a, b) => a * b);
    if (bits.length != expected) {
      throw ArgumentError(
        'fromFp16Bits: shape $shape expects $expected values, got '
        '${bits.length}',
      );
    }
    return Tensor._cpuF16(shape, Uint16List.fromList(bits));
  }

  /// Materialise `this` as an fp32 CPU tensor. If already fp32, this
  /// clones for safety (callers of a low-level "give me fp32" method
  /// should not observe aliasing). Detached from the autograd graph.
  Tensor toFp32() {
    if (dtype == DType.fp32) {
      return device == Device.CPU
          ? Tensor._cpu(shape, Float32List.fromList(_cpuData!))
          : Tensor._cpu(shape, Float32List.fromList(toList()));
    }
    // fp16 CPU → fp32 CPU (bulk decode).
    return Tensor._cpu(shape, decodeFp16Bulk(_cpuF16Bits!));
  }

  /// Encode `this` (must be fp32 CPU) into an fp16 CPU tensor. Uses
  /// round-to-nearest-even; values outside fp16 dynamic range saturate
  /// to ±Inf. Detached from the autograd graph.
  Tensor toFp16() {
    if (device != Device.CPU) {
      throw StateError(
        'toFp16: only supported on CPU tensors (source device=$device)',
      );
    }
    if (dtype == DType.fp16) {
      return Tensor._cpuF16(shape, Uint16List.fromList(_cpuF16Bits!));
    }
    return Tensor._cpuF16(shape, encodeFp16Bulk(_cpuData!));
  }

  /// Materialise this tensor's data as a fp32 `Float32List` for the
  /// duration of one op. On fp32 CPU tensors this is a zero-copy view
  /// of the underlying storage; on fp16 tensors this **allocates and
  /// decodes** each call. On GPU tensors this throws.
  ///
  /// Ops that read weight tensors (matmul, layernorm, rmsnorm,
  /// embedding, elementwise-add of bias) route their reads through
  /// this method so they transparently accept both fp32 and fp16
  /// weights. Ops that mutate storage (in-place assign, optimizer
  /// step) must not use this — they only ever see fp32.
  Float32List _readAsFp32() {
    if (device != Device.CPU) {
      throw StateError('_readAsFp32: not a CPU tensor (device=$device)');
    }
    if (dtype == DType.fp32) return _cpuData!;
    return decodeFp16Bulk(_cpuF16Bits!);
  }

  /// Returns a new tensor placed on [target]. Returns `this` unchanged
  /// when already on that device. Involves an H<->D copy when crossing
  /// devices. The result is detached from the autograd graph.
  ///
  /// fp16 → GPU auto-promotes to fp32 (GPU storage is fp32-only).
  Tensor to(Device target) {
    if (device == target && dtype == DType.fp32) return this;
    if (target == Device.CPU) {
      if (device == Device.CPU) {
        // Same device but dtype differs (fp16 CPU → asked for CPU).
        // Callers who want fp32 should use `.toFp32()`; here we
        // stay in the same storage format.
        return this;
      }
      final ptr = calloc<ffi.Float>(length);
      engine.getTensorData(_handle!, ptr);
      final data = Float32List.fromList(ptr.asTypedList(length));
      calloc.free(ptr);
      return Tensor._cpu(shape, data);
    }
    // target == GPU.
    final srcFp32 = dtype == DType.fp32 ? _cpuData! : _readAsFp32();
    return Tensor._gpu(shape, _uploadToGpu(shape, srcFp32));
  }

  /// Copies tensor data to host as a plain `List<double>`, regardless of
  /// current device. fp16 storage is decoded to fp32 on the fly.
  List<double> toList() {
    if (device == Device.CPU) {
      if (dtype == DType.fp16) return decodeFp16Bulk(_cpuF16Bits!).toList();
      return _cpuData!.toList();
    }
    final ptr = calloc<ffi.Float>(length);
    engine.getTensorData(_handle!, ptr);
    final out = ptr.asTypedList(length).toList();
    calloc.free(ptr);
    return out;
  }

  /// Returns a data-copy of this tensor with no autograd links.
  Tensor clone() {
    if (device == Device.CPU) {
      if (dtype == DType.fp16) {
        return Tensor._cpuF16(shape, Uint16List.fromList(_cpuF16Bits!));
      }
      return Tensor._cpu(shape, Float32List.fromList(_cpuData!));
    }
    return Tensor._gpu(shape, _uploadToGpu(shape, toList()));
  }

  /// Detach from the autograd graph — same data, `requiresGrad = false`.
  Tensor detach() => clone();

  /// Overwrite this tensor's storage with `source`'s data, in place.
  /// Shape must match; `source` is consumed (its GPU handle is adopted
  /// and its handle field is nulled, so callers must not use it after).
  /// Autograd state on `this` (requiresGrad, grad) is preserved.
  ///
  /// fp16 tensors are read-only — assigning into (or from) an fp16
  /// tensor throws. Promote with `.toFp32()` first if you need
  /// mutability.
  void assign(Tensor source) {
    if (source.length != length) {
      throw ArgumentError(
        'assign: length mismatch — got ${source.length}, expected $length',
      );
    }
    if (source.device != device) {
      throw ArgumentError(
        'assign: device mismatch — got ${source.device}, expected $device. '
        'Call .to(...) on source first.',
      );
    }
    if (dtype == DType.fp16 || source.dtype == DType.fp16) {
      throw StateError(
        'assign: fp16 tensors are read-only. Promote with .toFp32() first.',
      );
    }
    if (device == Device.CPU) {
      _cpuData!.setAll(0, source._cpuData!);
    } else {
      // Detach both finalizers to prevent a double-free once the
      // handles change owners.
      engine.destroyTensorFinalizer.detach(this);
      engine.destroyTensorFinalizer.detach(source);
      engine.destroyTensor(_handle!);
      _handle = source._handle;
      source._handle = null;
      // Re-attach so the newly adopted handle is freed if `this` is
      // itself GC'd without an explicit `dispose()`.
      engine.destroyTensorFinalizer.attach(
        this,
        _handle!,
        detach: this,
        externalSize: length * 4,
      );
    }
  }

  /// **In-place** replace this tensor's CPU storage with `source`'s
  /// CPU storage, including its dtype. Both tensors must be CPU and
  /// their `length` must match; `shape` is enforced element-count-only
  /// (rank/layout on `this` is preserved).
  ///
  /// Unlike [assign], this transfers ownership of the backing buffer
  /// rather than copying values, and it accepts (and produces) fp16
  /// storage. Autograd state on `this` is cleared — the caller is
  /// expected to be swapping in read-only weight data (e.g. from a
  /// safetensors loader) at model-init time, before any training
  /// begins.
  ///
  /// After the call, `source` is left in a zombie state — its
  /// backing pointers have been moved out — and must not be used.
  void adoptCpuStorageFrom(Tensor source) {    if (device != Device.CPU || source.device != Device.CPU) {
      throw StateError(
        'adoptCpuStorageFrom: both tensors must be CPU '
        '(dst=$device, src=${source.device})',
      );
    }
    if (source.length != length) {
      throw ArgumentError(
        'adoptCpuStorageFrom: length mismatch — got ${source.length}, '
        'expected $length',
      );
    }
    _cpuData = source._cpuData;
    _cpuF16Bits = source._cpuF16Bits;
    dtype = source.dtype;
    _grad?.dispose();
    _grad = null;
    _children = const [];
    _backward = null;
    requiresGrad = false;
    source._cpuData = null;
    source._cpuF16Bits = null;
  }

  /// Extract a row-slice `[start, end)` of a rank-2 CPU tensor into a
  /// fresh CPU tensor of the same dtype. fp16 storage is preserved
  /// bit-for-bit (no fp32 round-trip), which matters when a HF
  /// loader has to split a big fp16 `[H·hd, D]` matrix into `H`
  /// per-head `[hd, D]` slices at model-load time.
  Tensor sliceRows(int start, int end) {
    if (shape.length != 2) {
      throw ArgumentError('sliceRows: expected rank 2, got $shape');
    }
    if (device != Device.CPU) {
      throw StateError('sliceRows: only supported on CPU tensors');
    }
    if (start < 0 || end > shape[0] || start > end) {
      throw ArgumentError(
        'sliceRows: [start=$start, end=$end) out of range for rows=${shape[0]}',
      );
    }
    final c = shape[1];
    final n = (end - start) * c;
    if (dtype == DType.fp16) {
      final src = _cpuF16Bits!;
      final out = Uint16List(n);
      for (int i = 0; i < n; i++) {
        out[i] = src[start * c + i];
      }
      return Tensor._cpuF16([end - start, c], out);
    }
    final src = _cpuData!;
    final out = Float32List(n);
    for (int i = 0; i < n; i++) {
      out[i] = src[start * c + i];
    }
    return Tensor._cpu([end - start, c], out);
  }

  /// Releases GPU memory when applicable. Also disposes any accumulated
  /// gradient. Idempotent; safe on CPU tensors.
  void dispose() {
    _grad?.dispose();
    _grad = null;
    if (device == Device.GPU && _handle != null) {
      // Detach the finalizer so it doesn't run later on a freed pointer.
      engine.destroyTensorFinalizer.detach(this);
      engine.destroyTensor(_handle!);
      _handle = null;
    }
  }

  // ---------- Autograd core ----------

  /// Trigger reverse-mode differentiation from this tensor. Typically
  /// called on a scalar loss. The root gradient is initialized to ones
  /// if not already set.
  ///
  /// When `freeGraph` is true (the default), every non-root
  /// intermediate node's data + grad are disposed as soon as its
  /// backward closure runs. This is essential for GPU training loops
  /// \u2014 without it, autograd-graph intermediates leak GPU memory
  /// (~200 handles per step at mid-model sizes) and VRAM is exhausted
  /// within a few dozen steps. Leaf tensors (parameters, inputs) are
  /// never freed. The root itself (`this`) also survives so callers
  /// can read the loss value after backward returns.
  ///
  /// Pass `freeGraph: false` for higher-order gradients or to inspect
  /// intermediate `.grad` values after backward.
  void backward({bool freeGraph = true}) {
    if (!requiresGrad) {
      throw StateError(
        'backward() called on a tensor with requiresGrad = false',
      );
    }
    _grad ??= Tensor.fill(shape, 1.0, device: device);

    final ordered = <Tensor>[];
    final visited = <Tensor>{};
    void visit(Tensor t) {
      if (visited.contains(t)) return;
      visited.add(t);
      for (final c in t._children) {
        visit(c);
      }
      ordered.add(t);
    }

    visit(this);
    for (int i = ordered.length - 1; i >= 0; i--) {
      final t = ordered[i];
      t._backward?.call();
      final hook = debugBackwardHook;
      if (hook != null) hook(ordered.length - 1 - i, t);

      if (freeGraph &&
          !identical(t, this) &&
          t._backward != null &&
          t.device == Device.GPU) {
        // Non-root, non-leaf GPU intermediate \u2014 safe to release.
        t._grad?.dispose();
        t._grad = null;
        if (t._handle != null) t.dispose();
        t._children = const [];
        t._backward = null;
      }
    }

    // Release root's backward machinery once the graph walk is done.
    // Its data (`_handle` / `_cpuData`) stays alive so the caller can
    // still read the loss value.
    if (freeGraph) {
      _children = const [];
      _backward = null;
    }
  }

  /// Clear the accumulated gradient. Call between training steps.
  void zeroGrad() {
    _grad?.dispose();
    _grad = null;
  }

  /// Wire `fn` as this tensor's backward closure, remember its parents,
  /// and mark this node as requiring grad. Called by every op that
  /// participates in autograd. Skipped entirely inside a [noGrad] scope.
  void _setBackward(List<Tensor> children, void Function() fn) {
    if (_noGradDepth > 0) return;
    _children = children;
    _backward = fn;
    requiresGrad = true;
  }

  static int _noGradDepth = 0;

  /// Run [body] with autograd disabled: every op skips backward wiring
  /// so no graph is built and no intermediates are retained for a
  /// future backward call. Use this around inference / generation loops
  /// where you would otherwise accumulate an unbounded graph and
  /// exhaust GPU memory.
  static T noGrad<T>(T Function() body) {
    _noGradDepth++;
    try {
      return body();
    } finally {
      _noGradDepth--;
    }
  }

  /// Add `contribution` into this tensor's gradient. Takes ownership of
  /// `contribution` — do not use it after passing it in.
  void _accumulateGrad(Tensor contribution) {
    if (contribution.shape.reduce((a, b) => a * b) !=
        shape.reduce((a, b) => a * b)) {
      throw StateError(
        '_accumulateGrad: shape mismatch — target $shape vs '
        'contribution ${contribution.shape}',
      );
    }
    if (_grad == null) {
      _grad = contribution;
    } else {
      final summed = _grad! + contribution;
      _grad!.dispose();
      contribution.dispose();
      _grad = summed;
    }
  }

  /// Reduce an incoming grad `g` back to `targetShape` when a forward
  /// op broadcast a smaller tensor into a larger output. Handles the
  /// scalar case (`length == 1`) and 2D row-broadcast (`[1, N]`).
  static Tensor _reduceForBroadcast(Tensor g, List<int> targetShape) {
    final targetLen = targetShape.reduce((a, b) => a * b);
    if (g.shape.reduce((a, b) => a * b) == targetLen) return g;
    if (targetLen == 1) {
      final s = g.sum();
      g.dispose();
      return s;
    }
    // Row broadcast [1, N] <- [M, N] — reduce along axis 0.
    if (targetShape.length == 2 &&
        targetShape[0] == 1 &&
        g.shape.length == 2 &&
        g.shape[1] == targetShape[1]) {
      final n = g.shape[1];
      final xs = g.toList();
      final out = Float32List(n);
      for (int i = 0; i < g.shape[0]; i++) {
        for (int j = 0; j < n; j++) {
          out[j] += xs[i * n + j];
        }
      }
      final reduced = Tensor._cpu(targetShape, out);
      final placed = g.device == Device.GPU ? reduced.to(Device.GPU) : reduced;
      g.dispose();
      return placed;
    }
    throw ArgumentError(
      '_reduceForBroadcast: no rule for ${g.shape} -> $targetShape',
    );
  }
}
