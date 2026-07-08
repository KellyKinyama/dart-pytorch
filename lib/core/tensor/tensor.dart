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

part 'mat_mul.dart';
part 'ops.dart';
part 'layer_norm.dart';
part 'softmax.dart';
part 'embedding.dart';
part 'attention.dart';
part 'dropout.dart';
part 'concat.dart';
part 'aft.dart';
part 'scatter.dart';

enum Device { CPU, GPU }

class Tensor {
  final List<int> shape;
  final int length;
  Device device;

  // Backing store — exactly one of these is populated (dictated by [device]).
  Float32List? _cpuData;
  ffi.Pointer<ffi.Void>? _handle;

  // Autograd state.
  bool requiresGrad;
  Tensor? _grad;
  void Function()? _backward;
  List<Tensor> _children = const [];

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
      _cpuData = data;

  Tensor._gpu(
    this.shape,
    ffi.Pointer<ffi.Void> handle, {
    this.requiresGrad = false,
  }) : length = shape.reduce((a, b) => a * b),
       device = Device.GPU,
       _handle = handle;

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

  /// Returns a new tensor placed on [target]. Returns `this` unchanged
  /// when already on that device. Involves an H<->D copy when crossing
  /// devices. The result is detached from the autograd graph.
  Tensor to(Device target) {
    if (device == target) return this;
    if (target == Device.CPU) {
      final ptr = calloc<ffi.Float>(length);
      engine.getTensorData(_handle!, ptr);
      final data = Float32List.fromList(ptr.asTypedList(length));
      calloc.free(ptr);
      return Tensor._cpu(shape, data);
    }
    return Tensor._gpu(shape, _uploadToGpu(shape, _cpuData!));
  }

  /// Copies tensor data to host as a plain `List<double>`, regardless of
  /// current device.
  List<double> toList() {
    if (device == Device.CPU) return _cpuData!.toList();
    final ptr = calloc<ffi.Float>(length);
    engine.getTensorData(_handle!, ptr);
    final out = ptr.asTypedList(length).toList();
    calloc.free(ptr);
    return out;
  }

  /// Returns a data-copy of this tensor with no autograd links.
  Tensor clone() {
    if (device == Device.CPU) {
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
    if (device == Device.CPU) {
      _cpuData!.setAll(0, source._cpuData!);
    } else {
      engine.destroyTensor(_handle!);
      _handle = source._handle;
      source._handle = null;
    }
  }

  /// Releases GPU memory when applicable. Also disposes any accumulated
  /// gradient. Idempotent; safe on CPU tensors.
  void dispose() {
    _grad?.dispose();
    _grad = null;
    if (device == Device.GPU && _handle != null) {
      engine.destroyTensor(_handle!);
      _handle = null;
    }
  }

  // ---------- Autograd core ----------

  /// Trigger reverse-mode differentiation from this tensor. Typically
  /// called on a scalar loss. The root gradient is initialized to ones
  /// if not already set.
  void backward() {
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
      ordered[i]._backward?.call();
    }
  }

  /// Clear the accumulated gradient. Call between training steps.
  void zeroGrad() {
    _grad?.dispose();
    _grad = null;
  }

  /// Wire `fn` as this tensor's backward closure, remember its parents,
  /// and mark this node as requiring grad. Called by every op that
  /// participates in autograd.
  void _setBackward(List<Tensor> children, void Function() fn) {
    _children = children;
    _backward = fn;
    requiresGrad = true;
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
