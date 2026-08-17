part of 'tensor.dart';

/// Local type aliases for the FFI-op signatures.
typedef _GpuOp1 = ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>);
typedef _GpuOp2 =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
    );

/// Basic tensor operations with autograd wiring.
///
/// Forward: dispatches by input [Tensor.device]. Binary ops require
/// both operands on the same device (see `doc/device-placement.md`).
///
/// Backward: every op wires a `_backward` closure on its output when at
/// least one input has `requiresGrad = true`. Gradient math is composed
/// of the same ops as the forward pass, so backward runs on whichever
/// device the tensors already live on — no separate CPU vs GPU code
/// paths for gradients.
extension TensorOps on Tensor {
  // ---------------------------------------------------------------------
  // Elementwise binary ops.
  // ---------------------------------------------------------------------

  Tensor operator +(dynamic o) {
    final out = _binaryFwd(
      o,
      cpuFn: (a, b) => a + b,
      gpuExact: engine.addTensors,
      gpuScalar: engine.addTensorScalar,
      gpuRowBroadcast: engine.addTensorRowBroadcast,
      opName: '+',
    );
    if (o is num) {
      if (requiresGrad) {
        final a = this;
        out._setBackward([a], () {
          a._accumulateGrad(out._grad!.clone());
        });
      }
    } else if (o is Tensor && (requiresGrad || o.requiresGrad)) {
      final a = this;
      final b = o;
      out._setBackward([a, b], () {
        final g = out._grad!;
        // Always clone before handing to _accumulateGrad: (a) both branches
        // may need g; (b) _reduceForBroadcast can pass its input through
        // unchanged, aliasing out._grad into the child's _grad, which
        // freeGraph would later dispose out from under us.
        if (a.requiresGrad) {
          a._accumulateGrad(Tensor._reduceForBroadcast(g.clone(), a.shape));
        }
        if (b.requiresGrad) {
          b._accumulateGrad(Tensor._reduceForBroadcast(g.clone(), b.shape));
        }
      });
    }
    return out;
  }

  Tensor operator -(dynamic o) {
    final out = _binaryFwd(
      o,
      cpuFn: (a, b) => a - b,
      gpuExact: engine.subTensors,
      gpuScalar: engine.subTensorScalar,
      opName: '-',
    );
    if (o is num) {
      if (requiresGrad) {
        final a = this;
        out._setBackward([a], () {
          a._accumulateGrad(out._grad!.clone());
        });
      }
    } else if (o is Tensor && (requiresGrad || o.requiresGrad)) {
      final a = this;
      final b = o;
      out._setBackward([a, b], () {
        final g = out._grad!;
        if (a.requiresGrad) {
          // Clone: _reduceForBroadcast can pass g through unchanged,
          // which would alias out._grad into a._grad and get disposed
          // when freeGraph releases out.
          a._accumulateGrad(Tensor._reduceForBroadcast(g.clone(), a.shape));
        }
        if (b.requiresGrad) {
          b._accumulateGrad(Tensor._reduceForBroadcast(g * -1.0, b.shape));
        }
      });
    }
    return out;
  }

  Tensor operator *(dynamic o) {
    final out = _binaryFwd(
      o,
      cpuFn: (a, b) => a * b,
      gpuExact: engine.mulTensors,
      gpuScalar: engine.mulTensorScalar,
      opName: '*',
    );
    if (o is num) {
      if (requiresGrad) {
        final a = this;
        final s = o.toDouble();
        out._setBackward([a], () {
          a._accumulateGrad(out._grad! * s);
        });
      }
    } else if (o is Tensor && (requiresGrad || o.requiresGrad)) {
      final a = this;
      final b = o;
      out._setBackward([a, b], () {
        final g = out._grad!;
        if (a.requiresGrad) {
          a._accumulateGrad(Tensor._reduceForBroadcast(g * b, a.shape));
        }
        if (b.requiresGrad) {
          b._accumulateGrad(Tensor._reduceForBroadcast(g * a, b.shape));
        }
      });
    }
    return out;
  }

  Tensor operator /(dynamic o) {
    final out = _binaryFwd(
      o,
      cpuFn: (a, b) => a / b,
      gpuExact: engine.divTensors,
      gpuScalar: engine.divTensorScalar,
      opName: '/',
    );
    if (o is num) {
      if (requiresGrad) {
        final a = this;
        final s = o.toDouble();
        out._setBackward([a], () {
          a._accumulateGrad(out._grad! / s);
        });
      }
    } else if (o is Tensor && (requiresGrad || o.requiresGrad)) {
      final a = this;
      final b = o;
      out._setBackward([a, b], () {
        final g = out._grad!;
        if (a.requiresGrad) {
          a._accumulateGrad(Tensor._reduceForBroadcast(g / b, a.shape));
        }
        if (b.requiresGrad) {
          // d/db (a / b) = -a / b^2
          final bSq = b * b;
          final contrib = (g * a * -1.0) / bSq;
          b._accumulateGrad(Tensor._reduceForBroadcast(contrib, b.shape));
        }
      });
    }
    return out;
  }

  // ---------------------------------------------------------------------
  // Forward-only binary dispatch.
  // ---------------------------------------------------------------------

  Tensor _binaryFwd(
    dynamic other, {
    required double Function(double, double) cpuFn,
    required _GpuOp2 gpuExact,
    required _GpuOp2 gpuScalar,
    _GpuOp2? gpuRowBroadcast,
    required String opName,
  }) {
    if (other is num) {
      final s = other.toDouble();
      if (device == Device.CPU) {
        final out = Float32List(length);
        final d = _cpuData!;
        for (int i = 0; i < length; i++) {
          out[i] = cpuFn(d[i], s);
        }
        return Tensor._cpu(shape, out);
      }
      final scalarT = Tensor.fill([1, 1], s, device: Device.GPU);
      final h = gpuScalar(_handle!, scalarT._handle!);
      scalarT.dispose();
      return Tensor._gpu(shape, h);
    }

    if (other is! Tensor) {
      throw ArgumentError('$opName: unsupported operand ${other.runtimeType}');
    }
    if (device != other.device) {
      throw ArgumentError(
        '$opName: mixed devices ($device vs ${other.device}). '
        'Call .to(...) on one operand first.',
      );
    }

    if (device == Device.CPU) {
      return _binaryCpu(other, cpuFn, opName);
    }
    return _binaryGpu(other, gpuExact, gpuScalar, gpuRowBroadcast, opName);
  }

  Tensor _binaryCpu(
    Tensor other,
    double Function(double, double) f,
    String opName,
  ) {
    final a = _readAsFp32();
    final b = other._readAsFp32();
    if (a.length == b.length) {
      final out = Float32List(a.length);
      for (int i = 0; i < a.length; i++) {
        out[i] = f(a[i], b[i]);
      }
      return Tensor._cpu(shape, out);
    }
    if (b.length == 1) {
      final s = b[0];
      final out = Float32List(a.length);
      for (int i = 0; i < a.length; i++) {
        out[i] = f(a[i], s);
      }
      return Tensor._cpu(shape, out);
    }
    if (a.length == 1) {
      final s = a[0];
      final out = Float32List(b.length);
      for (int i = 0; i < b.length; i++) {
        out[i] = f(s, b[i]);
      }
      return Tensor._cpu(other.shape, out);
    }
    if (shape.length == 2 &&
        other.shape.length == 2 &&
        other.shape[0] == 1 &&
        other.shape[1] == shape[1]) {
      final m = shape[0];
      final n = shape[1];
      final out = Float32List(a.length);
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          out[i * n + j] = f(a[i * n + j], b[j]);
        }
      }
      return Tensor._cpu(shape, out);
    }
    throw ArgumentError(
      '$opName: incompatible shapes $shape vs ${other.shape}',
    );
  }

  Tensor _binaryGpu(
    Tensor other,
    _GpuOp2 gpuExact,
    _GpuOp2 gpuScalar,
    _GpuOp2? gpuRowBroadcast,
    String opName,
  ) {
    if (length == other.length) {
      return Tensor._gpu(shape, gpuExact(_handle!, other._handle!));
    }
    if (other.length == 1) {
      return Tensor._gpu(shape, gpuScalar(_handle!, other._handle!));
    }
    if (gpuRowBroadcast != null &&
        shape.length == 2 &&
        other.shape.length == 2 &&
        other.shape[0] == 1 &&
        other.shape[1] == shape[1]) {
      return Tensor._gpu(shape, gpuRowBroadcast(_handle!, other._handle!));
    }
    throw ArgumentError(
      '$opName: incompatible shapes $shape vs ${other.shape} on GPU '
      '(row-broadcast on GPU is only wired for `+`)',
    );
  }

  // ---------------------------------------------------------------------
  // Unary elementwise (activations + math).
  // ---------------------------------------------------------------------

  /// ReLU. Backward: `dX = dOut * (X > 0)`.
  Tensor relu() {
    final out = _unaryFwd(
      cpuFn: (x) => x < 0 ? 0.0 : x,
      gpu: engine.reluTensor,
    );
    if (requiresGrad) {
      final x = this;
      out._setBackward([x], () {
        if (x.device == Device.GPU) {
          final gX = Tensor.fill(x.shape, 0.0, device: Device.GPU);
          engine.reluBackwardOp(x._handle!, out._grad!._handle!, gX._handle!);
          x._accumulateGrad(gX);
        } else {
          final mask = x._reluMaskCpu();
          x._accumulateGrad(out._grad! * mask);
        }
      });
    }
    return out;
  }

  /// Sigmoid. Backward: `dY = dOut * (y - y*y)` — pure composition.
  Tensor sigmoid() {
    final out = _unaryFwd(
      cpuFn: (x) => 1.0 / (1.0 + math.exp(-x)),
      gpu: engine.sigmoidTensor,
    );
    if (requiresGrad) {
      final x = this;
      out._setBackward([x], () {
        final y = out;
        final g = out._grad!;
        x._accumulateGrad(g * (y - y * y));
      });
    }
    return out;
  }

  /// Tanh. Backward: `dY = dOut - dOut * y * y` — pure composition.
  ///
  /// Uses the numerically stable formulation
  ///   `tanh(x) = sign(x) * (1 - 2 / (exp(2|x|) + 1))`,
  /// which avoids `exp(2x)` overflowing to `+inf` (giving `inf/inf = NaN`)
  /// for large positive `x`, and symmetrically underflow for large
  /// negative `x`. For `|x| > 20` the result is already within fp32 ULP
  /// of `±1`, so we short-circuit.
  Tensor tanh() {
    final out = _unaryFwd(
      cpuFn: (x) {
        if (x > 20.0) return 1.0;
        if (x < -20.0) return -1.0;
        final ax = x.abs();
        final t = 1.0 - 2.0 / (math.exp(2.0 * ax) + 1.0);
        return x >= 0 ? t : -t;
      },
      gpu: engine.tanhTensor,
    );
    if (requiresGrad) {
      final x = this;
      out._setBackward([x], () {
        final y = out;
        final g = out._grad!;
        x._accumulateGrad(g - g * y * y);
      });
    }
    return out;
  }

  /// Absolute value. Backward: `dX = dOut * sign(X)` (with sign(0)=0).
  Tensor abs() {
    final out = _unaryFwd(cpuFn: (x) => x.abs(), gpu: engine.absTensor);
    if (requiresGrad) {
      final x = this;
      out._setBackward([x], () {
        if (x.device == Device.GPU) {
          final gX = Tensor.fill(x.shape, 0.0, device: Device.GPU);
          engine.absBackwardOp(x._handle!, out._grad!._handle!, gX._handle!);
          x._accumulateGrad(gX);
        } else {
          final s = x._signMaskCpu();
          x._accumulateGrad(out._grad! * s);
        }
      });
    }
    return out;
  }

  /// Natural log. Backward: `dY = dOut / x`.
  Tensor log() {
    final out = _unaryFwd(cpuFn: (x) => math.log(x), gpu: engine.logTensor);
    if (requiresGrad) {
      final x = this;
      out._setBackward([x], () {
        x._accumulateGrad(out._grad! / x);
      });
    }
    return out;
  }

  /// Power. Backward: `dY = dOut * exp * x^(exp-1)`.
  Tensor pow(double exp) {
    if (device == Device.CPU) {
      final d = _cpuData!;
      final out = Float32List(d.length);
      for (int i = 0; i < d.length; i++) {
        out[i] = math.pow(d[i], exp).toDouble();
      }
      final result = Tensor._cpu(shape, out);
      _wirePowBackward(result, exp);
      return result;
    }
    final result = Tensor._gpu(shape, engine.powTensor(_handle!, exp));
    _wirePowBackward(result, exp);
    return result;
  }

  void _wirePowBackward(Tensor out, double exp) {
    if (!requiresGrad) return;
    final x = this;
    out._setBackward([x], () {
      x._accumulateGrad(out._grad! * x.pow(exp - 1) * exp);
    });
  }

  Tensor _unaryFwd({
    required double Function(double) cpuFn,
    required _GpuOp1 gpu,
  }) {
    if (device == Device.CPU) {
      final d = _cpuData!;
      final out = Float32List(d.length);
      for (int i = 0; i < d.length; i++) {
        out[i] = cpuFn(d[i]);
      }
      return Tensor._cpu(shape, out);
    }
    return Tensor._gpu(shape, gpu(_handle!));
  }

  Tensor _reluMaskCpu() {
    final d = _cpuData!;
    final out = Float32List(d.length);
    for (int i = 0; i < d.length; i++) {
      out[i] = d[i] > 0 ? 1.0 : 0.0;
    }
    return Tensor._cpu(shape, out);
  }

  Tensor _signMaskCpu() {
    final d = _cpuData!;
    final out = Float32List(d.length);
    for (int i = 0; i < d.length; i++) {
      out[i] = d[i] > 0
          ? 1.0
          : d[i] < 0
          ? -1.0
          : 0.0;
    }
    return Tensor._cpu(shape, out);
  }

  // ---------------------------------------------------------------------
  // Rearrangement.
  // ---------------------------------------------------------------------

  /// Return a view of this tensor with a new shape. The product of the
  /// new shape must equal [length]. CPU tensors share the underlying
  /// `Float32List` (zero-copy); GPU tensors copy their contents to a
  /// fresh handle to keep ownership simple.
  ///
  /// Backward reshapes the outgoing gradient back to the source shape.
  Tensor reshape(List<int> newShape) {
    final n = newShape.fold<int>(1, (a, b) => a * b);
    if (n != length) {
      throw ArgumentError(
        'reshape: new shape $newShape has $n elements, expected $length',
      );
    }
    final out = device == Device.CPU
        ? Tensor._cpu(List<int>.of(newShape), _cpuData!)
        : Tensor._gpu(
            List<int>.of(newShape),
            Tensor._uploadToGpu(newShape, toList()),
          );
    if (requiresGrad) {
      final x = this;
      out._setBackward([x], () {
        x._accumulateGrad(out._grad!.reshape(x.shape));
      });
    }
    return out;
  }

  /// 2D transpose. Backward: `dX = dOut.transpose()`.
  Tensor transpose() {
    if (shape.length != 2) {
      throw ArgumentError('transpose() requires a 2D tensor; got $shape');
    }
    final r = shape[0];
    final c = shape[1];
    final out = device == Device.CPU
        ? _transposeCpu(r, c)
        : Tensor._gpu([c, r], engine.transposeTensor(_handle!));
    if (requiresGrad) {
      final x = this;
      out._setBackward([x], () {
        x._accumulateGrad(out._grad!.transpose());
      });
    }
    return out;
  }

  Tensor _transposeCpu(int r, int c) {
    // fp16 transpose keeps the storage in fp16 — it's just a
    // rearrangement of 16-bit values, no math involved. This is
    // what lets `Linear.forward` (which calls `weight.transpose()`)
    // work on fp16 weights without a fp32 blow-up.
    if (dtype == DType.fp16) {
      final src = _cpuF16Bits!;
      final out = Uint16List(length);
      for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
          out[j * r + i] = src[i * c + j];
        }
      }
      return Tensor._cpuF16([c, r], out);
    }
    final src = _cpuData!;
    final out = Float32List(length);
    for (int i = 0; i < r; i++) {
      for (int j = 0; j < c; j++) {
        out[j * r + i] = src[i * c + j];
      }
    }
    return Tensor._cpu([c, r], out);
  }

  // ---------------------------------------------------------------------
  // Reductions. Result shape: [1, 1]. Backward broadcasts back.
  // ---------------------------------------------------------------------

  /// Sum of all elements. Backward: `dX = ones_like(X) * dOut`.
  Tensor sum() {
    Tensor out;
    if (device == Device.CPU) {
      final d = _cpuData!;
      double s = 0;
      for (int i = 0; i < d.length; i++) {
        s += d[i];
      }
      out = Tensor._cpu([1, 1], Float32List.fromList([s]));
    } else {
      out = Tensor._gpu([1, 1], engine.sumTensor(_handle!));
    }
    if (requiresGrad) {
      final x = this;
      out._setBackward([x], () {
        final ones = Tensor.fill(x.shape, 1.0, device: x.device);
        x._accumulateGrad(ones * out._grad!);
      });
    }
    return out;
  }

  /// Mean of all elements. Backward: `dX = ones_like(X) * (dOut / n)`.
  Tensor mean() {
    Tensor out;
    if (device == Device.CPU) {
      final d = _cpuData!;
      double s = 0;
      for (int i = 0; i < d.length; i++) {
        s += d[i];
      }
      out = Tensor._cpu([1, 1], Float32List.fromList([s / d.length]));
    } else {
      out = Tensor._gpu([1, 1], engine.meanTensor(_handle!));
    }
    if (requiresGrad) {
      final x = this;
      final n = x.length.toDouble();
      out._setBackward([x], () {
        final ones = Tensor.fill(x.shape, 1.0, device: x.device);
        x._accumulateGrad(ones * (out._grad! / n));
      });
    }
    return out;
  }
}
