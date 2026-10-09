/// 3D parallelism demo: tensor-parallel (within a host, across its GPUs)
/// combined with data-parallel (across hosts, over TCP via [Dist]).
///
/// Each rank is a host. Within a rank the model is tensor-parallel across
/// that host's visible GPUs (weights sharded per card). Across ranks the
/// gradients are averaged with the socket all-reduce in bin/ddp_dist.dart.
/// Because tensor-parallel weight gradients are already local to each
/// card and every replica shares the same shard layout, flattening
/// `model.parameters()` and all-reducing element-wise correctly averages
/// corresponding shards across the data-parallel replicas — no special
/// collective is needed. The averaged grads are scattered back onto each
/// shard's own GPU.
///
/// Single process (tensor-parallel only, all-reduce is a no-op):
///   dart run bin/tensor_parallel_ddp_demo.dart
///
/// Two hosts × their GPUs (data × tensor parallel):
///   RANK=0 WORLD_SIZE=2 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
///     dart run bin/tensor_parallel_ddp_demo.dart     # host 0
///   RANK=1 WORLD_SIZE=2 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
///     dart run bin/tensor_parallel_ddp_demo.dart     # host 1
///
/// Requires the multi-GPU native lib (rebuild libmat_mul from
/// lib/native/src/engine.cu).
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';

import 'ddp_dist.dart';

const int _model = 64;
const int _hidden = 256;
const int _tokens = 8;
const int _steps = 40;

/// Parse a `TP_DEVICES` topology string like "0,1,2" into GPU ordinals.
/// Null/empty means "use every visible GPU on this host".
List<int>? _parseTpDevices(String? s) {
  if (s == null || s.trim().isEmpty) return null;
  return s.split(',').map((e) => int.parse(e.trim())).toList();
}

/// Download every parameter's grad into one host buffer (device-aware).
void _gatherGrads(List<Tensor> params, Float32List flat) {
  var off = 0;
  for (final p in params) {
    final g = p.grad;
    if (g != null) {
      flat.setAll(off, g.toFloat32List());
    } else {
      flat.fillRange(off, off + p.length, 0.0);
    }
    off += p.length;
  }
}

/// Write averaged grads back onto each parameter's own GPU.
void _scatterGrads(List<Tensor> params, Float32List flat) {
  var off = 0;
  for (final p in params) {
    final n = p.length;
    final slice = Float32List.sublistView(flat, off, off + n);
    final gradT = p.device == Device.GPU
        ? Tensor.onGpu(
            p.gpuIndex,
            () => Tensor.fromFloat32List(p.shape, slice, device: Device.GPU),
          )
        : Tensor.fromFloat32List(p.shape, slice, device: Device.CPU);
    p.grad?.assign(gradT);
    off += n;
  }
}

Future<void> main() async {
  await ensureNativeLib();
  final dist = await Dist.init();
  void log(String m) {
    if (dist.isMaster) print(m);
  }

  // The GPU set this rank shards across (rank→GPU-set topology).
  final tpDevices = _parseTpDevices(Platform.environment['TP_DEVICES']);
  log('=== 3D-parallel (tensor × data) training demo ===');
  log('world size (hosts): ${dist.worldSize} | visible GPUs this host: '
      '${Tensor.gpuCount} | TP devices this rank: '
      '${tpDevices ?? 'all'} | model: $_model, hidden: $_hidden');

  // Tensor-parallel model across this rank's assigned GPUs.
  final rng = math.Random(0);
  final upW = Tensor.fromList(
    [_hidden, _model],
    List<double>.generate(_hidden * _model, (_) => (rng.nextDouble() * 2 - 1) * 0.1),
  );
  final downW = Tensor.fromList(
    [_model, _hidden],
    List<double>.generate(_model * _hidden, (_) => (rng.nextDouble() * 2 - 1) * 0.1),
  );
  final mlp = TensorParallelMLP.fromWeights(
    upWeight: upW,
    downWeight: downW,
    trainable: true,
    devices: tpDevices,
  );
  final params = mlp.parameters();
  final totalScalars = params.fold<int>(0, (a, p) => a + p.length);
  final flat = Float32List(totalScalars);

  // Each rank (host) sees a different data shard — seed by rank.
  final dataRng = math.Random(1000 + dist.rank);
  final outDev = mlp.down.outputDevice;
  final x = Tensor.fromList(
    [_tokens, _model],
    List<double>.generate(_tokens * _model, (_) => (dataRng.nextDouble() * 2 - 1) * 0.5),
    device: Device.GPU,
  ).toGpu(outDev);
  final target = Tensor.fromList(
    [_tokens, _model],
    List<double>.generate(_tokens * _model, (_) => (dataRng.nextDouble() * 2 - 1) * 0.5),
    device: Device.GPU,
  ).toGpu(outDev);

  final opt = SGD(params, lr: 0.1, momentum: 0.9);

  for (var step = 0; step < _steps; step++) {
    final out = mlp(x);
    final diff = out - target;
    final loss = (diff * diff).mean();
    final v = loss.toList().first;

    opt.zeroGrad();
    loss.backward();

    // Cross-host data-parallel gradient averaging (no-op when world==1).
    _gatherGrads(params, flat);
    await dist.allReduceMean(flat);
    _scatterGrads(params, flat);

    opt.step();
    if (step % 10 == 0 || step == _steps - 1) {
      log('step ${step.toString().padLeft(3)}  loss ${v.toStringAsExponential(4)}');
    }
  }

  log('done — tensor-parallel within host, data-parallel across '
      '${dist.worldSize} host(s).');
  await dist.close();
}
