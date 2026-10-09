/// One model, several GPUs — in a single process.
///
/// Demonstrates true in-process model (pipeline) parallelism: a stack of
/// linear layers is split into contiguous groups, one group per physical
/// GPU. Each group's weights are *allocated on their own card* with
/// [Tensor.onGpu]; the activation is handed from one card to the next with
/// [Tensor.toGpu] (a direct device-to-device peer copy). Nothing but the
/// boundary activation ever crosses between GPUs, so a model far larger
/// than any single card's memory can run end to end.
///
///   dart run bin/multi_gpu_pipeline_demo.dart
///
/// Requires the multi-GPU-capable native lib (rebuild libmat_mul from
/// lib/native/src/engine.cu). Falls back to a single device gracefully:
/// with one visible GPU the whole stack simply runs on device 0.
library;

import 'package:dart_pytorch/dart_pytorch.dart';

const int _dim = 512; // hidden width
const int _layers = 8; // total linear layers in the stack

Future<void> main() async {
  await ensureNativeLib();

  final gpus = Tensor.gpuCount;
  if (!_multiGpuAvailable()) {
    print('Native lib has no multi-GPU support; running on device 0 only. '
        'Rebuild libmat_mul from lib/native/src/engine.cu to enable it.');
  }
  print('=== multi-GPU pipeline demo ===');
  print('visible GPUs: $gpus, layers: $_layers, dim: $_dim');

  // Assign each layer to a GPU, splitting the stack into `gpus` contiguous
  // stages so stage j owns a near-equal slice of the layers.
  final layerGpu = List<int>.generate(
    _layers,
    (i) => gpus <= 1 ? 0 : (i * gpus) ~/ _layers,
  );

  // Build each layer's weight matrix directly on its assigned GPU.
  final weights = <Tensor>[];
  for (var i = 0; i < _layers; i++) {
    final g = layerGpu[i];
    weights.add(
      Tensor.onGpu(g, () {
        final vals = List<double>.generate(
          _dim * _dim,
          (k) => ((k + i) % 13 - 6) * 0.01,
        );
        return Tensor.fromList([_dim, _dim], vals, device: Device.GPU);
      }),
    );
    print('layer $i  -> GPU $g');
  }

  // A single input row, starting on the first stage's GPU.
  final x0vals = List<double>.generate(_dim, (k) => (k % 7 - 3) * 0.1);
  var x = Tensor.onGpu(
    layerGpu.first,
    () => Tensor.fromList([1, _dim], x0vals, device: Device.GPU),
  );

  // Forward pass. Move the activation to each layer's card, then run that
  // layer there. The matmul + relu allocate their outputs on the current
  // device, so they stay resident on the right GPU for the next op.
  for (var i = 0; i < _layers; i++) {
    final g = layerGpu[i];
    x = x.toGpu(g); // peer copy only when crossing a GPU boundary
    x = Tensor.onGpu(g, () => x.matmul(weights[i]).relu());
  }

  // Pull the result back to the host to read it.
  final out = x.toList();
  var sum = 0.0;
  for (final v in out) {
    sum += v;
  }
  print('output[0..4]: ${out.take(5).map((v) => v.toStringAsFixed(4)).toList()}');
  print('output sum:   ${sum.toStringAsFixed(4)}  (len ${out.length})');
  print('done — model ran across ${layerGpu.toSet().length} device(s).');
}

bool _multiGpuAvailable() {
  // gpuCount > 1 implies the device-management symbols loaded; but even on
  // a single card the demo is valid, so this only drives the hint message.
  return Tensor.gpuCount > 1;
}
