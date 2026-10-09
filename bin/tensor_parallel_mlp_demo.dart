/// Tensor-parallel transformer MLP across several GPUs — in one process.
///
/// Builds a [TensorParallelMLP] whose up-projection is split column-wise
/// and whose down-projection is split row-wise across every visible GPU,
/// then checks its output against a plain single-device reference. The
/// wide hidden layer is never materialised on a single card — each GPU
/// only ever holds its own `1/G` slice of the weights.
///
///   dart run bin/tensor_parallel_mlp_demo.dart
///
/// Requires the multi-GPU native lib (rebuild libmat_mul from
/// lib/native/src/engine.cu). On a single visible GPU it still runs,
/// degenerating to an ordinary MLP on device 0.
library;

import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

const int _model = 256; // model (input/output) dim
const int _hidden = 1024; // wide MLP hidden dim (the big matmul)
const int _tokens = 8; // rows in the input batch

Future<void> main() async {
  await ensureNativeLib();

  final gpus = Tensor.gpuCount;
  print('=== tensor-parallel MLP demo ===');
  print('visible GPUs: $gpus | model: $_model, hidden: $_hidden, '
      'tokens: $_tokens');

  final rng = math.Random(0);
  List<double> rand(int n, double s) =>
      List<double>.generate(n, (_) => (rng.nextDouble() * 2 - 1) * s);

  // Full weights live on the host; the layer shards them onto the GPUs.
  final upW = Tensor.fromList([_hidden, _model], rand(_hidden * _model, 0.05));
  final upB = Tensor.fromList([1, _hidden], rand(_hidden, 0.02));
  final downW =
      Tensor.fromList([_model, _hidden], rand(_model * _hidden, 0.05));
  final downB = Tensor.fromList([1, _model], rand(_model, 0.02));
  final x = Tensor.fromList([_tokens, _model], rand(_tokens * _model, 0.1));

  // Tensor-parallel layer across all visible GPUs.
  final mlp = TensorParallelMLP.fromWeights(
    upWeight: upW,
    upBias: upB,
    downWeight: downW,
    downBias: downB,
  );
  final tpOut = mlp(x).to(Device.CPU).toFloat32List();
  print('up split across GPUs: ${mlp.up.devices}');
  print('down split across GPUs: ${mlp.down.devices}');

  // Single-device reference: relu(x @ Up.T + ub) @ Down.T + db on GPU 0.
  final refOut = Tensor.onGpu(0, () {
    final xg = x.toGpu(0);
    final h = (xg.matmul(upW.toGpu(0).transpose()) + upB.toGpu(0)).relu();
    return h.matmul(downW.toGpu(0).transpose()) + downB.toGpu(0);
  }).to(Device.CPU).toFloat32List();

  var maxDiff = 0.0;
  for (var i = 0; i < refOut.length; i++) {
    final d = (tpOut[i] - refOut[i]).abs();
    if (d > maxDiff) maxDiff = d;
  }
  print('output shape: [$_tokens, $_model]');
  print('max abs diff vs single-GPU reference: ${maxDiff.toStringAsExponential(3)}');
  print(maxDiff < 1e-3
      ? 'OK — tensor-parallel output matches the reference.'
      : 'WARNING — mismatch exceeds tolerance.');
}
