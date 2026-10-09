/// Training a tensor-parallel MLP across several GPUs — distributed
/// backward in one process. Shards an MLP across every visible GPU with
/// `trainable: true`, then overfits it to a fixed target with SGD.
///
/// Each shard's weight gradient is computed **locally on its own card**
/// (that is the point of tensor parallelism); only the activation and
/// its gradient cross GPU boundaries, carried by the differentiable
/// [Tensor.toGpu]. The optimizer steps every shard in place on its card.
///
///   dart run bin/tensor_parallel_train_demo.dart
///
/// Requires the multi-GPU native lib (rebuild libmat_mul from
/// lib/native/src/engine.cu). Runs on a single GPU too.
library;

import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

const int _model = 64;
const int _hidden = 256;
const int _tokens = 8;
const int _steps = 60;

Future<void> main() async {
  await ensureNativeLib();

  print('=== tensor-parallel MLP training demo ===');
  print('visible GPUs: ${Tensor.gpuCount} | model: $_model, hidden: $_hidden, '
      'tokens: $_tokens, steps: $_steps');

  final rng = math.Random(0);
  List<double> rand(int n, double s) =>
      List<double>.generate(n, (_) => (rng.nextDouble() * 2 - 1) * s);

  final upW = Tensor.fromList([_hidden, _model], rand(_hidden * _model, 0.1));
  final downW = Tensor.fromList([_model, _hidden], rand(_model * _hidden, 0.1));

  final mlp = TensorParallelMLP.fromWeights(
    upWeight: upW,
    downWeight: downW,
    trainable: true,
  );
  print('trainable params: ${mlp.parameters().length} shard tensor(s) across '
      '${mlp.up.devices} / ${mlp.down.devices}');

  final outDev = mlp.down.outputDevice;
  // Fixed input + target, resident on the output device.
  final x = Tensor.fromList(
    [_tokens, _model],
    rand(_tokens * _model, 0.5),
    device: Device.GPU,
  ).toGpu(outDev);
  final target = Tensor.fromList(
    [_tokens, _model],
    rand(_tokens * _model, 0.5),
    device: Device.GPU,
  ).toGpu(outDev);

  final opt = SGD(mlp.parameters(), lr: 0.1, momentum: 0.9);

  var first = 0.0;
  var last = 0.0;
  for (var step = 0; step < _steps; step++) {
    final out = mlp(x);
    final diff = out - target;
    final loss = (diff * diff).mean();
    final v = loss.toList().first;
    if (step == 0) first = v;
    last = v;

    opt.zeroGrad();
    loss.backward();
    opt.step();

    if (step % 10 == 0 || step == _steps - 1) {
      print('step ${step.toString().padLeft(3)}  loss ${v.toStringAsExponential(4)}');
    }
  }

  // Confirm every shard actually received a gradient on its own card.
  final gradded = mlp.parameters().where((p) => p.grad != null).length;
  print('params with gradients after training: $gradded / ${mlp.parameters().length}');
  print('loss: ${first.toStringAsExponential(4)} -> ${last.toStringAsExponential(4)}');
  print(last < first * 0.5
      ? 'OK — tensor-parallel training reduced the loss.'
      : 'WARNING — loss did not decrease as expected.');
}
