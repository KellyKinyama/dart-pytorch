import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

const _weightsPath = 'models/silero_vad/silero_vad.dpt';

void main() {
  group('SileroVad on GPU', () {
    if (!File(_weightsPath).existsSync()) {
      test(
        'weights missing → GPU tests skipped',
        () {},
        skip: 'Run scripts/extract_silero_vad.py first.',
      );
      return;
    }

    late SileroVad cpuModel;
    late SileroVad gpuModel;
    setUpAll(() {
      cpuModel = SileroVad();
      SileroVadReader.loadFile(cpuModel, _weightsPath);
      gpuModel = SileroVad(device: Device.GPU);
      SileroVadReader.loadFile(gpuModel, _weightsPath);
    });

    test('CPU and GPU forward agree on silence', () {
      final silence = Tensor.fill(
        [1, SileroVad.chunkSize],
        0.0,
        device: Device.CPU,
      );
      final rCpu = cpuModel.callChunk(
        input: silence,
        state: cpuModel.zeroState(),
        context: cpuModel.zeroContext(),
      );
      final rGpu = gpuModel.callChunk(
        input: silence,
        state: gpuModel.zeroState(),
        context: gpuModel.zeroContext(),
      );
      expect(rGpu.prob.toList()[0], closeTo(rCpu.prob.toList()[0], 1e-4));
    });

    test('CPU and GPU forward agree on 440 Hz sine sequence', () {
      var stateCpu = cpuModel.zeroState();
      var ctxCpu = cpuModel.zeroContext();
      var stateGpu = gpuModel.zeroState();
      var ctxGpu = gpuModel.zeroContext();

      for (int i = 0; i < 6; i++) {
        final chunk = Float32List(SileroVad.chunkSize);
        if (i != 0) {
          for (int k = 0; k < chunk.length; k++) {
            final t = (k + i * chunk.length) / 16000.0;
            chunk[k] = 0.3 * math.sin(2 * math.pi * 440 * t);
          }
        }
        final inputCpu = Tensor.fromFloat32List(
          [1, SileroVad.chunkSize],
          Float32List.fromList(chunk),
          device: Device.CPU,
        );
        final inputGpu = Tensor.fromFloat32List(
          [1, SileroVad.chunkSize],
          Float32List.fromList(chunk),
          device: Device.CPU,
        );
        final rCpu = cpuModel.callChunk(
          input: inputCpu,
          state: stateCpu,
          context: ctxCpu,
        );
        final rGpu = gpuModel.callChunk(
          input: inputGpu,
          state: stateGpu,
          context: ctxGpu,
        );
        stateCpu = rCpu.state;
        ctxCpu = rCpu.context;
        stateGpu = rGpu.state;
        ctxGpu = rGpu.context;
        expect(
          rGpu.prob.toList()[0],
          closeTo(rCpu.prob.toList()[0], 1e-3),
          reason: 'chunk $i',
        );
      }
    });
  });
}
