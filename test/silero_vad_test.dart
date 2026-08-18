import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:test/test.dart';

const _weightsPath = 'models/silero_vad/silero_vad.dpt';

void main() {
  group('SileroVad', () {
    final hasWeights = File(_weightsPath).existsSync();
    if (!hasWeights) {
      test('weights missing → tests skipped', () {}, skip:
          'Run `python3 scripts/extract_silero_vad.py '
          'models/silero_vad/silero_vad.onnx '
          'models/silero_vad/silero_vad.dpt` first.');
      return;
    }

    late SileroVad model;
    setUpAll(() {
      model = SileroVad();
      SileroVadReader.loadFile(model, _weightsPath);
    });

    test('load produces 14 parameter tensors', () {
      expect(model.parameters().length, equals(14));
    });

    test('silence gives near-zero probability', () {
      var state = model.zeroState();
      var ctx = model.zeroContext();
      final silence = Tensor.fill(
        [1, SileroVad.chunkSize],
        0.0,
        device: Device.CPU,
      );
      // Warm one chunk, then check.
      final r0 = model.callChunk(input: silence, state: state, context: ctx);
      state = r0.state;
      ctx = r0.context;
      final r1 = model.callChunk(input: silence, state: state, context: ctx);
      final p = r1.prob.toList()[0];
      expect(p, lessThan(0.05));
    });

    test('440 Hz sine → matches ONNX reference within tolerance', () {
      // Reference values captured from onnxruntime on the same input.
      const ref = [
        0.001670,
        0.066041,
        0.028639,
        0.013030,
        0.009366,
        0.006642,
      ];
      var state = model.zeroState();
      var ctx = model.zeroContext();
      for (int i = 0; i < ref.length; i++) {
        final chunk = Float32List(SileroVad.chunkSize);
        if (i != 0) {
          for (int k = 0; k < chunk.length; k++) {
            final t = (k + i * chunk.length) / 16000.0;
            chunk[k] = 0.3 * math.sin(2 * math.pi * 440 * t);
          }
        }
        final input = Tensor.fromFloat32List(
          [1, SileroVad.chunkSize],
          chunk,
          device: Device.CPU,
        );
        final r = model.callChunk(input: input, state: state, context: ctx);
        state = r.state;
        ctx = r.context;
        final p = r.prob.toList()[0];
        // 20 % relative tolerance covers FP32 exp/tanh ordering differences.
        expect(
          p,
          closeTo(ref[i], 0.2 * math.max(ref[i], 0.01)),
          reason: 'chunk $i',
        );
      }
    });

    test('state persists across chunks (not reset each call)', () {
      final silence = Tensor.fill(
        [1, SileroVad.chunkSize],
        0.0,
        device: Device.CPU,
      );
      final state0 = model.zeroState();
      final ctx0 = model.zeroContext();
      // One call from zero state.
      final r1 = model.callChunk(
        input: silence,
        state: state0,
        context: ctx0,
      );
      // Second call from the returned state — result should differ, because
      // LSTM state advances.
      final r2 = model.callChunk(
        input: silence,
        state: r1.state,
        context: r1.context,
      );
      // Compare LSTM h vectors.
      final h1 = r1.state.h.toList();
      final h2 = r2.state.h.toList();
      var maxDiff = 0.0;
      for (int i = 0; i < h1.length; i++) {
        final d = (h1[i] - h2[i]).abs();
        if (d > maxDiff) maxDiff = d;
      }
      expect(maxDiff, greaterThan(1e-6),
          reason: 'state should evolve between chunks');
    });
  });
}
