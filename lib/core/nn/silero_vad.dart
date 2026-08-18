/// Silero VAD v5 architecture port.
///
/// Faithful reimplementation of the model shipped in
/// `silero-vad/data/silero_vad.onnx` (16 kHz branch). Reads a 512-
/// sample audio window at 16 kHz and returns the probability that
/// the window contains speech. State (`h`, `c`) is carried between
/// windows for streaming.
///
/// Pipeline (per 512-sample chunk):
///
///   1. Pad audio on the left with 64 zeros (or with the previous
///      window's tail — kept in [context]).
///   2. Learned STFT: Conv1d(1, 258, kernel=256, stride=128).
///   3. Magnitude: split output into (real, imag) 129-channel halves
///      and compute sqrt(real^2 + imag^2 + eps).
///   4. Encoder: 4× Conv1d(k=3, pad=1) + ReLU.
///   5. LSTMCell: hidden=128. State (h, c) persists across chunks.
///   6. Head: ReLU + Conv1d(128, 1, k=1) + Sigmoid.
///   7. Mean over the time dimension.
///
/// Load pretrained weights with [SileroVadReader.readFile] and pass
/// them into [SileroVad.loadFromDpt].
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../tensor/tensor.dart';
import 'conv1d.dart';
import 'lstm_cell.dart';
import 'module.dart';

/// Recurrent state carried across chunks. `h` and `c` are both shape
/// `[batch, 128]`. [SileroVad.zeroState] returns fresh zeros.
class SileroState {
  final Tensor h;
  final Tensor c;
  const SileroState(this.h, this.c);
}

class SileroVad extends Module {
  static const int sampleRate = 16000;
  static const int chunkSize = 512;
  static const int stftKernel = 256;
  static const int stftStride = 128;
  static const int stftBins = 129; // (256/2) + 1
  static const int contextSize = 64;
  static const int hiddenSize = 128;
  static const double magEps = 1e-9;

  late Conv1d stftConv;
  late Conv1d enc0;
  late Conv1d enc1;
  late Conv1d enc2;
  late Conv1d enc3;
  late LSTMCell lstm;
  late Conv1d head;

  /// Device holding all module weights and running matmuls / adds.
  /// Element-wise CPU-only steps (magnitude, sigmoid, tanh, etc.) run
  /// in Dart regardless — they're not on the hot path for VAD.
  final Device device;

  SileroVad({this.device = Device.CPU}) {
    stftConv = Conv1d(
      inChannels: 1,
      outChannels: 2 * stftBins,
      kernelSize: stftKernel,
      stride: stftStride,
      bias: false,
      device: device,
    );
    enc0 = Conv1d(
      inChannels: stftBins,
      outChannels: 128,
      kernelSize: 3,
      padding: 1,
      device: device,
    );
    enc1 = Conv1d(
      inChannels: 128,
      outChannels: 64,
      kernelSize: 3,
      stride: 2,
      padding: 1,
      device: device,
    );
    enc2 = Conv1d(
      inChannels: 64,
      outChannels: 64,
      kernelSize: 3,
      stride: 2,
      padding: 1,
      device: device,
    );
    enc3 = Conv1d(
      inChannels: 64,
      outChannels: 128,
      kernelSize: 3,
      padding: 1,
      device: device,
    );
    lstm = LSTMCell(128, hiddenSize, device: device);
    head = Conv1d(
      inChannels: 128,
      outChannels: 1,
      kernelSize: 1,
      device: device,
    );
  }

  /// Fresh zero state.
  SileroState zeroState({int batch = 1}) {
    return SileroState(
      Tensor.fill([batch, hiddenSize], 0.0, device: device),
      Tensor.fill([batch, hiddenSize], 0.0, device: device),
    );
  }

  /// Fresh zero context (the 64-sample tail carried across calls).
  Float32List zeroContext({int batch = 1}) => Float32List(batch * contextSize);

  /// Run one 512-sample chunk. `input` is `[batch, chunkSize]` on CPU.
  /// `context` is the previous chunk's last 64 samples (or zeros on
  /// the first call). Returns `(probability, newState, newContext)`.
  ///
  ///   probability: `[batch, 1]`
  ///   newState:    LSTM (h, c) for next call
  ///   newContext:  the last 64 samples of `input`, to prepend next
  ///                time
  ({Tensor prob, SileroState state, Float32List context}) callChunk({
    required Tensor input,
    required SileroState state,
    required Float32List context,
  }) {
    if (input.shape.length != 2 || input.shape[1] != chunkSize) {
      throw ArgumentError('input must be [B, $chunkSize]; got ${input.shape}');
    }
    final b = input.shape[0];
    if (context.length != b * contextSize) {
      throw ArgumentError(
        'context length ${context.length} != $b × $contextSize',
      );
    }

    // Prepend `context` to input → concat then reflect-pad the RIGHT
    // side by 64 samples so the STFT produces 4 frames per chunk
    // (matches the official silero_vad.onnx layout).
    final inputData = input.toFloat32List();
    const paddedLen = contextSize + chunkSize + contextSize; // 64+512+64=640
    final full = Float32List(b * 1 * paddedLen);
    for (int bi = 0; bi < b; bi++) {
      final ctxOff = bi * contextSize;
      final inOff = bi * chunkSize;
      final rowBase = bi * paddedLen;
      for (int i = 0; i < contextSize; i++) {
        full[rowBase + i] = context[ctxOff + i];
      }
      for (int i = 0; i < chunkSize; i++) {
        full[rowBase + contextSize + i] = inputData[inOff + i];
      }
      // Reflect padding on the right: mirror the last `contextSize`
      // samples of the concatenated signal, EXCLUDING the boundary
      // sample itself (PyTorch/numpy reflect convention).
      for (int i = 0; i < contextSize; i++) {
        // reflected index into signal[0..contextSize+chunkSize-1]:
        // for right pad j∈[0..63], signal[(chunkSize+contextSize) - 2 - j]
        final src = (contextSize + chunkSize) - 2 - i;
        full[rowBase + contextSize + chunkSize + i] = full[rowBase + src];
      }
    }
    final padded = Tensor.fromFloat32List(
      [b, 1, paddedLen],
      full,
      device: device,
    );

    // STFT: [B, 258, Lstft] where Lstft = (576 - 256) / 128 + 1 = 3.
    final stft = stftConv(padded);

    // Magnitude: sqrt(real^2 + imag^2 + eps).
    final mag = _magnitude(stft, b);

    // Encoder.
    var h = enc0(mag).relu();
    h = enc1(h).relu();
    h = enc2(h).relu();
    h = enc3(h).relu();
    // h: [B, 128, Lstft]

    // For each timestep in Lstft, run the LSTM cell. Silero unrolls
    // the recurrence over the encoded frames.
    final lstmSteps = h.shape[2];
    var hState = state.h;
    var cState = state.c;
    final headInputs = <Tensor>[];
    for (int t = 0; t < lstmSteps; t++) {
      final slice = _sliceTime(h, t); // [B, 128]
      final res = lstm(slice, hState, cState);
      hState = res.h;
      cState = res.c;
      headInputs.add(hState);
    }
    // Stack back to [B, 128, Lstft].
    final headIn = _stackTime(headInputs, b, 128, lstmSteps);

    // Head: ReLU + Conv1d(128, 1, k=1) + Sigmoid.
    final logits = head(headIn.relu());
    final prob = _sigmoidTensor(logits);

    // Mean over time → [B, 1].
    final probData = prob.toFloat32List();
    final out = Float32List(b);
    for (int bi = 0; bi < b; bi++) {
      var s = 0.0;
      for (int t = 0; t < lstmSteps; t++) {
        s += probData[bi * lstmSteps + t];
      }
      out[bi] = s / lstmSteps;
    }
    final probOut = Tensor.fromFloat32List([b, 1], out, device: device);

    // New context: last 64 samples of input.
    final newCtx = Float32List(b * contextSize);
    for (int bi = 0; bi < b; bi++) {
      for (int i = 0; i < contextSize; i++) {
        newCtx[bi * contextSize + i] =
            inputData[bi * chunkSize + (chunkSize - contextSize + i)];
      }
    }

    return (prob: probOut, state: SileroState(hState, cState), context: newCtx);
  }

  Tensor _magnitude(Tensor stft, int b) {
    // stft: [B, 258, L]. Split into 129 real + 129 imag.
    final data = stft.toFloat32List();
    final l = stft.shape[2];
    final mag = Float32List(b * stftBins * l);
    for (int bi = 0; bi < b; bi++) {
      for (int c = 0; c < stftBins; c++) {
        for (int t = 0; t < l; t++) {
          final re = data[bi * 2 * stftBins * l + c * l + t];
          final im = data[bi * 2 * stftBins * l + (stftBins + c) * l + t];
          mag[bi * stftBins * l + c * l + t] = math.sqrt(
            re * re + im * im + magEps,
          );
        }
      }
    }
    return Tensor.fromFloat32List([b, stftBins, l], mag, device: device);
  }

  Tensor _sliceTime(Tensor bcl, int t) {
    // bcl: [B, C, L] → [B, C] at time t.
    final b = bcl.shape[0];
    final c = bcl.shape[1];
    final l = bcl.shape[2];
    final src = bcl.toFloat32List();
    final out = Float32List(b * c);
    for (int bi = 0; bi < b; bi++) {
      for (int ci = 0; ci < c; ci++) {
        out[bi * c + ci] = src[bi * c * l + ci * l + t];
      }
    }
    return Tensor.fromFloat32List([b, c], out, device: device);
  }

  Tensor _stackTime(List<Tensor> steps, int b, int c, int l) {
    final out = Float32List(b * c * l);
    for (int t = 0; t < l; t++) {
      final s = steps[t].toFloat32List();
      for (int bi = 0; bi < b; bi++) {
        for (int ci = 0; ci < c; ci++) {
          out[bi * c * l + ci * l + t] = s[bi * c + ci];
        }
      }
    }
    return Tensor.fromFloat32List([b, c, l], out, device: device);
  }

  Tensor _sigmoidTensor(Tensor t) {
    final d = t.toFloat32List();
    final out = Float32List(d.length);
    for (int i = 0; i < d.length; i++) {
      final x = d[i];
      out[i] = x >= 0
          ? 1.0 / (1.0 + math.exp(-x))
          : (math.exp(x) / (1.0 + math.exp(x)));
    }
    return Tensor.fromFloat32List(t.shape, out, device: device);
  }

  @override
  List<Tensor> parameters() {
    return [
      ...stftConv.parameters(),
      ...enc0.parameters(),
      ...enc1.parameters(),
      ...enc2.parameters(),
      ...enc3.parameters(),
      ...lstm.parameters(),
      ...head.parameters(),
    ];
  }
}
