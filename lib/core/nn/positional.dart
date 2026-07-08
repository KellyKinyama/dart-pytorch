/// Positional encodings for sequence models.
///
/// Two variants, both `Module`s that take a `[seqLen, embedDim]` input
/// and return `[seqLen, embedDim]` with a position-dependent bias
/// added:
///
///   * [SinusoidalPositionalEncoding] — fixed, non-trainable, from
///     "Attention Is All You Need" (Vaswani et al., 2017). No
///     parameters; the encoding is recomputed on each forward for the
///     exact sequence length (cheap, O(N * D), avoids needing a
///     slice op).
///   * [LearnedPositionalEmbedding] — a trainable table of shape
///     `[maxLen, embedDim]`, gathered by position indices
///     `[0, 1, ..., seqLen-1]` via the existing `Embedding` op.
///
/// Both variants expect input shape `[seqLen, embedDim]` (single
/// sequence — the same 2D convention used by the rest of `dart_pytorch`).
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../tensor/tensor.dart';
import 'embedding.dart';
import 'module.dart';

class SinusoidalPositionalEncoding extends Module {
  final int embedDim;

  SinusoidalPositionalEncoding(this.embedDim);

  /// Adds sinusoidal PE to `x`. `startPos` is the position of the
  /// first row of `x` (used by autoregressive / cached inference so
  /// that a single new token gets the encoding for its true position
  /// rather than position 0).
  Tensor call(Tensor x, {int startPos = 0}) {
    if (x.shape.length != 2 || x.shape[1] != embedDim) {
      throw ArgumentError(
        'SinusoidalPositionalEncoding: expected [*, $embedDim]; '
        'got ${x.shape}',
      );
    }
    final n = x.shape[0];
    final pe = _computePE(n, startPos, x.device);
    return x + pe;
  }

  Tensor _computePE(int n, int startPos, Device device) {
    final data = Float32List(n * embedDim);
    for (int row = 0; row < n; row++) {
      final pos = row + startPos;
      for (int i = 0; i < embedDim; i++) {
        final pairIdx = i ~/ 2;
        // freq = 1 / 10000^(2i/d)
        final freq = math.pow(10000.0, -2.0 * pairIdx / embedDim);
        final angle = pos * freq;
        data[row * embedDim + i] = (i.isEven
            ? math.sin(angle)
            : math.cos(angle));
      }
    }
    return Tensor.fromList([n, embedDim], data, device: device);
  }

  @override
  List<Tensor> parameters() => const [];
}

class LearnedPositionalEmbedding extends Module {
  final int maxLen;
  final int embedDim;
  final Embedding table;

  LearnedPositionalEmbedding(
    this.maxLen,
    this.embedDim, {
    Device device = Device.CPU,
    int seed = 0,
  }) : table = Embedding(maxLen, embedDim, device: device, seed: seed);

  Tensor call(Tensor x, {int startPos = 0}) {
    if (x.shape.length != 2 || x.shape[1] != embedDim) {
      throw ArgumentError(
        'LearnedPositionalEmbedding: expected [*, $embedDim]; '
        'got ${x.shape}',
      );
    }
    final n = x.shape[0];
    if (startPos + n > maxLen) {
      throw ArgumentError(
        'LearnedPositionalEmbedding: startPos+seqLen ${startPos + n} '
        'exceeds maxLen $maxLen',
      );
    }
    final positions = Tensor.fromList(
      [n],
      List<double>.generate(n, (i) => (i + startPos).toDouble()),
      device: x.device,
    );
    return x + table(positions);
  }

  @override
  List<Tensor> parameters() => table.parameters();

  @override
  List<Module> submodules() => [table];
}
