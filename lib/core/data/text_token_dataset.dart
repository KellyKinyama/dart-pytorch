/// Language-model sliding-window dataset.
///
/// Reads an entire text file (or a plain string), tokenizes it with a
/// caller-supplied tokenizer, and exposes one training example per
/// valid starting index: for each position `i`, the input is
/// `tokens[i .. i + blockSize)` and the target is
/// `tokens[i + 1 .. i + blockSize + 1)` (shifted-by-one next-token
/// prediction, as in GPT-style causal LM training).
///
/// The dataset length is therefore `numTokens - blockSize`. Each item
/// is a [LmSample] of two `[blockSize]` [Tensor]s on the requested
/// device.
///
/// The tokenizer only needs to expose `List<int> encode(String)` —
/// both [CharTokenizer] and [BpeTokenizer] satisfy this.
library;

import 'dart:io';

import '../tensor/tensor.dart';
import 'dataset.dart';

/// The interface [TextTokenDataset] needs from a tokenizer. Both
/// [CharTokenizer] and [BpeTokenizer] satisfy this via structural
/// typing (Dart uses subtyping, so we accept anything with an
/// `encode` method matching this shape).
abstract class TextEncoder {
  List<int> encode(String s);
}

/// One `(input, target)` pair for next-token prediction training.
class LmSample {
  final Tensor input; // [blockSize] float ids
  final Tensor target; // [blockSize] float ids
  const LmSample(this.input, this.target);
}

class TextTokenDataset extends Dataset<LmSample> {
  final List<int> tokens;
  final int blockSize;
  final Device device;

  int get numTokens => tokens.length;

  TextTokenDataset._(this.tokens, this.blockSize, this.device) {
    if (blockSize < 1) {
      throw ArgumentError('blockSize must be >= 1, got $blockSize');
    }
    if (tokens.length <= blockSize) {
      throw ArgumentError(
        'need > blockSize tokens for at least one example, got '
        '${tokens.length} tokens with blockSize=$blockSize',
      );
    }
  }

  /// Tokenize `text` and expose sliding-window LM examples.
  factory TextTokenDataset.fromText(
    String text, {
    required dynamic tokenizer,
    required int blockSize,
    Device device = Device.CPU,
  }) {
    // Structural typing: accept anything with `List<int> encode(String)`.
    final List<int> ids;
    try {
      ids = (tokenizer as dynamic).encode(text) as List<int>;
    } catch (e) {
      throw ArgumentError(
        'tokenizer must expose `List<int> encode(String)`; '
        'got ${tokenizer.runtimeType} ($e)',
      );
    }
    return TextTokenDataset._(ids, blockSize, device);
  }

  /// Tokenize the contents of a UTF-8 text file. Convenience wrapper
  /// over [TextTokenDataset.fromText].
  factory TextTokenDataset.fromFile(
    String path, {
    required dynamic tokenizer,
    required int blockSize,
    Device device = Device.CPU,
  }) {
    final text = File(path).readAsStringSync();
    return TextTokenDataset.fromText(
      text,
      tokenizer: tokenizer,
      blockSize: blockSize,
      device: device,
    );
  }

  /// Build directly from a pre-tokenized id list — bypasses the
  /// tokenizer requirement, useful when tokens live in a compact
  /// binary file.
  factory TextTokenDataset.fromTokens(
    List<int> tokens, {
    required int blockSize,
    Device device = Device.CPU,
  }) =>
      TextTokenDataset._(List<int>.from(tokens), blockSize, device);

  @override
  int get length => tokens.length - blockSize;

  @override
  LmSample operator [](int index) {
    if (index < 0 || index >= length) {
      throw RangeError.range(index, 0, length - 1, 'index');
    }
    final xs = List<double>.generate(
      blockSize,
      (k) => tokens[index + k].toDouble(),
    );
    final ys = List<double>.generate(
      blockSize,
      (k) => tokens[index + 1 + k].toDouble(),
    );
    return LmSample(
      Tensor.fromList([blockSize], xs, device: device),
      Tensor.fromList([blockSize], ys, device: device),
    );
  }
}
