/// Shared helpers for the tiny-Shakespeare demos in this folder.
///
/// Loads the character-level corpus from `data/tiny_shakespeare.txt`,
/// builds a [CharTokenizer], batches windows for training, and
/// implements minimal temperature / top-K sampling from a logits row.
///
/// This file is imported by `shakespeare_gpt.dart`,
/// `shakespeare_transformer_lm.dart`, `shakespeare_aft.dart`, and
/// `shakespeare_moe.dart`. It has no `main`.
library;

import 'dart:io';
import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

/// Path to the character-level corpus, resolved relative to the
/// repository root.
const String kShakespearePath = 'data/tiny_shakespeare.txt';

/// Load the tiny-Shakespeare corpus. `maxChars` truncates to the
/// leading `maxChars` characters (default: the full file) — useful
/// for keeping demo runtimes short.
String loadCorpus({int? maxChars}) {
  final f = File(kShakespearePath);
  if (!f.existsSync()) {
    stderr.writeln(
      'shakespeare_util: could not find $kShakespearePath. '
      'Run demos from the repository root.',
    );
    exit(1);
  }
  final text = f.readAsStringSync();
  if (maxChars == null || maxChars >= text.length) return text;
  return text.substring(0, maxChars);
}

/// Sample a random `[batch, blockSize+1]` window from `ids` and return
/// `(x, y)` where `x` = tokens `[start .. start+blockSize)` and `y` =
/// next tokens `[start+1 .. start+blockSize+1)`. Both are flat 1D
/// tensors of length `batch * blockSize`.
///
/// For simplicity (and to match the rest of the library, most of which
/// is 2D-only), we return **one** window at a time (`batch == 1`) as a
/// 1D `[blockSize]` tensor pair. Multi-window batching is left to the
/// caller.
(Tensor, Tensor) getWindow(List<int> ids, int blockSize, math.Random rng) {
  final maxStart = ids.length - blockSize - 1;
  if (maxStart <= 0) {
    throw ArgumentError(
      'getWindow: corpus (${ids.length}) too short for blockSize $blockSize',
    );
  }
  final start = rng.nextInt(maxStart);
  final x = List<double>.generate(blockSize, (i) => ids[start + i].toDouble());
  final y = List<double>.generate(
    blockSize,
    (i) => ids[start + i + 1].toDouble(),
  );
  return (Tensor.fromList([blockSize], x), Tensor.fromList([blockSize], y));
}

/// Sample one token index from a raw logits row using temperature +
/// optional top-K nucleus.  `temperature <= 0` (or `topK == 1`)
/// collapses to greedy argmax.
int sampleFromLogits(
  List<double> logits, {
  double temperature = 1.0,
  int? topK,
  math.Random? rng,
}) {
  final v = logits.length;
  final greedy = temperature <= 0.0 || (topK != null && topK <= 1);
  if (greedy) {
    var bestI = 0;
    var bestV = logits[0];
    for (int i = 1; i < v; i++) {
      if (logits[i] > bestV) {
        bestV = logits[i];
        bestI = i;
      }
    }
    return bestI;
  }
  final r = rng ?? math.Random();
  final scaled = List<double>.generate(v, (i) => logits[i] / temperature);
  if (topK != null && topK < v) {
    final sorted = List<int>.generate(v, (i) => i)
      ..sort((a, b) => scaled[b].compareTo(scaled[a]));
    final keep = sorted.take(topK).toSet();
    for (int i = 0; i < v; i++) {
      if (!keep.contains(i)) scaled[i] = double.negativeInfinity;
    }
  }
  var maxV = -double.infinity;
  for (final s in scaled) {
    if (s > maxV) maxV = s;
  }
  final probs = List<double>.filled(v, 0.0);
  var sum = 0.0;
  for (int i = 0; i < v; i++) {
    probs[i] = math.exp(scaled[i] - maxV);
    sum += probs[i];
  }
  final u = r.nextDouble() * sum;
  var acc = 0.0;
  for (int i = 0; i < v; i++) {
    acc += probs[i];
    if (u <= acc) return i;
  }
  return v - 1;
}

/// Autoregressive sampling loop for a 1D language model. `stepFn`
/// takes the current context (List of ids as doubles) and returns the
/// logits row for the *last* position (length = vocab). This is the
/// portable no-cache path shared across all four demos.
List<int> generateText(
  List<int> prompt, {
  required int maxNewTokens,
  required int maxCtx,
  required int vocabSize,
  required List<double> Function(List<double> ctx) stepFn,
  double temperature = 1.0,
  int? topK,
  math.Random? rng,
}) {
  final out = List<int>.of(prompt);
  final r = rng ?? math.Random();
  for (int step = 0; step < maxNewTokens; step++) {
    final start = out.length > maxCtx ? out.length - maxCtx : 0;
    final ctxD = List<double>.generate(
      out.length - start,
      (i) => out[start + i].toDouble(),
    );
    final logits = stepFn(ctxD);
    final row = logits.length == vocabSize
        ? logits
        : logits.sublist(logits.length - vocabSize);
    final next = sampleFromLogits(row, temperature: temperature, topK: topK, rng: r);
    out.add(next);
  }
  return out;
}

/// Convenience: build the last-row logits `List<double>` from a full
/// forward pass output of shape `[seqLen, vocab]`.
List<double> lastRowLogits(Tensor logits, int vocabSize) {
  final flat = logits.toList();
  final n = flat.length ~/ vocabSize;
  final base = (n - 1) * vocabSize;
  return List<double>.generate(vocabSize, (i) => flat[base + i]);
}

/// Total scalar parameter count, useful for demo headers.
int paramScalarCount(List<Tensor> params) =>
    params.fold<int>(0, (a, p) => a + p.length);
