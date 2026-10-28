/// Greedy / sampled autoregressive decoding for the Needle 3 port.
///
/// The reference engine forces `<think>` as the first token of every
/// assistant turn, then decodes the one-line derivation and the
/// `<tool_call>[...]</tool_call>`. A port can stop at greedy decoding and
/// still reproduce the model; the byte-level grammar and confidence gates
/// are an optional layer on top (not implemented here).
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'needle_forward.dart';

/// Argmax index of [v] over `[0, length)`.
int argmax(List<double> v) {
  var best = 0;
  for (var i = 1; i < v.length; i++) {
    if (v[i] > v[best]) best = i;
  }
  return best;
}

/// Greedy (or temperature-sampled) decoding driven by a [lastLogits]
/// closure that returns the `[vocab]` logits for a given context.
///
/// Returns only the newly generated ids (prompt excluded). Stops when a
/// token in [stopIds] is produced or [maxNewTokens] is reached. If
/// [forceFirst] is non-null it is emitted as the first new token (the
/// engine forces `<think>`).
List<int> generateGreedy({
  required List<int> promptIds,
  required int vocab,
  required List<double> Function(List<int> ctx) lastLogits,
  int maxNewTokens = 64,
  Set<int> stopIds = const {},
  int? forceFirst,
  double temperature = 0.0,
  math.Random? rng,
}) {
  final ctx = List<int>.of(promptIds);
  final out = <int>[];
  final r = rng ?? math.Random();
  for (var step = 0; step < maxNewTokens; step++) {
    int next;
    if (step == 0 && forceFirst != null) {
      next = forceFirst;
    } else {
      final logits = lastLogits(ctx);
      next = temperature <= 0
          ? argmax(logits)
          : _sample(logits, temperature, r);
    }
    ctx.add(next);
    out.add(next);
    if (stopIds.contains(next)) break;
  }
  return out;
}

int _sample(List<double> logits, double temp, math.Random r) {
  var mx = double.negativeInfinity;
  for (final v in logits) {
    if (v > mx) mx = v;
  }
  final w = Float64List(logits.length);
  var sum = 0.0;
  for (var i = 0; i < logits.length; i++) {
    final e = math.exp((logits[i] - mx) / temp);
    w[i] = e;
    sum += e;
  }
  var x = r.nextDouble() * sum;
  for (var i = 0; i < w.length; i++) {
    x -= w[i];
    if (x <= 0) return i;
  }
  return w.length - 1;
}

/// Generate from a loaded [NeedleModelF]. Recomputes the full sequence each
/// step (no KV cache) — fine for the CPU port. Returns the new token ids.
List<int> generate(
  NeedleModelF model, {
  required List<int> promptIds,
  int maxNewTokens = 64,
  Set<int> stopIds = const {},
  int? forceFirst,
  double temperature = 0.0,
  math.Random? rng,
}) {
  final v = model.config.vocab;
  return generateGreedy(
    promptIds: promptIds,
    vocab: v,
    lastLogits: (ctx) {
      final all = model.logits(ctx);
      return all.sublist((ctx.length - 1) * v, ctx.length * v);
    },
    maxNewTokens: maxNewTokens,
    stopIds: stopIds,
    forceFirst: forceFirst,
    temperature: temperature,
    rng: rng,
  );
}
