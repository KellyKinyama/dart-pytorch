/// Train an **Attention-Free Transformer** ([AFTLanguageModel]) to generate
/// chess moves in **UCI**, then sample a continuation and check how many of
/// its moves are legal with the `chess` package.
///
/// This is the AFT counterpart to `bin/chess_gpt_demo.dart` (standard GPT)
/// and `bin/chess_moves_lm_demo.dart` (context-free linear-attention
/// [CharStateLM]). All three read the same Lichess UCI corpus
/// (data/chess_uci.txt), use the same flags and the same legality check, so
/// the models can be compared head-to-head. AFT scales linearly with
/// sequence length per layer (no `T x T` attention matrix).
///
///   dart run bin/chess_aft_demo.dart
///   dart run bin/chess_aft_demo.dart --user DrNykterstein --games 800 --steps 2000
///   dart run bin/chess_aft_demo.dart --prompt "e2e4 e7e5 " --temp 0.5 --topk 20
///
/// CPU by default.
library;

import 'dart:convert';
import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

import 'chess_util.dart';
import 'shakespeare_util.dart'
    show paramScalarCount, generateText, lastRowLogits;

int _argInt(List<String> a, String name, int fallback) {
  final i = a.indexOf(name);
  if (i >= 0 && i + 1 < a.length) return int.tryParse(a[i + 1]) ?? fallback;
  return fallback;
}

String _argStr(List<String> a, String name, String fallback) {
  final i = a.indexOf(name);
  return (i >= 0 && i + 1 < a.length) ? a[i + 1] : fallback;
}

double _argDouble(List<String> a, String name, double fallback) {
  final i = a.indexOf(name);
  if (i >= 0 && i + 1 < a.length) return double.tryParse(a[i + 1]) ?? fallback;
  return fallback;
}

/// Sample a continuation from [prompt] and report how many of the moves
/// (prompt + generated) are legal from the starting position.
void _sampleAndCheck(
  AFTLanguageModel model,
  CharTokenizer tok,
  int maxCtx,
  String prompt,
  int genLen,
  double temp,
  int? topK,
) {
  List<double> stepFn(List<double> ctx) {
    final x = Tensor.fromList([ctx.length], ctx, device: Device.CPU);
    return lastRowLogits(model(x), tok.vocabSize);
  }

  final gen = generateText(
    tok.encode(prompt),
    maxNewTokens: genLen,
    maxCtx: maxCtx,
    vocabSize: tok.vocabSize,
    stepFn: stepFn,
    temperature: temp,
    topK: topK,
    rng: math.Random(7),
  );
  final text = tok.decode(gen);
  print('');
  print('prompt:     ${jsonEncode(prompt)}');
  print('generated:  $text');

  final rep = checkUciLegality(text);
  print(rep);
  if (rep.legal > 0) print('position after legal moves:\n${rep.boardAscii}');
}

Future<void> main(List<String> args) async {
  final user = _argStr(args, '--user', 'DrNykterstein');
  final games = _argInt(args, '--games', 400);
  final plies = _argInt(args, '--plies', 40);
  final regen = args.contains('--regen');
  final steps = _argInt(args, '--steps', 400);
  final seqLen = _argInt(args, '--seq', 64);
  final embedDim = _argInt(args, '--dim', 128);
  final numLayers = _argInt(args, '--layers', 3);
  final prompt = _argStr(args, '--prompt', 'e2e4 ');
  final genLen = _argInt(args, '--gen', 80);
  final temp = _argDouble(args, '--temp', 0.6);
  final topKArg = _argInt(args, '--topk', 0);
  final topK = topKArg > 0 ? topKArg : null;
  final lr = _argDouble(args, '--lr', 3e-3);

  final text = await loadChessUci(
    user: user,
    maxGames: games,
    maxPlies: plies,
    regenerate: regen,
  );
  final tok = CharTokenizer.fromText(text);
  final ids = tok.encode(text);
  print('=== chess-move LM (Attention-Free Transformer, UCI) ===');
  print('corpus: ${ids.length} chars, vocab: ${tok.vocabSize} | '
      'dim: $embedDim, layers: $numLayers, seq: $seqLen');

  final model = AFTLanguageModel(
    vocabSize: tok.vocabSize,
    embedDim: embedDim,
    numLayers: numLayers,
    maxLen: seqLen,
    seed: 1,
  );
  final params = model.parameters();
  print('model : AFTLanguageModel ${paramScalarCount(params)} scalars');

  final opt = Adam(params, lr: lr);
  final rng = math.Random(0);
  final sw = Stopwatch()..start();

  for (var step = 0; step < steps; step++) {
    final start = rng.nextInt(ids.length - seqLen - 1);
    final x = Tensor.fromList(
      [seqLen],
      [for (var i = 0; i < seqLen; i++) ids[start + i].toDouble()],
    );
    final y = Tensor.fromList(
      [seqLen],
      [for (var i = 0; i < seqLen; i++) ids[start + i + 1].toDouble()],
    );

    final logits = model(x); // [seqLen, vocab]
    final loss = logits.crossEntropy(y).mean();
    final v = loss.toList().first;

    opt.zeroGrad();
    loss.backward();
    clipGradNorm(params, 1.0);
    opt.step();

    if (step % 25 == 0 || step == steps - 1) {
      final rate = (step + 1) / (sw.elapsedMilliseconds / 1000);
      print('step ${step.toString().padLeft(4)}  loss ${v.toStringAsFixed(4)}'
          '  (${rate.toStringAsFixed(1)} it/s)');
    }
  }

  model.eval();
  _sampleAndCheck(model, tok, seqLen, prompt, genLen, temp, topK);
}
