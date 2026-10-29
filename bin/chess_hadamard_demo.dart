/// Train a small decoder LM whose feed-forward sublayer is the
/// near-parameter-free [HadamardFFN] (Cactus Needle's Hadamard MLP idea) on
/// Lichess UCI games, then sample a continuation and check legality.
///
/// This is the Hadamard-FFN counterpart to `bin/chess_gpt_demo.dart`: same
/// corpus, same attention, same flags and legality check — only the FFN is
/// swapped, so you can compare parameter count and loss directly. Each layer's
/// FFN uses `3*dim` params instead of a dense FFN's ~`8*dim^2`.
///
///   dart run bin/chess_hadamard_demo.dart --games 200 --steps 400
///   dart run bin/chess_hadamard_demo.dart --prompt "e2e4 e7e5 " --temp 0.5
///
/// `--dim` must be a power of two. CPU by default.
library;

import 'dart:convert';
import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

import 'chess_util.dart';
import 'shakespeare_util.dart' show paramScalarCount, generateText, lastRowLogits;

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

/// A minimal pre-LN decoder LM: token + learned positional embeddings, a stack
/// of (LayerNorm → attention → residual, LayerNorm → HadamardFFN → residual)
/// blocks, a final LayerNorm, and a weight-tied output head.
class HadamardLM extends Module {
  final int maxCtx, embedDim, numLayers, numHeads;
  final Embedding tokenEmb;
  final LearnedPositionalEmbedding posEmb;
  final List<LayerNorm> ln1 = [];
  final List<MultiHeadAttention> attn = [];
  final List<LayerNorm> ln2 = [];
  final List<HadamardFFN> ffn = [];
  final LayerNorm finalLn;

  HadamardLM({
    required int vocabSize,
    required this.maxCtx,
    required this.embedDim,
    required this.numLayers,
    required this.numHeads,
    int seed = 0,
  })  : tokenEmb = Embedding(vocabSize, embedDim, seed: seed),
        posEmb = LearnedPositionalEmbedding(maxCtx, embedDim, seed: seed + 11),
        finalLn = LayerNorm(embedDim) {
    for (var l = 0; l < numLayers; l++) {
      ln1.add(LayerNorm(embedDim));
      attn.add(MultiHeadAttention(embedDim, numHeads, seed: seed + 100 * l));
      ln2.add(LayerNorm(embedDim));
      ffn.add(HadamardFFN(embedDim, seed: seed + 200 * l));
    }
  }

  Tensor call(Tensor ids) {
    final n = ids.shape[0];
    var h = posEmb(tokenEmb(ids));
    final mask = causalMask(n, device: h.device);
    for (var l = 0; l < numLayers; l++) {
      h = h + attn[l](ln1[l](h), mask: mask);
      h = h + ffn[l](ln2[l](h));
    }
    h = finalLn(h);
    return h.matmul(tokenEmb.weight.transpose());
  }

  @override
  List<Tensor> parameters() => [
        ...tokenEmb.parameters(),
        ...posEmb.parameters(),
        for (final m in ln1) ...m.parameters(),
        for (final m in attn) ...m.parameters(),
        for (final m in ln2) ...m.parameters(),
        for (final m in ffn) ...m.parameters(),
        ...finalLn.parameters(),
      ];
}

void _sampleAndCheck(
  HadamardLM model,
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
  final numHeads = _argInt(args, '--heads', 4);
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
  print('=== chess-move LM (Hadamard-FFN decoder, UCI) ===');
  print('corpus: ${ids.length} chars, vocab: ${tok.vocabSize} | '
      'dim: $embedDim, layers: $numLayers, heads: $numHeads, seq: $seqLen');

  final model = HadamardLM(
    vocabSize: tok.vocabSize,
    maxCtx: seqLen,
    embedDim: embedDim,
    numLayers: numLayers,
    numHeads: numHeads,
    seed: 1,
  );
  final params = model.parameters();
  final ffnParams = 3 * embedDim * numLayers;
  print('model : HadamardLM ${paramScalarCount(params)} scalars  '
      '(FFN = $ffnParams, vs dense ~${8 * embedDim * embedDim * numLayers})');

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

  _sampleAndCheck(model, tok, seqLen, prompt, genLen, temp, topK);
}
