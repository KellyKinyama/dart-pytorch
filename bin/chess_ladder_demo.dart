/// Train one **laddered** chess LM that is deployable at any depth, after
/// Cactus Needle's intelligence ladder: every prefix-of-blocks (selected by
/// endpoint-preserving bisection) is a working sub-network, so a single set of
/// weights runs at depth 1..L without retraining.
///
/// Uses the [HadamardFFN] mixer; each step trains the full depth plus one
/// random shallower rung, so all depths learn. After training it reports the
/// validation loss at every depth (the "ladder") and generates at `--depth`.
///
///   dart run bin/chess_ladder_demo.dart --games 200 --steps 400 --layers 6
///   dart run bin/chess_ladder_demo.dart --depth 3 --prompt "e2e4 "
///
/// `--dim` must be a power of two. CPU by default.
library;

import 'dart:convert';
import 'dart:math' as math;

import 'package:chess/chess.dart' as ch;
import 'package:dart_pytorch/dart_pytorch.dart';

import 'chess_util.dart';
import 'shakespeare_util.dart' show paramScalarCount, generateText, lastRowLogits;

const String _startFen =
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

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

/// Endpoint-preserving bisection order and per-depth block selection come from
/// `ladderLayerOrder` / `ladderBlocks` in the library.

/// Laddered Hadamard-FFN decoder: `call(ids, depth: d)` runs only the ladder
/// blocks for depth `d` (shared weights), then the shared final norm + head.
class LadderLM extends Module {
  final int maxCtx, embedDim, maxLayers, numHeads;
  final Embedding tokenEmb;
  final LearnedPositionalEmbedding posEmb;
  final List<LayerNorm> ln1 = [];
  final List<MultiHeadAttention> attn = [];
  final List<LayerNorm> ln2 = [];
  final List<HadamardFFN> ffn = [];
  final LayerNorm finalLn;

  LadderLM({
    required int vocabSize,
    required this.maxCtx,
    required this.embedDim,
    required this.maxLayers,
    required this.numHeads,
    int seed = 0,
  })  : tokenEmb = Embedding(vocabSize, embedDim, seed: seed),
        posEmb = LearnedPositionalEmbedding(maxCtx, embedDim, seed: seed + 11),
        finalLn = LayerNorm(embedDim) {
    for (var l = 0; l < maxLayers; l++) {
      ln1.add(LayerNorm(embedDim));
      attn.add(MultiHeadAttention(embedDim, numHeads, seed: seed + 100 * l));
      ln2.add(LayerNorm(embedDim));
      ffn.add(HadamardFFN(embedDim, seed: seed + 200 * l));
    }
  }

  Tensor call(Tensor ids, {int? depth}) {
    final d = depth ?? maxLayers;
    final keep = ladderBlocks(maxLayers, d);
    final n = ids.shape[0];
    var h = posEmb(tokenEmb(ids));
    final mask = causalMask(n, device: h.device);
    for (final bi in keep) {
      h = h + attn[bi](ln1[bi](h), mask: mask);
      h = h + ffn[bi](ln2[bi](h));
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
  LadderLM model,
  CharTokenizer tok,
  int maxCtx,
  int depth,
  String prompt,
  int genLen,
  double temp,
  int? topK,
) {
  List<double> stepFn(List<double> ctx) {
    final x = Tensor.fromList([ctx.length], ctx, device: Device.CPU);
    return lastRowLogits(model(x, depth: depth), tok.vocabSize);
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
  print('depth $depth  prompt: ${jsonEncode(prompt)}');
  print('generated:  $text');

  final board = ch.Chess.fromFEN(_startFen);
  board.reset();
  final tokens = text.split(RegExp(r'\s+')).where((t) => t.isNotEmpty).toList();
  var legal = 0;
  String? firstIllegal;
  for (final uci in tokens) {
    final mv = _uciToMove(uci);
    if (board.game_over || mv == null || !board.move(mv)) {
      firstIllegal = uci;
      break;
    }
    legal++;
  }
  print('legal move prefix: $legal / ${tokens.length}'
      '${firstIllegal != null ? "  (stopped at \"$firstIllegal\")" : ""}');
}

Map<String, String>? _uciToMove(String u) {
  if (u.length < 4) return null;
  final m = {'from': u.substring(0, 2), 'to': u.substring(2, 4)};
  if (u.length > 4) m['promotion'] = u.substring(4, 5);
  return m;
}

double _evalLoss(
  LadderLM model,
  List<int> ids,
  int seqLen,
  int depth,
  int windows,
  math.Random rng,
) {
  var total = 0.0;
  for (var w = 0; w < windows; w++) {
    final start = rng.nextInt(ids.length - seqLen - 1);
    final x = Tensor.fromList(
      [seqLen],
      [for (var i = 0; i < seqLen; i++) ids[start + i].toDouble()],
    );
    final y = Tensor.fromList(
      [seqLen],
      [for (var i = 0; i < seqLen; i++) ids[start + i + 1].toDouble()],
    );
    total += Tensor.noGrad(
      () => model(x, depth: depth).crossEntropy(y).mean().toList().first,
    );
  }
  return total / windows;
}

Future<void> main(List<String> args) async {
  final user = _argStr(args, '--user', 'DrNykterstein');
  final games = _argInt(args, '--games', 400);
  final plies = _argInt(args, '--plies', 40);
  final regen = args.contains('--regen');
  final steps = _argInt(args, '--steps', 400);
  final seqLen = _argInt(args, '--seq', 64);
  final embedDim = _argInt(args, '--dim', 128);
  final maxLayers = _argInt(args, '--layers', 6);
  final numHeads = _argInt(args, '--heads', 4);
  final genDepth = _argInt(args, '--depth', maxLayers);
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
  print('=== chess-move LM (intelligence ladder, UCI) ===');
  print('corpus: ${ids.length} chars, vocab: ${tok.vocabSize} | '
      'dim: $embedDim, maxLayers: $maxLayers, heads: $numHeads, seq: $seqLen');
  for (var d = 1; d <= maxLayers; d++) {
    print('  depth $d -> blocks ${ladderBlocks(maxLayers, d)}');
  }

  final model = LadderLM(
    vocabSize: tok.vocabSize,
    maxCtx: seqLen,
    embedDim: embedDim,
    maxLayers: maxLayers,
    numHeads: numHeads,
    seed: 1,
  );
  final params = model.parameters();
  print('model : LadderLM ${paramScalarCount(params)} scalars');

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

    // train full depth + one random shallower rung so every depth learns
    final rung = 1 + rng.nextInt(maxLayers);
    final lossFull = model(x, depth: maxLayers).crossEntropy(y).mean();
    final lossRung = model(x, depth: rung).crossEntropy(y).mean();
    final loss = lossFull + lossRung;
    final v = lossFull.toList().first;

    opt.zeroGrad();
    loss.backward();
    clipGradNorm(params, 1.0);
    opt.step();

    if (step % 25 == 0 || step == steps - 1) {
      final rate = (step + 1) / (sw.elapsedMilliseconds / 1000);
      print('step ${step.toString().padLeft(4)}  lossL ${v.toStringAsFixed(4)}'
          '  (${rate.toStringAsFixed(1)} it/s)');
    }
  }

  print('\n--- the ladder: validation loss per depth (one model) ---');
  final evalRng = math.Random(123);
  for (var d = 1; d <= maxLayers; d++) {
    final l = _evalLoss(model, ids, seqLen, d, 30, math.Random(evalRng.nextInt(1 << 30)));
    print('  depth ${d.toString().padLeft(2)}  loss ${l.toStringAsFixed(4)}');
  }

  _sampleAndCheck(model, tok, seqLen, genDepth, prompt, genLen, temp, topK);
}
