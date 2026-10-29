/// Play chess with the Attention-Free Transformer ([AFTLanguageModel]).
///
/// Unlike `bin/chess_aft_demo.dart` (which trains and samples a free-form
/// continuation that is usually illegal), this program *plays* a real game:
/// at every turn it enumerates the board's **legal** moves, scores each one
/// under the model, and picks a move — so every move is legal and a full
/// game is always produced (constrained / legal decoding).
///
/// Weights are loaded from a checkpoint so you play instead of train. If the
/// checkpoint is missing, it trains one quickly, saves it, and then plays;
/// subsequent runs just load and play.
///
///   dart run bin/chess_aft_play.dart                 # AFT self-play
///   dart run bin/chess_aft_play.dart --vs white      # you are White vs AFT
///   dart run bin/chess_aft_play.dart --temp 0.0      # strongest (greedy) play
///   dart run bin/chess_aft_play.dart --train 4000    # (re)train then save
///
/// CPU only.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:chess/chess.dart' as ch;
import 'package:dart_pytorch/dart_pytorch.dart';

import 'chess_util.dart';
import 'shakespeare_util.dart' show paramScalarCount;

const String _defaultCkpt = 'models/chess_aft.json';

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

/// Parse/format UCI moves and enumerate legal moves come from chess_util
/// (`uciToMove`, `moveToUci`, `legalUciMoves`).

void _save(String path, AFTLanguageModel m, CharTokenizer tok, int maxLen) {
  final obj = {
    'config': {
      'vocab': m.vocabSize,
      'dim': m.embedDim,
      'layers': m.numLayers,
      'maxLen': maxLen,
    },
    'chars': tok.chars,
    'params': [
      for (final p in m.parameters()) {'shape': p.shape, 'data': p.toList()},
    ],
  };
  final f = File(path);
  f.parent.createSync(recursive: true);
  f.writeAsStringSync(jsonEncode(obj));
}

(AFTLanguageModel, CharTokenizer, int) _load(String path) {
  final obj = jsonDecode(File(path).readAsStringSync()) as Map<String, dynamic>;
  final c = obj['config'] as Map<String, dynamic>;
  final maxLen = c['maxLen'] as int;
  final tok = CharTokenizer.fromVocab((obj['chars'] as List).cast<String>());
  final model = AFTLanguageModel(
    vocabSize: c['vocab'] as int,
    embedDim: c['dim'] as int,
    numLayers: c['layers'] as int,
    maxLen: maxLen,
    seed: 1,
  );
  final params = model.parameters();
  final saved = obj['params'] as List;
  for (var i = 0; i < params.length; i++) {
    final data = [
      for (final v in (saved[i] as Map)['data'] as List) (v as num).toDouble(),
    ];
    params[i].assign(Tensor.fromList(params[i].shape, data, device: Device.CPU));
  }
  return (model, tok, maxLen);
}

/// Train a fresh AFT on the Lichess UCI corpus and save it.
Future<(AFTLanguageModel, CharTokenizer, int)> _trainAndSave({
  required String user,
  required int games,
  required int plies,
  required int steps,
  required int seqLen,
  required int embedDim,
  required int numLayers,
  required double lr,
  required String savePath,
  int saveEvery = 500,
}) async {
  final text =
      await loadChessUci(user: user, maxGames: games, maxPlies: plies);
  final tok = CharTokenizer.fromText(text);
  final ids = tok.encode(text);
  print('training AFT: corpus ${ids.length} chars, vocab ${tok.vocabSize}, '
      '${steps} steps ...');

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

  // Save the best checkpoint by a smoothed (EMA) loss, and allow a graceful
  // Ctrl+C to stop early with the best weights already on disk.
  var ema = double.nan;
  var best = double.infinity;
  var savedAny = false;
  var stop = false;
  final sigint = ProcessSignal.sigint.watch().listen((_) {
    if (stop) exit(130); // second Ctrl+C: force quit
    stop = true;
    stdout.writeln('\n[interrupt] stopping after this step; '
        'best checkpoint is saved at $savePath');
  });

  void checkpoint(int step) {
    if (ema.isNaN || ema >= best) return;
    best = ema;
    _save(savePath, model, tok, seqLen);
    savedAny = true;
    print('  step ${step.toString().padLeft(5)}  '
        'ema ${ema.toStringAsFixed(4)}  ✔ saved best -> $savePath');
  }

  for (var step = 0; step < steps && !stop; step++) {
    final start = rng.nextInt(ids.length - seqLen - 1);
    final x = Tensor.fromList(
      [seqLen],
      [for (var i = 0; i < seqLen; i++) ids[start + i].toDouble()],
    );
    final y = Tensor.fromList(
      [seqLen],
      [for (var i = 0; i < seqLen; i++) ids[start + i + 1].toDouble()],
    );
    final loss = model(x).crossEntropy(y).mean();
    opt.zeroGrad();
    loss.backward();
    clipGradNorm(params, 1.0);
    opt.step();

    final v = loss.toList().first;
    ema = ema.isNaN ? v : 0.98 * ema + 0.02 * v;

    if (step % 100 == 0 || step == steps - 1) {
      final rate = (step + 1) / (sw.elapsedMilliseconds / 1000);
      print('  step ${step.toString().padLeft(5)}  loss ${v.toStringAsFixed(4)}'
          '  ema ${ema.toStringAsFixed(4)}  (${rate.toStringAsFixed(1)} it/s)');
    }
    if (step > 0 && step % saveEvery == 0) checkpoint(step);
  }
  checkpoint(steps - 1); // final best check
  if (!savedAny) {
    _save(savePath, model, tok, seqLen); // ensure a checkpoint always exists
    print('saved checkpoint to $savePath');
  }
  await sigint.cancel();
  return (model, tok, seqLen);
}

/// Log-probability the model assigns to emitting [moveStr] next, given the
/// running game context [ctxIds]. Context is truncated from the front so the
/// scored sequence fits in [maxLen].
double _scoreMove(
  AFTLanguageModel model,
  CharTokenizer tok,
  List<int> ctxIds,
  String moveStr,
  int maxLen,
) {
  final v = tok.vocabSize;
  final sIds = tok.encode(moveStr);
  var ctx = ctxIds;
  final maxCtxLen = maxLen - sIds.length;
  if (ctx.length > maxCtxLen) ctx = ctx.sublist(ctx.length - maxCtxLen);
  final seq = [...ctx, ...sIds];
  final x = Tensor.fromList(
    [seq.length],
    [for (final t in seq) t.toDouble()],
    device: Device.CPU,
  );
  final flat = model(x).toList(); // [seq.length, vocab] row-major
  var lp = 0.0;
  for (var k = 0; k < sIds.length; k++) {
    final row = (ctx.length - 1 + k) * v; // position predicting sIds[k]
    var mx = double.negativeInfinity;
    for (var j = 0; j < v; j++) {
      final z = flat[row + j];
      if (z > mx) mx = z;
    }
    var sum = 0.0;
    for (var j = 0; j < v; j++) {
      sum += math.exp(flat[row + j] - mx);
    }
    lp += (flat[row + sIds[k]] - mx) - math.log(sum);
  }
  return lp;
}

/// Choose a legal move for the side to move. `temp <= 0` is greedy (argmax
/// score); otherwise sample from softmax(scores / temp).
String? _chooseMove(
  AFTLanguageModel model,
  CharTokenizer tok,
  List<int> ctxIds,
  ch.Chess board,
  int maxLen,
  double temp,
  math.Random rng,
) {
  final legal = board.generate_moves();
  if (legal.isEmpty) return null;
  final cands = [for (final m in legal) moveToUci(m)];
  final scores = [
    for (final s in cands) _scoreMove(model, tok, ctxIds, s, maxLen),
  ];
  if (temp <= 0) {
    var best = 0;
    for (var i = 1; i < scores.length; i++) {
      if (scores[i] > scores[best]) best = i;
    }
    return cands[best];
  }
  var mx = double.negativeInfinity;
  for (final s in scores) {
    if (s > mx) mx = s;
  }
  final weights = [for (final s in scores) math.exp((s - mx) / temp)];
  var total = 0.0;
  for (final w in weights) {
    total += w;
  }
  var r = rng.nextDouble() * total;
  for (var i = 0; i < cands.length; i++) {
    r -= weights[i];
    if (r <= 0) return cands[i];
  }
  return cands.last;
}

String _result(ch.Chess board) {
  if (board.in_checkmate) {
    return board.turn == ch.Color.WHITE ? '0-1 (Black mates)' : '1-0 (White mates)';
  }
  if (board.in_stalemate) return '1/2-1/2 (stalemate)';
  if (board.in_threefold_repetition) return '1/2-1/2 (repetition)';
  if (board.insufficient_material) return '1/2-1/2 (insufficient material)';
  if (board.in_draw) return '1/2-1/2 (draw)';
  return '* (move limit)';
}

Future<void> main(List<String> args) async {
  final loadPath = _argStr(args, '--load', _defaultCkpt);
  final trainSteps = _argInt(args, '--train', 0); // force (re)train if > 0
  final user = _argStr(args, '--user', 'DrNykterstein');
  final games = _argInt(args, '--games', 400);
  final plies = _argInt(args, '--plies', 40);
  final seqLen = _argInt(args, '--seq', 64);
  final embedDim = _argInt(args, '--dim', 128);
  final numLayers = _argInt(args, '--layers', 3);
  final lr = _argDouble(args, '--lr', 3e-3);
  final saveEvery = _argInt(args, '--save-every', 500);
  final temp = _argDouble(args, '--temp', 0.3);
  final maxGamePlies = _argInt(args, '--max', 120);
  final vs = _argStr(args, '--vs', ''); // 'white' | 'black' | '' (self-play)
  final seed = _argInt(args, '--seed', 7);

  AFTLanguageModel model;
  CharTokenizer tok;
  int maxLen;

  if (trainSteps > 0 || !File(loadPath).existsSync()) {
    final steps = trainSteps > 0 ? trainSteps : 4000;
    if (trainSteps == 0) {
      print('no checkpoint at $loadPath — training one first ...');
    }
    (model, tok, maxLen) = await _trainAndSave(
      user: user,
      games: games,
      plies: plies,
      steps: steps,
      seqLen: seqLen,
      embedDim: embedDim,
      numLayers: numLayers,
      lr: lr,
      savePath: loadPath,
      saveEvery: saveEvery,
    );
  } else {
    (model, tok, maxLen) = _load(loadPath);
    print('loaded AFT checkpoint $loadPath '
        '(vocab ${tok.vocabSize}, dim ${model.embedDim}, '
        'layers ${model.numLayers}, ctx $maxLen)');
  }
  model.eval();

  final humanWhite = vs == 'white';
  final humanBlack = vs == 'black';
  final interactive = humanWhite || humanBlack;
  final bos = tok.chars.contains('\n')
      ? '\n'
      : (tok.chars.contains(' ') ? ' ' : tok.chars.first);

  print('\n=== AFT plays chess (constrained legal decoding, temp=$temp) ===');
  print(interactive
      ? 'you are ${humanWhite ? "White" : "Black"} — enter moves in UCI (e.g. e2e4, e7e8q). "quit" to stop.'
      : 'AFT self-play.');

  final board = ch.Chess();
  board.reset();
  final rng = math.Random(seed);
  var ctxStr = bos;
  var ply = 0;

  while (!board.game_over && ply < maxGamePlies) {
    final whiteToMove = board.turn == ch.Color.WHITE;
    final humanTurn =
        interactive && ((whiteToMove && humanWhite) || (!whiteToMove && humanBlack));

    String uci;
    if (humanTurn) {
      stdout.write('your move (${whiteToMove ? "White" : "Black"}): ');
      final line = stdin.readLineSync()?.trim() ?? '';
      if (line.isEmpty) continue;
      if (line == 'quit' || line == 'q') break;
      final mv = uciToMove(line);
      if (mv == null || !board.move(mv)) {
        print('  illegal move "$line" — legal examples: '
            '${board.generate_moves().take(6).map(moveToUci).join(" ")} ...');
        continue;
      }
      uci = line;
    } else {
      final chosen =
          _chooseMove(model, tok, tok.encode(ctxStr), board, maxLen, temp, rng);
      if (chosen == null) break;
      board.move(uciToMove(chosen));
      uci = chosen;
      print('  AFT (${whiteToMove ? "White" : "Black"}): $uci');
    }

    ctxStr += '$uci ';
    ply++;
    if (!interactive && ply % 2 == 0) {
      // print the board after each full move in self-play
      print(board.ascii);
    }
  }

  print('\nfinal position after $ply plies:');
  print(board.ascii);
  print('result: ${_result(board)}');
}
