/// Train the context-free character LM ([CharStateLM]) to generate chess
/// moves in **UCI**, then sample a continuation and check how many of its
/// moves are legal with the `chess` package.
///
/// The corpus is real games **downloaded from Lichess** and converted to
/// UCI (one game per line), cached at data/chess_uci.txt. If the download
/// is unavailable it falls back to synthetic random legal games.
///
///   dart run bin/chess_moves_lm_demo.dart
///   dart run bin/chess_moves_lm_demo.dart --user DrNykterstein --games 800 --steps 800
///   dart run bin/chess_moves_lm_demo.dart --prompt "e2e4 e7e5 " --temp 0.5
///
/// CPU by default.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:chess/chess.dart' as ch;
import 'package:dart_pytorch/dart_pytorch.dart';

import 'chess_util.dart';

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

void _save(String path, CharStateLM m, CharTokenizer tok) {
  final obj = {
    'config': {
      'vocab': m.vocabSize,
      'dim': m.embedDim,
      'layers': m.blocks.length,
      'heads': m.numHeads,
      'useDecay': m.useDecay,
    },
    'chars': tok.chars,
    'params': [
      for (final p in m.parameters()) {'shape': p.shape, 'data': p.toList()},
    ],
  };
  File(path).writeAsStringSync(jsonEncode(obj));
}

(CharStateLM, CharTokenizer) _load(String path) {
  final obj = jsonDecode(File(path).readAsStringSync()) as Map<String, dynamic>;
  final c = obj['config'] as Map<String, dynamic>;
  final tok = CharTokenizer.fromVocab((obj['chars'] as List).cast<String>());
  final model = CharStateLM(
    vocabSize: c['vocab'] as int,
    embedDim: c['dim'] as int,
    numLayers: c['layers'] as int,
    numHeads: c['heads'] as int,
    useDecay: c['useDecay'] as bool,
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
  return (model, tok);
}

/// Sample a continuation from [prompt] and report how many of the moves
/// (prompt + generated) are legal from the starting position.
void _sampleAndCheck(
  CharStateLM model,
  CharTokenizer tok,
  String prompt,
  int genLen,
  double temp,
) {
  final gen = model.generate(
    tok.encode(prompt),
    genLen,
    temperature: temp,
    rng: math.Random(7),
  );
  final text = tok.decode(gen);
  print('');
  print('prompt:     ${jsonEncode(prompt)}');
  print('generated:  $text');

  // Replay the moves on a board; count the legal prefix.
  final board = ch.Chess.fromFEN(_startFen);
  board.reset(); // start from the initial position
  final tokens =
      text.split(RegExp(r'\s+')).where((t) => t.isNotEmpty).toList();
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
  if (legal > 0) print('position after legal moves:\n${board.ascii}');
}

/// Parse a UCI move string ("e2e4", "e7e8q") into a `chess` move map.
Map<String, String>? _uciToMove(String u) {
  if (u.length < 4) return null;
  final m = {'from': u.substring(0, 2), 'to': u.substring(2, 4)};
  if (u.length > 4) m['promotion'] = u.substring(4, 5);
  return m;
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
  final loadPath = _argStr(args, '--load', '');
  final savePath = _argStr(args, '--save', '');

  if (loadPath.isNotEmpty) {
    final (model, tok) = _load(loadPath);
    print('=== chess-move LM — loaded $loadPath ===');
    _sampleAndCheck(model, tok, prompt, genLen, temp);
    return;
  }

  final text = await loadChessUci(
    user: user,
    maxGames: games,
    maxPlies: plies,
    regenerate: regen,
  );
  final tok = CharTokenizer.fromText(text);
  final ids = tok.encode(text);
  print('=== chess-move LM (context-free char LM, UCI) ===');
  print('corpus: ${ids.length} chars, vocab: ${tok.vocabSize} | '
      'dim: $embedDim, layers: $numLayers, heads: $numHeads, seq: $seqLen');

  final model = CharStateLM(
    vocabSize: tok.vocabSize,
    embedDim: embedDim,
    numLayers: numLayers,
    numHeads: numHeads,
    useDecay: true,
    seed: 1,
  );
  final opt = Adam(model.parameters(), lr: 3e-3);
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

    final logits = model.callSeq(x);
    final loss = logits.crossEntropy(y).mean();
    final v = loss.toList().first;

    opt.zeroGrad();
    loss.backward();
    opt.step();

    if (step % 25 == 0 || step == steps - 1) {
      final rate = (step + 1) / (sw.elapsedMilliseconds / 1000);
      print('step ${step.toString().padLeft(4)}  loss ${v.toStringAsFixed(4)}'
          '  (${rate.toStringAsFixed(1)} it/s)');
    }
  }

  if (savePath.isNotEmpty) {
    _save(savePath, model, tok);
    print('saved checkpoint to $savePath');
  }

  _sampleAndCheck(model, tok, prompt, genLen, temp);
}
