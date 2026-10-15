/// Train a context-free, state-based character language model on
/// tiny-Shakespeare and sample from it. The model is a decoder-only
/// transformer whose attention is linear (recurrent) with multiple heads
/// and a per-head RWKV/RetNet-style decay gate: it carries a fixed-size
/// state instead of a context window, so there is no positional embedding
/// and generation is O(1) per character. See lib/core/nn/char_state_lm.dart.
///
///   dart run bin/char_state_lm_demo.dart
///   dart run bin/char_state_lm_demo.dart --steps 600 --chars 10000
///
/// CPU by default — the per-character recurrence builds a small graph per
/// step; the defaults keep a single run to a few minutes.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

import 'shakespeare_util.dart' show loadCorpus;

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

/// Serialize config + tokenizer vocab + all parameters (flat, in
/// `parameters()` order) to a JSON file.
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

/// Rebuild a model + tokenizer from a file written by [_save] and copy
/// the saved weights into it (`parameters()` order is deterministic).
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
  if (saved.length != params.length) {
    throw StateError('checkpoint has ${saved.length} params; '
        'model expects ${params.length}');
  }
  for (var i = 0; i < params.length; i++) {
    final data = [
      for (final v in (saved[i] as Map)['data'] as List) (v as num).toDouble(),
    ];
    // Force CPU — Tensor.fromList auto-places large tensors on GPU.
    params[i].assign(
      Tensor.fromList(params[i].shape, data, device: Device.CPU),
    );
  }
  return (model, tok);
}

void _generate(
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
    rng: math.Random(42),
  );
  print('');
  print('prompt:    ${jsonEncode(prompt)}');
  print('generated:\n${tok.decode(gen)}');
}

Future<void> main(List<String> args) async {
  final maxChars = _argInt(args, '--chars', 5000);
  final steps = _argInt(args, '--steps', 300);
  final seqLen = _argInt(args, '--seq', 48);
  final embedDim = _argInt(args, '--dim', 96);
  final numLayers = _argInt(args, '--layers', 2);
  final numHeads = _argInt(args, '--heads', 4);
  final overfit = args.contains('--overfit');
  final prompt = _argStr(args, '--prompt', 'ROMEO:');
  final genLen = _argInt(args, '--gen', 240);
  final temp = _argDouble(args, '--temp', 0.8);
  final loadPath = _argStr(args, '--load', '');
  final savePath = _argStr(args, '--save', '');

  // Load a trained model and prompt it immediately — no retraining.
  if (loadPath.isNotEmpty) {
    final (model, tok) = _load(loadPath);
    print('=== context-free char LM — loaded $loadPath ===');
    print('vocab: ${tok.vocabSize}, dim: ${model.embedDim}, '
        'layers: ${model.blocks.length}, heads: ${model.numHeads}');
    _generate(model, tok, prompt, genLen, temp);
    return;
  }

  final text = loadCorpus(maxChars: maxChars);
  final tok = CharTokenizer.fromText(text);
  final ids = tok.encode(text);
  print('=== context-free (state-based) char LM — tiny-Shakespeare ===');
  print('corpus: ${ids.length} chars, vocab: ${tok.vocabSize} | '
      'dim: $embedDim, layers: $numLayers, heads: $numHeads, seq: $seqLen'
      '${overfit ? ' | MODE: overfit one window' : ''}');

  final model = CharStateLM(
    vocabSize: tok.vocabSize,
    embedDim: embedDim,
    numLayers: numLayers,
    numHeads: numHeads,
    useDecay: true, // RWKV-style per-head state decay
    seed: 1,
  );
  final opt = Adam(model.parameters(), lr: 3e-3);
  final rng = math.Random(0);
  final sw = Stopwatch()..start();

  // Overfit mode trains on ONE fixed window every step — the standard
  // "can the model memorize a tiny sample" sanity check. Loss should
  // collapse toward ~0 and greedy decoding should reproduce it exactly.
  const fixedStart = 0;
  Tensor fixedX() => Tensor.fromList(
        [seqLen],
        [for (var i = 0; i < seqLen; i++) ids[fixedStart + i].toDouble()],
      );
  Tensor fixedY() => Tensor.fromList(
        [seqLen],
        [for (var i = 0; i < seqLen; i++) ids[fixedStart + i + 1].toDouble()],
      );

  var lastLoss = double.nan;
  for (var step = 0; step < steps; step++) {
    final Tensor x, y;
    if (overfit) {
      x = fixedX();
      y = fixedY();
    } else {
      final start = rng.nextInt(ids.length - seqLen - 1);
      x = Tensor.fromList(
        [seqLen],
        [for (var i = 0; i < seqLen; i++) ids[start + i].toDouble()],
      );
      y = Tensor.fromList(
        [seqLen],
        [for (var i = 0; i < seqLen; i++) ids[start + i + 1].toDouble()],
      );
    }

    final logits = model.callSeq(x); // [seqLen, vocab]
    final loss = logits.crossEntropy(y).mean();
    lastLoss = loss.toList().first;

    opt.zeroGrad();
    loss.backward();
    opt.step();

    if (step % 25 == 0 || step == steps - 1) {
      final rate = (step + 1) / (sw.elapsedMilliseconds / 1000);
      print('step ${step.toString().padLeft(4)}  '
          'loss ${lastLoss.toStringAsFixed(4)}  (${rate.toStringAsFixed(1)} it/s)');
    }
  }

  if (savePath.isNotEmpty) {
    _save(savePath, model, tok);
    print('saved checkpoint to $savePath');
  }

  if (overfit) {
    // Greedily regenerate the window from its first character. A fully
    // memorized model reproduces it exactly (argmax = true next char, so
    // the fed chars keep the recurrent state on the memorized trajectory).
    final target = ids.sublist(fixedStart, fixedStart + seqLen);
    final recon = model.generate([target.first], seqLen - 1, temperature: 0.0);
    var matched = 0;
    for (var i = 0; i < seqLen; i++) {
      if (i < recon.length && recon[i] == target[i]) matched++;
    }
    print('');
    print('final loss: ${lastLoss.toStringAsFixed(6)}');
    print('target:     ${jsonEncode(tok.decode(target))}');
    print('greedy:     ${jsonEncode(tok.decode(recon))}');
    print('chars matched: $matched / $seqLen '
        '${matched == seqLen ? '— OK, memorized exactly' : ''}');
    return;
  }

  // Stream characters from the recurrent state (no context window).
  _generate(model, tok, prompt, genLen, temp);
}
