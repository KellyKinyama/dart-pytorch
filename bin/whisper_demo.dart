/// End-to-end Whisper tiny.en transcription demo.
///
///   dart run bin/whisper_demo.dart \
///       [--wav PATH] [--weights PATH] [--tokenizer PATH]
///       [--max-len N]
///
/// Defaults:
///   --wav        models/silero_vad/warmup_audio.wav
///   --weights    models/whisper-tiny.en/model.safetensors
///   --tokenizer  models/whisper-tiny.en/tokenizer.json
///   --max-len    50
library;

import 'dart:io';

import 'package:dart_pytorch/core/audio/whisper_mel.dart';
import 'package:dart_pytorch/core/data/hf_bpe_tokenizer.dart';
import 'package:dart_pytorch/core/nn/whisper.dart';
import 'package:dart_pytorch/core/nn/whisper_decoder.dart';
import 'package:dart_pytorch/core/nn/whisper_hf_loader.dart';
import 'package:dart_pytorch/core/tensor/tensor.dart';

// tiny.en special tokens (from generation_config.json).
const int _sot = 50257; // <|startoftranscript|>
const int _eot = 50256; // <|endoftext|>
const int _noTimestamps = 50362; // <|notimestamps|>
const int _spaceTok = 220; // " " — Whisper suppresses this as first sample.

Future<void> main(List<String> args) async {
  var wavPath = 'models/silero_vad/warmup_audio.wav';
  var weightsPath = 'models/whisper-tiny.en/model.safetensors';
  var tokenizerPath = 'models/whisper-tiny.en/tokenizer.json';
  var maxLen = 50;

  for (int i = 0; i < args.length; i++) {
    final a = args[i];
    if (a == '--wav' && i + 1 < args.length) {
      wavPath = args[++i];
    } else if (a == '--weights' && i + 1 < args.length) {
      weightsPath = args[++i];
    } else if (a == '--tokenizer' && i + 1 < args.length) {
      tokenizerPath = args[++i];
    } else if (a == '--max-len' && i + 1 < args.length) {
      maxLen = int.parse(args[++i]);
    }
  }

  for (final p in [wavPath, weightsPath, tokenizerPath]) {
    if (!File(p).existsSync()) {
      stderr.writeln('missing: $p');
      exit(2);
    }
  }

  final swTotal = Stopwatch()..start();

  print('== log-mel ==');
  final swMel = Stopwatch()..start();
  final mel = await WhisperMel().logMelFromFile(wavPath);
  swMel.stop();
  print('  ${swMel.elapsedMilliseconds} ms  ($wavPath)');
  final melT = Tensor.fromFloat32List([1, 80, 3000], mel, device: Device.CPU);

  print('');
  print('== build + load encoder ==');
  final swEnc = Stopwatch()..start();
  final encoder = WhisperEncoder(
    nMels: 80,
    embedDim: 384,
    numHeads: 6,
    numLayers: 4,
    nCtx: 1500,
  );
  final encReport = WhisperHFLoader.loadFile(encoder, weightsPath);
  swEnc.stop();
  print('  ${swEnc.elapsedMilliseconds} ms  $encReport');

  print('');
  print('== build + load decoder ==');
  final swDec = Stopwatch()..start();
  final decoder = WhisperDecoder(
    vocabSize: 51864,
    embedDim: 384,
    numHeads: 6,
    numLayers: 4,
    nCtx: 448,
  );
  final decReport = WhisperHFLoader.loadDecoderFile(decoder, weightsPath);
  swDec.stop();
  print('  ${swDec.elapsedMilliseconds} ms  $decReport');
  if (decReport.unusedKeys.isNotEmpty) {
    print('  unused (up to 10):');
    for (final k in decReport.unusedKeys.take(10)) {
      print('    - $k');
    }
  }

  print('');
  print('== tokenizer ==');
  final tokenizer = HFBpeTokenizer.loadFile(tokenizerPath);
  print('  loaded  vocab=${tokenizer.vocab.length}');

  print('');
  print('== encoder forward ==');
  final swFwdE = Stopwatch()..start();
  final memory = encoder(melT);
  swFwdE.stop();
  print('  ${swFwdE.elapsedMilliseconds} ms  → ${memory.shape}');

  print('');
  print('== prime cross-attn ==');
  final swPrime = Stopwatch()..start();
  decoder.primeCrossAttn(memory);
  swPrime.stop();
  print('  ${swPrime.elapsedMilliseconds} ms');

  print('');
  print('== greedy decode ==');
  final swGen = Stopwatch()..start();
  final tokens = decoder.greedyDecode(
    startTokens: [_sot, _noTimestamps],
    eot: _eot,
    maxLen: maxLen,
    initialSuppress: [_spaceTok, _eot],
  );
  swGen.stop();
  print('  ${swGen.elapsedMilliseconds} ms  → ${tokens.length} tokens '
      '(${tokens.length - 2} sampled)');
  print('  raw ids: $tokens');

  // Strip prefix (SOT, NOTIMESTAMPS); tokenizer decodes byte-BPE cleanly.
  final sampled = tokens.sublist(2);
  final text = tokenizer.decode(sampled);
  print('');
  print('== transcript ==');
  print('  "$text"');

  swTotal.stop();
  print('');
  print('total wall = ${swTotal.elapsedMilliseconds} ms');
}
