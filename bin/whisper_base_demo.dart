/// Whisper `base.en` demo — 74M-param English-only ASR. Same encoder/
/// decoder architecture as `tiny.en` (see `bin/whisper_demo.dart`),
/// scaled to 6 layers × 8 heads × d_model=512 in both stacks.
/// Meaningfully higher accuracy than tiny.en on real speech,
/// especially non-studio audio.
///
///   dart run bin/whisper_base_demo.dart                 # CPU
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib \
///     dart run bin/whisper_base_demo.dart --gpu          # GPU
///   dart run bin/whisper_base_demo.dart --wav path/to/x.wav
///
/// Weights (one-time, ~278 MB):
///   mkdir -p models/whisper-base.en
///   for f in model.safetensors config.json tokenizer.json \
///            generation_config.json; do
///     curl -L -o "models/whisper-base.en/$f" \
///       "https://huggingface.co/openai/whisper-base.en/resolve/main/$f"
///   done
library;

import 'dart:io';

import 'package:dart_pytorch/core/audio/whisper_mel.dart';
import 'package:dart_pytorch/core/data/hf_bpe_tokenizer.dart';
import 'package:dart_pytorch/core/nn/whisper.dart';
import 'package:dart_pytorch/core/nn/whisper_decoder.dart';
import 'package:dart_pytorch/core/nn/whisper_hf_loader.dart';
import 'package:dart_pytorch/core/tensor/tensor.dart';

const int _sot = 50257;
const int _eot = 50256;
const int _noTimestamps = 50362;
const int _spaceTok = 220;

Future<void> main(List<String> args) async {
  var wavPath = 'data/jfk.wav';
  var weightsPath = 'models/whisper-base.en/model.safetensors';
  var tokenizerPath = 'models/whisper-base.en/tokenizer.json';
  var maxLen = 100;
  var useGpu = false;

  for (int i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--gpu':
        useGpu = true;
        break;
      case '--wav':
        wavPath = args[++i];
        break;
      case '--weights':
        weightsPath = args[++i];
        break;
      case '--tokenizer':
        tokenizerPath = args[++i];
        break;
      case '--max-len':
        maxLen = int.parse(args[++i]);
        break;
    }
  }
  for (final p in [wavPath, weightsPath, tokenizerPath]) {
    if (!File(p).existsSync()) {
      stderr.writeln('missing: $p');
      exit(2);
    }
  }

  final device = useGpu ? Device.GPU : Device.CPU;
  final swTotal = Stopwatch()..start();

  print('== log-mel ==');
  final swMel = Stopwatch()..start();
  final mel = await WhisperMel().logMelFromFile(wavPath);
  swMel.stop();
  print('  ${swMel.elapsedMilliseconds} ms  ($wavPath)');
  final melT = Tensor.fromFloat32List([1, 80, 3000], mel, device: device);

  print('');
  print('== build + load ${useGpu ? '(GPU)' : '(CPU)'} ==');
  final swBuild = Stopwatch()..start();
  final encoder = WhisperEncoder(
    nMels: 80,
    embedDim: 512,
    numHeads: 8,
    numLayers: 6,
    nCtx: 1500,
    device: device,
  );
  final encReport = WhisperHFLoader.loadFile(encoder, weightsPath);
  final decoder = WhisperDecoder(
    vocabSize: 51864,
    embedDim: 512,
    numHeads: 8,
    numLayers: 6,
    nCtx: 448,
    device: device,
  );
  final decReport = WhisperHFLoader.loadDecoderFile(decoder, weightsPath);
  swBuild.stop();
  print('  encoder: $encReport');
  print('  decoder: $decReport');
  print('  build+load wall: ${swBuild.elapsedMilliseconds} ms');

  final tokenizer = HFBpeTokenizer.loadFile(tokenizerPath);

  print('');
  print('== encoder forward ==');
  final swE = Stopwatch()..start();
  final memory = encoder(melT);
  swE.stop();
  print('  ${swE.elapsedMilliseconds} ms  → ${memory.shape}');

  print('');
  print('== prime cross-attn ==');
  final swP = Stopwatch()..start();
  decoder.primeCrossAttn(memory);
  swP.stop();
  print('  ${swP.elapsedMilliseconds} ms');

  print('');
  print('== greedy decode ==');
  final swG = Stopwatch()..start();
  final tokens = decoder.greedyDecode(
    startTokens: [_sot, _noTimestamps],
    eot: _eot,
    maxLen: maxLen,
    initialSuppress: [_spaceTok, _eot],
  );
  swG.stop();
  final sampled = tokens.sublist(2);
  print('  ${swG.elapsedMilliseconds} ms  → ${sampled.length} tokens');

  print('');
  print('== transcript ==');
  print('  "${tokenizer.decode(sampled)}"');

  swTotal.stop();
  print('');
  print('total wall = ${swTotal.elapsedMilliseconds} ms');
}
