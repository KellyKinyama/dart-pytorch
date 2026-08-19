/// Distil-Whisper `distil-small.en` demo. Same architecture family
/// as Whisper (`WhisperEncoder` / `WhisperDecoder`), just a bigger
/// config: 12 encoder layers + 4 decoder layers at d_model=768.
/// About 3× more parameters than tiny.en (166M vs 39M) but ~5× the
/// quality gap on WER benchmarks, and ~2× faster than the equivalent
/// `whisper-small.en` thanks to the distilled shallow decoder.
///
///   dart run bin/distil_whisper_demo.dart                    # CPU
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib \
///     dart run bin/distil_whisper_demo.dart --gpu            # GPU
///   dart run bin/distil_whisper_demo.dart --wav path/to/x.wav
///
/// One-time weight download:
///   mkdir -p models/distil-small.en
///   for f in model.safetensors config.json tokenizer.json \
///            generation_config.json; do
///     curl -L -o "models/distil-small.en/$f" \
///       "https://huggingface.co/distil-whisper/distil-small.en/resolve/main/$f"
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
  var weightsPath = 'models/distil-small.en/model.safetensors';
  var tokenizerPath = 'models/distil-small.en/tokenizer.json';
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
    embedDim: 768,
    numHeads: 12,
    numLayers: 12,
    nCtx: 1500,
    device: device,
  );
  final encReport = WhisperHFLoader.loadFile(encoder, weightsPath);
  final decoder = WhisperDecoder(
    vocabSize: 51864,
    embedDim: 768,
    numHeads: 12,
    numLayers: 4,
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
