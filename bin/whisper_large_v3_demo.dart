/// Whisper `large-v3` demo — OpenAI's biggest and best ASR encoder
/// (~1.55 B params). Same encoder/decoder architecture as tiny.en /
/// base.en (`WhisperEncoder` / `WhisperDecoder`), just a much bigger
/// config: 32 encoder + 32 decoder layers at d_model=1280 and
/// **128 mel channels** (up from 80 in v1/v2/tiny/base).
///
/// **VRAM warning.** large-v3 needs ≥6 GB VRAM in fp32 for a full
/// forward on 30 s of audio. On a 6 GB WSL laptop this is on the
/// edge — pass `--cpu` and expect ~5–10 minutes per 30 s clip on
/// modern hardware.
///
///   dart run bin/whisper_large_v3_demo.dart --wav path/to/audio.wav
///
/// One-time setup (~3 GB fp32 checkpoint, safetensors):
///   mkdir -p models/whisper-large-v3
///   for f in model.safetensors tokenizer.json; do
///     curl -L -o "models/whisper-large-v3/$f" \
///       "https://huggingface.co/openai/whisper-large-v3/resolve/main/$f"
///   done
///
/// large-v3 is multilingual — pass `--lang xx` (2-letter code) via
/// the language-suppression token if you want a language other than
/// the default (auto-detect via first-token prediction is not wired
/// up in this demo; it uses the "no timestamps" prefix only).
library;

import 'dart:io';

import 'package:dart_pytorch/dart_pytorch.dart';

// Whisper `large-v3` uses the same GPT-2 BPE tokenizer as tiny.en but
// with a slightly bigger vocab (51 866 including new special tokens
// for lang detection). These constants are stable across all v3
// variants and match `openai/whisper-large-v3` on HF.
const int _sot = 50258; // <|startoftranscript|>
const int _eot = 50257; // <|endoftext|>
const int _noTimestamps = 50364; // <|notimestamps|>
const int _spaceTok = 220; // ' '

const _weightsDefault = 'models/whisper-large-v3/model.safetensors';
const _tokenizerDefault = 'models/whisper-large-v3/tokenizer.json';

Future<void> main(List<String> args) async {
  var weightsPath = _weightsDefault;
  var tokenizerPath = _tokenizerDefault;
  String? wavPath;
  var useGpu = false;
  var maxLen = 100;
  for (int i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--gpu':
        useGpu = true;
        break;
      case '--weights':
        weightsPath = args[++i];
        break;
      case '--tokenizer':
        tokenizerPath = args[++i];
        break;
      case '--wav':
        wavPath = args[++i];
        break;
      case '--max-len':
        maxLen = int.parse(args[++i]);
        break;
    }
  }
  if (wavPath == null) {
    stderr.writeln(
      'usage: dart run bin/whisper_large_v3_demo.dart '
      '--wav PATH [--gpu] [--max-len N]',
    );
    exit(64);
  }
  for (final p in [weightsPath, tokenizerPath, wavPath]) {
    if (!File(p).existsSync()) {
      stderr.writeln('missing: $p');
      exit(2);
    }
  }

  final device = useGpu ? Device.GPU : Device.CPU;
  final swTotal = Stopwatch()..start();

  print('== log-mel (128 channels for v3) ==');
  final swMel = Stopwatch()..start();
  final mel = await WhisperMel(
    const WhisperMelConfig(nMels: 128),
  ).logMelFromFile(wavPath);
  swMel.stop();
  print('  ${swMel.elapsedMilliseconds} ms  ($wavPath)');
  final melT = Tensor.fromFloat32List([1, 128, 3000], mel, device: device);

  print('');
  print('== build + load ${useGpu ? '(GPU)' : '(CPU)'} ==');
  final swBuild = Stopwatch()..start();
  final encoder = WhisperEncoder(
    nMels: 128,
    embedDim: 1280,
    numHeads: 20,
    numLayers: 32,
    nCtx: 1500,
    device: device,
  );
  final encReport = WhisperHFLoader.loadFile(encoder, weightsPath);
  final decoder = WhisperDecoder(
    vocabSize: 51866,
    embedDim: 1280,
    numHeads: 20,
    numLayers: 32,
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
