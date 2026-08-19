/// Whisper base (multilingual) — 74 M ASR for any of Whisper's 99
/// languages.
///
/// Same arch as `bin/whisper_base_demo.dart` (which loads the
/// `.en` variant); this binary loads `openai/whisper-base` (no
/// `.en`) and prepends a language + task token to the decoder
/// prompt so the model transcribes in the requested language.
///
///   dart run bin/whisper_multilingual_demo.dart               # CPU, en
///   dart run bin/whisper_multilingual_demo.dart --lang sw     # Swahili
///   dart run bin/whisper_multilingual_demo.dart --translate   # to English
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib \
///     dart run bin/whisper_multilingual_demo.dart --gpu
///
/// One-time weight download (~290 MB fp32):
///   mkdir -p models/whisper-base
///   for f in model.safetensors tokenizer.json config.json; do
///     curl -L -o "models/whisper-base/$f" \
///       "https://huggingface.co/openai/whisper-base/resolve/main/$f"
///   done
///
/// **Zambian languages note**: Whisper's language list covers 99
/// languages but NOT Bemba, Nyanja/Chichewa, Tonga, or Lozi.
/// Community fine-tunes exist (e.g., `chiyo123/whisper-small-bemba`
/// on HuggingFace) that hijack an existing language token
/// (typically `<|sw|>` for Swahili) at training time. To use one,
/// download it, set `--lang sw` (or whichever token the fine-tune
/// uses — check its `generation_config.json`), and point
/// `--weights` at the fine-tune's safetensors.
library;

import 'dart:io';

import 'package:dart_pytorch/dart_pytorch.dart';

// Multilingual Whisper base special tokens (verified against
// models/whisper-base/tokenizer.json).
const int _sot = 50258; // <|startoftranscript|>
const int _eot = 50257; // <|endoftext|>
const int _transcribe = 50359;
const int _translate = 50358;
const int _noTimestamps = 50363;
const int _spaceTok = 220;

// Whisper's language-token base id — <|en|> = 50259, offsets follow
// Whisper's official language list order.
const int _langBase = 50259;
const Map<String, int> _langOffsets = {
  'en': 0, 'zh': 1, 'de': 2, 'es': 3, 'ru': 4, 'ko': 5, 'fr': 6,
  'ja': 7, 'pt': 8, 'tr': 9, 'pl': 10, 'ca': 11, 'nl': 12, 'ar': 13,
  'sv': 14, 'it': 15, 'id': 16, 'hi': 17, 'fi': 18, 'vi': 19, 'iw': 20,
  'uk': 21, 'el': 22, 'ms': 23, 'cs': 24, 'ro': 25, 'da': 26, 'hu': 27,
  'ta': 28, 'no': 29, 'th': 30, 'ur': 31, 'hr': 32, 'bg': 33, 'lt': 34,
  'la': 35, 'mi': 36, 'ml': 37, 'cy': 38, 'sk': 39, 'te': 40, 'fa': 41,
  'lv': 42, 'bn': 43, 'sr': 44, 'az': 45, 'sl': 46, 'kn': 47, 'et': 48,
  'mk': 49, 'br': 50, 'eu': 51, 'is': 52, 'hy': 53, 'ne': 54, 'mn': 55,
  'bs': 56, 'kk': 57, 'sq': 58, 'sw': 59, 'gl': 60, 'mr': 61, 'pa': 62,
  'si': 63, 'km': 64, 'sn': 65, 'yo': 66, 'so': 67, 'af': 68, 'oc': 69,
  'ka': 70, 'be': 71, 'tg': 72, 'sd': 73, 'gu': 74, 'am': 75, 'yi': 76,
  'lo': 77, 'uz': 78, 'fo': 79, 'ht': 80, 'ps': 81, 'tk': 82, 'nn': 83,
  'mt': 84, 'sa': 85, 'lb': 86, 'my': 87, 'bo': 88, 'tl': 89, 'mg': 90,
  'as': 91, 'tt': 92, 'haw': 93, 'ln': 94, 'ha': 95, 'ba': 96, 'jw': 97,
  'su': 98,
};

int _langTokenId(String lang) {
  final off = _langOffsets[lang];
  if (off == null) {
    throw ArgumentError(
      'unknown language "$lang" — Whisper multilingual supports 99 codes; '
      'try en / zh / sw / de / fr etc. See --help for the full list.',
    );
  }
  return _langBase + off;
}

Future<void> main(List<String> args) async {
  var weightsPath = 'models/whisper-base/model.safetensors';
  var tokenizerPath = 'models/whisper-base/tokenizer.json';
  var wavPath = 'data/jfk.wav';
  var lang = 'en';
  var task = 'transcribe';
  var useGpu = false;
  var maxLen = 200;
  for (int i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--weights':
        weightsPath = args[++i];
        break;
      case '--tokenizer':
        tokenizerPath = args[++i];
        break;
      case '--wav':
        wavPath = args[++i];
        break;
      case '--lang':
        lang = args[++i];
        break;
      case '--translate':
        task = 'translate';
        break;
      case '--gpu':
        useGpu = true;
        break;
      case '--max-len':
        maxLen = int.parse(args[++i]);
        break;
    }
  }
  for (final p in [weightsPath, tokenizerPath, wavPath]) {
    if (!File(p).existsSync()) {
      stderr.writeln('missing: $p');
      exit(2);
    }
  }
  final device = useGpu ? Device.GPU : Device.CPU;
  final swTotal = Stopwatch()..start();

  print('== log-mel (80 channels) ==');
  final swMel = Stopwatch()..start();
  final mel = await WhisperMel().logMelFromFile(wavPath);
  swMel.stop();
  print('  ${swMel.elapsedMilliseconds} ms  ($wavPath)');
  final melT = Tensor.fromFloat32List([1, 80, 3000], mel, device: device);

  print('');
  print('== build + load ${useGpu ? "(GPU)" : "(CPU)"} ==');
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
    vocabSize: 51865,
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

  final langTok = _langTokenId(lang);
  final taskTok = task == 'translate' ? _translate : _transcribe;
  final startTokens = [_sot, langTok, taskTok, _noTimestamps];
  print('');
  print('== greedy decode (lang=$lang, task=$task) ==');
  print('  prompt: [SOT, <|$lang|>=$langTok, <|$task|>=$taskTok, <|notimestamps|>=$_noTimestamps]');
  final swG = Stopwatch()..start();
  final tokens = decoder.greedyDecode(
    startTokens: startTokens,
    eot: _eot,
    maxLen: maxLen,
    initialSuppress: [_spaceTok, _eot],
  );
  swG.stop();
  final sampled = tokens.sublist(startTokens.length);
  print('  ${swG.elapsedMilliseconds} ms  → ${sampled.length} tokens');

  print('');
  print('== transcript ==');
  print('  "${tokenizer.decode(sampled)}"');
  swTotal.stop();
  print('');
  print('total wall = ${swTotal.elapsedMilliseconds} ms');
}
