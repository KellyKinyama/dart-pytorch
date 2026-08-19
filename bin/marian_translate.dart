/// Generic Marian NMT runner — pick any Helsinki-NLP/opus-mt-* pair
/// via `--pair`.
///
/// Every opus-mt pair we support uses the same architecture
/// (6+6 layers, dModel=512, 8 heads, SiLU FFN) and only differs in
/// vocab size + pad/decoder-start ids. This binary works for all of
/// them.
///
///   dart run bin/marian_translate.dart --pair en-de --text "Hello world."
///   dart run bin/marian_translate.dart --pair en-zh --text "The dog runs fast."
///   dart run bin/marian_translate.dart --pair zh-en --text "你好世界。"
///
/// Weights + vocab for the pair must live under
/// `models/opus-mt-<pair>/`. See T3 in commands.md for one-time
/// download / convert commands.
library;

import 'dart:io';

import 'package:dart_pytorch/core/data/marian_vocab_tokenizer.dart';
import 'package:dart_pytorch/dart_pytorch.dart';

MarianConfig _configForPair(String pair, Device device) {
  switch (pair) {
    case 'en-de':
      return MarianHFLoader.opusMtEnDeConfig(device: device);
    case 'en-zh':
      return MarianHFLoader.opusMtEnZhConfig(device: device);
    case 'zh-en':
      return MarianHFLoader.opusMtZhEnConfig(device: device);
    default:
      throw ArgumentError(
        'unknown pair "$pair"; use en-de | en-zh | zh-en',
      );
  }
}

Future<void> main(List<String> args) async {
  var pair = 'en-de';
  String? text;
  var maxNew = 60;
  var numBeams = 1;
  var lengthPenalty = 0.6;
  var useGpu = false;
  String? weightsOverride;
  String? vocabOverride;
  for (int i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--pair':
        pair = args[++i];
        break;
      case '--text':
        text = args[++i];
        break;
      case '--max-new':
        maxNew = int.parse(args[++i]);
        break;
      case '--beams':
        numBeams = int.parse(args[++i]);
        break;
      case '--length-penalty':
        lengthPenalty = double.parse(args[++i]);
        break;
      case '--gpu':
        useGpu = true;
        break;
      case '--weights':
        weightsOverride = args[++i];
        break;
      case '--vocab':
        vocabOverride = args[++i];
        break;
    }
  }
  text ??= _defaultTextFor(pair);
  final weightsPath =
      weightsOverride ?? 'models/opus-mt-$pair/model.safetensors';
  final vocabPath = vocabOverride ?? 'models/opus-mt-$pair/vocab.json';
  for (final p in [weightsPath, vocabPath]) {
    if (!File(p).existsSync()) {
      stderr.writeln('missing: $p');
      stderr.writeln(
        'download + convert with:\n'
        '  mkdir -p models/opus-mt-$pair\n'
        '  for f in pytorch_model.bin vocab.json config.json; do\n'
        '    curl -L -o "models/opus-mt-$pair/\$f" \\\n'
        '      "https://huggingface.co/Helsinki-NLP/opus-mt-$pair/resolve/main/\$f"\n'
        '  done\n'
        '  python3 scripts/convert_marian_pt_to_safetensors.py \\\n'
        '    models/opus-mt-$pair/pytorch_model.bin \\\n'
        '    models/opus-mt-$pair/model.safetensors',
      );
      exit(2);
    }
  }

  final device = useGpu ? Device.GPU : Device.CPU;
  final cfg = _configForPair(pair, device);
  print(
    '== opus-mt-$pair '
    '(vocab=${cfg.vocabSize}, dModel=${cfg.dModel}, '
    'layers=${cfg.numLayers}+${cfg.numDecoderLayers}, heads=${cfg.numHeads}) ==',
  );

  final swBuild = Stopwatch()..start();
  final model = MarianModel(cfg);
  swBuild.stop();
  print('build: ${swBuild.elapsedMilliseconds} ms');

  final swLoad = Stopwatch()..start();
  final report = MarianHFLoader.loadFile(model, weightsPath);
  swLoad.stop();
  print('load : ${swLoad.elapsedMilliseconds} ms  $report');

  final tok = MarianVocabTokenizer.loadFile(vocabPath);
  final ids = tok.encode(text);
  print('');
  print('== source ==');
  print('  "$text"');
  print('  -> $ids');

  print('');
  print('== ${numBeams == 1 ? "greedy" : "beam=$numBeams"} decode ==');
  final swG = Stopwatch()..start();
  final out = numBeams == 1
      ? model.generate(ids, maxNewTokens: maxNew)
      : model.generateBeam(
          ids,
          numBeams: numBeams,
          maxNewTokens: maxNew,
          lengthPenalty: lengthPenalty,
        );
  swG.stop();
  final newTokens = out.sublist(1);
  print(
    '  ${swG.elapsedMilliseconds} ms  '
    '(${newTokens.length} tokens, '
    '${(newTokens.length * 1000.0 / swG.elapsedMilliseconds).toStringAsFixed(1)} tok/s)',
  );

  print('');
  print('== translation ==');
  print('  "${tok.decode(newTokens)}"');
  print('  ids: $out');
}

String _defaultTextFor(String pair) {
  switch (pair) {
    case 'zh-en':
      return '你好，世界。今天天气很好。';
    case 'en-zh':
    case 'en-de':
    default:
      return 'The weather is nice today.';
  }
}
