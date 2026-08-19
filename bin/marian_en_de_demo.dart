/// Marian NMT (Helsinki-NLP Opus-MT) English -> German translation.
///
/// ~74 M dense encoder-decoder. Genuinely better than FLAN-T5-small
/// for pure translation.
///
///   dart run bin/marian_en_de_demo.dart           # CPU
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib \
///     dart run bin/marian_en_de_demo.dart --gpu   # GPU
///   dart run bin/marian_en_de_demo.dart \
///     --text "The weather is nice today."
///
/// One-time setup (~300 MB fp32):
///   mkdir -p models/opus-mt-en-de
///   for f in pytorch_model.bin vocab.json config.json; do
///     curl -L -o "models/opus-mt-en-de/$f" \
///       "https://huggingface.co/Helsinki-NLP/opus-mt-en-de/resolve/main/$f"
///   done
///   python3 scripts/convert_marian_pt_to_safetensors.py \
///     models/opus-mt-en-de/pytorch_model.bin \
///     models/opus-mt-en-de/model.safetensors
///
/// Tokenizer note: Marian ships a SentencePiece BPE model in
/// `source.spm` (binary protobuf). We don't parse it; instead we
/// read the plain-text `vocab.json` and do greedy longest-match
/// with the `▁` metaspace marker. Close to SPM output for most
/// well-formed English inputs, occasional divergence on rare
/// substrings.
library;

import 'dart:io';

import 'package:dart_pytorch/core/data/marian_vocab_tokenizer.dart';
import 'package:dart_pytorch/dart_pytorch.dart';

const _weightsDefault = 'models/opus-mt-en-de/model.safetensors';
const _vocabDefault = 'models/opus-mt-en-de/vocab.json';

Future<void> main(List<String> args) async {
  var weightsPath = _weightsDefault;
  var vocabPath = _vocabDefault;
  var text = 'The weather is nice today.';
  var maxNew = 60;
  var useGpu = false;
  var noCache = false;
  var numBeams = 1;
  var lengthPenalty = 0.6;
  for (int i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--gpu':
        useGpu = true;
        break;
      case '--no-cache':
        noCache = true;
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
      case '--weights':
        weightsPath = args[++i];
        break;
      case '--vocab':
        vocabPath = args[++i];
        break;
    }
  }
  for (final p in [weightsPath, vocabPath]) {
    if (!File(p).existsSync()) {
      stderr.writeln('missing: $p');
      stderr.writeln(
        'download + convert with:\n'
        '  mkdir -p models/opus-mt-en-de\n'
        '  for f in pytorch_model.bin vocab.json config.json; do\n'
        '    curl -L -o "models/opus-mt-en-de/\$f" \\\n'
        '      "https://huggingface.co/Helsinki-NLP/opus-mt-en-de/resolve/main/\$f"\n'
        '  done\n'
        '  python3 scripts/convert_marian_pt_to_safetensors.py \\\n'
        '    models/opus-mt-en-de/pytorch_model.bin \\\n'
        '    models/opus-mt-en-de/model.safetensors',
      );
      exit(2);
    }
  }

  final device = useGpu ? Device.GPU : Device.CPU;
  final cfg = MarianHFLoader.opusMtEnDeConfig(device: device);
  print(
    '== opus-mt-en-de '
    '(dModel=${cfg.dModel}, layers=${cfg.numLayers}+${cfg.numDecoderLayers}, '
    'heads=${cfg.numHeads}, ffn=${cfg.ffnDim}, act=${cfg.activation.name}) ==',
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
  print('== prompt ==');
  print('  "$text"');
  print('  -> $ids');

  print('');
  print(
    '== ${numBeams == 1 ? "greedy" : "beam=$numBeams"} decode '
    '(${noCache && numBeams == 1 ? "no cache" : "KV cache"}) ==',
  );
  final swG = Stopwatch()..start();
  final out = numBeams == 1
      ? model.generate(ids, maxNewTokens: maxNew, useCache: !noCache)
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
