/// T5-small / FLAN-T5-small text-to-text inference demo.
///
///   dart run bin/t5_small_demo.dart                                  # CPU
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib \
///     dart run bin/t5_small_demo.dart --gpu                          # GPU
///   dart run bin/t5_small_demo.dart \
///     --preset flan-t5-small \
///     --input-ids "8774,25,10,4712,5"       # "Hello, world."
///     --max-new 20
///
/// **Tokenizer note.** T5 uses SentencePiece (Unigram), which is not
/// implemented here yet. Until that lands, pass `--input-ids` with a
/// comma-separated list of int token ids from an external encoder
/// (e.g. Python: `AutoTokenizer.from_pretrained("google/flan-t5-small").
/// encode("summarize: ...")` then paste the ids). The demo runs the
/// full pipeline: encoder forward → greedy decoder loop → returns
/// integer id list.
///
/// One-time weight download (~240 MB fp32 for flan-t5-small):
///   mkdir -p models/flan-t5-small
///   for f in model.safetensors config.json tokenizer.json; do
///     curl -L -o "models/flan-t5-small/$f" \
///       "https://huggingface.co/google/flan-t5-small/resolve/main/$f"
///   done
library;

import 'dart:io';

import 'package:dart_pytorch/core/data/t5_sp_tokenizer.dart';
import 'package:dart_pytorch/dart_pytorch.dart';

const _weightsDefault = 'models/flan-t5-small/model.safetensors';
const _tokenizerDefault = 'models/flan-t5-small/tokenizer.json';

Future<void> main(List<String> args) async {
  var weightsPath = _weightsDefault;
  var tokenizerPath = _tokenizerDefault;
  var preset = 'flan-t5-small';
  var maxNew = 20;
  var useGpu = false;
  String? text;
  List<int>? inputIds;
  for (int i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--gpu':
        useGpu = true;
        break;
      case '--preset':
        preset = args[++i];
        break;
      case '--weights':
        weightsPath = args[++i];
        break;
      case '--tokenizer':
        tokenizerPath = args[++i];
        break;
      case '--text':
        text = args[++i];
        break;
      case '--max-new':
        maxNew = int.parse(args[++i]);
        break;
      case '--input-ids':
        inputIds = args[++i]
            .split(',')
            .map((s) => int.parse(s.trim()))
            .toList();
        break;
    }
  }

  final device = useGpu ? Device.GPU : Device.CPU;
  T5Config cfg;
  switch (preset) {
    case 't5-small':
      cfg = T5HFLoader.t5SmallConfig(device: device);
      break;
    case 't5-v1_1-small':
      cfg = T5HFLoader.t5V11SmallConfig(device: device);
      break;
    case 'flan-t5-small':
      cfg = T5HFLoader.flanT5SmallConfig(device: device);
      break;
    case 'flan-t5-base':
      cfg = T5HFLoader.flanT5BaseConfig(device: device);
      break;
    case 'codet5p-220m':
      cfg = T5HFLoader.codeT5pBaseConfig(device: device);
      break;
    default:
      stderr.writeln(
        'unknown preset "$preset"; use '
        't5-small | t5-v1_1-small | flan-t5-small | flan-t5-base | '
        'codet5p-220m',
      );
      exit(64);
  }

  print(
    '== $preset '
    '(dModel=${cfg.dModel}, layers=${cfg.numLayers}, heads=${cfg.numHeads}, '
    'dKv=${cfg.dKv}, ffn=${cfg.feedForwardProj.name}) ==',
  );

  final swBuild = Stopwatch()..start();
  final model = T5Model(cfg);
  swBuild.stop();
  print('build: ${swBuild.elapsedMilliseconds} ms');

  if (!File(weightsPath).existsSync()) {
    stderr.writeln('');
    stderr.writeln(
      'warning: $weightsPath not found — running with random init',
    );
    stderr.writeln('         (output will be gibberish; download real weights');
    stderr.writeln('         via the header docstring for real inference).');
  } else {
    final swLoad = Stopwatch()..start();
    final report = T5HFLoader.loadFile(model, weightsPath);
    swLoad.stop();
    print('load : ${swLoad.elapsedMilliseconds} ms  $report');
  }

  if (inputIds == null) {
    if (text != null && File(tokenizerPath).existsSync()) {
      final tok = T5SpTokenizer.loadFile(tokenizerPath);
      inputIds = tok.encode(text);
      print('');
      print('== tokenizer ($tokenizerPath) ==');
      print('  "$text" -> $inputIds');
    } else {
      inputIds = [37, 3, 1200, 6, 1];
      print('');
      print('(no --text or tokenizer; using synthetic $inputIds)');
    }
  }

  print('');
  print('== encode + greedy decode ==');
  final swG = Stopwatch()..start();
  final out = model.generate(inputIds, maxNewTokens: maxNew);
  swG.stop();
  final newTokens = out.sublist(1);
  print(
    '${swG.elapsedMilliseconds} ms  '
    '(${newTokens.length} tokens, '
    '${(newTokens.length * 1000.0 / swG.elapsedMilliseconds).toStringAsFixed(1)} tok/s)',
  );

  if (File(tokenizerPath).existsSync()) {
    final tok = T5SpTokenizer.loadFile(tokenizerPath);
    print('');
    print('== decoded ==');
    print('  "${tok.decode(newTokens)}"');
  }
  print('');
  print('output ids: $out');
}
