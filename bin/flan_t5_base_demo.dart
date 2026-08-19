/// FLAN-T5-base end-to-end text-to-text demo (250 M params).
///
/// Same encoder-decoder architecture as [bin/t5_small_demo.dart] but
/// the base-size checkpoint (12+12 layers, dModel=768, 12 heads,
/// gated-GELU FFN). Delivers noticeably better translation and
/// summarization than -small at ~3x the wall time per token on CPU.
///
///   dart run bin/flan_t5_base_demo.dart              # CPU, greedy
///   LD_LIBRARY_PATH=/usr/lib/wsl/lib \
///     dart run bin/flan_t5_base_demo.dart --gpu     # GPU
///   dart run bin/flan_t5_base_demo.dart \
///     --text "translate English to German: I love machine learning."
///
/// One-time weight download (~990 MB fp32):
///   mkdir -p models/flan-t5-base
///   for f in model.safetensors tokenizer.json config.json; do
///     curl -L -o "models/flan-t5-base/$f" \
///       "https://huggingface.co/google/flan-t5-base/resolve/main/$f"
///   done
library;

import 'dart:io';

import 'package:dart_pytorch/core/data/t5_sp_tokenizer.dart';
import 'package:dart_pytorch/dart_pytorch.dart';

const _weightsDefault = 'models/flan-t5-base/model.safetensors';
const _tokenizerDefault = 'models/flan-t5-base/tokenizer.json';

Future<void> main(List<String> args) async {
  var weightsPath = _weightsDefault;
  var tokenizerPath = _tokenizerDefault;
  var text = 'translate English to German: I love machine learning.';
  var maxNew = 40;
  var useGpu = false;
  var noCache = false;
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
      case '--weights':
        weightsPath = args[++i];
        break;
      case '--tokenizer':
        tokenizerPath = args[++i];
        break;
    }
  }
  for (final p in [weightsPath, tokenizerPath]) {
    if (!File(p).existsSync()) {
      stderr.writeln('missing: $p');
      stderr.writeln(
        'download flan-t5-base with:\n'
        '  mkdir -p models/flan-t5-base\n'
        '  for f in model.safetensors tokenizer.json; do\n'
        '    curl -L -o "models/flan-t5-base/\$f" \\\n'
        '      "https://huggingface.co/google/flan-t5-base/resolve/main/\$f"\n'
        '  done',
      );
      exit(2);
    }
  }

  final device = useGpu ? Device.GPU : Device.CPU;
  final cfg = T5HFLoader.flanT5BaseConfig(device: device);
  print(
    '== flan-t5-base '
    '(dModel=${cfg.dModel}, layers=${cfg.numLayers}, heads=${cfg.numHeads}, '
    'dKv=${cfg.dKv}, ffn=${cfg.feedForwardProj.name}) ==',
  );

  final swBuild = Stopwatch()..start();
  final model = T5Model(cfg);
  swBuild.stop();
  print('build: ${swBuild.elapsedMilliseconds} ms');

  final swLoad = Stopwatch()..start();
  final report = T5HFLoader.loadFile(model, weightsPath);
  swLoad.stop();
  print('load : ${swLoad.elapsedMilliseconds} ms  $report');

  final tok = T5SpTokenizer.loadFile(tokenizerPath);
  final ids = tok.encode(text);
  print('');
  print('== prompt ==');
  print('  "$text"');
  print('  -> $ids');

  print('');
  print('== greedy decode (${noCache ? "no cache" : "KV cache"}) ==');
  final swG = Stopwatch()..start();
  final out = model.generate(
    ids,
    maxNewTokens: maxNew,
    useCache: !noCache,
  );
  swG.stop();
  final newTokens = out.sublist(1);
  print(
    '  ${swG.elapsedMilliseconds} ms  '
    '(${newTokens.length} tokens, '
    '${(newTokens.length * 1000.0 / swG.elapsedMilliseconds).toStringAsFixed(1)} tok/s)',
  );

  print('');
  print('== output ==');
  print('  "${tok.decode(newTokens)}"');
  print('  ids: $out');
}
